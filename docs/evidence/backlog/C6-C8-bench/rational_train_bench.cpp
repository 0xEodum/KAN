// Backlog C6-C8: complete training steps (train_step) of rational networks,
// the rational counterpart of C9-bench/train_bench.cpp (matched protocol: the
// input and upstream gradient stay resident; one step = forward + backward +
// SGD captured as one CUDA graph, status checked every step).
// Every layer is RationalConfig{3, 2} with the given denominator policy,
// initialized by kan::initialize(VarianceScaling, seed 1, denominator radius 4).
// Usage: rational_train_bench <f64|f32> <guarded|absolute|smooth> [windows] [case]
// Output: one CSV row per timing window; the checksum (sum of all trained
// parameters after the last window) must agree between builds that compute
// bitwise-identical steps.
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <variant>
#include <vector>

namespace {
struct Case {
    const char* name;
    std::vector<std::size_t> widths;
    std::size_t batch;
    int steps_f64, steps_f32; // timed steps per window
};

kan::Network network(const Case& c, kan::DenominatorPolicy policy) {
    std::vector<kan::NetworkLayer> layers;
    for (std::size_t j = 0; j + 1 < c.widths.size(); ++j) {
        kan::RationalConfig config;
        config.denominator_policy = policy;
        layers.emplace_back(kan::Layer(c.widths[j], c.widths[j+1], config));
    }
    kan::Network n(std::move(layers));
    kan::VarianceScaling init;
    init.seed = 1;
    // Pole-free for |z| <= 4 (hidden activations stay well inside), so the
    // guarded policy trains through every timed step.
    init.denominators.radius = 4;
    kan::initialize(n, init);
    return n;
}
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: rational_train_bench f64|f32 guarded|absolute|smooth [windows] [case]\n"); return 2; }
    const std::string precision_name = argv[1], policy_name = argv[2];
    const int windows = argc > 3 ? std::atoi(argv[3]) : 3;
    const std::string only = argc > 4 ? argv[4] : "";
    const bool f64 = precision_name == "f64";
    const auto precision = f64 ? kan::cuda::Precision::Float64 : kan::cuda::Precision::Float32;
    const auto policy = policy_name == "absolute" ? kan::DenominatorPolicy::Absolute
                      : policy_name == "smooth"   ? kan::DenominatorPolicy::Smooth
                                                  : kan::DenominatorPolicy::Guarded;
    const Case cases[] = {
        {"64x64x32x16-b1024", {64, 64, 32, 16}, 1024, 50, 200},
        {"256x256x256x10-b2048", {256, 256, 256, 10}, 2048, 3, 20},
        {"256x256x256x10-b8192", {256, 256, 256, 10}, 8192, 0, 5},
    };
    // A small rate: the upstream is fixed, so a large one drifts the guarded
    // denominators into a pole within the timed steps; the work is the same.
    constexpr double rate = 1e-5;
    std::printf("impl,precision,policy,case,window,steps,ms_per_step,checksum\n");
    for (const auto& c : cases) {
        if (!only.empty() && only != c.name) continue;
        const int steps = f64 ? c.steps_f64 : c.steps_f32;
        if (steps == 0) continue; // FP64 caches of the 8192 batch exceed the 24 GB device
        try {
            kan::cuda::ResidentNetwork r(network(c, policy), c.batch, precision);
            const auto inputs = c.widths.front(), outputs = c.widths.back();
            r.upload_input(wave(c.batch*inputs, 0.95, 0.113, 0.0), c.batch);
            r.upload_output_gradient(wave(c.batch*outputs, 1.0/static_cast<double>(c.batch), 0.07, 0.1));
            const int warmup = std::max(2, steps/10);
            for (int i = 0; i < warmup; ++i) r.train_step(rate);
            r.synchronize();
            for (int w = 0; w < windows; ++w) {
                const auto t = std::chrono::steady_clock::now();
                for (int i = 0; i < steps; ++i) r.train_step(rate);
                r.synchronize();
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count()/steps;
                double checksum = 0;
                if (w + 1 == windows) {
                    const auto trained = r.download_parameters();
                    for (const auto& stage : trained.layers()) {
                        const auto& l = std::get<kan::Layer>(stage);
                        for (double v : l.coefficients()) checksum += v;
                        for (double v : std::get<kan::RationalEdges>(l.carrier()).denominators) checksum += v;
                    }
                }
                std::printf("kan,%s,%s,%s,%d,%d,%.4f,%.17g\n", precision_name.c_str(), policy_name.c_str(), c.name, w, steps, ms, checksum);
                std::fflush(stdout);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s %s %s: %s\n", precision_name.c_str(), policy_name.c_str(), c.name, e.what());
        }
    }
}

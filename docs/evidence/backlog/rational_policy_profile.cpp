// M2 profiling harness: one denominator policy, the m4_benchmark fixture
// (64x64x32x16, batch 1024, scale 1.2, center 0.1), degrees [1/1], [3/2] or
// [6/4]. Prints the median of `repeats` full CPU and resident steps
// (forward + backward + SGD). Under ncu, the resident steps give one launch of
// each rational kernel per layer and step.
// Usage: rational_policy_profile <guarded|absolute|smooth> <0|1|2> [repeats] [cpu|resident|all]
// Build: build_profile.cmd <build-dir> <output-exe>
#include "kan/families.hpp"
#include "kan/network.hpp"
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#ifdef KAN_NO_POLICY
namespace kan { enum class DenominatorPolicy { Guarded, Absolute, Smooth }; }
#endif

namespace {
using Clock = std::chrono::steady_clock;
kan::Network network(kan::DenominatorPolicy policy, int family) {
    kan::RationalConfig config;
    config.numerator_degree = family == 0 ? 1 : family == 1 ? 3 : 6;
    config.denominator_degree = family == 0 ? 1 : family == 1 ? 2 : 4;
    config.center = 0.1; config.scale = 1.2;
#ifndef KAN_NO_POLICY // defined to build against the pre-M2 baseline (Guarded only)
    config.denominator_policy = policy;
#endif
    const std::size_t widths[] = {64, 64, 32, 16};
    std::vector<kan::Layer> layers;
    for (std::size_t l = 1; l < 4; ++l) {
        kan::Layer layer(widths[l-1], widths[l], config);
        std::vector<double> a(layer.coefficients().size()), bias(layer.outputs());
        std::vector<double> b(std::get<kan::RationalEdges>(layer.carrier()).denominators.size());
        for (std::size_t j = 0; j < a.size(); ++j)
            a[j] = 0.02 * std::sin(static_cast<double>((j + 1) * (l + 1))) /
                   (static_cast<double>(layer.inputs()) * static_cast<double>(1 + j % (config.numerator_degree + 1)));
        for (std::size_t j = 0; j < bias.size(); ++j) bias[j] = 0.01 * std::cos(static_cast<double>(j + l));
        for (std::size_t j = 0; j < b.size(); ++j) b[j] = 0.01 * std::cos(static_cast<double>((j + 3) * (l + 1)));
        kan::set_rational_parameters(layer, a, b, bias);
        layers.push_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
std::vector<double> data(std::size_t count, double scale, std::size_t offset) {
    std::vector<double> result(count);
    for (std::size_t j = 0; j < count; ++j) result[j] = scale * std::sin(static_cast<double>((j + offset) % 997) * 0.071);
    return result;
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
double best(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
} // namespace

int run(int argc, char** argv) {
    if (argc < 3) { std::puts("usage: rational_policy_profile <guarded|absolute|smooth> <0|1|2> [repeats] [cpu|resident|all]"); return 2; }
    const std::string name = argv[1];
    const auto policy = name == "absolute" ? kan::DenominatorPolicy::Absolute
                      : name == "smooth" ? kan::DenominatorPolicy::Smooth : kan::DenominatorPolicy::Guarded;
    const int family = std::atoi(argv[2]);
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 15;
    const std::string backend = argc > 4 ? argv[4] : "all";
    const std::size_t batch = 1024;
    const auto input = data(batch * 64, 0.75, 1), upstream = data(batch * 16, 0.02 / static_cast<double>(batch), 23);
    if (backend != "resident") {
        auto cpu = network(policy, family);
        std::vector<double> times;
        for (int r = 0; r < repeats; ++r) {
            const auto t = Clock::now();
            cpu.forward(input, batch);
            cpu.sgd(cpu.backward(input, batch, upstream), 0.001);
            times.push_back(ms(t));
        }
        std::printf("%s [%d] cpu_step_ms median %.3f best %.3f\n", name.c_str(), family, median(times), best(times));
    }
    if (backend != "cpu") {
        kan::cuda::ResidentNetwork gpu(network(policy, family), batch);
        gpu.upload_input(input, batch); gpu.upload_output_gradient(upstream);
        std::vector<double> times;
        for (int r = 0; r < repeats + 2; ++r) {
            const auto t = Clock::now();
            gpu.forward(); gpu.backward(); gpu.sgd(0.001); gpu.synchronize();
            if (r >= 2) times.push_back(ms(t));
        }
        gpu.forward(); // SGD invalidated the output of the last step
        const auto out = gpu.download_output();
        double checksum = 0;
        for (double v : out) checksum += v;
        std::printf("%s [%d] resident_step_ms median %.3f best %.3f checksum %.17g\n", name.c_str(), family,
                    median(times), best(times), checksum);
    }
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

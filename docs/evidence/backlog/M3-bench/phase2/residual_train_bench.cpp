// Backlog M3 phase 2: full train_step timing with the SiLU residual branch
// disabled or enabled on every KAN layer. Protocol of C9's `matched` mode
// (C9-bench/train_bench.cpp): x and the upstream gradient stay resident, a
// step is forward + backward + SGD as one graph replay, status interval 1.
// Topologies: the four C9 Chebyshev K=7 cases, plus a rational [3/2] Smooth
// and a trainable-RBF (8 centers) 256x256x256x10 network at batch 2048.
// Usage: residual_train_bench <f64|f32|tf32> <branch 0|1> [windows] [case]
// One CSV row per timing window. The same source is built against the
// baseline (2437421, branch 0 only: it rejects the branch) and phase 2.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

namespace {
enum class Carrier { Chebyshev, Rational, TrainableRbf };
struct Case {
    const char* name;
    Carrier carrier;
    std::vector<std::size_t> widths;
    std::size_t batch;
    int steps64, steps32; // timed steps per window (FP64, FP32/TF32)
    double rate;
};
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
kan::Network network(const Case& c, bool branch) {
    std::vector<kan::NetworkLayer> layers;
    std::uint64_t state = 12345;
    auto normal = [&] {
        auto uniform = [&] { state = state*6364136223846793005ULL + 1442695040888963407ULL; return ((state >> 11) + 0.5)/9007199254740992.0; };
        return std::sqrt(-2*std::log(uniform()))*std::cos(6.283185307179586*uniform());
    };
    for (std::size_t j = 0; j + 1 < c.widths.size(); ++j) {
        const auto in = c.widths[j], out = c.widths[j+1];
        kan::Layer layer = [&] {
            if (c.carrier == Carrier::Chebyshev) return kan::Layer(in, out, kan::ChebyshevConfig{7});
            if (c.carrier == Carrier::TrainableRbf) {
                kan::TrainableRbfConfig rbf{wave(8, 1.0, 0.0, 0.0), std::vector<double>(8, std::log(0.3))};
                for (std::size_t k = 0; k < 8; ++k) rbf.centers[k] = -1.0 + 2.0*static_cast<double>(k)/7.0;
                return kan::Layer(in, out, rbf);
            }
            kan::RationalConfig r;
            r.numerator_degree = 3; r.denominator_degree = 2; r.denominator_policy = kan::DenominatorPolicy::Smooth;
            return kan::Layer(in, out, r);
        }();
        std::vector<double> coefficients(layer.coefficients().size()), bias(out, 0.0);
        const double scale = 0.1/std::sqrt(static_cast<double>(in*layer.terms()));
        for (auto& v : coefficients) v = normal()*scale;
        if (c.carrier == Carrier::Rational)
            kan::set_rational_parameters(layer, coefficients, wave(in*out*2, 0.1, 0.37, 0.2), bias);
        else
            layer.set_parameters(coefficients, bias);
        if (branch) layer.set_residual(kan::SiluResidual{wave(in*out, 0.1/std::sqrt(static_cast<double>(in)), 0.917, 0.3)});
        layers.emplace_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: residual_train_bench f64|f32|tf32 0|1 [windows] [case]\n"); return 2; }
    const std::string precision_name = argv[1];
    const bool branch = std::atoi(argv[2]) != 0;
    const int windows = argc > 3 ? std::atoi(argv[3]) : 3;
    const std::string only = argc > 4 ? argv[4] : "";
    const auto precision = precision_name == "f32"    ? kan::cuda::Precision::Float32
                           : precision_name == "tf32" ? kan::cuda::Precision::TensorFloat32
                                                      : kan::cuda::Precision::Float64;
    const bool f64 = precision == kan::cuda::Precision::Float64;
    const Case cases[] = {
        {"64x64x32x16-b1024", Carrier::Chebyshev, {64, 64, 32, 16}, 1024, 200, 400, 1e-3},
        {"16x24x8-b1024", Carrier::Chebyshev, {16, 24, 8}, 1024, 200, 400, 1e-3},
        {"256x256x256x10-b8192", Carrier::Chebyshev, {256, 256, 256, 10}, 8192, 10, 50, 1e-3},
        {"1024x1024x1024-b4096", Carrier::Chebyshev, {1024, 1024, 1024}, 4096, 2, 20, 1e-3},
        {"rational-256x256x256x10-b2048", Carrier::Rational, {256, 256, 256, 10}, 2048, 3, 30, 1e-5},
        {"rbf-256x256x256x10-b2048", Carrier::TrainableRbf, {256, 256, 256, 10}, 2048, 10, 50, 1e-3},
    };
    std::printf("precision,branch,case,window,steps,ms_per_step,checksum\n");
    for (const auto& c : cases) {
        if (!only.empty() && only != c.name) continue;
        const int steps = f64 ? c.steps64 : c.steps32;
        kan::cuda::ResidentNetwork r(network(c, branch), c.batch, precision);
        const auto inputs = c.widths.front(), outputs = c.widths.back();
        r.upload_input(wave(c.batch*inputs, 0.95, 0.113, 0.0), c.batch);
        r.upload_output_gradient(wave(c.batch*outputs, 1.0/static_cast<double>(c.batch), 0.07, 0.1));
        const int warmup = std::max(3, steps/10);
        for (int i = 0; i < warmup; ++i) r.train_step(c.rate);
        r.synchronize();
        for (int w = 0; w < windows; ++w) {
            const auto t = std::chrono::steady_clock::now();
            for (int i = 0; i < steps; ++i) r.train_step(c.rate);
            r.synchronize();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count()/steps;
            double checksum = 0;
            if (w + 1 == windows) {
                const auto trained = r.download_parameters(); // keep the network alive while iterating
                for (const auto& stage : trained.layers())
                    for (double v : std::get<kan::Layer>(stage).coefficients()) checksum += v;
            }
            std::printf("%s,%d,%s,%d,%d,%.4f,%.10g\n", precision_name.c_str(), branch ? 1 : 0, c.name, w, steps, ms, checksum);
            std::fflush(stdout);
        }
    }
}

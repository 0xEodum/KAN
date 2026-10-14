// Backlog M5: complete-call cost of normalized Hermite functions against the
// physicists' Hermite polynomials on the resident executor (train_step) and the
// CPU (Network backward + SGD).
//
// Network: KAN layers of the given widths, every one HermiteConfig{terms} with
// the given normalization, a tanh input map between consecutive KAN layers;
// parameters from VarianceScaling{1, Uniform, seed 7} (each normalization under
// its own reference moments). Inputs 0.95*sin(...), upstream 1e-3*cos(...).
// Inputs stay in (-1, 1), so the Hermite-function rows never rescale (they do
// only from |x| ~ 9 in FP32 / 27 in FP64); the cost compared is the per-term
// recurrence.
// Protocol (C9 matched, as C5-bench/spline_bench.cpp): input and upstream
// uploaded once; a step is train_step(rate, 0, Loss::OutputGradient); steps
// per window chosen so a window lasts about 0.4 s (3..200). `cpu` times
// Network::backward + Network::sgd per step on the host instead.
// Usage: hermite_bench <f64|f32|tf32|cpu> <physicists|function> <widths> <batch> <terms> [windows=3] [rate=1e-3]
// (the timing sweep uses rate 1e-12: physicists' K = 32 diverges at 1e-3 in FP32 within
// three steps, see the conditioning demonstration; the step cost does not depend on it).
// Build: scripts/link_harness.sh <build> docs/evidence/backlog/M5-bench/hermite_bench.cpp <exe>
// One CSV row per window: precision,normalization,topology,batch,terms,window,steps,step_ms,checksum
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::vector<std::size_t> widths_of(const std::string& text) {
    std::vector<std::size_t> widths;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, 'x')) widths.push_back(std::stoul(item));
    return widths;
}
double checksum_of(const kan::Network& network) {
    double sum = 0;
    for (const auto& stage : network.layers())
        if (const auto* l = std::get_if<kan::Layer>(&stage))
            for (double v : l->coefficients()) sum += v;
    return sum;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: hermite_bench f64|f32|tf32|cpu physicists|function <widths> <batch> <terms> [windows] [rate]\n");
        return 2;
    }
    const std::string precision_name = argv[1], normalization_name = argv[2], topology = argv[3];
    const std::size_t batch = std::stoul(argv[4]), terms = std::stoul(argv[5]);
    const int windows = argc > 6 ? std::atoi(argv[6]) : 3;
    const double rate = argc > 7 ? std::atof(argv[7]) : 1e-3;
    const auto normalization = normalization_name == "function" ? kan::HermiteNormalization::Function
                                                                : kan::HermiteNormalization::Physicists;
    const auto widths = widths_of(topology);
    std::vector<kan::NetworkLayer> layers;
    for (std::size_t j = 0; j + 1 < widths.size(); ++j) {
        if (j) layers.emplace_back(kan::InputMap(widths[j], kan::TanhMap{}));
        layers.emplace_back(kan::Layer(widths[j], widths[j + 1], kan::HermiteConfig{terms, normalization}));
    }
    kan::Network network(std::move(layers));
    kan::initialize(network, kan::VarianceScaling{1, kan::Distribution::Uniform, 7, {}});
    std::vector<double> x(batch * widths.front()), dy(batch * widths.back());
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = 0.95 * std::sin(0.377 * static_cast<double>(k) + 0.2);
    for (std::size_t k = 0; k < dy.size(); ++k) dy[k] = 1e-3 * std::cos(0.613 * static_cast<double>(k));
    std::printf("precision,normalization,topology,batch,terms,window,steps,step_ms,checksum\n");
    if (precision_name == "cpu") {
        const auto step = [&] { network.sgd(network.backward(x, batch, dy), rate); };
        step();
        auto t = Clock::now();
        step();
        const double probe = since(t);
        step();
        const double checksum = checksum_of(network);
        const int steps = static_cast<int>(std::clamp(400.0 / std::max(probe, 1e-3), 3.0, 200.0));
        for (int w = 0; w < windows; ++w) {
            t = Clock::now();
            for (int i = 0; i < steps; ++i) step();
            std::printf("cpu,%s,%s,%zu,%zu,%d,%d,%.4f,%.12g\n", normalization_name.c_str(), topology.c_str(), batch, terms, w,
                        steps, since(t) / steps, checksum);
        }
        return 0;
    }
    const auto precision = precision_name == "f64" ? kan::cuda::Precision::Float64
                         : precision_name == "f32" ? kan::cuda::Precision::Float32 : kan::cuda::Precision::TensorFloat32;
    kan::cuda::ResidentNetwork gpu(network, batch, precision);
    gpu.upload_input(x, batch);
    gpu.upload_output_gradient(dy);
    gpu.train_step(rate); gpu.train_step(rate); gpu.synchronize();
    auto t = Clock::now();
    gpu.train_step(rate); gpu.synchronize();
    const double probe = since(t);
    const double checksum = checksum_of(gpu.download_parameters());
    const int steps = static_cast<int>(std::clamp(400.0 / std::max(probe, 1e-3), 3.0, 200.0));
    for (int w = 0; w < windows; ++w) {
        t = Clock::now();
        for (int i = 0; i < steps; ++i) gpu.train_step(rate);
        gpu.synchronize();
        std::printf("%s,%s,%s,%zu,%zu,%d,%d,%.4f,%.12g\n", precision_name.c_str(), normalization_name.c_str(), topology.c_str(),
                    batch, terms, w, steps, since(t) / steps, checksum);
    }
    return 0;
}

// Backlog C5: complete-call benchmark of B-spline networks on the resident
// executor, dense rows against the sparse path (forced through the private
// test hook src/backends/cuda/resident_testing.hpp), for the crossover sweep.
//
// Network: KAN layers of the given widths, every one a clamped B-spline of the
// given degree and term count on [-1, 1] (uniform interior knots), with a tanh
// input map between consecutive KAN layers so hidden inputs stay in the domain;
// deterministic coefficients 0.3*sin(...)/sqrt(inputs*(degree+1)), bias 0.01.
// Protocol (C9 matched): input and upstream uploaded once; a step is
// train_step(rate, 0, Loss::OutputGradient) (forward + backward + SGD as one
// graph replay, status interval 1). `forward` times eager forward() calls
// (including their status synchronization) on the same executor.
// Steps per window are chosen from one timed step so that a window lasts about
// 0.4 s (at least 3, at most 200). The checksum is the sum of all coefficients
// after the first three steps, comparable between paths and builds.
// Usage: spline_bench <f64|f32|tf32> <auto|dense|sparse> <widths, e.g. 256x256x256x10> <batch>
//                     <degree> <terms> [windows=3]
// Build: scripts/link_harness.sh <build> docs/evidence/backlog/C5-bench/spline_bench.cpp <exe> -I src
// One CSV row per window:
// precision,path,topology,batch,degree,terms,sparse_layers,arena_mb,window,steps,step_ms,forward_ms,checksum
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
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

kan::BSplineConfig spline(std::size_t degree, std::size_t terms) {
    kan::BSplineConfig b{degree, std::vector<double>(degree + 1, -1.0)};
    const auto interior = terms - degree - 1;
    for (std::size_t k = 1; k <= interior; ++k) b.knots.push_back(-1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(interior + 1));
    b.knots.insert(b.knots.end(), degree + 1, 1.0);
    return b;
}
std::vector<std::size_t> widths_of(const std::string& text) {
    std::vector<std::size_t> widths;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, 'x')) widths.push_back(std::stoul(item));
    return widths;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: spline_bench f64|f32|tf32 auto|dense|sparse <widths> <batch> <degree> <terms> [windows]\n");
        return 2;
    }
    const std::string precision_name = argv[1], path_name = argv[2], topology = argv[3];
    const std::size_t batch = std::stoul(argv[4]), degree = std::stoul(argv[5]), terms = std::stoul(argv[6]);
    const int windows = argc > 7 ? std::atoi(argv[7]) : 3;
    const auto precision = precision_name == "f64" ? kan::cuda::Precision::Float64
                         : precision_name == "f32" ? kan::cuda::Precision::Float32 : kan::cuda::Precision::TensorFloat32;
    const auto path = path_name == "dense" ? kan::cuda::testing::SplinePath::Dense
                    : path_name == "sparse" ? kan::cuda::testing::SplinePath::Sparse : kan::cuda::testing::SplinePath::Automatic;
    const auto widths = widths_of(topology);
    std::vector<kan::NetworkLayer> layers;
    for (std::size_t j = 0; j + 1 < widths.size(); ++j) {
        if (j) layers.emplace_back(kan::InputMap(widths[j], kan::TanhMap{}));
        kan::Layer layer(widths[j], widths[j + 1], spline(degree, terms));
        std::vector<double> c(layer.coefficients().size());
        const double scale = 0.3 / std::sqrt(static_cast<double>(widths[j] * (degree + 1)));
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = scale * std::sin(0.731 * static_cast<double>(k) + static_cast<double>(j));
        layer.set_parameters(c, std::vector<double>(layer.outputs(), 0.01));
        layers.emplace_back(std::move(layer));
    }
    const kan::Network network(std::move(layers));
    std::vector<double> x(batch * widths.front()), dy(batch * widths.back());
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = 0.95 * std::sin(0.377 * static_cast<double>(k) + 0.2);
    for (std::size_t k = 0; k < dy.size(); ++k) dy[k] = 1e-3 * std::cos(0.613 * static_cast<double>(k));

    kan::cuda::testing::ResidentLayout layout;
    kan::cuda::testing::ResidentTestScope scope(path, &layout);
    kan::cuda::ResidentNetwork gpu(network, batch, precision);
    std::size_t sparse_layers = 0;
    for (bool s : layout.sparse) sparse_layers += s;
    gpu.upload_input(x, batch);
    gpu.upload_output_gradient(dy);
    constexpr double rate = 1e-3;
    gpu.train_step(rate); gpu.train_step(rate); gpu.synchronize();
    auto t = Clock::now();
    gpu.train_step(rate); gpu.synchronize();
    const double probe = since(t);
    // Checksum: coefficient sum after exactly these three steps (comparable across paths).
    double checksum = 0;
    const auto trained = gpu.download_parameters();
    for (const auto& stage : trained.layers())
        if (const auto* l = std::get_if<kan::Layer>(&stage))
            for (double v : l->coefficients()) checksum += v;
    const int steps = static_cast<int>(std::clamp(400.0 / std::max(probe, 1e-3), 3.0, 200.0));
    std::printf("precision,path,topology,batch,degree,terms,sparse_layers,arena_mb,window,steps,step_ms,forward_ms,checksum\n");
    for (int w = 0; w < windows; ++w) {
        t = Clock::now();
        for (int i = 0; i < steps; ++i) gpu.train_step(rate);
        gpu.synchronize();
        const double step_ms = since(t) / steps;
        gpu.forward(); // warm the eager path
        t = Clock::now();
        for (int i = 0; i < steps; ++i) gpu.forward();
        const double forward_ms = since(t) / steps;
        std::printf("%s,%s,%s,%zu,%zu,%zu,%zu,%.1f,%d,%d,%.4f,%.4f,%.12g\n", precision_name.c_str(), path_name.c_str(), topology.c_str(),
                    batch, degree, terms, sparse_layers, static_cast<double>(layout.arena_bytes) / 1048576.0, w, steps, step_ms,
                    forward_ms, checksum);
    }
    return 0;
}

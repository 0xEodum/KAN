// M1 profiling harness: resident full steps (forward, backward, SGD) of the
// R3 reference topology 64->64->32->16, Chebyshev K=7, with an optional
// input map in front. Usage:
//   map_profile <none|affine|tanh|layernorm|layernorm-affine> [batch] [steps] [features]
// Prints the median step time over `steps` timed steps after 5 warmups.
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int run(int argc, char** argv) {
    const std::string kind = argc > 1 ? argv[1] : "none";
    const std::size_t batch = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1024;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 50;
    const std::size_t features = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 64;
    std::vector<kan::NetworkLayer> layers;
    std::vector<double> ones(features, 1.0), zeros(features, 0.0), small(features, 0.01);
    if (kind == "affine") layers.emplace_back(kan::InputMap(features, kan::AffineMap{small, zeros}));
    else if (kind == "tanh") layers.emplace_back(kan::InputMap(features, kan::TanhMap{0.01}));
    else if (kind == "layernorm") layers.emplace_back(kan::InputMap(features, kan::LayerNormMap{}));
    else if (kind == "layernorm-affine") layers.emplace_back(kan::InputMap(features, kan::LayerNormMap{1e-5, ones, zeros}));
    else if (kind != "none") { std::fprintf(stderr, "unknown map\n"); return 2; }
    const std::size_t widths[] = {features, 64, 32, 16};
    for (std::size_t l = 0; l + 1 < 4; ++l) {
        kan::Layer layer(widths[l], widths[l + 1], kan::ChebyshevConfig{7});
        std::vector<double> c(layer.coefficients().size()), b(layer.bias().size(), 0.01);
        // Scaled so that every layer output stays inside [-1, 1] for all map kinds.
        const double scale = 0.5 / static_cast<double>(widths[l] * 7);
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = scale * std::sin(0.37 * static_cast<double>(k));
        layer.set_parameters(c, b);
        layers.emplace_back(std::move(layer));
    }
    kan::Network network(std::move(layers));
    std::vector<double> x(batch * features), u(batch * 16);
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = 0.9 * std::sin(0.61 * static_cast<double>(k));
    for (std::size_t k = 0; k < u.size(); ++k) u[k] = 1e-3 * std::cos(0.7 * static_cast<double>(k));
    kan::cuda::ResidentNetwork gpu(network, batch);
    gpu.upload_input(x, batch);
    gpu.upload_output_gradient(u);
    std::vector<double> times;
    for (int step = 0; step < steps + 5; ++step) {
        const auto start = std::chrono::steady_clock::now();
        gpu.forward();
        gpu.backward();
        gpu.sgd(1e-3);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (step >= 5) times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    std::printf("%s batch=%zu features=%zu median_step_ms=%.4f\n", kind.c_str(), batch, features, times[times.size() / 2]);
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}

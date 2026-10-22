// Backlog M6: complete train_step of B-spline networks with one shared grid
// per layer against per-input grids (identical start functions: every input's
// grid is the shared grid stretched to its own domain), and the cost of a
// grid refit: device ResidentNetwork::refit_grids against the host route
// (download_parameters, CPU Network::refit_grids, executor reconstruction).
// Protocol as C9/M7 (input and output gradient uploaded once, a step is
// train_step(rate), status interval 1); per window `steps` steps, median of
// windows. The hash is FNV-1a over the downloaded parameters after 5 steps.
// Usage: grid_step <f64|f32|tf32> <widths, e.g. 256x256x256x10> <batch> <intervals> <degree>
//                  <shared|replicated|input> <auto|dense|sparse> [windows=5] [mode=step|refit]
// Build: scripts/link_harness.sh <build> docs/evidence/backlog/M6-bench/grid_step.cpp <exe> -Isrc
//        (base c06b2ca: add -DM6_BASE; shared grids and step mode only)
// CSV: precision,grid,path,widths,batch,intervals,degree,window,value_ms,hash
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double ms(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }
std::vector<std::size_t> widths_of(const std::string& text) {
    std::vector<std::size_t> w; std::stringstream s(text); std::string item;
    while (std::getline(s, item, 'x')) w.push_back(std::stoul(item));
    return w;
}
// Uniform clamped knots on [lo, hi].
std::vector<double> uniform(std::size_t p, std::size_t G, double lo, double hi) {
    std::vector<double> t(p, lo);
    for (std::size_t j = 0; j <= G; ++j) t.push_back(lo + (hi - lo) * double(j) / double(G));
    t.insert(t.end(), p, hi);
    return t;
}
// grid: "shared" (one grid on [-1, 1]), "replicated" (that grid repeated per
// input: the same functions on per-input grids) or "input" (input i on
// [-s_i, s_i], s_i = 1 + 0.02 (i mod 7)).
kan::Network network(const std::vector<std::size_t>& w, std::size_t G, std::size_t p, const std::string& grid) {
    std::vector<kan::NetworkLayer> layers;
    std::uint64_t state = 12345;
    auto normal = [&] {
        auto u = [&] { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return ((state >> 11) + 0.5) / 9007199254740992.0; };
        return std::sqrt(-2 * std::log(u())) * std::cos(6.283185307179586 * u());
    };
    for (std::size_t j = 0; j + 1 < w.size(); ++j) {
        kan::BSplineConfig c{p, uniform(p, G, -1, 1)};
#ifndef M6_BASE
        if (grid == "replicated") c = kan::per_input_grids(c, w[j]);
        if (grid == "input") {
            c.knots.clear(); c.grids = w[j];
            for (std::size_t i = 0; i < w[j]; ++i) {
                const double s = 1 + 0.02 * double(i % 7);
                const auto t = uniform(p, G, -s, s);
                c.knots.insert(c.knots.end(), t.begin(), t.end());
            }
        }
#endif
        kan::Layer layer(w[j], w[j + 1], c);
        std::vector<double> coefficients(layer.coefficients().size()), bias(w[j + 1], 0.0);
        const double scale = 0.5 / std::sqrt(double(w[j]));
        for (auto& v : coefficients) v = normal() * scale;
        layer.set_parameters(coefficients, bias);
        layers.emplace_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
std::uint64_t fnv(const kan::Network& n) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto& l : n.layers())
        for (double x : std::get<kan::Layer>(l).coefficients()) {
            std::uint64_t bits; std::memcpy(&bits, &x, sizeof bits);
            for (int b = 0; b < 8; ++b) { h ^= (bits >> (8 * b)) & 0xff; h *= 1099511628211ULL; }
        }
    return h;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: grid_step f64|f32|tf32 widths batch intervals degree shared|input auto|dense|sparse [windows] [step|refit]\n"); return 2; }
    const std::string prec = argv[1], grid = argv[6], path_name = argv[7];
    const auto w = widths_of(argv[2]);
    const std::size_t batch = std::stoul(argv[3]), G = std::stoul(argv[4]), p = std::stoul(argv[5]);
    const int windows = argc > 8 ? std::stoi(argv[8]) : 5;
    const std::string mode = argc > 9 ? argv[9] : "step";
    const auto precision = prec == "f64" ? kan::cuda::Precision::Float64 : prec == "f32" ? kan::cuda::Precision::Float32
                                                                                        : kan::cuda::Precision::TensorFloat32;
    using kan::cuda::testing::SplinePath;
    const auto path = path_name == "dense" ? SplinePath::Dense : path_name == "sparse" ? SplinePath::Sparse : SplinePath::Automatic;
    kan::cuda::testing::ResidentTestScope scope(path);
    const auto net = network(w, G, p, grid);
    std::vector<double> x(batch * w.front()), u(batch * w.back());
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = 0.95 * std::sin(0.7 * double(k) + 0.1 * double(k % 13));
    for (std::size_t k = 0; k < u.size(); ++k) u[k] = 1e-3 * std::cos(0.37 * double(k));
    std::uint64_t hash = 0;
    {
        kan::cuda::ResidentNetwork r(net, batch, precision);
        r.upload_input(x, batch); r.upload_output_gradient(u);
        for (int k = 0; k < 5; ++k) r.train_step(1e-3);
        hash = fnv(r.download_parameters());
    }
    kan::cuda::ResidentNetwork r(net, batch, precision);
    r.upload_input(x, batch); r.upload_output_gradient(u);
    const auto row = [&](int window, double value) {
        std::printf("%s,%s,%s,%s,%zu,%zu,%zu,%d,%.6f,%016llx\n", prec.c_str(), grid.c_str(), path_name.c_str(), argv[2], batch, G, p,
                    window, value, static_cast<unsigned long long>(hash));
    };
    if (mode == "step") {
        // Steps per window: about 200 ms of work.
        for (int k = 0; k < 10; ++k) r.train_step(1e-3);
        r.synchronize();
        auto t = clk::now();
        for (int k = 0; k < 10; ++k) r.train_step(1e-3);
        r.synchronize();
        const int steps = std::max(10, static_cast<int>(200.0 / std::max(ms(t) / 10, 1e-3)));
        for (int window = 0; window < windows; ++window) {
            t = clk::now();
            for (int k = 0; k < steps; ++k) r.train_step(1e-3);
            r.synchronize();
            row(window, ms(t) / steps);
        }
#ifndef M6_BASE
    } else if (mode == "refit") {
        // Device refit (per window: one refit, the executor otherwise idle).
        r.refit_grids(); r.synchronize();
        for (int window = 0; window < windows; ++window) {
            const auto t = clk::now();
            r.refit_grids();
            r.synchronize();
            row(window, ms(t));
        }
    } else {
        // Host route: download, CPU refit on the same input, rebuild, first upload.
        for (int window = 0; window < windows; ++window) {
            const auto t = clk::now();
            auto cpu = r.download_parameters();
            cpu.refit_grids(x, batch);
            kan::cuda::ResidentNetwork rebuilt(cpu, batch, precision);
            rebuilt.upload_input(x, batch); rebuilt.upload_output_gradient(u);
            rebuilt.synchronize();
            row(window, ms(t));
        }
#endif
    }
}

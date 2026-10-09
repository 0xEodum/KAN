// Backlog M3 evidence harness: the five-layer x1*x2 demonstration of
// tests/residual_training_test.cpp (and M4's tests/initializer_training_test.cpp)
// run longer, with and without the residual branch, printing the MSE curve.
// Usage: residual_training_long <carrier> <init> <branch 0|1> <epochs>
//   carrier: chebyshev | bspline | guarded | absolute | smooth
//   init:    noise | variance | zero
// Build: build_harness.cmd <build-dir> residual_training_long.cpp <exe>
#include "kan/initializers.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
struct Data { std::vector<double> x, target; std::size_t batch; };
Data product_data() {
    Data d{{}, {}, 64};
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j) {
            const double a = -0.9 + 1.8 * double(i) / 7, b = -0.9 + 1.8 * double(j) / 7;
            d.x.insert(d.x.end(), {a, b});
            d.target.push_back(a * b);
        }
    return d;
}
template<class Config> kan::Network deep_network(const Config& config, bool branch) {
    std::vector<kan::NetworkLayer> stages;
    std::size_t in = 2;
    for (std::size_t l = 0; l <= 4; ++l) {
        const std::size_t out = l == 4 ? 1 : 8;
        kan::Layer layer(in, out, config);
        if (branch) layer.set_residual(kan::SiluResidual{std::vector<double>(in * out, 0.0)});
        stages.emplace_back(kan::InputMap(in, kan::TanhMap{1.0}));
        stages.emplace_back(std::move(layer));
        in = out;
    }
    return kan::Network(std::move(stages));
}
bool report(std::size_t epoch) {
    for (std::size_t e : {0, 100, 300, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000})
        if (epoch == e) return true;
    return false;
}
template<class Config> void run(const Config& config, const std::string& init, bool branch, std::size_t epochs) {
    const auto d = product_data();
    auto net = deep_network(config, branch);
    if (init == "noise") kan::initialize(net, kan::NoiseInit{0.3, kan::Distribution::Uniform, 1, {}});
    else if (init == "variance") kan::initialize(net, kan::VarianceScaling{1, kan::Distribution::Uniform, 1, {}});
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t epoch = 0; epoch <= epochs; ++epoch) {
        const auto y = net.forward(d.x, d.batch);
        std::vector<double> upstream(d.batch);
        double s = 0;
        for (std::size_t b = 0; b < d.batch; ++b) {
            upstream[b] = 2 * (y[b] - d.target[b]) / double(d.batch);
            s += (y[b] - d.target[b]) * (y[b] - d.target[b]);
        }
        if (report(epoch) || epoch == epochs) std::printf(" %zu:%.3e", epoch, s / double(d.batch));
        if (epoch < epochs) net.sgd(net.backward(d.x, d.batch, upstream), 0.03);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("  (%.1f s)\n", seconds);
}
kan::RationalConfig rational(kan::DenominatorPolicy policy) {
    kan::RationalConfig c;
    c.denominator_policy = policy;
    return c;
}
kan::BSplineConfig spline() {
    std::vector<double> knots{-1, -1, -1};
    for (int j = 0; j <= 5; ++j) knots.push_back(-1 + 0.4 * j);
    knots.insert(knots.end(), {1, 1, 1});
    return {3, knots};
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 5) { std::puts("usage: residual_training_long <carrier> <init> <branch 0|1> <epochs>"); return 2; }
    const std::string carrier = argv[1], init = argv[2];
    const bool branch = std::string(argv[3]) == "1";
    const std::size_t epochs = std::strtoull(argv[4], nullptr, 10);
    std::printf("curve %-9s %-8s branch=%d", carrier.c_str(), init.c_str(), int(branch));
    if (carrier == "chebyshev") run(kan::ChebyshevConfig{5}, init, branch, epochs);
    else if (carrier == "bspline") run(spline(), init, branch, epochs);
    else if (carrier == "guarded") run(rational(kan::DenominatorPolicy::Guarded), init, branch, epochs);
    else if (carrier == "absolute") run(rational(kan::DenominatorPolicy::Absolute), init, branch, epochs);
    else if (carrier == "smooth") run(rational(kan::DenominatorPolicy::Smooth), init, branch, epochs);
    else return 2;
    return 0;
}

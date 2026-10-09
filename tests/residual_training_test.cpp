// Backlog M3 demonstration (closes the M4 open point): the five-layer x1*x2
// network of tests/initializer_training_test.cpp, every KAN layer with the
// SiLU residual branch. pykan-style NoiseInit, which stays at the mean
// predictor (MSE 0.1205) without the branch, leaves that saddle with it:
// within 3000 epochs for Chebyshev and B-spline edges, between 3000 and 10000
// for rational edges (see docs/evidence/backlog/M3.md). CPU, deterministic.
#include "kan/initializers.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <cstdio>
#include <string>

namespace {
constexpr std::size_t width = 8, hidden_layers = 4, epochs = 3000, rational_noise_epochs = 10000;

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

// [Tanh, KAN 2->8, (Tanh, KAN 8->8) x 3, Tanh, KAN 8->1], each KAN layer with
// a (zero, to be initialized) residual branch.
template<class Config>
kan::Network deep_network(const Config& config) {
    std::vector<kan::NetworkLayer> stages;
    std::size_t in = 2;
    for (std::size_t l = 0; l <= hidden_layers; ++l) {
        const std::size_t out = l == hidden_layers ? 1 : width;
        kan::Layer layer(in, out, config);
        layer.set_residual(kan::SiluResidual{std::vector<double>(in * out, 0.0)});
        stages.emplace_back(kan::InputMap(in, kan::TanhMap{1.0}));
        stages.emplace_back(std::move(layer));
        in = out;
    }
    return kan::Network(std::move(stages));
}

// Full-batch SGD on the mean squared error; prints the curve.
double train(kan::Network net, const Data& d, double rate, const std::string& label, std::size_t epochs) {
    std::printf("curve %-34s", label.c_str());
    double last = 0;
    for (std::size_t epoch = 0; epoch <= epochs; ++epoch) {
        const auto y = net.forward(d.x, d.batch);
        std::vector<double> upstream(d.batch);
        double s = 0;
        for (std::size_t b = 0; b < d.batch; ++b) {
            upstream[b] = 2 * (y[b] - d.target[b]) / double(d.batch);
            s += (y[b] - d.target[b]) * (y[b] - d.target[b]);
        }
        last = s / double(d.batch);
        if (epoch == 0 || epoch == 100 || epoch == 300 || epoch == 1000 || epoch == 3000 || epoch == 5000 || epoch == epochs)
            std::printf(" %zu:%.3e", epoch, last);
        if (epoch < epochs) net.sgd(net.backward(d.x, d.batch, upstream), rate);
    }
    std::printf("\n");
    return last;
}

double target_variance(const Data& d) {
    double mean = 0, s = 0;
    for (double t : d.target) mean += t;
    mean /= double(d.batch);
    for (double t : d.target) s += (t - mean) * (t - mean);
    return s / double(d.batch);
}

template<class Config>
void demonstrate(const Config& config, const char* name, double rate, std::size_t noise_epochs = epochs) {
    const auto d = product_data();
    const double variance = target_variance(d);
    auto noisy = deep_network(config);
    kan::initialize(noisy, kan::NoiseInit{0.3, kan::Distribution::Uniform, 1, {}});
    const double noise = train(noisy, d, rate, std::string(name) + " noise+residual", noise_epochs);
    auto scaled = deep_network(config);
    kan::initialize(scaled, kan::VarianceScaling{1, kan::Distribution::Uniform, 1, {}});
    const double variance_scaled = train(scaled, d, rate, std::string(name) + " variance+residual", epochs);
    REQUIRE(noise < 0.5 * variance);           // NoiseInit leaves the saddle with the branch
    REQUIRE(variance_scaled < 0.05 * variance); // VarianceScaling (zero branch weights) still trains
}

kan::BSplineConfig spline() {
    std::vector<double> knots{-1, -1, -1};
    for (int j = 0; j <= 5; ++j) knots.push_back(-1 + 0.4 * j);
    knots.insert(knots.end(), {1, 1, 1});
    return {3, knots};
}

kan::RationalConfig rational(kan::DenominatorPolicy policy) {
    kan::RationalConfig c;
    c.denominator_policy = policy;
    return c;
}
} // namespace

TEST(deep_chebyshev_with_residual_trains_from_noise) { demonstrate(kan::ChebyshevConfig{5}, "chebyshev", 0.03); }
TEST(deep_bspline_with_residual_trains_from_noise) { demonstrate(spline(), "bspline", 0.03); }
TEST(deep_rational_guarded_with_residual_trains_from_noise) {
    demonstrate(rational(kan::DenominatorPolicy::Guarded), "rational guarded", 0.03, rational_noise_epochs);
}
TEST(deep_rational_absolute_with_residual_trains_from_noise) {
    demonstrate(rational(kan::DenominatorPolicy::Absolute), "rational absolute", 0.03, rational_noise_epochs);
}
TEST(deep_rational_smooth_with_residual_trains_from_noise) {
    demonstrate(rational(kan::DenominatorPolicy::Smooth), "rational smooth", 0.03, rational_noise_epochs);
}

int main() { return test::run(); }

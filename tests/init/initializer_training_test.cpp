// Backlog M4 demonstration: deep KAN networks that do not train from zero
// initialization train from the explicit initializers (CPU, deterministic).
#include "kan/initializers.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <cstdio>
#include <optional>
#include <string>

namespace {
constexpr std::size_t width = 8, hidden_layers = 4, epochs = 3000;

struct Data { std::vector<double> x, target; std::size_t batch; };

// f(x1, x2) = x1 * x2 on an 8 x 8 grid in [-0.9, 0.9]^2: not of the form
// g(h1(x1) + h2(x2)), so it needs more than one effective hidden unit.
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

template<class Config>
kan::Network deep_network(const Config& config) {
    std::vector<kan::NetworkLayer> stages;
    std::size_t in = 2;
    for (std::size_t l = 0; l <= hidden_layers; ++l) {
        const std::size_t out = l == hidden_layers ? 1 : width;
        stages.emplace_back(kan::InputMap(in, kan::TanhMap{1.0}));
        stages.emplace_back(kan::Layer(in, out, config));
        in = out;
    }
    return kan::Network(std::move(stages));
}

double loss(const kan::Network& net, const Data& d) {
    const auto y = net.forward(d.x, d.batch);
    double s = 0;
    for (std::size_t b = 0; b < d.batch; ++b) s += (y[b] - d.target[b]) * (y[b] - d.target[b]);
    return s / double(d.batch);
}

// Full-batch SGD on the mean squared error; prints the curve.
double train(kan::Network net, const Data& d, double rate, const std::string& label) {
    std::printf("curve %-28s", label.c_str());
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
        if (epoch == 0 || epoch == 100 || epoch == 300 || epoch == 1000 || epoch == epochs)
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
void demonstrate(const Config& config, const char* name, double rate) {
    const auto d = product_data();
    const double variance = target_variance(d);
    const double zero = train(deep_network(config), d, rate, std::string(name) + " zero");
    auto net = deep_network(config);
    kan::initialize(net, kan::VarianceScaling{1, kan::Distribution::Uniform, 1, {}});
    const double scaled = train(net, d, rate, std::string(name) + " variance");
    auto noisy = deep_network(config);
    kan::initialize(noisy, kan::NoiseInit{0.3, kan::Distribution::Uniform, 1, {}});
    train(noisy, d, rate, std::string(name) + " noise");
    REQUIRE(zero > 0.5 * variance);    // zero initialization does not learn the product
    REQUIRE(scaled < 0.05 * variance); // variance scaling does
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

TEST(deep_chebyshev_trains_only_from_explicit_initialization) { demonstrate(kan::ChebyshevConfig{5}, "chebyshev", 0.03); }
TEST(deep_bspline_trains_only_from_explicit_initialization) { demonstrate(spline(), "bspline", 0.03); }
TEST(deep_rational_guarded_trains_only_from_explicit_initialization) {
    demonstrate(rational(kan::DenominatorPolicy::Guarded), "rational guarded", 0.03);
}
TEST(deep_rational_absolute_trains_only_from_explicit_initialization) {
    demonstrate(rational(kan::DenominatorPolicy::Absolute), "rational absolute", 0.03);
}
TEST(deep_rational_smooth_trains_only_from_explicit_initialization) {
    demonstrate(rational(kan::DenominatorPolicy::Smooth), "rational smooth", 0.03);
}

int main() { return test::run(); }

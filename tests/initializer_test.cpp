// Backlog M4: explicit typed initializers.
#include "kan/families.hpp"
#include "kan/initializers.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>

namespace {
constexpr std::uint64_t golden_gamma = 0x9E3779B97F4A7C15ull;

// Independent reference SplitMix64 (Steele, Lea, Flood 2014; Vigna's code).
std::uint64_t mix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
double symmetric_uniform(std::uint64_t seed, std::uint64_t index) { // draw `index` (0-based)
    const double u = static_cast<double>(mix(seed + (index + 1) * golden_gamma) >> 11) * 0x1p-53;
    return 2 * u - 1;
}

void fnv(std::uint64_t& h, std::span<const double> values) {
    for (double v : values) {
        const auto bits = std::bit_cast<std::uint64_t>(v);
        for (int b = 0; b < 8; ++b) h = (h ^ ((bits >> (8 * b)) & 0xff)) * 0x100000001b3ull;
    }
}
std::uint64_t digest(const kan::Layer& layer) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    fnv(h, layer.coefficients());
    fnv(h, test::denominators(layer));
    fnv(h, layer.bias());
    return h;
}
std::uint64_t digest(const kan::Network& network) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const auto& stage : network.layers()) {
        if (const auto* l = std::get_if<kan::Layer>(&stage)) {
            fnv(h, l->coefficients());
            fnv(h, test::denominators(*l));
            fnv(h, l->bias());
        } else if (const auto* ln = std::get_if<kan::LayerNormMap>(&std::get<kan::InputMap>(stage).map())) {
            fnv(h, ln->gain);
            fnv(h, ln->bias);
        }
    }
    return h;
}
std::string hex(std::uint64_t v) {
    char buffer[24];
    std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(v));
    return buffer;
}

std::vector<double> linspace(double a, double b, std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = a + (b - a) * double(i) / double(n - 1);
    return v;
}
kan::BSplineConfig uniform_spline(std::size_t degree, std::size_t intervals, double a = -1, double b = 1) {
    std::vector<double> knots(degree, a);
    for (double t : linspace(a, b, intervals + 1)) knots.push_back(t);
    knots.insert(knots.end(), degree, b);
    return {degree, knots};
}

// Independent second moments: composite midpoint rule of w(x) phi_k(x)^2 over
// [a, b] divided by the integral of w.
template<class Weight>
std::vector<double> numeric_moments(const kan::BasisConfig& c, double a, double b, Weight w, std::size_t n = 200000) {
    const auto terms = kan::basis_size(c);
    std::vector<double> m(terms, 0.0);
    double mass = 0;
    const double h = (b - a) / double(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double x = a + (double(j) + 0.5) * h, weight = w(x);
        const auto v = kan::evaluate_basis(c, x).values;
        for (std::size_t k = 0; k < terms; ++k) m[k] += weight * v[k] * v[k];
        mass += weight;
    }
    for (auto& value : m) value /= mass;
    return m;
}
void same_moments(const std::vector<double>& actual, const std::vector<double>& expected, double tolerance) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t k = 0; k < actual.size(); ++k) test::near(actual[k] / expected[k], 1.0, tolerance);
}

double mean_square(std::span<const double> v) {
    double s = 0;
    for (double x : v) s += x * x;
    return s / double(v.size());
}
} // namespace

TEST(constructors_keep_zero_initialization) {
    kan::Layer basis(3, 2, kan::ChebyshevConfig{5});
    kan::Layer rational(3, 2, kan::RationalConfig{});
    for (double c : basis.coefficients()) REQUIRE(c == 0);
    for (double c : rational.coefficients()) REQUIRE(c == 0);
    for (double b : test::denominators(rational)) REQUIRE(b == 0);
    REQUIRE(kan::VarianceScaling{}.gain == 1.0);
    REQUIRE(kan::VarianceScaling{}.distribution == kan::Distribution::Uniform);
    REQUIRE(kan::NoiseInit{}.scale == 0.3);
    REQUIRE(kan::DenominatorInit{}.bound == 0.5 && kan::DenominatorInit{}.radius == 1.0);
}

TEST(polynomial_reference_moments_are_closed_forms) {
    const auto cheb = kan::reference_moments(kan::ChebyshevConfig{6});
    test::near(cheb.variance, 0.5, 0);
    REQUIRE(cheb.second_moments == std::vector<double>({1, 0.5, 0.5, 0.5, 0.5, 0.5}));
    const auto leg = kan::reference_moments(kan::LegendreConfig{5});
    test::near(leg.variance, 1.0 / 3, 1e-15);
    for (std::size_t k = 0; k < 5; ++k) test::near(leg.second_moments[k], 1.0 / double(2 * k + 1), 1e-15);
    const auto her = kan::reference_moments(kan::HermiteConfig{6});
    test::near(her.variance, 0.5, 0);
    REQUIRE(her.second_moments == std::vector<double>({1, 2, 8, 48, 384, 3840})); // 2^n n!
    const auto fou = kan::reference_moments(kan::FourierConfig{5, 2.0});
    test::near(fou.variance, std::numbers::pi * std::numbers::pi / 12, 1e-15);
    REQUIRE(fou.second_moments == std::vector<double>({1, 0.5, 0.5, 0.5, 0.5}));
    // Jacobi under its normalized weight, against numerical integration.
    for (auto [alpha, beta] : {std::pair{1.0, 0.5}, std::pair{0.0, 0.0}, std::pair{2.5, 1.0}, std::pair{0.25, 3.0}}) {
        const kan::JacobiConfig c{6, alpha, beta};
        const auto ref = kan::reference_moments(c);
        const auto w = [=](double x) { return std::pow(1 - x, alpha) * std::pow(1 + x, beta); };
        same_moments(ref.second_moments, numeric_moments(c, -1, 1, w), 1e-6);
        const double ab = alpha + beta;
        test::near(ref.variance, 4 * (alpha + 1) * (beta + 1) / ((ab + 2) * (ab + 2) * (ab + 3)), 1e-14);
    }
    // alpha + beta = -1 has a removable 0/0 in the gamma-function form.
    const kan::JacobiConfig odd{5, -0.5, -0.5};
    const auto ref = kan::reference_moments(odd);
    const auto cheb_w = [](double x) { return 1 / std::sqrt(1 - x * x); };
    same_moments(ref.second_moments, numeric_moments(odd, -1, 1, cheb_w, 2000000), 2e-3);
    test::near(ref.variance, 0.5, 1e-15);
    // Hermite: numerical check under exp(-x^2).
    const auto hw = [](double x) { return std::exp(-x * x); };
    same_moments(her.second_moments, numeric_moments(kan::HermiteConfig{6}, -12, 12, hw), 1e-9);
}

TEST(localized_reference_moments_match_numerical_integration) {
    const auto one = [](double) { return 1.0; };
    for (const auto& spline : {uniform_spline(3, 5), uniform_spline(0, 4), uniform_spline(2, 3, 0, 7),
                               kan::BSplineConfig{2, {-1, -1, -1, -0.7, 0.1, 0.1, 0.9, 1.4, 1.4, 1.4}}}) {
        const auto ref = kan::reference_moments(spline);
        const double a = spline.knots.front(), b = spline.knots.back();
        same_moments(ref.second_moments, numeric_moments(spline, a, b, one), 1e-8);
        test::near(ref.variance, (b - a) * (b - a) / 12, 1e-14);
    }
    const kan::GaussianRbfConfig rbf{{-1, -0.2, 0.5, 1}, 0.4};
    auto ref = kan::reference_moments(rbf);
    same_moments(ref.second_moments, numeric_moments(rbf, -1, 1, one), 1e-8);
    test::near(ref.variance, 4.0 / 12, 1e-14);
    const kan::TrainableRbfConfig trainable{{0, 2, 3}, {-1, 0.2, -0.5}};
    ref = kan::reference_moments(trainable);
    same_moments(ref.second_moments, numeric_moments(trainable, 0, 3, one), 1e-8);
    const kan::MexicanHatConfig hat{{-2, 0, 1}, {0.5, 1.5, 0.3}};
    ref = kan::reference_moments(hat);
    same_moments(ref.second_moments, numeric_moments(hat, -2, 1, one), 1e-8);
    test::near(ref.variance, 9.0 / 12, 1e-14);
    // Coincident centers: [c - h, c + h] with the largest width or scale.
    const kan::GaussianRbfConfig single{{0.5}, 0.25};
    same_moments(kan::reference_moments(single).second_moments, numeric_moments(single, 0.25, 0.75, one), 1e-8);
    const kan::MexicanHatConfig coincident{{1, 1}, {0.5, 2}};
    ref = kan::reference_moments(coincident);
    same_moments(ref.second_moments, numeric_moments(coincident, -1, 3, one), 1e-8);
    test::near(ref.variance, 16.0 / 12, 1e-14);
    kan::reference_moments(kan::GaussianRbfConfig{{0, 1}, 1e-9}); // narrow terms stay finite
    test::throws<std::invalid_argument>([] { kan::reference_moments(kan::ChebyshevConfig{0}); });
}

TEST(uniform_draws_follow_the_specified_transform) {
    // Coefficient j of a fresh layer is sqrt(3)*sigma_k*(2u-1), u the j-th
    // SplitMix64 output >> 11 times 2^-53.
    kan::Layer layer(3, 2, kan::LegendreConfig{4});
    const std::uint64_t seed = 12345;
    kan::initialize(layer, kan::VarianceScaling{1.5, kan::Distribution::Uniform, seed, {}});
    const auto c = layer.coefficients();
    for (std::size_t j = 0; j < c.size(); ++j) {
        const auto k = j % 4;
        const double m = 1.0 / double(2 * k + 1);
        const double sigma = std::sqrt(1.5 * 1.5 * (1.0 / 3) / (3.0 * 4.0 * m));
        test::near(c[j], std::sqrt(3.0) * sigma * symmetric_uniform(seed, j), 1e-15);
    }
    for (double b : layer.bias()) REQUIRE(b == 0);
    // Noise: U(-a/2, a/2), a = scale / (G sqrt(inputs)), G = terms - degree.
    kan::Layer spline(4, 3, uniform_spline(3, 5));
    kan::initialize(spline, kan::NoiseInit{0.5, kan::Distribution::Uniform, 7, {}});
    const double a = 0.5 / (5.0 * std::sqrt(4.0));
    const auto s = spline.coefficients();
    for (std::size_t j = 0; j < s.size(); ++j) test::near(s[j], 0.5 * a * symmetric_uniform(7, j), 1e-15);
}

TEST(normal_draws_have_the_stated_moments) {
    kan::Layer layer(64, 64, kan::ChebyshevConfig{2});
    kan::initialize(layer, kan::NoiseInit{1.0, kan::Distribution::Normal, 3, {}});
    const auto c = layer.coefficients();
    const double a = 1.0 / (2.0 * 8.0), variance = a * a / 12;
    double mean = 0, m2 = 0, m4 = 0;
    for (double v : c) { mean += v; m2 += v * v; m4 += v * v * v * v; }
    mean /= double(c.size()); m2 /= double(c.size()); m4 /= double(c.size());
    test::near(mean / std::sqrt(variance), 0, 0.05);
    test::near(m2 / variance, 1, 0.05);
    test::near(m4 / (m2 * m2), 3, 0.15); // Gaussian kurtosis, uniform would give 1.8
    kan::Layer uniform(64, 64, kan::ChebyshevConfig{2});
    kan::initialize(uniform, kan::NoiseInit{1.0, kan::Distribution::Uniform, 3, {}});
    const auto u = uniform.coefficients();
    REQUIRE(std::all_of(u.begin(), u.end(), [&](double v) { return std::abs(v) <= a / 2; }));
    test::near(mean_square(u) / variance, 1, 0.05);
}

TEST(variance_scaling_preserves_the_reference_second_moment) {
    // One wide layer, inputs drawn from the reference measure: E[y^2] = gain^2 * variance.
    std::uint64_t state = 99;
    const auto next_u = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull;
                              return (double(state >> 11) + 0.5) * 0x1p-53; };
    struct Case { kan::BasisConfig basis; std::function<double(double)> inverse_cdf; };
    const double pi = std::numbers::pi;
    std::vector<Case> cases{
        {kan::ChebyshevConfig{6}, [&](double u) { return -std::cos(pi * u); }},
        {kan::LegendreConfig{6}, [](double u) { return 2 * u - 1; }},
        {kan::JacobiConfig{5, 1, 0}, [](double u) { return 1 - 2 * std::sqrt(1 - u); }}, // density (1-x)/2
        {kan::FourierConfig{7, 1.5}, [&](double u) { return (2 * u - 1) * pi / 1.5; }},
        {uniform_spline(3, 6), [](double u) { return 2 * u - 1; }},
        {kan::GaussianRbfConfig{linspace(-1, 1, 6), 0.4}, [](double u) { return 2 * u - 1; }},
        {kan::TrainableRbfConfig{linspace(-2, 2, 5), std::vector<double>(5, -0.3)}, [](double u) { return 4 * u - 2; }},
        {kan::MexicanHatConfig{linspace(-1, 1, 5), std::vector<double>(5, 0.5)}, [](double u) { return 2 * u - 1; }},
    };
    for (std::size_t family = 0; family < cases.size(); ++family) {
        const std::size_t inputs = 32, outputs = 32, batch = 2048;
        for (auto distribution : {kan::Distribution::Uniform, kan::Distribution::Normal}) {
            kan::Layer layer(inputs, outputs, cases[family].basis);
            kan::initialize(layer, kan::VarianceScaling{0.8, distribution, 11 + family, {}});
            std::vector<double> x(inputs * batch);
            for (auto& v : x) v = cases[family].inverse_cdf(next_u());
            const double target = 0.64 * kan::reference_moments(cases[family].basis).variance;
            test::near(mean_square(layer.forward(x, batch)) / target, 1, 0.15);
        }
    }
    // Hermite: x ~ N(0, 1/2) by Box-Muller.
    kan::Layer hermite(32, 32, kan::HermiteConfig{5});
    kan::initialize(hermite, kan::VarianceScaling{1, kan::Distribution::Normal, 5, {}});
    std::vector<double> x(32 * 2048);
    for (auto& v : x) v = std::sqrt(0.5) * std::sqrt(-2 * std::log(next_u())) * std::cos(2 * pi * next_u());
    test::near(mean_square(hermite.forward(x, 2048)) / 0.5, 1, 0.2);
}

TEST(variance_stays_bounded_through_eight_layers) {
    // Bounded-domain families get a TanhMap in front of every KAN layer
    // (inputs in (-1, 1)); bounded bases stack directly.
    const std::size_t width = 16, batch = 512, depth = 8;
    std::vector<double> x(width * batch);
    for (std::size_t j = 0; j < x.size(); ++j) x[j] = std::sin(0.37 * double(j) + 0.1 * double(j % 7));
    struct Case { const char* name; kan::BasisConfig basis; bool tanh; };
    const std::vector<Case> cases{
        {"chebyshev", kan::ChebyshevConfig{5}, true}, {"legendre", kan::LegendreConfig{5}, true},
        {"jacobi", kan::JacobiConfig{5, 0.5, 0.5}, true}, {"bspline", uniform_spline(3, 5), true},
        {"hermite", kan::HermiteConfig{4}, true}, {"fourier", kan::FourierConfig{5, 1.0}, false},
        {"rbf", kan::GaussianRbfConfig{linspace(-1, 1, 6), 0.4}, false},
        {"mexican_hat", kan::MexicanHatConfig{linspace(-1, 1, 5), std::vector<double>(5, 0.5)}, false},
    };
    for (const auto& c : cases) {
        std::vector<kan::NetworkLayer> stages;
        for (std::size_t l = 0; l < depth; ++l) {
            if (c.tanh) stages.emplace_back(kan::InputMap(width, kan::TanhMap{1.0}));
            stages.emplace_back(kan::Layer(width, width, c.basis));
        }
        const double target = kan::reference_moments(c.basis).variance;
        for (int mode = 0; mode < 3; ++mode) {
            kan::Network net(stages);
            if (mode == 1) kan::initialize(net, kan::VarianceScaling{1, kan::Distribution::Uniform, 2024, {}});
            if (mode == 2) kan::initialize(net, kan::NoiseInit{0.3, kan::Distribution::Uniform, 2024, {}});
            auto y = x;
            std::printf("depth %-11s %-8s", c.name, mode == 0 ? "zero" : mode == 1 ? "variance" : "noise");
            for (const auto& stage : net.layers()) {
                y = std::visit([&](const auto& s) { return s.forward(y, batch); }, stage);
                if (!std::holds_alternative<kan::Layer>(stage)) continue;
                const double ratio = mean_square(y) / target;
                std::printf(" %.3g", ratio);
                if (mode == 0) REQUIRE(ratio == 0);
                if (mode == 1) REQUIRE(ratio > 0.05 && ratio < 20);
            }
            std::printf("\n");
            if (mode == 2) REQUIRE(mean_square(y) / target < 1e-4); // pykan noise decays without a base branch
        }
    }
}

TEST(rational_denominators_are_bounded_nonzero_and_pole_free) {
    for (auto policy : {kan::DenominatorPolicy::Guarded, kan::DenominatorPolicy::Absolute, kan::DenominatorPolicy::Smooth})
        for (std::size_t n : {1u, 2u, 5u, 16u})
            for (double radius : {1.0, 0.25, 3.0}) {
                kan::RationalConfig config;
                config.numerator_degree = 3;
                config.denominator_degree = n;
                config.center = 0.2;
                config.scale = 1.5;
                config.denominator_policy = policy;
                const kan::DenominatorInit d{0.6, radius};
                for (int which = 0; which < 2; ++which) {
                    kan::Layer layer(4, 3, config);
                    if (which == 0) kan::initialize(layer, kan::VarianceScaling{1, kan::Distribution::Normal, 8, d});
                    else kan::initialize(layer, kan::NoiseInit{0.3, kan::Distribution::Uniform, 8, d});
                    const auto b = test::denominators(layer);
                    for (std::size_t e = 0; e < 12; ++e) {
                        double bound = 0, power = 1;
                        for (std::size_t k = 1; k <= n; ++k) {
                            power *= radius;
                            const double magnitude = std::abs(b[e * n + k - 1]) * power;
                            REQUIRE(magnitude <= 0.6 / double(n) * (1 + 1e-15));
                            REQUIRE(magnitude >= 0.3 / double(n) * (1 - 1e-15));
                            bound += magnitude;
                        }
                        REQUIRE(bound <= 0.6 * (1 + 1e-14)); // |S(z)| <= 0.6 for |z| <= radius
                    }
                    // Executing the whole domain never reports a pole.
                    const auto xs = linspace(config.center - radius * config.scale, config.center + radius * config.scale, 801);
                    std::vector<double> x;
                    for (double v : xs) x.insert(x.end(), 4, v);
                    const auto grads = layer.backward(x, xs.size(), std::vector<double>(3 * xs.size(), 1.0));
                    // Safe policies are not stationary at the initial point.
                    REQUIRE(std::any_of(test::denominators(grads).begin(), test::denominators(grads).end(),
                                        [](double g) { return g != 0; }));
                }
            }
}

TEST(rational_numerators_preserve_variance_on_the_domain) {
    for (auto policy : {kan::DenominatorPolicy::Guarded, kan::DenominatorPolicy::Absolute, kan::DenominatorPolicy::Smooth}) {
        kan::RationalConfig config;
        config.center = -0.5;
        config.scale = 2.0;
        config.denominator_policy = policy;
        kan::Layer layer(32, 32, config);
        kan::initialize(layer, kan::VarianceScaling{1, kan::Distribution::Uniform, 21, {0.5, 1.0}});
        std::vector<double> x(32 * 2048);
        std::uint64_t state = 5;
        for (auto& v : x) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            v = config.center + config.scale * (2 * (double(state >> 11) + 0.5) * 0x1p-53 - 1);
        }
        const double target = config.scale * config.scale / 3; // x uniform on center +- radius*scale
        test::near(mean_square(layer.forward(x, 2048)) / target, 1, 0.15);
    }
}

TEST(guarded_bound_must_respect_the_pole_guard) {
    kan::RationalConfig config;
    config.epsilon = 0.5;
    kan::Layer layer(2, 2, config);
    const auto before = digest(layer);
    // epsilon * (1 + bound) must stay below 1 - bound.
    test::throws<std::invalid_argument>([&] { kan::initialize(layer, kan::VarianceScaling{1, {}, 0, {0.5, 1}}); });
    REQUIRE(digest(layer) == before);
    kan::initialize(layer, kan::VarianceScaling{1, {}, 0, {0.3, 1}});
    config.denominator_policy = kan::DenominatorPolicy::Smooth; // safe policies ignore epsilon
    kan::Layer smooth(2, 2, config);
    kan::initialize(smooth, kan::VarianceScaling{1, {}, 0, {0.9, 1}});
}

TEST(invalid_initializers_are_rejected_atomically) {
    kan::Layer layer(2, 3, kan::ChebyshevConfig{4});
    kan::initialize(layer, kan::VarianceScaling{});
    const auto before = digest(layer);
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    for (double gain : {0.0, -1.0, nan, inf})
        test::throws<std::invalid_argument>([&] { kan::initialize(layer, kan::VarianceScaling{gain, {}, 0, {}}); });
    for (double scale : {0.0, -0.1, nan, inf})
        test::throws<std::invalid_argument>([&] { kan::initialize(layer, kan::NoiseInit{scale, {}, 0, {}}); });
    for (double bound : {0.0, 1.0, -0.5, nan})
        test::throws<std::invalid_argument>([&] { kan::initialize(layer, kan::VarianceScaling{1, {}, 0, {bound, 1}}); });
    for (double radius : {0.0, -1.0, nan, inf})
        test::throws<std::invalid_argument>([&] { kan::initialize(layer, kan::NoiseInit{0.3, {}, 0, {0.5, radius}}); });
    test::throws<std::invalid_argument>([&] {
        kan::initialize(layer, kan::VarianceScaling{1, static_cast<kan::Distribution>(7), 0, {}});
    });
    REQUIRE(digest(layer) == before);
    kan::Layer moved(2, 2, kan::ChebyshevConfig{3});
    kan::Layer sink = std::move(moved);
    test::throws<std::invalid_argument>([&] { kan::initialize(moved, kan::VarianceScaling{}); });
    // A radius whose powers leave the double range cannot bound the denominators.
    kan::RationalConfig r;
    r.denominator_degree = 16;
    kan::Layer rational(1, 1, r);
    test::throws<std::overflow_error>([&] { kan::initialize(rational, kan::VarianceScaling{1, {}, 0, {0.5, 1e300}}); });
    test::throws<std::overflow_error>([&] { kan::initialize(rational, kan::VarianceScaling{1, {}, 0, {0.5, 1e-300}}); });
}

TEST(network_initialization_uses_layer_seeds_and_resets_layer_norm) {
    for (std::size_t p = 0; p < 4; ++p) REQUIRE(kan::layer_seed(77, p) == mix(77 + (p + 1) * golden_gamma));
    kan::RationalConfig rational;
    rational.denominator_policy = kan::DenominatorPolicy::Absolute;
    const kan::TrainableRbfConfig rbf{{-1, 0, 1}, {-0.5, -0.4, -0.3}};
    kan::Network net({kan::InputMap(3, kan::AffineMap{{2, 2, 2}, {0, 0, 0}}), kan::Layer(3, 4, kan::ChebyshevConfig{4}),
                      kan::InputMap(4, kan::LayerNormMap{1e-5, {3, 3, 3, 3}, {1, 1, 1, 1}}),
                      kan::Layer(4, 2, rbf), kan::Layer(2, 2, rational), kan::InputMap(2, kan::TanhMap{2})});
    const kan::Initializer init = kan::VarianceScaling{1.2, kan::Distribution::Normal, 77, {}};
    kan::initialize(net, init);
    for (std::size_t p : {1u, 3u, 4u}) {
        auto expected = test::layer(net, p);
        auto per_layer = std::get<kan::VarianceScaling>(init);
        per_layer.seed = kan::layer_seed(77, p);
        kan::initialize(expected, per_layer);
        REQUIRE(digest(expected) == digest(test::layer(net, p)));
    }
    REQUIRE(test::input_map(net, 0).map() == kan::InputMapKind(kan::AffineMap{{2, 2, 2}, {0, 0, 0}}));
    REQUIRE(test::input_map(net, 2).map() == kan::InputMapKind(kan::LayerNormMap{1e-5, {1, 1, 1, 1}, {0, 0, 0, 0}}));
    REQUIRE(test::trainable(test::layer(net, 3)) == rbf);
    // Atomic: a failure in a later layer leaves the whole network unchanged.
    const auto before = digest(net);
    rational.denominator_policy = kan::DenominatorPolicy::Guarded;
    rational.epsilon = 0.9;
    kan::Network bad({kan::Layer(3, 4, kan::ChebyshevConfig{4}), kan::Layer(4, 2, rational)});
    const auto bad_before = digest(bad);
    test::throws<std::invalid_argument>([&] { kan::initialize(bad, kan::VarianceScaling{}); });
    REQUIRE(digest(bad) == bad_before);
    REQUIRE(digest(net) == before);
}

TEST(seeds_are_deterministic_and_pinned_across_platforms) {
    // Pinned digests of the parameter bits. The same values must result from
    // MSVC and GCC builds (backlog M4 portability requirement).
    kan::RationalConfig smooth;
    smooth.denominator_policy = kan::DenominatorPolicy::Smooth;
    smooth.numerator_degree = 4;
    smooth.denominator_degree = 3;
    struct Case { kan::Layer layer; kan::Initializer init; const char* pinned; };
    std::vector<Case> cases{
        {kan::Layer(5, 4, kan::ChebyshevConfig{6}), kan::VarianceScaling{1, kan::Distribution::Uniform, 1, {}}, "PIN0"},
        {kan::Layer(5, 4, kan::HermiteConfig{5}), kan::VarianceScaling{1, kan::Distribution::Normal, 2, {}}, "PIN1"},
        {kan::Layer(5, 4, kan::JacobiConfig{5, 0.3, -0.4}), kan::VarianceScaling{0.7, kan::Distribution::Normal, 3, {}}, "PIN2"},
        {kan::Layer(5, 4, uniform_spline(3, 7)), kan::NoiseInit{0.3, kan::Distribution::Uniform, 4, {}}, "PIN3"},
        {kan::Layer(5, 4, uniform_spline(2, 4)), kan::VarianceScaling{1, kan::Distribution::Normal, 5, {}}, "PIN4"},
        {kan::Layer(5, 4, kan::TrainableRbfConfig{{-1, 0.1, 1}, {-0.7, -1.1, 0.3}}),
         kan::VarianceScaling{1, kan::Distribution::Normal, 6, {}}, "PIN5"},
        {kan::Layer(5, 4, kan::MexicanHatConfig{{-1, 0, 2}, {0.5, 0.7, 1.1}}),
         kan::VarianceScaling{1, kan::Distribution::Uniform, 7, {}}, "PIN6"},
        {kan::Layer(5, 4, kan::FourierConfig{5, 2.5}), kan::NoiseInit{0.3, kan::Distribution::Normal, 8, {}}, "PIN7"},
        {kan::Layer(5, 4, smooth), kan::VarianceScaling{1, kan::Distribution::Normal, 9, {0.4, 2}}, "PIN8"},
    };
    std::string failures;
    for (auto& c : cases) {
        auto copy = c.layer;
        kan::initialize(c.layer, c.init);
        kan::initialize(copy, c.init);
        REQUIRE(digest(c.layer) == digest(copy));
        auto other = c.init;
        std::visit([](auto& i) { ++i.seed; }, other);
        kan::initialize(copy, other);
        REQUIRE(digest(c.layer) != digest(copy));
        const auto value = hex(digest(c.layer));
        std::printf("pinned %s\n", value.c_str());
        if (value != c.pinned) failures += value + " != " + c.pinned + "; ";
    }
    if (!failures.empty()) throw std::runtime_error(failures);
}

int main() { return test::run(); }

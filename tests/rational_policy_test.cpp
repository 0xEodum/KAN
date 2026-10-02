// M2: denominator (singularity) policies of rational edges.
#include "kan/families.hpp"
#include "kan/layer.hpp"
#include "kan/network.hpp"
#include "kan/rational.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace {
using kan::DenominatorPolicy;
constexpr DenominatorPolicy all_policies[] = {DenominatorPolicy::Guarded, DenominatorPolicy::Absolute,
                                              DenominatorPolicy::Smooth};
constexpr DenominatorPolicy safe_policies[] = {DenominatorPolicy::Absolute, DenominatorPolicy::Smooth};

kan::RationalConfig config(DenominatorPolicy policy, std::size_t m, std::size_t n) {
    kan::RationalConfig c;
    c.numerator_degree = m; c.denominator_degree = n; c.denominator_policy = policy;
    return c;
}
double value(const kan::RationalConfig& c, double x, const std::vector<double>& a, const std::vector<double>& b) {
    return kan::evaluate_rational(c, x, a, b).value;
}

// Central differences of every VJP of one edge.
void check_edge(const kan::RationalConfig& c, double x, const std::vector<double>& a, const std::vector<double>& b,
                double tolerance) {
    const double h = 1e-6;
    const auto r = kan::evaluate_rational(c, x, a, b);
    test::near(r.input_derivative, (value(c, x + h, a, b) - value(c, x - h, a, b)) / (2 * h), tolerance);
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto p = a, q = a; p[i] += h; q[i] -= h;
        test::near(r.numerator_derivatives[i], (value(c, x, p, b) - value(c, x, q, b)) / (2 * h), tolerance);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        auto p = b, q = b; p[i] += h; q[i] -= h;
        test::near(r.denominator_derivatives[i], (value(c, x, a, p) - value(c, x, a, q)) / (2 * h), tolerance);
    }
}
} // namespace

TEST(default_policy_is_guarded_and_invalid_policy_is_rejected) {
    REQUIRE(kan::RationalConfig{}.denominator_policy == DenominatorPolicy::Guarded);
    auto bad = kan::RationalConfig{};
    bad.denominator_policy = static_cast<DenominatorPolicy>(3);
    test::throws<std::invalid_argument>([&] { kan::validate_rational(bad); });
    test::throws<std::invalid_argument>([&] { kan::Layer(1, 1, bad); });
    // Epsilon is validated for every policy, used only by the guarded one.
    for (auto policy : safe_policies) {
        auto c = config(policy, 1, 1); c.epsilon = 0;
        test::throws<std::invalid_argument>([&] { kan::validate_rational(c); });
    }
}

TEST(independent_closed_forms_per_policy) {
    // r = (a0 + a1 z) / Q, S = b z, z = (x - 0.25) / 2.
    const std::vector<double> a{1, 0.5}, b{-0.5};
    for (double x : {-1.7, -0.3, 0.9, 2.6}) {
        const double z = (x - 0.25) / 2, s = -0.5 * z, p = 1 + 0.5 * z;
        {
            auto c = config(DenominatorPolicy::Absolute, 1, 1); c.center = 0.25; c.scale = 2;
            const auto r = kan::evaluate_rational(c, x, a, b);
            const double q = 1 + std::abs(s), sign = s > 0 ? 1 : -1, dq = sign * -0.5;
            test::near(r.value, p / q);
            test::near(r.input_derivative, (0.5 / q - p / q * dq / q) / 2);
            test::near(r.numerator_derivatives[0], 1 / q);
            test::near(r.numerator_derivatives[1], z / q);
            test::near(r.denominator_derivatives[0], -p / q * sign * z / q);
        }
        {
            auto c = config(DenominatorPolicy::Smooth, 1, 1); c.center = 0.25; c.scale = 2;
            const auto r = kan::evaluate_rational(c, x, a, b);
            const double q = 1 + s * s, dq = 2 * s * -0.5;
            test::near(r.value, p / q);
            test::near(r.input_derivative, (0.5 / q - p / q * dq / q) / 2);
            test::near(r.numerator_derivatives[0], 1 / q);
            test::near(r.numerator_derivatives[1], z / q);
            test::near(r.denominator_derivatives[0], -p / q * 2 * s * z / q);
        }
    }
}

TEST(finite_difference_vjps_all_policies_and_orders) {
    for (auto policy : all_policies)
        for (auto orders : {std::pair{0u, 3u}, std::pair{4u, 1u}, std::pair{3u, 5u}, std::pair{2u, 2u}, std::pair{16u, 16u}}) {
            auto c = config(policy, orders.first, orders.second); c.center = 0.1; c.scale = 1.4;
            std::vector<double> a(c.numerator_degree + 1), b(c.denominator_degree);
            for (std::size_t i = 0; i < a.size(); ++i) a[i] = 0.07 * static_cast<double>(i + 1);
            for (std::size_t i = 0; i < b.size(); ++i) b[i] = (i % 2 ? -0.13 : 0.11) * static_cast<double>(i + 1);
            for (double x : {-0.7, 0.0, 0.8, 1.9}) check_edge(c, x, a, b, 3e-7);
        }
}

TEST(safe_policies_never_report_a_pole) {
    // Guarded: Q = 1 - z = 0 at x = 1 (exact pole).
    test::throws<std::domain_error>([] {
        kan::evaluate_rational(config(DenominatorPolicy::Guarded, 1, 1), 1, std::vector<double>{1, -1}, std::vector<double>{-1});
    });
    for (auto policy : safe_policies)
        for (double x : {1.0, 1.0 - 1e-9, 1e-3, -4.0}) {
            auto c = config(policy, 1, 1); c.epsilon = 0.5; // a huge epsilon must not matter
            const auto r = kan::evaluate_rational(c, x, std::vector<double>{1, -1}, std::vector<double>{-1});
            const double s = -x, q = policy == DenominatorPolicy::Absolute ? 1 + std::abs(s) : 1 + s * s;
            test::near(r.value, (1 - x) / q);
            REQUIRE(q >= 1);
        }
}

TEST(absolute_subgradient_is_zero_where_the_sum_vanishes) {
    // S(z) = z - z^2 = 0 at z = 1: Q = 1 under both safe policies.
    for (auto policy : safe_policies) {
        const auto c = config(policy, 2, 2);
        const auto r = kan::evaluate_rational(c, 1, std::vector<double>{0.5, 2, 3}, std::vector<double>{1, -1});
        test::near(r.value, 5.5);
        test::near(r.input_derivative, 2 + 6); // P'(1), Q'(1) = 0
        for (double d : r.denominator_derivatives) REQUIRE(d == 0);
        test::near(r.numerator_derivatives[2], 1);
    }
    // Absolute one-sided derivatives in b differ; the documented subgradient
    // sign(0) = 0 is their midpoint.
    const auto c = config(DenominatorPolicy::Absolute, 0, 1);
    const std::vector<double> a{1};
    const double h = 1e-7;
    const double right = (value(c, 2, a, {h}) - value(c, 2, a, {0})) / h;
    const double left = (value(c, 2, a, {0}) - value(c, 2, a, {-h})) / h;
    test::near(right, -2, 1e-6); test::near(left, 2, 1e-6);
    REQUIRE(kan::evaluate_rational(c, 2, a, std::vector<double>{0}).denominator_derivatives[0] == 0);
}

TEST(safe_policies_keep_overflow_checks) {
    for (auto policy : all_policies) {
        auto c = config(policy, 3, 2);
        test::throws<std::overflow_error>([&] {
            kan::evaluate_rational(c, 1e200, std::vector<double>{0, 0, 0, 1}, std::vector<double>{0, 0});
        });
    }
    // S^2 overflows although S is finite.
    test::throws<std::overflow_error>([] {
        kan::evaluate_rational(config(DenominatorPolicy::Smooth, 0, 1), 1e10, std::vector<double>{1}, std::vector<double>{1e200});
    });
    const auto r = kan::evaluate_rational(config(DenominatorPolicy::Absolute, 0, 1), 1e10, std::vector<double>{1},
                                          std::vector<double>{1e200});
    test::near(r.value / 1e-210, 1);
}

TEST(representable_vjps_survive_underflowing_slopes) {
    // Smooth: Q' = 2 S S' = 2 b^2 z underflows to a subnormal while
    // r Q'/Q is representable; the log-space path keeps full precision.
    const double b = 1.2345e-160, p = 1e300;
    auto c = config(DenominatorPolicy::Smooth, 0, 1);
    auto r = kan::evaluate_rational(c, 1, std::vector<double>{p}, std::vector<double>{b});
    test::near(r.input_derivative / (-(p * b) * 2 * b), 1, 1e-12);
    test::near(r.denominator_derivatives[0] / (-p * 2 * b), 1, 1e-12);
    // Underflowing powers: dr/db_k = -r g z^k / Q with tiny z^k.
    c = config(DenominatorPolicy::Smooth, 0, 2);
    r = kan::evaluate_rational(c, 1e-200, std::vector<double>{1e300}, std::vector<double>{1e100, 0});
    test::near(r.denominator_derivatives[1] / -2e-200, 1, 1e-10);
    c = config(DenominatorPolicy::Absolute, 0, 2);
    r = kan::evaluate_rational(c, 1e-200, std::vector<double>{1e300}, std::vector<double>{-1e-50, 0});
    test::near(r.denominator_derivatives[1] / 1e-100, 1, 1e-10);
}

TEST(log_space_paths_with_negative_signs) {
    // Smooth, z = -1e-200: g = 2S < 0, z^2 > 0, so dr/db_2 = -r g z^2/Q > 0.
    auto c = config(DenominatorPolicy::Smooth, 0, 2);
    auto r = kan::evaluate_rational(c, -1e-200, std::vector<double>{1e300}, std::vector<double>{1e100, 0});
    test::near(r.denominator_derivatives[0] / -2.0, 1, 1e-12);
    test::near(r.denominator_derivatives[1] / 2e-200, 1, 1e-10);
    // Smooth, z = -1: Q' = 2 S S' = -2 b^2 underflows; dr/dx = -r Q'/Q > 0.
    const double b = 1.2345e-160, p = 1e300;
    c = config(DenominatorPolicy::Smooth, 0, 1);
    r = kan::evaluate_rational(c, -1, std::vector<double>{p}, std::vector<double>{b});
    test::near(r.input_derivative / ((p * b) * 2 * b), 1, 1e-12);
    // Absolute, odd power of a negative tiny z (z^3 underflows):
    // S < 0, g = -1, dr/db_3 = -r g z^3/Q = -1e300 * 1e-600 < 0.
    c = config(DenominatorPolicy::Absolute, 0, 3);
    r = kan::evaluate_rational(c, -1e-200, std::vector<double>{1e300}, std::vector<double>{1, 0, 0});
    test::near(r.denominator_derivatives[0] / -1e100, 1, 1e-12);
    test::near(r.denominator_derivatives[2] / -1e-300, 1, 1e-10);
}

TEST(set_carrier_rejects_an_unknown_policy) {
    kan::Layer layer(1, 1, config(DenominatorPolicy::Absolute, 1, 1));
    auto edges = std::get<kan::RationalEdges>(layer.carrier());
    edges.config.denominator_policy = static_cast<DenominatorPolicy>(-1);
    test::throws<std::invalid_argument>([&] { layer.set_carrier(edges); });
    REQUIRE(std::get<kan::RationalEdges>(layer.carrier()).config.denominator_policy == DenominatorPolicy::Absolute);
}

namespace {
// evaluate_rational results of the pre-M2 code (5dc6819) as exact bit
// patterns: the default policy must reproduce them bit for bit.
struct Pinned {
    double value, input;
    std::vector<double> a, b;
};
void require_bits(double actual, double expected) {
    REQUIRE(std::bit_cast<std::uint64_t>(actual) == std::bit_cast<std::uint64_t>(expected));
}
void require_pinned(const kan::RationalConfig& c, double x, const std::vector<double>& a,
                    const std::vector<double>& b, const Pinned& expected) {
    REQUIRE(c.denominator_policy == DenominatorPolicy::Guarded);
    const auto r = kan::evaluate_rational(c, x, a, b);
    require_bits(r.value, expected.value);
    require_bits(r.input_derivative, expected.input);
    for (std::size_t k = 0; k < a.size(); ++k) require_bits(r.numerator_derivatives[k], expected.a[k]);
    for (std::size_t k = 0; k < b.size(); ++k) require_bits(r.denominator_derivatives[k], expected.b[k]);
}
} // namespace

TEST(default_policy_is_bit_identical_to_pre_m2_formulas) {
    const kan::RationalConfig c{3, 2, 0.1, 1.3, 1e-8};
    const std::vector<double> a{0.3, -0.2, 0.11, 0.05}, b{0.4, -0.25};
    require_pinned(c, 0.7, a, b, {0x1.ab48294361de5p-3, -0x1.1b8d2fcf27fd6p-4,
        {0x1.c48d639d74c0dp-1, 0x1.a1bd97077f76ep-2, 0x1.819b5055b0bc8p-3, 0x1.63f1d400545f3p-4},
        {-0x1.5c9e7dc8a69ccp-4, -0x1.41cd606a72695p-5}});
    require_pinned(c, -1.9, a, b, {-0x1.a7f9b2ce60195p+1, -0x1.b683912ed57cep+3,
        {-0x1.3507507507509p+2, 0x1.db6db6db6db6fp+2, -0x1.6db6db6db6db8p+3, 0x1.1951951951952p+4},
        {0x1.89b101767dce7p+4, -0x1.2ed6ed6ed6ed9p+5}});
    // Log-space paths (underflowing powers and quotients).
    require_pinned({0, 2, 0, 1, 1e-8}, 1e-200, {1e300}, {0, 0},
        {0x1.7e43c8800759cp+996, 0, {1}, {-0x1.249ad2594c37dp+332, -0x1.bff2ee48e0319p-333}});
    require_pinned({0, 1, 0, 1, 1e-8}, 1e300, {1e-300}, {1e-270},
        {0, 0, {0x1.4484bfeebc29fp-100}, {-0x1.9b604aaaca649p-200}});
    require_pinned({6, 4, -0.2, 0.9, 1e-8}, 1.3, {0.1, 0.2, -0.3, 0.05, 0.01, -0.02, 0.003}, {0.1, -0.05, 0.02, 0.01},
        {-0x1.e622d64d105cap-3, -0x1.3e8eff2a8dccap-1,
         {0x1.ab8be054741fbp-1, 0x1.6449e59bb61a6p+0, 0x1.28e83f5717c0ap+1, 0x1.eed8699127964p+1,
          0x1.9c5f02a3a0fd3p+2, 0x1.57a4823306285p+3, 0x1.1e5e6c7fda76ep+4},
         {0x1.524a62fb90aa0p-2, 0x1.19e8a7d1a3385p-1, 0x1.d5d917b2bab31p-1, 0x1.878a3e6a463fep+0}});
}

namespace {
// One 1 -> 1 rational edge trained on a single sample towards target 4.
// Guarded SGD with rate 1/16 moves b from -1/2 exactly to -1: a pole at x = 1.
kan::Layer pole_layer(DenominatorPolicy policy) {
    kan::Layer layer(1, 1, config(policy, 0, 1));
    kan::set_rational_parameters(layer, std::vector<double>{1}, std::vector<double>{-0.5}, std::vector<double>{0});
    return layer;
}
double loss_step(kan::Layer& layer, double rate) {
    const std::vector<double> x{1};
    const double y = layer.forward(x, 1)[0], u = y - 4;
    layer.sgd(layer.backward(x, 1, std::vector<double>{u}), rate);
    return 0.5 * u * u;
}
} // namespace

TEST(sgd_into_a_pole_stops_guarded_but_continues_under_safe_policies) {
    auto guarded = pole_layer(DenominatorPolicy::Guarded);
    loss_step(guarded, 0.0625);
    REQUIRE(test::denominators(guarded)[0] == -1);
    test::throws<std::domain_error>([&] { loss_step(guarded, 0.0625); });
    for (auto policy : safe_policies) {
        auto layer = pole_layer(policy);
        const double first = loss_step(layer, 0.0625);
        double last = first;
        for (int step = 0; step < 200; ++step) {
            last = loss_step(layer, 0.0625);
            REQUIRE(std::isfinite(last));
        }
        REQUIRE(last < 1e-6 * first);
    }
}

TEST(layer_and_network_vjps_under_safe_policies) {
    for (auto policy : safe_policies) {
        auto c = config(policy, 3, 2); c.center = 0.1; c.scale = 1.3;
        kan::Layer first(2, 3, c), second(3, 1, config(policy, 1, 3));
        for (auto* layer : {&first, &second}) {
            std::vector<double> a(layer->coefficients().size()), b(test::denominators(*layer).size());
            for (std::size_t k = 0; k < a.size(); ++k) a[k] = 0.3 * std::sin(static_cast<double>(k + 1));
            for (std::size_t k = 0; k < b.size(); ++k) b[k] = 0.4 * std::cos(static_cast<double>(2 * k + 1));
            kan::set_rational_parameters(*layer, a, b, std::vector<double>(layer->outputs(), 0.01));
        }
        kan::Network network({first, second});
        const std::vector<double> x{-0.5, 0.2, 0.8, -0.3, 0.1, 0.45}, u{0.2, -0.1, 0.3};
        const auto g = network.backward(x, 3, u);
        auto objective = [&](const kan::Network& n) {
            const auto y = n.forward(x, 3);
            double s = 0;
            for (std::size_t k = 0; k < y.size(); ++k) s += y[k] * u[k];
            return s;
        };
        const double h = 1e-6;
        for (std::size_t j = 0; j < 2; ++j) {
            const auto& layer = test::layer(network, j);
            const auto d = test::denominators(layer);
            for (std::size_t k = 0; k < d.size(); ++k) {
                std::vector<double> plus(d.begin(), d.end()), minus = plus;
                plus[k] += h; minus[k] -= h;
                auto lp = layer, lm = layer;
                const std::vector<double> a(layer.coefficients().begin(), layer.coefficients().end());
                const std::vector<double> bias(layer.bias().begin(), layer.bias().end());
                kan::set_rational_parameters(lp, a, plus, bias); kan::set_rational_parameters(lm, a, minus, bias);
                auto p = test::layers(network), m = p;
                p[j] = lp; m[j] = lm;
                test::near(test::denominators(test::grad(g, j))[k],
                           (objective(kan::Network(p)) - objective(kan::Network(m))) / (2 * h), 1e-7);
            }
        }
        for (std::size_t k = 0; k < x.size(); ++k) {
            auto p = x, m = x; p[k] += h; m[k] -= h;
            const auto yp = network.forward(p, 3), ym = network.forward(m, 3);
            double s = 0;
            for (std::size_t o = 0; o < yp.size(); ++o) s += (yp[o] - ym[o]) * u[o];
            test::near(g.input[k], s / (2 * h), 1e-7);
        }
    }
}

int main() { return test::run(); }

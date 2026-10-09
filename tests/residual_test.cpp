// Backlog M3: the optional SiLU residual branch of a Layer,
// y[b,o] = bias[o] + carrier_o(x_b) + sum_i w[o,i] silu(x[b,i]), on every carrier.
#include "kan/families.hpp"
#include "kan/initializers.hpp"
#include "kan/network.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

namespace {
constexpr double inf = std::numeric_limits<double>::infinity();

double silu(double x) { return x / (1 + std::exp(-x)); }
double silu_derivative(double x) {
    const double s = 1 / (1 + std::exp(-x));
    return s * (1 + x * (1 - s));
}

kan::BSplineConfig spline(std::size_t degree = 3) {
    kan::BSplineConfig b{degree, {}};
    b.knots.assign(degree + 1, -1);
    b.knots.insert(b.knots.end(), {-0.3, 0.4});
    b.knots.insert(b.knots.end(), degree + 1, 1);
    return b;
}

std::vector<double> pattern(std::size_t n, double scale, std::size_t period = 7) {
    std::vector<double> v(n);
    for (std::size_t j = 0; j < n; ++j) v[j] = scale * (double(j % period) - double(period / 2)) + 0.01 * double(j % 3);
    return v;
}

std::vector<double> std_vector(std::span<const double> s) { return {s.begin(), s.end()}; }

kan::SiluResidual weights(const kan::Layer& layer, double scale = 0.3) {
    return {pattern(layer.inputs() * layer.outputs(), scale, 5)};
}

// Fixtures with nonzero parameters, one per carrier.
kan::Layer linear(std::size_t inputs, std::size_t outputs, kan::BasisConfig basis) {
    kan::Layer l(inputs, outputs, std::move(basis));
    l.set_parameters(pattern(l.coefficients().size(), 0.05), pattern(outputs, 0.04, 3));
    return l;
}
kan::Layer rational(std::size_t inputs, std::size_t outputs) {
    kan::RationalConfig c;
    c.numerator_degree = 3;
    c.denominator_degree = 2;
    c.center = 0.1;
    c.scale = 1.3;
    c.denominator_policy = kan::DenominatorPolicy::Smooth;
    kan::Layer l(inputs, outputs, c);
    kan::set_rational_parameters(l, pattern(l.coefficients().size(), 0.04), pattern(test::denominators(l).size(), 0.1, 3),
                                 pattern(outputs, 0.03, 3));
    return l;
}
kan::Layer with_residual(kan::Layer layer, double scale = 0.3) {
    layer.set_residual(weights(layer, scale));
    return layer;
}

// Every trainable parameter group of a layer, readable, writable and with its gradient.
enum class Group { Coefficients, Bias, Residual, Centers, LogWidths, Denominators };
std::vector<double> values(const kan::Layer& l, Group g) {
    switch (g) {
    case Group::Coefficients: return std_vector(l.coefficients());
    case Group::Bias: return std_vector(l.bias());
    case Group::Residual: return l.residual() ? l.residual()->weights : std::vector<double>{};
    case Group::Centers: return std::holds_alternative<kan::TrainableRbfEdges>(l.carrier()) ? test::trainable(l).centers : std::vector<double>{};
    case Group::LogWidths: return std::holds_alternative<kan::TrainableRbfEdges>(l.carrier()) ? test::trainable(l).log_widths : std::vector<double>{};
    case Group::Denominators: return std_vector(test::denominators(l));
    }
    return {};
}
void assign(kan::Layer& l, Group g, const std::vector<double>& v) {
    const bool is_rational = std::holds_alternative<kan::RationalEdges>(l.carrier());
    auto c = values(l, Group::Coefficients), b = values(l, Group::Bias), d = values(l, Group::Denominators);
    switch (g) {
    case Group::Coefficients: c = v; break;
    case Group::Bias: b = v; break;
    case Group::Denominators: d = v; break;
    case Group::Residual: l.set_residual(kan::SiluResidual{v}); return;
    case Group::Centers: kan::set_rbf_parameters(l, v, test::trainable(l).log_widths); return;
    case Group::LogWidths: kan::set_rbf_parameters(l, test::trainable(l).centers, v); return;
    }
    if (is_rational) kan::set_rational_parameters(l, c, d, b);
    else l.set_parameters(c, b);
}
std::span<const double> gradient(const kan::LayerGradients& g, Group group) {
    switch (group) {
    case Group::Coefficients: return g.coefficients;
    case Group::Bias: return g.bias;
    case Group::Residual: return g.residual;
    case Group::Centers: return test::centers(g);
    case Group::LogWidths: return test::log_widths(g);
    case Group::Denominators: return test::denominators(g);
    }
    return {};
}

double objective(const kan::Network& n, const std::vector<double>& x, std::size_t batch, const std::vector<double>& u) {
    const auto y = n.forward(x, batch);
    return std::inner_product(y.begin(), y.end(), u.begin(), 0.0);
}

// Central differences of <u, f(x)> against every analytic VJP of the network.
void check_network(const std::vector<kan::NetworkLayer>& stages, const std::vector<double>& x, std::size_t batch) {
    const kan::Network n(stages);
    const auto u = pattern(batch * n.outputs(), 0.3, 5);
    const double h = 1e-6;
    const auto g = n.backward(x, batch, u);
    for (std::size_t j = 0; j < x.size(); ++j) {
        auto p = x, m = x;
        p[j] += h;
        m[j] -= h;
        test::near(g.input[j], (objective(n, p, batch, u) - objective(n, m, batch, u)) / (2 * h), 3e-7);
    }
    for (std::size_t k = 0; k < stages.size(); ++k) {
        const auto* layer = std::get_if<kan::Layer>(&stages[k]);
        if (!layer) continue;
        for (auto group : {Group::Coefficients, Group::Bias, Group::Residual, Group::Centers, Group::LogWidths, Group::Denominators}) {
            const auto v = values(*layer, group);
            const auto analytic = gradient(test::grad(g, k), group);
            REQUIRE(analytic.size() == v.size());
            for (std::size_t j = 0; j < v.size(); ++j) {
                auto plus = stages, minus = stages;
                auto vp = v, vm = v;
                vp[j] += h;
                vm[j] -= h;
                assign(std::get<kan::Layer>(plus[k]), group, vp);
                assign(std::get<kan::Layer>(minus[k]), group, vm);
                const double fd = (objective(kan::Network(plus), x, batch, u) - objective(kan::Network(minus), x, batch, u)) / (2 * h);
                test::near(analytic[j], fd, 3e-7);
            }
        }
    }
}

std::vector<kan::Layer> carriers(std::size_t inputs, std::size_t outputs) {
    return {linear(inputs, outputs, kan::ChebyshevConfig{4}), linear(inputs, outputs, spline()),
            linear(inputs, outputs, kan::TrainableRbfConfig{{-0.6, 0.1, 0.8}, {-0.3, 0.2, -0.1}}), rational(inputs, outputs)};
}
} // namespace

TEST(constructors_have_no_branch_and_gradients_leave_it_empty) {
    for (const auto& l : carriers(3, 2)) {
        REQUIRE(!l.residual());
        const auto g = l.backward(pattern(6, 0.2), 2, pattern(4, 0.3));
        REQUIRE(g.residual.empty());
        REQUIRE(l.regularization(0.1).gradients.residual.empty());
    }
}

TEST(forward_adds_the_branch_after_the_carrier_in_ascending_input_order) {
    const std::vector<double> x{-2.5, -0.4, 0.3, 0.9, 1.7, -0.05};
    for (const auto& plain : carriers(3, 2)) {
        const auto l = with_residual(plain);
        const auto& w = l.residual()->weights;
        const auto base = plain.forward(x, 2), y = l.forward(x, 2);
        for (std::size_t b = 0; b < 2; ++b)
            for (std::size_t o = 0; o < 2; ++o) {
                double r = 0;
                for (std::size_t i = 0; i < 3; ++i) r += w[o * 3 + i] * silu(x[b * 3 + i]);
                test::near(y[b * 2 + o], base[b * 2 + o] + r, 1e-15);
            }
    }
}

TEST(all_vjps_of_each_carrier_with_the_branch) {
    const std::vector<double> x{-0.7, 0.25, 0.6, 0.45, -0.15, -0.85};
    for (const auto& l : carriers(3, 2)) check_network({with_residual(l)}, x, 2);
}

TEST(all_vjps_of_mixed_networks_with_and_without_the_branch) {
    const std::vector<double> x{-0.7, 0.25, 0.6, 0.45};
    check_network({with_residual(rational(2, 3)), with_residual(linear(3, 2, kan::TrainableRbfConfig{{-0.6, 0.1, 0.8}, {-0.3, 0.2, -0.1}})),
                   linear(2, 1, spline())}, x, 2);
    check_network({kan::InputMap(2, kan::TanhMap{1.0}), with_residual(linear(2, 3, spline())), kan::InputMap(3, kan::TanhMap{1.0}),
                   with_residual(linear(3, 1, kan::ChebyshevConfig{3}), -0.4)}, x, 2);
}

// The backlog's motivation: localized carriers have zero value and zero
// gradient outside their support; the branch keeps a gradient path there.
TEST(branch_is_the_only_gradient_path_outside_localized_supports) {
    const std::vector<double> x{45.0, -41.0, 60.0, -52.0}; // outside [-1, 1] and 80+ widths from every center
    const std::vector<double> u{0.7, -0.4, 0.3, 0.9};
    for (const auto& plain : {linear(2, 2, spline()), linear(2, 2, kan::GaussianRbfConfig{{-1, 0, 1}, 0.5}),
                              linear(2, 2, kan::TrainableRbfConfig{{-1, 0, 1}, {-0.7, -0.7, -0.7}})}) {
        // The carrier itself is exactly zero there: the output is the bias.
        const auto y = plain.forward(x, 2);
        for (std::size_t b = 0; b < 2; ++b)
            for (std::size_t o = 0; o < 2; ++o) REQUIRE(y[b * 2 + o] == plain.bias()[o]);
        const auto without = plain.backward(x, 2, u);
        for (double v : without.input) REQUIRE(v == 0);
        const auto l = with_residual(plain);
        const auto with = l.backward(x, 2, u);
        const auto& w = l.residual()->weights;
        for (std::size_t b = 0; b < 2; ++b)
            for (std::size_t i = 0; i < 2; ++i) {
                const double t = u[b * 2] * w[i] + u[b * 2 + 1] * w[2 + i];
                REQUIRE(with.input[b * 2 + i] != 0);
                test::near(with.input[b * 2 + i], silu_derivative(x[b * 2 + i]) * t, 1e-14);
            }
    }
}

TEST(branch_restores_first_layer_gradients_behind_a_saturated_localized_layer) {
    // Layer 1 sees in-domain inputs; its bias moves its outputs far outside
    // layer 2's support, so without the branch nothing reaches layer 1.
    const std::vector<double> x{-0.5, 0.2, 0.35, -0.6};
    for (const auto& second : {linear(2, 1, spline()), linear(2, 1, kan::GaussianRbfConfig{{-1, 0, 1}, 0.5}),
                               linear(2, 1, kan::TrainableRbfConfig{{-1, 0, 1}, {-0.7, -0.7, -0.7}})}) {
        auto first = linear(2, 2, spline());
        first.set_parameters(std_vector(first.coefficients()), std::vector<double>{40, -40});
        const std::vector<double> u{1.0, -0.5};
        const auto without = kan::Network({first, second}).backward(x, 2, u);
        for (double v : test::grad(without, 0).coefficients) REQUIRE(v == 0);
        for (double v : test::grad(without, 0).bias) REQUIRE(v == 0);
        for (double v : without.input) REQUIRE(v == 0);
        const auto with = kan::Network({first, with_residual(second)}).backward(x, 2, u);
        double norm = 0;
        for (double v : test::grad(with, 0).coefficients) norm += v * v;
        REQUIRE(norm > 0);
        for (double v : test::grad(with, 0).bias) REQUIRE(v != 0);
        // Layer 1 itself has no branch: its gradient has no residual field.
        REQUIRE(test::grad(with, 0).residual.empty());
        REQUIRE(test::grad(with, 1).residual.size() == 2);
    }
}

TEST(set_residual_validates_shape_and_finiteness_atomically) {
    auto l = linear(3, 2, kan::ChebyshevConfig{4});
    test::throws<std::invalid_argument>([&] { l.set_residual(kan::SiluResidual{std::vector<double>(5)}); });
    test::throws<std::invalid_argument>([&] { l.set_residual(kan::SiluResidual{{}}); });
    REQUIRE(!l.residual());
    l.set_residual(kan::SiluResidual{pattern(6, 0.2)});
    REQUIRE(l.residual()->weights == pattern(6, 0.2));
    auto bad = pattern(6, 0.2);
    bad[4] = inf;
    test::throws<std::invalid_argument>([&] { l.set_residual(kan::SiluResidual{bad}); });
    bad[4] = std::numeric_limits<double>::quiet_NaN();
    test::throws<std::invalid_argument>([&] { l.set_residual(kan::SiluResidual{bad}); });
    test::throws<std::invalid_argument>([&] { l.set_residual(kan::SiluResidual{std::vector<double>(7)}); });
    REQUIRE(l.residual()->weights == pattern(6, 0.2));
    l.set_residual(std::nullopt);
    REQUIRE(!l.residual());
    // A moved-from layer is rejected.
    auto source = with_residual(linear(3, 2, kan::ChebyshevConfig{4}));
    auto target = std::move(source);
    test::throws<std::invalid_argument>([&] { source.set_residual(kan::SiluResidual{pattern(6, 0.2)}); });
    test::throws<std::invalid_argument>([&] { (void)source.forward(pattern(3, 0.1), 1); });
    REQUIRE(target.residual()->weights.size() == 6);
}

TEST(branch_survives_every_carrier_replacement_and_copies) {
    const std::vector<double> samples{-0.8, -0.75, -0.7, 0.1, 0.2};
    auto s = with_residual(linear(2, 2, spline()));
    const auto w = *s.residual();
    kan::insert_knot(s, 0.1);
    REQUIRE(*s.residual() == w);
    kan::adapt_grid(s, samples);
    REQUIRE(*s.residual() == w);
    s.set_parameters(pattern(s.coefficients().size(), 0.02), std::vector<double>{0, 0});
    REQUIRE(*s.residual() == w);
    s.set_carrier(kan::BasisEdges{kan::ChebyshevConfig{3}, pattern(12, 0.01)});
    REQUIRE(*s.residual() == w);
    auto r = with_residual(linear(2, 2, kan::TrainableRbfConfig{{-0.6, 0.1, 0.8}, {-0.3, 0.2, -0.1}}));
    kan::set_rbf_parameters(r, std::vector<double>{-0.5, 0, 0.5}, std::vector<double>{0, 0, 0});
    REQUIRE(*r.residual() == w);
    auto q = with_residual(rational(2, 2));
    kan::set_rational_parameters(q, std_vector(q.coefficients()), pattern(test::denominators(q).size(), 0.05), std::vector<double>{1, 2});
    REQUIRE(*q.residual() == w);
    // Copies are independent values.
    auto copy = q;
    REQUIRE(*copy.residual() == *q.residual());
    copy.set_residual(kan::SiluResidual{pattern(4, 0.7)});
    REQUIRE(*q.residual() == w);
    REQUIRE(!(*copy.residual() == *q.residual()));
    REQUIRE((kan::SiluResidual{{1, 2}} == kan::SiluResidual{{1, 2}}));
    // Network copies keep the branch; a failed carrier replacement keeps it too.
    const kan::Network n({q, copy});
    REQUIRE(*test::layer(n, 0).residual() == w);
    test::throws<std::invalid_argument>([&] { q.set_carrier(kan::BasisEdges{kan::ChebyshevConfig{3}, {}}); });
    REQUIRE(*q.residual() == w);
}

TEST(sgd_updates_the_branch_and_validates_its_gradient) {
    for (const auto& plain : carriers(3, 2)) {
        auto l = with_residual(plain);
        const auto x = pattern(6, 0.2), u = pattern(4, 0.3);
        const auto g = l.backward(x, 2, u);
        REQUIRE(g.residual.size() == 6);
        const auto before = l;
        auto wrong = g;
        wrong.residual.clear();
        test::throws<std::invalid_argument>([&] { l.sgd(wrong, 0.1); });
        wrong = g;
        wrong.residual.push_back(0);
        test::throws<std::invalid_argument>([&] { l.sgd(wrong, 0.1); });
        wrong = g;
        wrong.residual[2] = inf;
        test::throws<std::invalid_argument>([&] { l.sgd(wrong, 0.1); });
        wrong = g;
        wrong.residual[2] = std::numeric_limits<double>::max();
        test::throws<std::overflow_error>([&] { l.sgd(wrong, 4); });
        REQUIRE(*l.residual() == *before.residual());
        REQUIRE(l.carrier() == before.carrier());
        l.sgd(g, 0.1);
        for (std::size_t j = 0; j < 6; ++j) REQUIRE(l.residual()->weights[j] == before.residual()->weights[j] - 0.1 * g.residual[j]);
        // A gradient with a residual field is rejected by a layer without the branch.
        auto without = plain;
        test::throws<std::invalid_argument>([&] { without.sgd(g, 0.1); });
        REQUIRE(without.carrier() == plain.carrier());
    }
}

TEST(l2_penalizes_coefficients_and_branch_weights) {
    for (const auto& plain : carriers(3, 2)) {
        const auto l = with_residual(plain);
        const auto c = std_vector(l.coefficients());
        const auto& w = l.residual()->weights;
        const auto none = plain.regularization(0.3), r = l.regularization(0.3);
        double expected = 0;
        for (double v : c) expected += v * v;
        for (double v : w) expected += v * v;
        test::near(r.value, 0.15 * expected, 1e-14);
        REQUIRE(r.value > none.value);
        REQUIRE(r.gradients.coefficients == none.gradients.coefficients);
        REQUIRE(r.gradients.residual.size() == w.size());
        for (std::size_t j = 0; j < w.size(); ++j) REQUIRE(r.gradients.residual[j] == 0.3 * w[j]);
        // The penalty gradient is a valid SGD gradient.
        auto trained = l;
        trained.sgd(r.gradients, 0.5);
        for (std::size_t j = 0; j < w.size(); ++j) test::near(trained.residual()->weights[j], 0.85 * w[j], 1e-15);
    }
}

TEST(network_backward_sgd_and_regularization_carry_the_branch) {
    kan::Network n({with_residual(linear(2, 3, spline())), linear(3, 1, kan::ChebyshevConfig{3})});
    const auto x = pattern(4, 0.2);
    const auto g = n.backward(x, 2, std::vector<double>{1, -1});
    REQUIRE(test::grad(g, 0).residual.size() == 6);
    REQUIRE(test::grad(g, 1).residual.empty());
    const auto penalty = n.regularization(0.2);
    REQUIRE(test::grad(penalty.gradients, 0).residual.size() == 6);
    REQUIRE(test::grad(penalty.gradients, 1).residual.empty());
    const auto w = test::layer(n, 0).residual()->weights;
    n.sgd(g, 0.1);
    for (std::size_t j = 0; j < 6; ++j) REQUIRE(test::layer(n, 0).residual()->weights[j] == w[j] - 0.1 * test::grad(g, 0).residual[j]);
    // A failed network step changes no layer.
    auto bad = g;
    test::grad(bad, 0).residual[0] = std::numeric_limits<double>::max();
    const auto before = test::layer(n, 0).residual()->weights;
    test::throws<std::overflow_error>([&] { n.sgd(bad, 4); });
    REQUIRE(test::layer(n, 0).residual()->weights == before);
}

TEST(nonfinite_branch_results_raise_overflow) {
    auto l = linear(1, 1, kan::ChebyshevConfig{2});
    l.set_residual(kan::SiluResidual{{std::numeric_limits<double>::max()}});
    test::throws<std::overflow_error>([&] { (void)l.forward(std::vector<double>{1e10}, 1); });
    test::throws<std::overflow_error>([&] { (void)l.backward(std::vector<double>{1e10}, 1, std::vector<double>{10}); });
    // Extreme finite inputs are fine with ordinary weights.
    l.set_residual(kan::SiluResidual{{0.5}});
    const auto y = l.forward(std::vector<double>{-std::numeric_limits<double>::max()}, 1);
    REQUIRE(std::isfinite(y[0]));
}

// ---- Initializers ----------------------------------------------------------

namespace {
void fnv(std::uint64_t& h, std::span<const double> values) {
    for (double v : values) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        for (int b = 0; b < 8; ++b) h = (h ^ ((bits >> (8 * b)) & 0xff)) * 0x100000001b3ull;
    }
}
std::uint64_t digest(const kan::Layer& layer) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    fnv(h, layer.coefficients());
    fnv(h, test::denominators(layer));
    fnv(h, layer.bias());
    if (layer.residual()) fnv(h, layer.residual()->weights);
    return h;
}
std::string hex(std::uint64_t v) {
    char buffer[17];
    std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(v));
    return buffer;
}
kan::BSplineConfig uniform_spline(std::size_t degree, std::size_t intervals) {
    kan::BSplineConfig c{degree, {}};
    c.knots.assign(degree, -1);
    for (std::size_t j = 0; j <= intervals; ++j) c.knots.push_back(-1 + 2 * double(j) / double(intervals));
    c.knots.insert(c.knots.end(), degree, 1);
    return c;
}
kan::Layer zero_branch(kan::Layer layer) {
    layer.set_residual(kan::SiluResidual{std::vector<double>(layer.inputs() * layer.outputs(), 0.0)});
    return layer;
}
} // namespace

TEST(initializers_leave_layers_without_the_branch_unchanged_in_kind) {
    auto l = kan::Layer(4, 3, kan::ChebyshevConfig{5});
    auto reference = l;
    kan::initialize(l, kan::NoiseInit{});
    REQUIRE(!l.residual());
    // The carrier draws come first: a layer with the branch draws the same coefficients.
    auto branched = zero_branch(reference);
    kan::initialize(branched, kan::NoiseInit{});
    REQUIRE(branched.carrier() == l.carrier());
    REQUIRE(branched.residual()->weights.size() == 12);
    auto rational_plain = rational(3, 2), rational_branched = zero_branch(rational(3, 2));
    kan::initialize(rational_plain, kan::NoiseInit{0.3, kan::Distribution::Normal, 3, {}});
    kan::initialize(rational_branched, kan::NoiseInit{0.3, kan::Distribution::Normal, 3, {}});
    REQUIRE(rational_plain.carrier() == rational_branched.carrier());
}

TEST(noise_init_draws_pykan_scale_base) {
    // pykan: scale_base = mu/sqrt(in) + sigma * U(-1, 1)/sqrt(in).
    auto l = zero_branch(kan::Layer(16, 64, kan::ChebyshevConfig{3}));
    kan::initialize(l, kan::NoiseInit{0.3, kan::Distribution::Uniform, 11, {}, 0.5, 2.0});
    const auto& w = l.residual()->weights;
    double mean = 0, second = 0;
    for (double v : w) {
        REQUIRE(v >= (0.5 - 2.0) / 4 && v < (0.5 + 2.0) / 4);
        mean += v;
    }
    mean /= double(w.size());
    for (double v : w) second += (v - mean) * (v - mean);
    second /= double(w.size());
    test::near(mean, 0.5 / 4, 0.03);
    test::near(second, 4.0 / 3 / 16, 0.03); // Var(sigma U(-1,1)/sqrt(in)) = sigma^2/(3 in)
    auto n = zero_branch(kan::Layer(16, 64, kan::ChebyshevConfig{3}));
    kan::initialize(n, kan::NoiseInit{0.3, kan::Distribution::Normal, 11, {}, 0.5, 2.0});
    mean = second = 0;
    for (double v : n.residual()->weights) mean += v;
    mean /= double(w.size());
    for (double v : n.residual()->weights) second += (v - mean) * (v - mean);
    second /= double(w.size());
    test::near(mean, 0.5 / 4, 0.03);
    test::near(second, 4.0 / 3 / 16, 0.03);
    // Zero spread gives exactly the mean.
    kan::initialize(n, kan::NoiseInit{0.3, kan::Distribution::Uniform, 1, {}, 0.8, 0.0});
    for (double v : n.residual()->weights) REQUIRE(v == 0.8 / 4);
    // Defaults: mean 0, spread 1.
    REQUIRE(kan::NoiseInit{}.residual_mean == 0 && kan::NoiseInit{}.residual_spread == 1);
}

TEST(noise_init_validates_branch_fields_before_any_change) {
    auto l = zero_branch(linear(3, 2, kan::ChebyshevConfig{3}));
    const auto before = l;
    test::throws<std::invalid_argument>([&] { kan::initialize(l, kan::NoiseInit{0.3, kan::Distribution::Uniform, 0, {}, inf, 1}); });
    test::throws<std::invalid_argument>([&] { kan::initialize(l, kan::NoiseInit{0.3, kan::Distribution::Uniform, 0, {}, 0, -1}); });
    test::throws<std::invalid_argument>([&] { kan::initialize(l, kan::NoiseInit{0.3, kan::Distribution::Uniform, 0, {}, 0, inf}); });
    REQUIRE(l.carrier() == before.carrier() && *l.residual() == *before.residual());
}

TEST(variance_scaling_sets_branch_weights_to_zero) {
    for (auto layer : carriers(3, 2)) {
        layer.set_residual(kan::SiluResidual{pattern(6, 0.4)});
        kan::initialize(layer, kan::VarianceScaling{1, kan::Distribution::Normal, 4, {}});
        for (double v : layer.residual()->weights) REQUIRE(v == 0);
        // Not a saddle: the branch gradient sum_b u silu(x) is nonzero.
        const auto g = layer.backward(pattern(6, 0.3), 2, pattern(4, 0.3));
        double norm = 0;
        for (double v : g.residual) norm += v * v;
        REQUIRE(norm > 0);
    }
}

TEST(network_initialization_keeps_branch_presence_per_layer) {
    kan::Network n({zero_branch(kan::Layer(2, 3, kan::ChebyshevConfig{3})), kan::Layer(3, 1, kan::ChebyshevConfig{3})});
    kan::initialize(n, kan::NoiseInit{});
    REQUIRE(test::layer(n, 0).residual());
    REQUIRE(!test::layer(n, 1).residual());
    double norm = 0;
    for (double v : test::layer(n, 0).residual()->weights) norm += v * v;
    REQUIRE(norm > 0);
}

TEST(branch_draws_are_pinned_across_platforms) {
    kan::RationalConfig smooth;
    smooth.denominator_policy = kan::DenominatorPolicy::Smooth;
    smooth.numerator_degree = 4;
    smooth.denominator_degree = 3;
    struct Case { kan::Layer layer; kan::Initializer init; const char* pinned; };
    std::vector<Case> cases{
        {zero_branch(kan::Layer(5, 4, uniform_spline(3, 7))), kan::NoiseInit{0.3, kan::Distribution::Uniform, 4, {}}, "0000000000000000"},
        {zero_branch(kan::Layer(5, 4, kan::FourierConfig{5, 2.5})), kan::NoiseInit{0.3, kan::Distribution::Normal, 8, {}, 0.2, 0.7}, "0000000000000000"},
        {zero_branch(kan::Layer(5, 4, smooth)), kan::NoiseInit{0.3, kan::Distribution::Normal, 9, {0.4, 2}}, "0000000000000000"},
    };
    std::string failures;
    for (auto& c : cases) {
        auto copy = c.layer;
        kan::initialize(c.layer, c.init);
        kan::initialize(copy, c.init);
        REQUIRE(digest(c.layer) == digest(copy));
        const auto value = hex(digest(c.layer));
        std::printf("pinned %s\n", value.c_str());
        if (value != c.pinned) failures += value + " != " + c.pinned + "; ";
    }
    if (!failures.empty()) throw std::runtime_error(failures);
}

int main() { return test::run(); }

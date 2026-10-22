#include "kan/families.hpp"
#include "kan/network.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

namespace {
kan::BSplineConfig spline(std::size_t degree = 3) {
    kan::BSplineConfig b{degree, {}};
    b.knots.assign(degree + 1, -1);
    b.knots.push_back(0);
    b.knots.insert(b.knots.end(), degree + 1, 1);
    return b;
}
kan::TrainableRbfConfig rbf() { return {{-0.6, 0.1, 0.8}, {-0.3, 0.2, -0.1}}; }
const std::vector<double>& centers(const kan::Layer& l) { return test::trainable(l).centers; }
const std::vector<double>& log_widths(const kan::Layer& l) { return test::trainable(l).log_widths; }
kan::Layer fixture(kan::BasisConfig b) {
    kan::Layer l(2, 2, b);
    std::vector<double> c(l.coefficients().size());
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = 0.07 * (static_cast<double>(i % 9) - 4);
    l.set_parameters(c, std::vector<double>{0.2, -0.1});
    return l;
}
double loss(const kan::Layer& l, const std::vector<double>& x, const std::vector<double>& u) {
    auto y = l.forward(x, 2);
    return std::inner_product(y.begin(), y.end(), u.begin(), 0.0);
}
void unchanged(const kan::Layer& a, const kan::Layer& b) {
    REQUIRE(
        std::equal(a.coefficients().begin(), a.coefficients().end(), b.coefficients().begin(), b.coefficients().end()));
    REQUIRE(a.carrier() == b.carrier());
}
}
TEST(trainable_shared_rbf_vjp_and_sgd) {
    auto l = fixture(rbf());
    const std::vector<double> x{-0.5, 0.2, 0.7, -0.1}, u{0.4, -0.8, 0.3, 0.2};
    auto g = l.backward(x, 2, u);
    REQUIRE(test::trainable(g).centers.size() == 3);
    REQUIRE(test::trainable(g).log_widths.size() == 3);
    for (std::size_t k = 0; k < 3; ++k) {
        auto p = l, m = l;
        auto pc = centers(l), mc = pc, w = log_widths(l);
        pc[k] += 1e-6;
        mc[k] -= 1e-6;
        kan::set_rbf_parameters(p, pc, w);
        kan::set_rbf_parameters(m, mc, w);
        test::near(test::trainable(g).centers[k], (loss(p, x, u) - loss(m, x, u)) / 2e-6, 2e-7);
        p = l;
        m = l;
        auto pw = w, mw = w;
        pw[k] += 1e-6;
        mw[k] -= 1e-6;
        kan::set_rbf_parameters(p, centers(l), pw);
        kan::set_rbf_parameters(m, centers(l), mw);
        test::near(test::trainable(g).log_widths[k], (loss(p, x, u) - loss(m, x, u)) / 2e-6, 2e-7);
    }
    auto before = l;
    l.sgd(g, 0.03);
    for (std::size_t k = 0; k < 3; ++k) {
        test::near(centers(l)[k], centers(before)[k] - 0.03 * test::trainable(g).centers[k]);
        test::near(log_widths(l)[k], log_widths(before)[k] - 0.03 * test::trainable(g).log_widths[k]);
    }
    auto z = l.backward({}, 0, {});
    for (double v : test::trainable(z).centers) test::near(v, 0);
    for (double v : test::trainable(z).log_widths) test::near(v, 0);
}
TEST(rbf_invalid_updates_are_atomic_across_network) {
    auto l = fixture(rbf());
    const auto before = l;
    test::throws<std::invalid_argument>(
        [&] { kan::set_rbf_parameters(l, std::vector<double>{1}, std::vector<double>{1}); });
    unchanged(l, before);
    auto g = l.backward(std::vector<double>{0.2, 0.4}, 1, std::vector<double>{1, 1});
    test::trainable(g).log_widths[1] = -1000;
    test::throws<std::overflow_error>([&] { l.sgd(g, 1); });
    unchanged(l, before);
    test::trainable(g).log_widths[1] = 1000;
    test::throws<std::overflow_error>([&] { l.sgd(g, 1); });
    unchanged(l, before);
    kan::Network n({l, l});
    auto ng = n.backward(std::vector<double>{0.2, 0.4}, 1, std::vector<double>{1, 1});
    test::trainable(test::grad(ng, 1)).centers[0] = std::numeric_limits<double>::infinity();
    test::throws<std::invalid_argument>([&] { n.sgd(ng, 0.1); });
    for (auto& q : test::layers(n)) unchanged(q, before);
    kan::Layer fixed(1, 1, {});
    test::throws<std::invalid_argument>([&] { kan::set_rbf_parameters(fixed, {}, {}); });
}
TEST(exact_knot_insertion_preserves_values_and_derivatives) {
    for (std::size_t p = 0; p <= 4; ++p) {
        auto l = fixture(spline(p)), before = l;
        kan::insert_knot(l, 0.3);
        REQUIRE(l.terms() == before.terms() + 1);
        for (double x : {-2., -1., -0.7, -0.001, 0., 0.15, 0.3, 0.65, 1., 2.}) {
            const std::vector<double> a{x, x};
            auto y = l.forward(a, 1), ref = before.forward(a, 1);
            for (std::size_t k = 0; k < y.size(); ++k) test::near(y[k], ref[k], 2e-12);
            // Exclude new degree-zero jump conventions (no derivative everywhere).
            auto g = l.backward(a, 1, std::vector<double>{0.3, 0.7}),
                 r = before.backward(a, 1, std::vector<double>{0.3, 0.7});
            for (std::size_t k = 0; k < g.input.size(); ++k) test::near(g.input[k], r.input[k], 2e-11);
        }
        const auto snapshot = l;
        test::throws<std::invalid_argument>([&] { kan::insert_knot(l, 1); });
        unchanged(l, snapshot);
        for (std::size_t count = 0; count < p; ++count) kan::insert_knot(l, 0.3);
        const auto full = l;
        test::throws<std::invalid_argument>([&] { kan::insert_knot(l, 0.3); });
        unchanged(l, full);
        auto stale = before.backward(std::vector<double>{0.1, 0.2}, 1, std::vector<double>{1, 1});
        test::throws<std::invalid_argument>([&] { l.sgd(stale, 0.1); });
    }
}
TEST(data_adaptation_and_indexed_network_refinement) {
    auto l = fixture(spline()), before = l;
    const double knot = kan::adapt_grid(l, std::vector<double>{-0.8, -0.6, -0.4, 0.7, 9});
    test::near(knot, -0.6);
    auto y = l.forward(std::vector<double>{-0.7, 0.4}, 1), ref = before.forward(std::vector<double>{-0.7, 0.4}, 1);
    for (std::size_t i = 0; i < y.size(); ++i) test::near(y[i], ref[i]);
    const auto snapshot = l;
    test::throws<std::invalid_argument>([&] { kan::adapt_grid(l, {}); });
    test::throws<std::invalid_argument>([&] { kan::adapt_grid(l, std::vector<double>{4}); });
    test::throws<std::invalid_argument>(
        [&] { kan::adapt_grid(l, std::vector<double>{std::numeric_limits<double>::quiet_NaN()}); });
    unchanged(l, snapshot);
    kan::Network n({before, before});
    auto output = n.forward(std::vector<double>{0.3, 0.4}, 1);
    n.insert_knot(1, 0.4);
    n.adapt_grid(0, std::vector<double>{-0.5});
    auto after = n.forward(std::vector<double>{0.3, 0.4}, 1);
    for (std::size_t i = 0; i < after.size(); ++i) test::near(after[i], output[i]);
    test::throws<std::invalid_argument>([&] { n.insert_knot(8, 0.2); });
}
TEST(l2_objective_vjp_and_zero_defaults) {
    auto l = fixture(rbf());
    auto r = l.regularization(0.3);
    double sum = 0;
    for (std::size_t k = 0; k < l.coefficients().size(); ++k) {
        sum += 0.15 * l.coefficients()[k] * l.coefficients()[k];
        test::near(r.gradients.coefficients[k], 0.3 * l.coefficients()[k]);
    }
    test::near(r.value, sum);
    for (double v : r.gradients.bias) test::near(v, 0);
    for (double v : test::trainable(r.gradients).centers) test::near(v, 0);
    for (double v : test::trainable(r.gradients).log_widths) test::near(v, 0);
    test::near(l.regularization(0).value, 0);
    auto before = l;
    test::throws<std::invalid_argument>([&] { l.regularization(-1); });
    unchanged(l, before);
    kan::Network n({l, l});
    auto nr = n.regularization(0.3);
    test::near(nr.value, 2 * sum);
    n.sgd(nr.gradients, 0.01);
    for (const auto& q : test::layers(n))
        for (std::size_t k = 0; k < q.coefficients().size(); ++k)
            test::near(q.coefficients()[k], 0.997 * l.coefficients()[k]);
}
// Backlog C5: the CPU evaluates B-spline layers on the degree+1 terms that can
// be nonzero (sparse engine). These cases pin it bitwise, signed zeros
// included, to the dense M1 loops over every term, recomputed here from
// kan::evaluate_basis in the same order: forward sum = bias, then
// += c[o,i,k]*Phi[i,k] for i, k ascending; backward per sample, for i, o, k:
// dC += u*Phi, dx += (u*c)*Phi'. Nonfinite dense results must stay nonfinite
// (Layer raises), including NaN from an overflowing u*c at a zero Phi'.
namespace {
struct DenseSpline {
    std::vector<double> output, input, coefficients;
    bool output_finite = true, gradient_finite = true;
};
DenseSpline dense_spline(const kan::Layer& l, const std::vector<double>& x, std::size_t batch,
                         const std::vector<double>& u) {
    const auto& basis = test::basis_of(l);
    const auto inputs = l.inputs(), outputs = l.outputs(), terms = l.terms();
    const auto& c = l.coefficients();
    DenseSpline r{std::vector<double>(batch * outputs), std::vector<double>(batch * inputs, 0.0),
                  std::vector<double>(c.size(), 0.0)};
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<kan::BasisValues> rows;
        for (std::size_t i = 0; i < inputs; ++i) rows.push_back(kan::evaluate_basis(basis, x[b * inputs + i]));
        for (std::size_t o = 0; o < outputs; ++o) {
            double sum = l.bias()[o];
            for (std::size_t i = 0; i < inputs; ++i)
                for (std::size_t k = 0; k < terms; ++k) sum += c[(o * inputs + i) * terms + k] * rows[i].values[k];
            r.output[b * outputs + o] = sum;
        }
        for (std::size_t i = 0; i < inputs; ++i) {
            double sum = r.input[b * inputs + i];
            for (std::size_t o = 0; o < outputs; ++o) {
                const double v = u[b * outputs + o];
                const auto edge = (o * inputs + i) * terms;
                for (std::size_t k = 0; k < terms; ++k) {
                    r.coefficients[edge + k] += v * rows[i].values[k];
                    sum += (v * c[edge + k]) * rows[i].derivatives[k];
                }
            }
            r.input[b * inputs + i] = sum;
        }
    }
    for (double v : r.output) r.output_finite = r.output_finite && std::isfinite(v);
    for (const auto* g : {&r.input, &r.coefficients})
        for (double v : *g) r.gradient_finite = r.gradient_finite && std::isfinite(v);
    return r;
}
bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
                                              [](double p, double q) { return std::memcmp(&p, &q, sizeof p) == 0; });
}
void require_dense_equivalence(const kan::Layer& l, const std::vector<double>& x, std::size_t batch,
                               const std::vector<double>& u) {
    const auto e = dense_spline(l, x, batch, u);
    if (e.output_finite)
        REQUIRE(same_bits(l.forward(x, batch), e.output));
    else
        test::throws<std::overflow_error>([&] { l.forward(x, batch); });
    if (e.gradient_finite) {
        const auto g = l.backward(x, batch, u);
        REQUIRE(same_bits(g.input, e.input));
        REQUIRE(same_bits(g.coefficients, e.coefficients));
    } else {
        test::throws<std::overflow_error>([&] { l.backward(x, batch, u); });
    }
}
kan::BSplineConfig clamped(std::size_t degree, std::vector<double> interior, double low = -1, double high = 1) {
    kan::BSplineConfig b{degree, std::vector<double>(degree + 1, low)};
    b.knots.insert(b.knots.end(), interior.begin(), interior.end());
    b.knots.insert(b.knots.end(), degree + 1, high);
    return b;
}
// Inputs at, between and outside the knots (outside: every term is zero).
std::vector<double> spline_inputs(const kan::BSplineConfig& b, std::size_t count) {
    const double low = b.knots[b.degree], high = b.knots[b.knots.size() - b.degree - 1];
    std::vector<double> x{low, high, low - 0.5, high + 0.25, -1e300, 1e300};
    for (double t : b.knots)
        if (std::isfinite(t)) x.push_back(t);
    for (std::size_t k = 0; x.size() < count; ++k)
        x.push_back((0.5 * low + 0.5 * high) +
                    (0.5 * high - 0.5 * low) * 1.2 * std::sin(0.37 * static_cast<double>(k)));
    x.resize(count);
    return x;
}
}
TEST(spline_sparse_engine_matches_dense_bitwise) {
    std::vector<kan::BSplineConfig> configs{clamped(0, {-0.25, 0.5}),
                                            clamped(1, {0}),
                                            clamped(2, {-0.5, 0, 0, 0.5}),
                                            clamped(3, {-0.5, 0, 0, 0.5}),
                                            clamped(3, {-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75}),
                                            clamped(3, {0, 0, 0, 0}),
                                            clamped(5, {-0.3, 0.1, 0.6}),
                                            clamped(16, {0}, -2, 2),
                                            clamped(2, {0}, -1e308, 1e308)};
    configs.push_back(clamped(3, [] {
        std::vector<double> t;
        for (int k = 1; k < 45; ++k) t.push_back(-1 + k / 22.5);
        return t;
    }()));
    for (const auto& b : configs) {
        for (std::size_t outputs : {1u, 3u}) {
            constexpr std::size_t inputs = 4, batch = 7;
            kan::Layer l(inputs, outputs, b);
            std::vector<double> c(l.coefficients().size()), bias(outputs);
            for (std::size_t k = 0; k < c.size(); ++k) c[k] = 0.3 * std::sin(1.7 * static_cast<double>(k) + 0.2);
            for (std::size_t o = 0; o < outputs; ++o) bias[o] = 0.01 * static_cast<double>(o);
            l.set_parameters(c, bias);
            std::vector<double> u(batch * outputs);
            for (std::size_t k = 0; k < u.size(); ++k) u[k] = std::cos(0.9 * static_cast<double>(k));
            require_dense_equivalence(l, spline_inputs(b, batch * inputs), batch, u);
        }
    }
}
// A -0 bias stays -0 only when every dense term is -0: an input outside the
// domain adds the +0 products c*0 of its nonnegative coefficients.
TEST(spline_sparse_engine_keeps_dense_signed_zeros) {
    const auto b = clamped(2, {0});
    kan::Layer l(2, 2, b);
    std::vector<double> c(l.coefficients().size(), -0.5);
    for (std::size_t k = 0; k < l.terms(); ++k) c[k] = 0.5; // output 0, input 0: positive coefficients
    l.set_parameters(c, std::vector<double>{-0.0, -0.0});
    const std::vector<double> x{3, -3, 0.25, 7};
    const auto y = l.forward(x, 2);
    REQUIRE(!std::signbit(y[0]));             // +0 from c*0 = +0 terms of input 0
    REQUIRE(std::signbit(y[1]) && y[1] == 0); // -0: only -0 terms
    require_dense_equivalence(l, x, 2, std::vector<double>{1, -1, 0.5, 2});
    l.set_parameters(std::vector<double>(c.size(), -0.0), std::vector<double>{-0.0, 0.0});
    require_dense_equivalence(l, x, 2, std::vector<double>{-0.0, 0.0, 1e-320, -1e-320});
}
// u*c overflows at a term whose Phi' is zero: the dense loop forms
// inf*0 = NaN and the layer reports it.
TEST(spline_sparse_engine_reports_overflow_at_zero_terms) {
    const auto b = clamped(1, {-0.5, 0, 0.5});
    kan::Layer l(1, 1, b);
    std::vector<double> c(l.terms(), 0.1);
    c.back() = 1e300; // nonzero only for x near the upper end
    l.set_parameters(c, std::vector<double>{0});
    const std::vector<double> x{-0.9};
    test::throws<std::overflow_error>([&] { l.backward(x, 1, std::vector<double>{1e10}); });
    require_dense_equivalence(l, x, 1, std::vector<double>{1e10});
    require_dense_equivalence(l, x, 1, std::vector<double>{1e-10}); // bound finite: sparse terms only
    require_dense_equivalence(l, std::vector<double>{4}, 1, std::vector<double>{1e10}); // outside the domain
}

int main() { return test::run(); }

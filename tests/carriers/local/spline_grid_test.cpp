// Backlog M6: per-input B-spline grids, quantile grid refit (pykan's
// update_grid_from_samples) and grid adaptation with samples propagated
// through the preceding layers of a network.
#include "kan/families.hpp"
#include "kan/initializers.hpp"
#include "kan/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <variant>

namespace {
// Clamped knots of degree p with the interior knots `interior` on [lo, hi].
std::vector<double> clamped(std::size_t p, double lo, double hi, const std::vector<double>& interior) {
    std::vector<double> t(p + 1, lo);
    t.insert(t.end(), interior.begin(), interior.end());
    t.insert(t.end(), p + 1, hi);
    return t;
}
std::vector<double> uniform_interior(double lo, double hi, std::size_t intervals) {
    std::vector<double> v;
    for (std::size_t j = 1; j < intervals; ++j) v.push_back(lo + (hi - lo) * double(j) / double(intervals));
    return v;
}
// Three different grids of degree p with 5 intervals each (8 terms for p = 3).
kan::BSplineConfig three_grids(std::size_t p = 3) {
    kan::BSplineConfig c{p, {}, 3};
    const std::vector<std::vector<double>> rows{clamped(p, -1, 1, {-0.6, -0.2, 0.2, 0.6}),
                                                clamped(p, -2, 0.5, {-1.5, -1.2, -0.3, 0.1}),
                                                clamped(p, 0, 4, {0.5, 1, 2, 3.5})};
    for (const auto& r : rows) c.knots.insert(c.knots.end(), r.begin(), r.end());
    return c;
}
double wave(std::size_t k, double a = 0.37) { return std::sin(a * double(k) + 0.3); }
kan::Layer seeded(kan::Layer layer, double scale = 0.2) {
    std::vector<double> c(layer.coefficients().size()), b(layer.outputs());
    for (std::size_t k = 0; k < c.size(); ++k) c[k] = scale * wave(k);
    for (std::size_t o = 0; o < b.size(); ++o) b[o] = 0.1 * wave(o, 1.3);
    layer.set_parameters(c, b);
    return layer;
}
// batch x inputs samples inside the domains of three_grids().
std::vector<double> samples(std::size_t batch, std::size_t inputs = 3) {
    const double lo[3] = {-1, -2, 0}, hi[3] = {1, 0.5, 4};
    std::vector<double> x(batch * inputs);
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t i = 0; i < inputs; ++i) {
            const double u = 0.5 + 0.49 * std::sin(1.7 * double(b) + 2.1 * double(i) + 0.4 * double(b * b % 7));
            x[b * inputs + i] = lo[i % 3] + (hi[i % 3] - lo[i % 3]) * u;
        }
    return x;
}
const kan::BSplineConfig& spline_of(const kan::Layer& l) {
    return std::get<kan::BSplineConfig>(std::get<kan::BasisEdges>(l.carrier()).basis);
}
double objective(const kan::Layer& l, const std::vector<double>& x, std::size_t batch, const std::vector<double>& u) {
    const auto y = l.forward(x, batch);
    return std::inner_product(y.begin(), y.end(), u.begin(), 0.0);
}
// e_m(values) / C(p, m): the blossom of x^m (Marsden), the B-spline
// coefficient k of x^m on knots t has values t[k+1..k+p].
double blossom(const double* values, std::size_t p, std::size_t m) {
    std::vector<double> e(m + 1, 0.0);
    e[0] = 1;
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = std::min(m, i + 1); j > 0; --j) e[j] += e[j - 1] * values[i];
    double binomial = 1;
    for (std::size_t j = 0; j < m; ++j) binomial = binomial * double(p - j) / double(j + 1);
    return e[m] / binomial;
}
// Coefficients of q(x) = sum_m a_m x^m on clamped knots t (degree p).
std::vector<double> polynomial_coefficients(const std::vector<double>& t, std::size_t p, const std::vector<double>& a) {
    const auto terms = t.size() - p - 1;
    std::vector<double> c(terms, 0.0);
    for (std::size_t k = 0; k < terms; ++k)
        for (std::size_t m = 0; m < a.size(); ++m) c[k] += a[m] * blossom(t.data() + k + 1, p, m);
    return c;
}
// pykan's quantile grid of one input (KANLayer.update_grid_from_samples):
// ids = int(n/G*j), mixed with the uniform grid, endpoints the extreme samples.
std::vector<double> pykan_knots(std::vector<double> x, std::size_t G, double eps) {
    std::sort(x.begin(), x.end());
    const auto n = x.size();
    std::vector<double> g(G + 1);
    const double lo = x.front(), hi = x.back(), h = (hi - lo) / double(G);
    for (std::size_t j = 0; j <= G; ++j) {
        const double q = j == G ? hi : x[static_cast<std::size_t>(double(n) / double(G) * double(j))];
        g[j] = eps * (lo + h * double(j)) + (1 - eps) * q;
    }
    g.front() = lo;
    g.back() = hi;
    return g;
}
std::vector<double> column(const std::vector<double>& x, std::size_t inputs, std::size_t i) {
    std::vector<double> v;
    for (std::size_t b = i; b < x.size(); b += inputs) v.push_back(x[b]);
    return v;
}
} // namespace

TEST(per_input_configuration_is_validated) {
    auto c = three_grids();
    REQUIRE(kan::basis_size(c) == 8);
    kan::validate_basis(c);
    test::throws<std::invalid_argument>([&] { kan::evaluate_basis(c, 0.1); });
    auto zero = c;
    zero.grids = 0;
    REQUIRE(kan::basis_size(zero) == 0);
    test::throws<std::invalid_argument>([&] { kan::validate_basis(zero); });
    auto ragged = c;
    ragged.knots.pop_back();
    test::throws<std::invalid_argument>([&] { kan::validate_basis(ragged); });
    auto unclamped = c;
    unclamped.knots[12] = -1.99; // grid 1 is no longer clamped at its lower end
    test::throws<std::invalid_argument>([&] { kan::validate_basis(unclamped); });
    // spline_grid and per_input_grids.
    const auto g1 = kan::spline_grid(c, 1);
    REQUIRE(g1.grids == 1 && g1.degree == 3);
    REQUIRE(std::equal(g1.knots.begin(), g1.knots.end(), c.knots.begin() + 12));
    test::throws<std::invalid_argument>([&] { kan::spline_grid(c, 3); });
    const auto shared = kan::spline_grid(c, 0);
    const auto replicated = kan::per_input_grids(shared, 3);
    REQUIRE(replicated.grids == 3 && replicated.knots.size() == 36);
    REQUIRE(kan::spline_grid(replicated, 2) == shared);
    test::throws<std::invalid_argument>([&] { kan::per_input_grids(c, 3); });
    test::throws<std::invalid_argument>([&] { kan::per_input_grids(shared, 0); });
    REQUIRE(!(replicated == kan::per_input_grids(shared, 1)));
    // A layer takes one grid or one grid per input.
    kan::Layer ok(3, 2, c);
    REQUIRE(ok.terms() == 8);
    test::throws<std::invalid_argument>([&] { kan::Layer(2, 2, c); });
    kan::Layer one(1, 2, shared);
    test::throws<std::invalid_argument>([&] { one.set_carrier(kan::BasisEdges{c, std::vector<double>(48)}); });
    // An existing two-field aggregate keeps one shared grid.
    const kan::BSplineConfig legacy{3, shared.knots};
    REQUIRE(legacy.grids == 1 && legacy == shared);
}

TEST(per_input_layer_is_the_sum_of_its_inputs) {
    const auto c = three_grids();
    const auto layer = seeded(kan::Layer(3, 2, c));
    const std::size_t batch = 17;
    auto x = samples(batch);
    x[0] = -1; x[1] = 0.5; x[5] = 4; x[3] = 1.5; x[4] = -3; // endpoints and outside the domain
    const auto y = layer.forward(x, batch);
    const auto& coefficients = std::get<kan::BasisEdges>(layer.carrier()).coefficients;
    // Input i alone: a one-input shared-grid layer on grid i with C[:, i, :].
    std::vector<std::vector<double>> parts;
    for (std::size_t i = 0; i < 3; ++i) {
        kan::Layer single(1, 2, kan::spline_grid(c, i));
        std::vector<double> ci(16);
        for (std::size_t o = 0; o < 2; ++o)
            for (std::size_t k = 0; k < 8; ++k) ci[o * 8 + k] = coefficients[(o * 3 + i) * 8 + k];
        single.set_parameters(ci, std::vector<double>(2, 0.0));
        parts.push_back(single.forward(column(x, 3, i), batch));
    }
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t o = 0; o < 2; ++o)
            test::near(y[b * 2 + o], layer.bias()[o] + parts[0][b * 2 + o] + parts[1][b * 2 + o] + parts[2][b * 2 + o], 1e-13);
    // Finite differences of the input and coefficient VJPs.
    std::vector<double> u(batch * 2);
    for (std::size_t k = 0; k < u.size(); ++k) u[k] = wave(k, 0.9);
    const auto g = layer.backward(x, batch, u);
    const double h = 1e-6;
    for (std::size_t e = 0; e < x.size(); ++e) {
        auto xp = x, xm = x;
        xp[e] += h; xm[e] -= h;
        const double fd = (objective(layer, xp, batch, u) - objective(layer, xm, batch, u)) / (2 * h);
        // Inputs at or across knots and endpoints are not differentiable there.
        if (e == 0 || e == 1 || e == 5) continue;
        test::near(g.input[e], fd, 1e-6);
    }
    for (std::size_t k = 0; k < coefficients.size(); ++k) {
        auto p = layer, m = layer;
        auto cp = coefficients, cm = coefficients;
        cp[k] += h; cm[k] -= h;
        p.set_parameters(cp, layer.bias()); m.set_parameters(cm, layer.bias());
        test::near(g.coefficients[k], (objective(p, x, batch, u) - objective(m, x, batch, u)) / (2 * h), 1e-7);
    }
}

TEST(identical_grids_match_the_shared_layer_bitwise) {
    for (std::size_t p : {0u, 1u, 2u, 3u, 5u}) {
        const auto shared = kan::spline_grid(three_grids(p), 0);
        const auto a = seeded(kan::Layer(3, 4, shared));
        auto b = kan::Layer(3, 4, kan::per_input_grids(shared, 3));
        b.set_parameters(a.coefficients(), a.bias());
        const std::size_t batch = 23;
        const auto x = samples(batch);
        std::vector<double> u(batch * 4);
        for (std::size_t k = 0; k < u.size(); ++k) u[k] = wave(k, 0.7);
        REQUIRE(a.forward(x, batch) == b.forward(x, batch));
        const auto ga = a.backward(x, batch, u), gb = b.backward(x, batch, u);
        REQUIRE(ga.input == gb.input && ga.coefficients == gb.coefficients && ga.bias == gb.bias);
    }
}

TEST(insert_knots_and_adapt_grid_per_input) {
    const auto c = three_grids();
    auto layer = seeded(kan::Layer(3, 2, c));
    const auto before = layer;
    const std::size_t batch = 40;
    const auto x = samples(batch);
    test::throws<std::invalid_argument>([&] { kan::insert_knot(layer, 0.1); });
    test::throws<std::invalid_argument>([&] { kan::adapt_grid(layer, std::vector<double>{0.1}); });
    test::throws<std::invalid_argument>([&] { kan::insert_knots(layer, std::vector<double>{0.1, -1}); });      // one per grid
    test::throws<std::invalid_argument>([&] { kan::insert_knots(layer, std::vector<double>{0.1, -1, 5}); });   // 5 outside grid 2
    REQUIRE(layer.carrier() == before.carrier());
    kan::insert_knots(layer, std::vector<double>{0.1, -1, 2.5});
    REQUIRE(layer.terms() == 9 && spline_of(layer).grids == 3);
    const auto row1 = kan::spline_grid(spline_of(layer), 1).knots;
    REQUIRE(row1.size() == 13 && std::count(row1.begin(), row1.end(), -1.0) == 1);
    const auto y0 = before.forward(x, batch), y1 = layer.forward(x, batch);
    for (std::size_t k = 0; k < y0.size(); ++k) test::near(y1[k], y0[k], 1e-12);
    // adapt_grid over layer inputs: per grid, the 1-D rule on its column.
    auto adapted = before;
    const auto knots = kan::adapt_grid(adapted, x, batch);
    REQUIRE(knots.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        kan::Layer single(1, 1, kan::spline_grid(c, i));
        REQUIRE(kan::adapt_grid(single, column(x, 3, i)) == knots[i]);
    }
    const auto y2 = adapted.forward(x, batch);
    for (std::size_t k = 0; k < y0.size(); ++k) test::near(y2[k], y0[k], 1e-12);
    // A shared grid pools the whole batch: the 1-D rule on all values.
    const auto shared = kan::spline_grid(three_grids(), 0);
    auto pooled = seeded(kan::Layer(3, 2, shared)), reference = pooled;
    const auto inserted = kan::adapt_grid(pooled, x, batch);
    REQUIRE(inserted.size() == 1 && inserted[0] == kan::adapt_grid(reference, x));
    REQUIRE(pooled.carrier() == reference.carrier());
    test::throws<std::invalid_argument>([&] { kan::adapt_grid(pooled, x, batch + 1); });
}

TEST(refit_knots_follow_pykan_quantiles) {
    const std::size_t batch = 37;
    const auto x = samples(batch);
    for (double eps : {0.0, 0.02, 0.5, 1.0})
        for (std::size_t G : {0u, 3u, 7u}) {
            auto layer = seeded(kan::Layer(3, 2, three_grids()));
            kan::refit_grid(layer, x, batch, kan::GridRefit{G, eps});
            const std::size_t intervals = G ? G : 5;
            const auto& s = spline_of(layer);
            REQUIRE(s.grids == 3 && s.degree == 3 && layer.terms() == intervals + 3);
            for (std::size_t i = 0; i < 3; ++i) {
                const auto g = pykan_knots(column(x, 3, i), intervals, eps);
                const auto expected = clamped(3, g.front(), g.back(), std::vector<double>(g.begin() + 1, g.end() - 1));
                const auto row = kan::spline_grid(s, i).knots;
                REQUIRE(row == expected);
            }
        }
    // A shared grid from all samples (per_input = false).
    auto layer = seeded(kan::Layer(3, 2, three_grids()));
    kan::refit_grid(layer, x, batch, kan::GridRefit{4, 0.02, 1e-8, false});
    const auto g = pykan_knots(x, 4, 0.02);
    REQUIRE(spline_of(layer).grids == 1);
    REQUIRE(spline_of(layer).knots == clamped(3, g.front(), g.back(), std::vector<double>(g.begin() + 1, g.end() - 1)));
}

TEST(refit_reproduces_polynomials_exactly) {
    // Each edge is a polynomial of the spline degree (Marsden coefficients);
    // every clamped grid represents it, so the least-squares refit with
    // ridge 0 reproduces it to rounding, for any new grid and term count.
    for (std::size_t p : {1u, 2u, 3u}) {
        auto c = three_grids(p);
        kan::Layer layer(3, 2, c);
        std::vector<double> coefficients;
        std::vector<std::vector<double>> polys;
        for (std::size_t o = 0; o < 2; ++o)
            for (std::size_t i = 0; i < 3; ++i) {
                std::vector<double> a{0.3 - 0.1 * double(o), -0.5 + 0.2 * double(i), 0.7, -0.2};
                a.resize(p + 1);
                polys.push_back(a);
                const auto ci = polynomial_coefficients(kan::spline_grid(c, i).knots, p, a);
                coefficients.insert(coefficients.end(), ci.begin(), ci.end());
            }
        layer.set_parameters(coefficients, std::vector<double>{0.25, -0.5});
        const std::size_t batch = 64;
        const auto x = samples(batch);
        for (std::size_t G : {0u, 2u, 9u}) {
            auto refit = layer;
            kan::refit_grid(refit, x, batch, kan::GridRefit{G, 0.02, 0.0});
            const auto& s = spline_of(refit);
            const auto& next = std::get<kan::BasisEdges>(refit.carrier()).coefficients;
            const auto terms = refit.terms();
            for (std::size_t o = 0; o < 2; ++o)
                for (std::size_t i = 0; i < 3; ++i) {
                    const auto expected = polynomial_coefficients(kan::spline_grid(s, i).knots, p, polys[o * 3 + i]);
                    for (std::size_t k = 0; k < terms; ++k) test::near(next[(o * 3 + i) * terms + k], expected[k], 1e-11);
                }
            REQUIRE(refit.bias()[0] == 0.25 && refit.bias()[1] == -0.5);
        }
    }
}

TEST(refit_onto_a_refined_grid_preserves_the_function) {
    // Uniform refit (grid_eps = 1) with samples reaching the old domain's
    // endpoints and twice the intervals: the new knots contain the old ones,
    // so any spline is reproduced (the refit is exact up to rounding).
    const auto shared = kan::BSplineConfig{3, clamped(3, -1, 1, uniform_interior(-1, 1, 4))};
    const auto layer = seeded(kan::Layer(2, 3, kan::per_input_grids(shared, 2)), 0.5);
    std::size_t batch = 200;
    auto x = samples(batch, 2);
    for (auto& v : x) v = std::clamp(v, -1.0, 1.0);
    x[0] = -1; x[1] = 1; x[2] = 1; x[3] = -1;
    auto refit = layer;
    kan::refit_grid(refit, x, batch, kan::GridRefit{8, 1.0, 0.0});
    REQUIRE(refit.terms() == 11);
    const auto y0 = layer.forward(x, batch), y1 = refit.forward(x, batch);
    for (std::size_t k = 0; k < y0.size(); ++k) test::near(y1[k], y0[k], 1e-12);
}

TEST(refit_validation_is_atomic) {
    auto layer = seeded(kan::Layer(3, 2, three_grids()));
    const auto before = layer;
    const std::size_t batch = 20;
    auto x = samples(batch);
    const auto unchanged = [&] { REQUIRE(layer.carrier() == before.carrier() && std::ranges::equal(layer.bias(), before.bias())); };
    test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, x, batch + 1); });
    test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, {}, 0); });
    for (double eps : {-0.1, 1.5, std::nan("")})
        test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, x, batch, kan::GridRefit{0, eps}); });
    test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, x, batch, kan::GridRefit{0, 0.02, -1}); });
    auto nonfinite = x;
    nonfinite[4] = INFINITY;
    test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, nonfinite, batch); });
    // An input whose samples span no interval.
    auto constant = x;
    for (std::size_t b = 0; b < batch; ++b) constant[b * 3 + 1] = 0.25;
    test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, constant, batch); });
    // Quantile knots that coincide (repeated samples, grid_eps = 0).
    auto repeated = x;
    for (std::size_t b = 0; b + 2 < batch; ++b) repeated[b * 3 + 2] = 1.0;
    test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, repeated, batch, kan::GridRefit{0, 0.0}); });
    kan::refit_grid(layer, repeated, batch, kan::GridRefit{0, 0.1}); // the uniform share separates them
    layer = before;
    // More intervals than samples: empty spans make ridge 0 singular.
    const std::vector<double> x3(x.begin(), x.begin() + 9);
    test::throws<std::invalid_argument>([&] { kan::refit_grid(layer, x3, 3, kan::GridRefit{12, 1.0, 0.0}); });
    unchanged();
    kan::refit_grid(layer, x3, 3, kan::GridRefit{12, 1.0, 1e-8});
    REQUIRE(layer.terms() == 15);
    // Other carriers.
    kan::Layer chebyshev(3, 2, kan::ChebyshevConfig{4});
    test::throws<std::invalid_argument>([&] { kan::refit_grid(chebyshev, x, batch); });
}

TEST(refit_keeps_bias_residual_and_handles_degree_zero) {
    auto layer = seeded(kan::Layer(3, 2, three_grids(0)));
    layer.set_residual(kan::SiluResidual{{0.1, -0.2, 0.3, 0.4, -0.5, 0.6}});
    const auto before = layer;
    const std::size_t batch = 50;
    const auto x = samples(batch);
    kan::refit_grid(layer, x, batch);
    REQUIRE(layer.residual() == before.residual() && std::ranges::equal(layer.bias(), before.bias()));
    REQUIRE(layer.terms() == 5 && spline_of(layer).degree == 0);
    // Shared grid stays shared with per_input = false; a one-input layer has one grid.
    auto shared = seeded(kan::Layer(1, 2, kan::spline_grid(three_grids(), 0)));
    kan::refit_grid(shared, column(x, 3, 0), batch);
    REQUIRE(spline_of(shared).grids == 1);
}

TEST(network_refit_propagates_samples_through_preceding_layers) {
    const auto spline = kan::spline_grid(three_grids(), 0);
    kan::Network network({kan::NetworkLayer(kan::InputMap(3, kan::AffineMap{{0.5, 1, 0.8}, {0.1, -0.2, 0}})),
                          kan::NetworkLayer(seeded(kan::Layer(3, 4, spline), 0.6)),
                          kan::NetworkLayer(seeded(kan::Layer(4, 3, kan::ChebyshevConfig{3}), 0.4)),
                          kan::NetworkLayer(seeded(kan::Layer(3, 2, kan::per_input_grids(spline, 3)), 0.3))});
    const std::size_t batch = 45;
    const auto x = samples(batch);
    // Manual pykan order: refit layer 1 on its activations, then layer 3 on the
    // activations of the refit network.
    auto layers = std::vector<kan::NetworkLayer>(network.layers().begin(), network.layers().end());
    const auto a0 = std::get<kan::InputMap>(layers[0]).forward(x, batch);
    kan::refit_grid(std::get<kan::Layer>(layers[1]), a0, batch);
    const auto a1 = std::get<kan::Layer>(layers[1]).forward(a0, batch);
    const auto a2 = std::get<kan::Layer>(layers[2]).forward(a1, batch);
    kan::refit_grid(std::get<kan::Layer>(layers[3]), a2, batch);
    auto refit = network;
    REQUIRE(refit.refit_grids(x, batch) == 2);
    for (std::size_t j = 0; j < 4; ++j) {
        if (const auto* l = std::get_if<kan::Layer>(&layers[j])) {
            const auto& r = std::get<kan::Layer>(refit.layers()[j]);
            REQUIRE(l->carrier() == r.carrier());
        }
    }
    REQUIRE(spline_of(std::get<kan::Layer>(refit.layers()[1])).grids == 3);
    // Atomic: a later layer that cannot refit (constant activations) leaves
    // the network unchanged.
    kan::Network failing({kan::NetworkLayer(seeded(kan::Layer(3, 4, spline), 0.6)),
                          kan::NetworkLayer(kan::Layer(4, 3, kan::ChebyshevConfig{3})),
                          kan::NetworkLayer(seeded(kan::Layer(3, 2, spline), 0.3))});
    const auto snapshot = failing;
    test::throws<std::invalid_argument>([&] { failing.refit_grids(x, batch); });
    for (std::size_t j = 0; j < 3; ++j)
        REQUIRE(std::get<kan::Layer>(failing.layers()[j]).carrier() == std::get<kan::Layer>(snapshot.layers()[j]).carrier());
    kan::Network none({kan::NetworkLayer(kan::Layer(3, 2, kan::ChebyshevConfig{3}))});
    test::throws<std::invalid_argument>([&] { none.refit_grids(x, batch); });
    test::throws<std::invalid_argument>([&] { refit.refit_grids(x, batch + 1); });
}

TEST(network_adapt_grid_propagates_samples) {
    const auto spline = kan::spline_grid(three_grids(), 0);
    kan::Network network({kan::NetworkLayer(seeded(kan::Layer(3, 3, kan::ChebyshevConfig{3}), 0.5)),
                          kan::NetworkLayer(seeded(kan::Layer(3, 2, kan::per_input_grids(spline, 3)), 0.3))});
    const std::size_t batch = 31;
    const auto x = samples(batch);
    const auto activations = std::get<kan::Layer>(network.layers()[0]).forward(x, batch);
    auto layer = std::get<kan::Layer>(network.layers()[1]);
    const auto expected = kan::adapt_grid(layer, activations, batch);
    const auto knots = network.adapt_grid(1, x, batch);
    REQUIRE(knots == expected);
    REQUIRE(std::get<kan::Layer>(network.layers()[1]).carrier() == layer.carrier());
    test::throws<std::invalid_argument>([&] { network.adapt_grid(0, x, batch); });
    test::throws<std::invalid_argument>([&] { network.adapt_grid(2, x, batch); });
}

TEST(initializers_use_each_inputs_grid) {
    const auto c = three_grids();
    test::throws<std::invalid_argument>([&] { kan::reference_moments(c); });
    for (const kan::Initializer init : {kan::Initializer(kan::VarianceScaling{1.3, kan::Distribution::Normal, 7}),
                                        kan::Initializer(kan::NoiseInit{0.3, kan::Distribution::Uniform, 9})}) {
        kan::Layer layer(3, 2, c);
        kan::initialize(layer, init);
        // Draws in layout order; input i's factors from its own grid: equal
        // to the coefficients of a shared-grid layer on grid i at (o, i, k).
        for (std::size_t i = 0; i < 3; ++i) {
            kan::Layer shared(3, 2, kan::spline_grid(c, i));
            kan::initialize(shared, init);
            for (std::size_t o = 0; o < 2; ++o)
                for (std::size_t k = 0; k < 8; ++k)
                    REQUIRE(layer.coefficients()[(o * 3 + i) * 8 + k] == shared.coefficients()[(o * 3 + i) * 8 + k]);
        }
        // Identical grids: the shared layer's parameters.
        const auto shared = kan::spline_grid(c, 1);
        kan::Layer a(3, 2, shared), b(3, 2, kan::per_input_grids(shared, 3));
        kan::initialize(a, init);
        kan::initialize(b, init);
        REQUIRE(std::ranges::equal(a.coefficients(), b.coefficients()));
    }
}

int main() { return test::run(); }

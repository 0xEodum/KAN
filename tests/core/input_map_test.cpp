// Backlog M1: typed input maps (affine, tanh, LayerNorm) as their own network
// layer kind. Finite-difference checks of every VJP, validation, SGD
// atomicity, heterogeneous Network semantics and the out-of-domain training
// demonstration.
#include "kan/input_map.hpp"
#include "kan/network.hpp"
#include "kan/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <cmath>
#include <limits>
#include <numeric>

namespace {
constexpr double h = 1e-6;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
std::vector<double> wave(std::size_t n, double scale, double phase) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * std::sin(phase + 0.83 * static_cast<double>(k));
    return v;
}
kan::LayerNormMap layer_norm(std::size_t features, bool affine) {
    kan::LayerNormMap m;
    m.epsilon = 1e-3;
    if (affine) { m.gain = wave(features, 0.4, 0.3); for (auto& g : m.gain) g += 1; m.bias = wave(features, 0.2, 1.1); }
    return m;
}
// Input VJP of a map against central differences of <forward(x), g>.
void check_input_vjp(const kan::InputMap& map, const std::vector<double>& x, std::size_t batch,
                     const std::vector<double>& g, double tolerance = 1e-7) {
    const auto analytic = map.backward(x, batch, g);
    REQUIRE(analytic.input.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto p = x, m = x;
        p[i] += h; m[i] -= h;
        test::near(analytic.input[i], (dot(map.forward(p, batch), g) - dot(map.forward(m, batch), g)) / (2 * h), tolerance);
    }
}
kan::Layer seeded(kan::Layer layer, double phase) {
    const auto c = wave(layer.coefficients().size(), 0.3, phase);
    const auto b = wave(layer.bias().size(), 0.05, phase + 1);
    layer.set_parameters(c, b);
    return layer;
}
kan::Layer rational(std::size_t in, std::size_t out) {
    kan::RationalConfig config{2, 1, 0.1, 1.3, 1e-8};
    kan::Layer l(in, out, config);
    kan::set_rational_parameters(l, wave(l.coefficients().size(), 0.3, 0.2),
                                 wave(in * out, 0.1, 0.7), wave(out, 0.05, 0.1));
    return l;
}
double objective(const kan::Network& n, const std::vector<double>& x, std::size_t batch, const std::vector<double>& g) {
    return dot(n.forward(x, batch), g);
}
// Every parameter of every stage of a network against central differences.
void check_network(const kan::Network& net, const std::vector<double>& x, double tolerance = 2e-6) {
    const auto batch = x.size() / net.inputs();
    std::vector<double> g = wave(batch * std::visit([](const auto& s) { return s.outputs(); }, net.layers().back()), 0.9, 0.4);
    const auto grad = net.backward(x, batch, g);
    REQUIRE(grad.layers.size() == net.layers().size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto p = x, m = x; p[i] += h; m[i] -= h;
        test::near(grad.input[i], (objective(net, p, batch, g) - objective(net, m, batch, g)) / (2 * h), tolerance);
    }
    const std::vector<kan::NetworkLayer> stages(net.layers().begin(), net.layers().end());
    for (std::size_t l = 0; l < stages.size(); ++l) {
        if (const auto* layer = std::get_if<kan::Layer>(&stages[l])) {
            const std::vector<double> c(layer->coefficients().begin(), layer->coefficients().end());
            const std::vector<double> b(layer->bias().begin(), layer->bias().end());
            for (std::size_t i = 0; i < c.size(); ++i) {
                auto plus = stages, minus = stages; auto pc = c, mc = c; pc[i] += h; mc[i] -= h;
                std::get<kan::Layer>(plus[l]).set_parameters(pc, b);
                std::get<kan::Layer>(minus[l]).set_parameters(mc, b);
                test::near(test::grad(grad, l).coefficients[i],
                           (objective(kan::Network(plus), x, batch, g) - objective(kan::Network(minus), x, batch, g)) / (2 * h), tolerance);
            }
        } else {
            const auto& map = std::get<kan::InputMap>(stages[l]);
            const auto& mg = test::map_grad(grad, l);
            const auto* ln = std::get_if<kan::LayerNormMap>(&map.map());
            if (!ln || ln->gain.empty()) { REQUIRE(mg.gain.empty()); REQUIRE(mg.bias.empty()); continue; }
            for (int which = 0; which < 2; ++which) {
                const auto& analytic = which == 0 ? mg.gain : mg.bias;
                REQUIRE(analytic.size() == map.features());
                for (std::size_t i = 0; i < analytic.size(); ++i) {
                    auto plus = stages, minus = stages;
                    auto pm = *ln, mm = *ln;
                    (which == 0 ? pm.gain : pm.bias)[i] += h;
                    (which == 0 ? mm.gain : mm.bias)[i] -= h;
                    std::get<kan::InputMap>(plus[l]).set_map(pm);
                    std::get<kan::InputMap>(minus[l]).set_map(mm);
                    test::near(analytic[i],
                               (objective(kan::Network(plus), x, batch, g) - objective(kan::Network(minus), x, batch, g)) / (2 * h), tolerance);
                }
            }
        }
    }
}
double mse(const std::vector<double>& y, const std::vector<double>& t) {
    double s = 0;
    for (std::size_t i = 0; i < y.size(); ++i) s += (y[i] - t[i]) * (y[i] - t[i]);
    return s / static_cast<double>(y.size());
}
// Full-batch gradient descent on the mean squared error; returns the final loss.
double train(kan::Network& net, const std::vector<double>& x, const std::vector<double>& t, int epochs, double rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto g = net.forward(x, x.size());
        for (std::size_t i = 0; i < g.size(); ++i) g[i] = 2 * (g[i] - t[i]) / static_cast<double>(g.size());
        net.sgd(net.backward(x, x.size(), g), rate);
    }
    return mse(net.forward(x, x.size()), t);
}
} // namespace

TEST(affine_map_values_and_vjp) {
    kan::InputMap map(2, kan::AffineMap{{2.0, -0.5}, {0.25, 1.0}});
    REQUIRE(map.inputs() == 2 && map.outputs() == 2 && map.features() == 2);
    const std::vector<double> x{0.5, -2, 1, 4};
    REQUIRE(map.forward(x, 2) == (std::vector<double>{1.25, 2.0, 2.25, -1.0}));
    const auto g = map.backward(x, 2, std::vector<double>{1, 1, -1, 2});
    REQUIRE(g.input == (std::vector<double>{2, -0.5, -2, -1}));
    REQUIRE(g.gain.empty() && g.bias.empty());
    check_input_vjp(map, wave(6, 3, 0.1), 3, wave(6, 1, 0.5));
}

TEST(tanh_map_values_and_vjp) {
    kan::InputMap map(3, kan::TanhMap{0.3});
    const auto x = wave(9, 4, 0.2);
    const auto y = map.forward(x, 3);
    for (std::size_t i = 0; i < x.size(); ++i) test::near(y[i], std::tanh(0.3 * x[i]), 1e-15);
    check_input_vjp(map, x, 3, wave(9, 1, 0.9));
    // Saturated inputs: exact zero derivative, never a nonfinite one.
    const auto far = map.backward(std::vector<double>{1e6, -1e300, 0}, 1, std::vector<double>{1, 1, 1});
    REQUIRE(far.input[0] == 0 && far.input[1] == 0);
    test::near(far.input[2], 0.3, 1e-15);
    REQUIRE(kan::InputMap(1, kan::TanhMap{}).forward(std::vector<double>{0.5}, 1)[0] == std::tanh(0.5));
}

TEST(layer_norm_normalizes_each_sample) {
    kan::InputMap map(4, layer_norm(4, false));
    const std::vector<double> x{1, 2, 3, 4, -100, 0, 100, 300};
    const auto y = map.forward(x, 2);
    for (std::size_t b = 0; b < 2; ++b) {
        double mean = 0, var = 0;
        for (std::size_t i = 0; i < 4; ++i) mean += x[b * 4 + i] / 4;
        for (std::size_t i = 0; i < 4; ++i) var += (x[b * 4 + i] - mean) * (x[b * 4 + i] - mean) / 4;
        for (std::size_t i = 0; i < 4; ++i) test::near(y[b * 4 + i], (x[b * 4 + i] - mean) / std::sqrt(var + 1e-3), 1e-14);
    }
    // One feature: x - mean is zero, the output is the bias.
    kan::InputMap single(1, kan::LayerNormMap{1e-5, {2.0}, {0.7}});
    REQUIRE(single.forward(std::vector<double>{123, -5}, 2) == (std::vector<double>{0.7, 0.7}));
    REQUIRE(single.backward(std::vector<double>{123}, 1, std::vector<double>{1}).input[0] == 0);
}

TEST(layer_norm_vjps_match_finite_differences) {
    for (bool affine : {false, true}) {
        kan::InputMap map(5, layer_norm(5, affine));
        const auto x = wave(15, 2, 0.4), g = wave(15, 1, 1.3);
        check_input_vjp(map, x, 3, g);
        const auto analytic = map.backward(x, 3, g);
        if (!affine) { REQUIRE(analytic.gain.empty() && analytic.bias.empty()); continue; }
        const auto base = std::get<kan::LayerNormMap>(map.map());
        for (int which = 0; which < 2; ++which)
            for (std::size_t i = 0; i < 5; ++i) {
                auto p = base, m = base;
                (which == 0 ? p.gain : p.bias)[i] += h;
                (which == 0 ? m.gain : m.bias)[i] -= h;
                const double fd = (dot(kan::InputMap(5, p).forward(x, 3), g) - dot(kan::InputMap(5, m).forward(x, 3), g)) / (2 * h);
                test::near((which == 0 ? analytic.gain : analytic.bias)[i], fd, 1e-7);
            }
    }
}

TEST(map_validation_and_moved_from) {
    using Error = std::invalid_argument;
    const double inf = std::numeric_limits<double>::infinity();
    test::throws<Error>([] { kan::InputMap(0, kan::TanhMap{}); });
    test::throws<Error>([] { kan::InputMap(2, kan::AffineMap{{1}, {0, 0}}); });
    test::throws<Error>([] { kan::InputMap(1, kan::AffineMap{{1}, {}}); });
    test::throws<Error>([&] { kan::InputMap(1, kan::AffineMap{{inf}, {0}}); });
    test::throws<Error>([] { kan::InputMap(2, kan::AffineMap{{1, 0}, {0, 0}}); });
    for (double s : {0.0, -1.0, inf, std::nan("")}) test::throws<Error>([&] { kan::InputMap(1, kan::TanhMap{s}); });
    for (double e : {0.0, -1e-5, inf}) test::throws<Error>([&] { kan::InputMap(2, kan::LayerNormMap{e, {}, {}}); });
    test::throws<Error>([] { kan::InputMap(2, kan::LayerNormMap{1e-5, {1, 1}, {}}); });
    test::throws<Error>([] { kan::InputMap(2, kan::LayerNormMap{1e-5, {1}, {0}}); });
    test::throws<Error>([&] { kan::InputMap(1, kan::LayerNormMap{1e-5, {inf}, {0}}); });
    kan::InputMap map(2, kan::TanhMap{});
    test::throws<Error>([&] { map.forward(std::vector<double>{1}, 1); });
    test::throws<Error>([&] { map.forward(std::vector<double>{1, inf}, 1); });
    test::throws<Error>([&] { map.backward(std::vector<double>{1, 2}, 1, std::vector<double>{1}); });
    test::throws<Error>([&] { map.set_map(kan::AffineMap{{1}, {0}}); });
    REQUIRE(std::holds_alternative<kan::TanhMap>(map.map()));
    REQUIRE(map.forward({}, 0).empty() && map.backward({}, 0, {}).input.empty());
    test::throws<std::overflow_error>([] { kan::InputMap(1, kan::AffineMap{{1e300}, {0}}).forward(std::vector<double>{1e300}, 1); });
    test::throws<std::overflow_error>([] { kan::InputMap(2, kan::LayerNormMap{}).forward(std::vector<double>{1e300, -1e300}, 1); });
    auto moved = std::move(map);
    REQUIRE(moved.forward(std::vector<double>{0, 1}, 1).size() == 2);
    test::throws<Error>([&] { map.forward(std::vector<double>{0, 1}, 1); });
    test::throws<Error>([&] { kan::Network invalid({kan::NetworkLayer(map)}); });
    map = moved;
    REQUIRE(map.forward(std::vector<double>{0, 1}, 1).size() == 2);
    kan::InputMap target(1, kan::AffineMap{{1}, {0}});
    target = std::move(moved);
    REQUIRE(target.features() == 2 && std::holds_alternative<kan::TanhMap>(target.map()));
    test::throws<Error>([&] { moved.backward(std::vector<double>{0, 1}, 1, std::vector<double>{1, 1}); });
    test::throws<Error>([&] { map.set_map(kan::TanhMap{-1}); });
    test::throws<Error>([&] { kan::Network({kan::NetworkLayer(map)}).regularization(-1); });
    REQUIRE((kan::AffineMap{{1}, {2}} == kan::AffineMap{{1}, {2}}));
    REQUIRE((kan::LayerNormMap{} == kan::LayerNormMap{1e-5, {}, {}}));
}

TEST(map_sgd_updates_trainable_parameters_atomically) {
    kan::InputMap map(3, layer_norm(3, true));
    const auto before = std::get<kan::LayerNormMap>(map.map());
    const auto x = wave(6, 1, 0.2);
    auto g = map.backward(x, 2, wave(6, 1, 0.8));
    auto bad = g; bad.gain.pop_back();
    test::throws<std::invalid_argument>([&] { map.sgd(bad, 0.1); });
    bad = g; bad.bias[1] = std::nan("");
    test::throws<std::invalid_argument>([&] { map.sgd(bad, 0.1); });
    bad = g; bad.gain[0] = std::numeric_limits<double>::max();
    test::throws<std::overflow_error>([&] { map.sgd(bad, 4); });
    test::throws<std::invalid_argument>([&] { map.sgd(g, 0); });
    REQUIRE(std::get<kan::LayerNormMap>(map.map()) == before);
    map.sgd(g, 0.1);
    const auto& after = std::get<kan::LayerNormMap>(map.map());
    for (std::size_t i = 0; i < 3; ++i) {
        test::near(after.gain[i], before.gain[i] - 0.1 * g.gain[i], 1e-15);
        test::near(after.bias[i], before.bias[i] - 0.1 * g.bias[i], 1e-15);
    }
    REQUIRE(after.epsilon == before.epsilon);
    // Fixed maps have no trainable parameters: SGD with empty vectors is a no-op.
    kan::InputMap fixed(2, kan::AffineMap{{2, 3}, {0, 1}});
    const auto snapshot = fixed.map();
    fixed.sgd(fixed.backward(std::vector<double>{1, 2}, 1, std::vector<double>{1, 1}), 0.5);
    REQUIRE(fixed.map() == snapshot);
    test::throws<std::invalid_argument>([&] { fixed.sgd(kan::InputMapGradients{{}, {1, 1}, {1, 1}}, 0.5); });
}

TEST(affine_helpers_map_samples_onto_a_domain) {
    // Feature 0 spans [-200, 600], feature 1 is constant.
    const std::vector<double> samples{-200, 5, 600, 5, 200, 5};
    const auto range = kan::affine_from_range(samples, 3, 2, -1, 1);
    REQUIRE(range.scale.size() == 2 && range.shift.size() == 2);
    const auto y = kan::InputMap(2, range).forward(samples, 3);
    test::near(y[0], -1, 1e-15); test::near(y[2], 1, 1e-15); test::near(y[4], 0, 1e-15);
    test::near(range.scale[1], 1, 0); test::near(y[1], 0, 1e-15);
    const auto spline = kan::affine_from_range(samples, 3, 2, 0, 4);
    const auto z = kan::InputMap(2, spline).forward(samples, 3);
    test::near(z[0], 0, 1e-15); test::near(z[2], 4, 1e-15); test::near(z[1], 2, 1e-15);
    const auto moments = kan::affine_from_moments(samples, 3, 2);
    const auto s = kan::InputMap(2, moments).forward(samples, 3);
    test::near(s[0] + s[2] + s[4], 0, 1e-14);
    test::near((s[0] * s[0] + s[2] * s[2] + s[4] * s[4]) / 3, 1, 1e-14);
    REQUIRE(s[1] == 0 && moments.scale[1] == 1);
    using Error = std::invalid_argument;
    test::throws<Error>([&] { kan::affine_from_range(samples, 3, 3); });
    test::throws<Error>([&] { kan::affine_from_range({}, 0, 2); });
    test::throws<Error>([&] { kan::affine_from_range(samples, 3, 2, 1, 1); });
    test::throws<Error>([&] { kan::affine_from_range(std::vector<double>{std::nan("")}, 1, 1); });
    test::throws<Error>([&] { kan::affine_from_moments(samples, 2, 2); });
    test::throws<std::overflow_error>([] { kan::affine_from_range(std::vector<double>{0, 1e-320}, 2, 1); });
    // An overflowing span would otherwise give a finite constant map.
    test::throws<std::overflow_error>([] { kan::affine_from_range(std::vector<double>{-1e308, 1e308}, 2, 1); });
    test::throws<std::overflow_error>([] { kan::affine_from_range(std::vector<double>{-1e300, 1e300}, 2, 1, 0, 1e-320); });
}

TEST(network_mixes_maps_and_layers) {
    kan::InputMap affine(2, kan::AffineMap{{0.01, 0.02}, {0, -0.5}});
    kan::Layer first = seeded(kan::Layer(2, 3, kan::ChebyshevConfig{4}), 0.1);
    kan::InputMap norm(3, layer_norm(3, true));
    kan::Layer spline = seeded(kan::Layer(3, 1, kan::BSplineConfig{2, {-3, -3, -3, 0, 3, 3, 3}}), 0.4);
    kan::Network net({affine, first, norm, spline});
    REQUIRE(net.layers().size() == 4 && net.inputs() == 2 && net.outputs() == 1);
    REQUIRE(std::holds_alternative<kan::InputMap>(net.layers()[0]));
    const auto x = wave(6, 50, 0.3);
    const auto manual = spline.forward(norm.forward(first.forward(affine.forward(x, 3), 3), 3), 3);
    REQUIRE(net.forward(x, 3) == manual);
    // Indices address positions in layers(); maps have no grid.
    test::throws<std::invalid_argument>([&] { net.insert_knot(2, 0.5); });
    test::throws<std::invalid_argument>([&] { net.adapt_grid(0, std::vector<double>{0.1}); });
    test::throws<std::invalid_argument>([&] { net.insert_knot(4, 0.5); });
    net.insert_knot(3, 1.0);
    REQUIRE(test::layer(net, 3).terms() == 5);
    // Regularization: KAN coefficients only; maps contribute zero-shaped gradients.
    const auto r = net.regularization(0.5);
    double expected = 0;
    for (std::size_t l : {1u, 3u}) for (double c : test::layer(net, l).coefficients()) expected += 0.25 * c * c;
    test::near(r.value, expected, 1e-14);
    REQUIRE(test::map_grad(r.gradients, 0).gain.empty());
    REQUIRE(test::map_grad(r.gradients, 2).gain == std::vector<double>(3, 0.0));
    REQUIRE(test::map_grad(r.gradients, 2).input.empty());
    // Dimension and kind checks.
    test::throws<std::invalid_argument>([&] { kan::Network({affine, norm}); });
    test::throws<std::invalid_argument>([&] { kan::Network({first, affine}); });
    auto g = net.backward(x, 3, std::vector<double>{1, -1, 0.5});
    auto swapped = g; std::swap(swapped.layers[0], swapped.layers[1]);
    test::throws<std::invalid_argument>([&] { net.sgd(swapped, 0.1); });
    // A map-only network is valid.
    kan::Network maps({kan::NetworkLayer(kan::InputMap(2, kan::TanhMap{2}))});
    REQUIRE(maps.forward(std::vector<double>{0.1, 0.2}, 1).size() == 2);
}

TEST(network_gradients_with_maps_match_finite_differences) {
    // Map in front of a polynomial layer and in front of a localized layer.
    check_network(kan::Network({kan::InputMap(2, kan::AffineMap{{0.02, -0.03}, {0.1, 0}}),
                                seeded(kan::Layer(2, 3, kan::ChebyshevConfig{4}), 0.1),
                                kan::InputMap(3, layer_norm(3, true)),
                                seeded(kan::Layer(3, 2, kan::BSplineConfig{2, {-3, -3, -3, -1, 0, 1, 3, 3, 3}}), 0.5)}),
                  wave(6, 40, 0.2));
    check_network(kan::Network({kan::InputMap(3, kan::TanhMap{0.05}),
                                seeded(kan::Layer(3, 2, kan::TrainableRbfConfig{{-0.5, 0, 0.6}, {-0.3, 0, -0.1}}), 0.3),
                                kan::InputMap(2, layer_norm(2, false)),
                                rational(2, 1)}),
                  wave(9, 30, 0.5));
    check_network(kan::Network({kan::InputMap(4, layer_norm(4, true)),
                                seeded(kan::Layer(4, 2, kan::MexicanHatConfig{{-1, 0, 1}, {0.8, 1, 0.6}}), 0.7)}),
                  wave(8, 25, 0.9));
}

TEST(network_sgd_with_maps_is_atomic) {
    kan::Network net({kan::InputMap(2, layer_norm(2, true)), seeded(kan::Layer(2, 1, kan::LegendreConfig{3}), 0.2)});
    const std::vector<double> x{3, -4, 1, 9};
    const auto before = net.forward(x, 2);
    auto g = net.backward(x, 2, std::vector<double>{1, -1});
    auto bad = g; test::map_grad(bad, 0).gain[1] = std::numeric_limits<double>::infinity();
    test::throws<std::invalid_argument>([&] { net.sgd(bad, 0.1); });
    bad = g; test::grad(bad, 1).coefficients[0] = std::numeric_limits<double>::max();
    test::throws<std::overflow_error>([&] { net.sgd(bad, 4); });
    REQUIRE(net.forward(x, 2) == before);
    net.sgd(g, 0.1);
    REQUIRE(net.forward(x, 2) != before);
    REQUIRE(std::get<kan::LayerNormMap>(test::input_map(net, 0).map()).gain != layer_norm(2, true).gain);
}

// Inputs far outside the basis domain: a polynomial explodes and a localized
// basis is dead without a map; the same layers train with an explicit map.
TEST(far_inputs_train_with_map_and_fail_without) {
    std::vector<double> x(33), t(33);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 100 + 400 * static_cast<double>(i) / 32; // [100, 500]: entirely outside [-1, 1]
        const double u = (x[i] - 300) / 200;
        t[i] = 0.2 + 0.7 * u - 0.4 * u * u;
    }
    const double initial = mse(std::vector<double>(t.size(), 0.0), t);

    // Polynomial without a map: T_4(500) ~ 5e11, SGD overflows or diverges.
    bool exploded = false;
    try {
        kan::Network raw({kan::Layer(1, 1, kan::ChebyshevConfig{5})});
        const double loss = train(raw, x, t, 50, 0.1);
        exploded = !std::isfinite(loss) || loss > 1e6 * initial;
    } catch (const std::overflow_error&) { exploded = true; }
    REQUIRE(exploded);
    kan::Network mapped({kan::InputMap(1, kan::affine_from_range(x, x.size(), 1)), kan::Layer(1, 1, kan::ChebyshevConfig{5})});
    REQUIRE(train(mapped, x, t, 400, 0.1) < 1e-8);
    kan::Network squashed({kan::InputMap(1, kan::TanhMap{1.0 / 200}), kan::Layer(1, 1, kan::ChebyshevConfig{5})});
    REQUIRE(train(squashed, x, t, 400, 0.1) < 1e-2 * initial);

    // B-spline on [-1, 1] without a map: zero output and exactly zero
    // coefficient gradients, so only the bias learns (the edge is dead).
    const kan::BSplineConfig spline{3, {-1, -1, -1, -1, -0.5, 0, 0.5, 1, 1, 1, 1}};
    kan::Network dead({kan::Layer(1, 1, spline)});
    const auto g = dead.backward(x, x.size(), std::vector<double>(x.size(), 1.0));
    for (double v : test::grad(g, 0).coefficients) REQUIRE(v == 0);
    const double dead_loss = train(dead, x, t, 400, 0.5);
    for (double c : test::layer(dead, 0).coefficients()) REQUIRE(c == 0);
    kan::Network live({kan::InputMap(1, kan::affine_from_range(x, x.size(), 1, -0.999, 0.999)), kan::Layer(1, 1, spline)});
    REQUIRE(train(live, x, t, 2000, 0.5) < 1e-3 * dead_loss);
}

int main() { return test::run(); }

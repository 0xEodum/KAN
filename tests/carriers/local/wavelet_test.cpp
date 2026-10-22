// Backlog M7: per-edge trainable Mexican-hat wavelets (Wav-KAN). Formula,
// tails, analytic VJPs against finite differences, the layer protocol
// (placement, validation, SGD, L2) and agreement with the fixed M3 dictionary.
#include "kan/families.hpp"
#include "kan/network.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <cmath>
#include <limits>
#include <variant>

namespace {
const double inf = std::numeric_limits<double>::infinity();
const double not_a_number = std::numeric_limits<double>::quiet_NaN();
// 2/(sqrt(3) pi^(1/4)), from the closed form in long double.
const long double amplitude = 2.0L / (std::sqrt(3.0L) * std::sqrt(std::sqrt(3.14159265358979323846264338327950288L)));

struct Closed {
    long double value, input, log_scale;
};
// Independent closed forms in long double, from q = (x - tau)/s.
Closed closed(double x, double tau, double l) {
    const long double s = std::exp(static_cast<long double>(l));
    const long double q = (static_cast<long double>(x) - tau) / s, q2 = q * q, e = std::exp(-q2 / 2);
    const long double a = amplitude / std::sqrt(s);
    return {a * (1 - q2) * e, a / s * q * (q2 - 3) * e, a * e * (-q2 * q2 + 3.5L * q2 - 0.5L)};
}

void close(double actual, long double expected, double relative, double absolute = 0) {
    if (!std::isfinite(actual) || std::abs(actual - static_cast<double>(expected)) >
                                      relative * std::abs(static_cast<double>(expected)) + absolute)
        throw std::runtime_error("actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(static_cast<double>(expected)));
}

std::vector<double> wave(std::size_t n, double scale, double frequency, double phase = 0) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * std::sin(frequency * static_cast<double>(k) + phase);
    return v;
}

// A wavelet layer with distinct, nonzero parameters.
kan::Layer wavelet(std::size_t in, std::size_t out, std::size_t size) {
    kan::Layer layer(in, out, kan::WaveletConfig{size});
    const auto count = in * out * size;
    kan::set_wavelet_parameters(layer, wave(count, 0.5, 0.9, 0.3), wave(count, 0.8, 1.7, 0.1),
                                wave(count, 0.4, 2.3, 0.2), wave(out, 0.1, 1.1));
    return layer;
}

const kan::WaveletEdges& edges_of(const kan::Layer& layer) { return std::get<kan::WaveletEdges>(layer.carrier()); }
const kan::WaveletGradients& nonlinear(const kan::LayerGradients& g) {
    return std::get<kan::WaveletGradients>(g.nonlinear);
}

// sum(u * layer.forward(x)), the scalar whose gradient backward returns.
double objective(const kan::Layer& layer, const std::vector<double>& x, std::size_t batch,
                 const std::vector<double>& u) {
    const auto y = layer.forward(x, batch);
    long double sum = 0;
    for (std::size_t j = 0; j < y.size(); ++j) sum += static_cast<long double>(u[j]) * y[j];
    return static_cast<double>(sum);
}
} // namespace

TEST(wavelet_configuration_and_log_scale_validation) {
    test::throws<std::invalid_argument>([] { kan::validate_wavelet(kan::WaveletConfig{0}); });
    kan::validate_wavelet(kan::WaveletConfig{1});
    kan::validate_wavelet(kan::WaveletConfig{7});
    REQUIRE(kan::WaveletConfig{} == kan::WaveletConfig{1});
    for (double l : {0.0, -700.0, 700.0, 709.0, -709.0}) REQUIRE(kan::valid_log_scale(l));
    for (double l : {710.0, -710.0, 1e300, inf, -inf, not_a_number}) REQUIRE(!kan::valid_log_scale(l));
    test::throws<std::invalid_argument>([] { kan::Layer(2, 3, kan::WaveletConfig{0}); });
    test::throws<std::invalid_argument>([] { kan::Layer(0, 3, kan::WaveletConfig{1}); });
    test::throws<std::invalid_argument>([] { kan::evaluate_wavelet(not_a_number, 0, 0); });
    test::throws<std::invalid_argument>([] { kan::evaluate_wavelet(0, inf, 0); });
    test::throws<std::invalid_argument>([] { kan::evaluate_wavelet(0, 0, 800); });
}

TEST(wavelet_matches_closed_forms_and_the_fixed_mexican_hat) {
    for (double tau : {-0.7, 0.0, 0.3})
        for (double l : {-2.0, -0.4, 0.0, 0.9})
            for (double x : {-3.0, -1.1, -0.2, 0.0, 0.25, 0.8, 2.5, 6.0}) {
                const auto r = kan::evaluate_wavelet(x, tau, l);
                const auto c = closed(x, tau, l);
                // exp(-q^2/2) inherits the rounding of q^2/2: relative error up to about q^2 eps.
                // Near a zero of a polynomial factor the rounding of q^2 (a few eps)
                // dominates: absolute floor eps-scale times the amplitude a = A s^(-1/2).
                const double q = (x - tau) / std::exp(l), relative = 1e-15 * (8 + q * q);
                const double a = static_cast<double>(amplitude) / std::sqrt(std::exp(l));
                close(r.value, c.value, relative, 1e-15 * a);
                close(r.input_derivative, c.input, relative, 1e-15 * a / std::exp(l));
                close(r.log_scale_derivative, c.log_scale, relative, 1e-15 * a);
                REQUIRE(r.translation_derivative == -r.input_derivative);
                // The M3 L2-normalized dictionary term with center tau and scale exp(l).
                const auto hat = kan::evaluate_basis(kan::MexicanHatConfig{{tau}, {std::exp(l)}}, x);
                test::near(r.value, hat.values[0], 1e-12);
                test::near(r.input_derivative, hat.derivatives[0], 1e-12);
            }
}

TEST(wavelet_derivatives_match_finite_differences) {
    for (double tau : {-0.5, 0.2})
        for (double l : {-0.8, 0.0, 0.6})
            for (double x : {-2.0, -0.6, 0.1, 0.45, 1.7}) {
                const auto r = kan::evaluate_wavelet(x, tau, l);
                const double h = 1e-6;
                const double dx =
                    (kan::evaluate_wavelet(x + h, tau, l).value - kan::evaluate_wavelet(x - h, tau, l).value) / (2 * h);
                const double dt =
                    (kan::evaluate_wavelet(x, tau + h, l).value - kan::evaluate_wavelet(x, tau - h, l).value) / (2 * h);
                const double dl =
                    (kan::evaluate_wavelet(x, tau, l + h).value - kan::evaluate_wavelet(x, tau, l - h).value) / (2 * h);
                test::near(r.input_derivative, dx, 1e-7);
                test::near(r.translation_derivative, dt, 1e-7);
                test::near(r.log_scale_derivative, dl, 1e-7);
            }
}

TEST(wavelet_tails_keep_representable_values_and_derivatives) {
    // A narrow wavelet (s = exp(-700)) evaluated 40 scales from its center:
    // exp(-q^2/2) = exp(-800) underflows, while s^(-1/2) = exp(350) and
    // 1/s = exp(700) bring value and derivatives back into range.
    const double l = -700, tau = 0, r = std::exp(700.0);
    const double x = 40 * std::exp(-700.0);
    const auto e = kan::evaluate_wavelet(x, tau, l);
    const double q = (x - tau) * r, q2 = q * q;
    const double log_a = static_cast<double>(std::log(amplitude)) + 350 - q2 / 2;
    close(e.value, -std::exp(log_a + std::log(q2 - 1)), 1e-11);
    close(e.input_derivative, std::exp(log_a + std::log(q) + std::log(q2 - 3) + 700), 1e-11);
    close(e.log_scale_derivative, -std::exp(log_a + std::log(q2 * q2 - 3.5 * q2 + 0.5)), 1e-11);
    REQUIRE(e.value != 0 && e.input_derivative != 0 && e.log_scale_derivative != 0);
    REQUIRE(e.translation_derivative == -e.input_derivative);
    const auto left = kan::evaluate_wavelet(tau - (x - tau), tau, l);
    REQUIRE(left.input_derivative == -e.input_derivative && left.value == e.value);
    // An infinite normalized distance is the exact zero tail.
    const auto far = kan::evaluate_wavelet(1e200, 0, -300);
    REQUIRE(far.value == 0 && far.input_derivative == 0 && far.log_scale_derivative == 0);
    // x - tau overflows, q is finite: still finite results.
    const auto wide = kan::evaluate_wavelet(1.5e308, -1.5e308, 700);
    REQUIRE(std::isfinite(wide.value) && std::isfinite(wide.input_derivative) &&
            std::isfinite(wide.log_scale_derivative));
    // The retried distance x r - tau r lands in the direct range: closed form.
    {
        const double x = 1e308, tau = -1e308, l = 709;
        const auto r = kan::evaluate_wavelet(x, tau, l);
        const long double s = std::exp(static_cast<long double>(l)), q = (static_cast<long double>(x) - tau) / s;
        const long double e = std::exp(-q * q / 2), a = amplitude / std::sqrt(s);
        close(r.value, a * (1 - q * q) * e, 1e-12);
        close(r.input_derivative, a / s * q * (q * q - 3) * e, 1e-12);
        close(r.log_scale_derivative, a * e * (-q * q * q * q + 3.5L * q * q - 0.5L), 1e-12);
        REQUIRE(r.value != 0 && r.log_scale_derivative != 0); // d/dx ~ 1e-463 underflows
    }
    // Moderate tail of an ordinary wavelet: values decay smoothly, never NaN.
    for (double d : {10.0, 37.0, 38.7, 39.0, 50.0, 1e10, 1e160}) {
        const auto t = kan::evaluate_wavelet(d, 0, 0);
        REQUIRE(std::isfinite(t.value) && std::isfinite(t.input_derivative) && std::isfinite(t.log_scale_derivative));
        REQUIRE(t.value <= 0);
    }
}

TEST(wavelet_layer_construction_places_terms_on_a_grid) {
    const kan::Layer layer(2, 3, kan::WaveletConfig{4});
    const auto* edges = std::get_if<kan::WaveletEdges>(&layer.carrier());
    REQUIRE(edges != nullptr);
    REQUIRE(edges->config == kan::WaveletConfig{4});
    REQUIRE(layer.terms() == 4);
    REQUIRE(edges->coefficients == std::vector<double>(24, 0.0));
    REQUIRE(layer.bias().size() == 3 && layer.bias()[0] == 0);
    REQUIRE(edges->translations.size() == 24 && edges->log_scales.size() == 24);
    for (std::size_t e = 0; e < 6; ++e)
        for (std::size_t k = 0; k < 4; ++k) {
            REQUIRE(edges->translations[e * 4 + k] == -1.0 + static_cast<double>(2 * k + 1) / 4.0);
            test::near(edges->log_scales[e * 4 + k], -std::log(4.0), 1e-15);
        }
    const kan::Layer single(1, 1, kan::WaveletConfig{});
    REQUIRE(edges_of(single).translations == std::vector<double>{0.0});
    REQUIRE(edges_of(single).log_scales == std::vector<double>{0.0});
    REQUIRE((kan::Layer(1, 1, kan::WaveletConfig{2}).carrier() == kan::Layer(1, 1, kan::WaveletConfig{2}).carrier()));
}

TEST(wavelet_layer_forward_sums_every_edge_term) {
    const auto layer = wavelet(3, 2, 3);
    const auto& e = edges_of(layer);
    const std::vector<double> x{0.3, -0.8, 1.2, -0.1, 0.05, 2.0};
    const auto y = layer.forward(x, 2);
    REQUIRE(y.size() == 4);
    for (std::size_t b = 0; b < 2; ++b)
        for (std::size_t o = 0; o < 2; ++o) {
            long double sum = layer.bias()[o];
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t k = 0; k < 3; ++k) {
                    const auto j = (o * 3 + i) * 3 + k;
                    sum += e.coefficients[j] * closed(x[b * 3 + i], e.translations[j], e.log_scales[j]).value;
                }
            test::near(y[b * 2 + o], static_cast<double>(sum), 1e-13);
        }
    REQUIRE(layer.forward({}, 0).empty());
    const auto g = layer.backward({}, 0, {});
    REQUIRE(nonlinear(g).translations == std::vector<double>(18, 0.0));
    REQUIRE(nonlinear(g).log_scales == std::vector<double>(18, 0.0));
}

TEST(wavelet_layer_vjps_match_finite_differences) {
    for (bool residual : {false, true}) {
        auto layer = wavelet(3, 2, 2);
        if (residual) layer.set_residual(kan::SiluResidual{wave(6, 0.3, 0.7, 0.5)});
        const std::size_t batch = 3;
        const auto x = wave(batch * 3, 1.1, 0.6, 0.2), u = wave(batch * 2, 0.9, 1.3, 0.4);
        const auto g = layer.backward(x, batch, u);
        const double h = 1e-6;
        for (std::size_t j = 0; j < x.size(); ++j) {
            auto a = x, b = x;
            a[j] += h;
            b[j] -= h;
            test::near(g.input[j], (objective(layer, a, batch, u) - objective(layer, b, batch, u)) / (2 * h), 1e-7);
        }
        const auto& e = edges_of(layer);
        const auto perturbed = [&](int which, std::size_t j, double delta) {
            auto next = e;
            (which == 0 ? next.coefficients : which == 1 ? next.translations : next.log_scales)[j] += delta;
            auto copy = layer;
            copy.set_carrier(next);
            return objective(copy, x, batch, u);
        };
        for (std::size_t j = 0; j < e.coefficients.size(); ++j) {
            test::near(g.coefficients[j], (perturbed(0, j, h) - perturbed(0, j, -h)) / (2 * h), 1e-7);
            test::near(nonlinear(g).translations[j], (perturbed(1, j, h) - perturbed(1, j, -h)) / (2 * h), 1e-7);
            test::near(nonlinear(g).log_scales[j], (perturbed(2, j, h) - perturbed(2, j, -h)) / (2 * h), 1e-7);
        }
        for (std::size_t o = 0; o < 2; ++o) test::near(g.bias[o], u[o] + u[2 + o] + u[4 + o], 1e-15);
        REQUIRE(g.residual.size() == (residual ? 6u : 0u));
    }
}

TEST(wavelet_layer_matches_the_fixed_dictionary) {
    // Every edge holding the same M3 dictionary: BasisEdges and WaveletEdges
    // represent the same function and have the same input and coefficient VJPs.
    const std::vector<double> centers{-0.6, 0.1, 0.7}, scales{0.4, 0.9, 0.3};
    kan::Layer fixed(2, 3, kan::MexicanHatConfig{centers, scales});
    kan::Layer trainable(2, 3, kan::WaveletConfig{3});
    const auto w = wave(18, 0.6, 0.8, 0.1), bias = wave(3, 0.2, 1.0, 0.3);
    fixed.set_parameters(w, bias);
    std::vector<double> tau(18), l(18);
    for (std::size_t j = 0; j < 18; ++j) {
        tau[j] = centers[j % 3];
        l[j] = std::log(scales[j % 3]);
    }
    kan::set_wavelet_parameters(trainable, w, tau, l, bias);
    const auto x = wave(8, 1.3, 0.5, 0.7), u = wave(12, 0.7, 0.9, 0.2);
    const auto a = fixed.forward(x, 4), b = trainable.forward(x, 4);
    for (std::size_t j = 0; j < a.size(); ++j) test::near(b[j], a[j], 1e-13);
    const auto ga = fixed.backward(x, 4, u), gb = trainable.backward(x, 4, u);
    for (std::size_t j = 0; j < ga.input.size(); ++j) test::near(gb.input[j], ga.input[j], 1e-12);
    for (std::size_t j = 0; j < ga.coefficients.size(); ++j) test::near(gb.coefficients[j], ga.coefficients[j], 1e-12);
}

TEST(wavelet_setters_validate_and_are_atomic) {
    auto layer = wavelet(2, 1, 2);
    const auto before = layer.carrier();
    const std::vector<double> four(4, 0.1), bias{0.5};
    test::throws<std::invalid_argument>(
        [&] { kan::set_wavelet_parameters(layer, std::vector<double>(3), four, four, bias); });
    test::throws<std::invalid_argument>(
        [&] { kan::set_wavelet_parameters(layer, four, std::vector<double>(5), four, bias); });
    test::throws<std::invalid_argument>(
        [&] { kan::set_wavelet_parameters(layer, four, four, four, std::vector<double>(2)); });
    test::throws<std::invalid_argument>(
        [&] { kan::set_wavelet_parameters(layer, four, std::vector<double>{0, inf, 0, 0}, four, bias); });
    test::throws<std::invalid_argument>(
        [&] { kan::set_wavelet_parameters(layer, four, four, std::vector<double>{0, 0, 710, 0}, bias); });
    test::throws<std::invalid_argument>(
        [&] { kan::set_wavelet_parameters(layer, four, four, std::vector<double>{not_a_number, 0, 0, 0}, bias); });
    test::throws<std::invalid_argument>(
        [&] { kan::set_wavelet_parameters(layer, std::vector<double>{inf, 0, 0, 0}, four, four, bias); });
    test::throws<std::invalid_argument>([&] {
        layer.set_carrier(kan::WaveletEdges{kan::WaveletConfig{2}, four, four, std::vector<double>{-800, 0, 0, 0}});
    });
    test::throws<std::invalid_argument>(
        [&] { layer.set_carrier(kan::WaveletEdges{kan::WaveletConfig{2}, four, four, std::vector<double>(3)}); });
    test::throws<std::invalid_argument>(
        [&] { layer.set_carrier(kan::WaveletEdges{kan::WaveletConfig{0}, {}, {}, {}}); });
    REQUIRE(layer.carrier() == before);
    kan::Layer chebyshev(2, 1, kan::ChebyshevConfig{2});
    test::throws<std::invalid_argument>([&] { kan::set_wavelet_parameters(chebyshev, four, four, four, bias); });
    kan::set_wavelet_parameters(layer, four, std::vector<double>{1, 2, 3, 4}, std::vector<double>{-1, 0, 1, 2}, bias);
    REQUIRE(edges_of(layer).translations == (std::vector<double>{1, 2, 3, 4}));
    REQUIRE(edges_of(layer).log_scales == (std::vector<double>{-1, 0, 1, 2}));
    REQUIRE(layer.bias()[0] == 0.5);
    // The setter keeps the residual branch; set_parameters keeps translations and log scales.
    layer.set_residual(kan::SiluResidual{{0.1, 0.2}});
    kan::set_wavelet_parameters(layer, four, four, four, bias);
    REQUIRE(layer.residual().has_value());
    layer.set_parameters(std::vector<double>{1, 1, 1, 1}, bias);
    REQUIRE(edges_of(layer).log_scales == four);
    kan::Layer moved = std::move(layer);
    test::throws<std::invalid_argument>([&] { kan::set_wavelet_parameters(layer, four, four, four, bias); });
    (void)moved;
}

TEST(wavelet_sgd_updates_parameters_and_validates_candidates) {
    auto layer = wavelet(2, 2, 2);
    const auto x = wave(6, 0.9, 0.7), u = wave(6, 0.5, 1.1);
    const auto g = layer.backward(x, 3, u);
    const auto before = edges_of(layer);
    auto next = layer;
    next.sgd(g, 0.05);
    const auto& after = edges_of(next);
    for (std::size_t j = 0; j < before.translations.size(); ++j) {
        REQUIRE(after.translations[j] == before.translations[j] - 0.05 * nonlinear(g).translations[j]);
        REQUIRE(after.log_scales[j] == before.log_scales[j] - 0.05 * nonlinear(g).log_scales[j]);
        REQUIRE(after.coefficients[j] == before.coefficients[j] - 0.05 * g.coefficients[j]);
    }
    // A candidate log scale whose exp leaves the double range: overflow, no change.
    auto huge = g;
    std::get<kan::WaveletGradients>(huge.nonlinear).log_scales[1] = -2e4;
    test::throws<std::overflow_error>([&] { layer.sgd(huge, 0.05); });
    REQUIRE(edges_of(layer) == before);
    // Wrong gradient alternative, shape or nonfinite gradient: invalid_argument.
    auto wrong = g;
    wrong.nonlinear = kan::RationalGradients{{0}};
    test::throws<std::invalid_argument>([&] { layer.sgd(wrong, 0.05); });
    wrong = g;
    std::get<kan::WaveletGradients>(wrong.nonlinear).translations.pop_back();
    test::throws<std::invalid_argument>([&] { layer.sgd(wrong, 0.05); });
    wrong = g;
    std::get<kan::WaveletGradients>(wrong.nonlinear).log_scales[0] = not_a_number;
    test::throws<std::invalid_argument>([&] { layer.sgd(wrong, 0.05); });
    REQUIRE(edges_of(layer) == before);
    // Another carrier rejects wavelet gradients.
    kan::Layer basis(2, 2, kan::LegendreConfig{2});
    auto b = basis.backward(x, 3, u);
    b.nonlinear = kan::WaveletGradients{};
    test::throws<std::invalid_argument>([&] { basis.sgd(b, 0.1); });
}

TEST(wavelet_regularization_penalizes_coefficients_only) {
    auto layer = wavelet(2, 1, 3);
    layer.set_residual(kan::SiluResidual{{0.4, -0.2}});
    const auto r = layer.regularization(0.5);
    const auto& w = edges_of(layer).coefficients;
    long double value = 0.5L * 0.5L * (0.4L * 0.4L + 0.2L * 0.2L);
    for (std::size_t j = 0; j < w.size(); ++j) {
        REQUIRE(r.gradients.coefficients[j] == 0.5 * w[j]);
        value += 0.25L * w[j] * w[j];
    }
    test::near(r.value, static_cast<double>(value), 1e-15);
    REQUIRE(nonlinear(r.gradients).translations == std::vector<double>(6, 0.0));
    REQUIRE(nonlinear(r.gradients).log_scales == std::vector<double>(6, 0.0));
    REQUIRE(r.gradients.bias == std::vector<double>{0});
}

TEST(wavelet_nonfinite_results_raise) {
    kan::Layer layer(1, 1, kan::WaveletConfig{2});
    kan::set_wavelet_parameters(layer, std::vector<double>{1.5e308, 1.5e308}, std::vector<double>{0, 0},
                                std::vector<double>{0, 0}, std::vector<double>{0});
    test::throws<std::overflow_error>([&] { layer.forward(std::vector<double>{0.0}, 1); });
    // A narrow wavelet with a large coefficient: the input derivative overflows.
    kan::set_wavelet_parameters(layer, std::vector<double>{1e300, 0}, std::vector<double>{0, 0},
                                std::vector<double>{-600, 0}, std::vector<double>{0});
    test::throws<std::overflow_error>([&] { layer.backward(std::vector<double>{1e-261}, 1, std::vector<double>{1}); });
}

TEST(wavelet_networks_differentiate_and_learn) {
    kan::Layer first(2, 3, kan::WaveletConfig{2});
    kan::set_wavelet_parameters(first, wave(12, 0.4, 0.8, 0.2), std::vector<double>(edges_of(first).translations),
                                std::vector<double>(edges_of(first).log_scales), std::vector<double>(3, 0.0));
    kan::Layer second(3, 1, kan::ChebyshevConfig{3});
    second.set_parameters(wave(9, 0.5, 1.2, 0.4), std::vector<double>{0});
    kan::Network network({first, second});
    const std::size_t batch = 32;
    std::vector<double> x(batch * 2), y(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        x[2 * b] = -1 + 2.0 * static_cast<double>(b) / (batch - 1);
        x[2 * b + 1] = std::sin(3.0 * static_cast<double>(b));
        y[b] = std::exp(-8 * (x[2 * b] - 0.35) * (x[2 * b] - 0.35)) - 0.3 * x[2 * b + 1];
    }
    // Input VJP of the network against finite differences.
    const auto u = wave(batch, 0.3, 0.9);
    const auto g = network.backward(x, batch, u);
    const auto loss = [&](const std::vector<double>& in) {
        const auto out = network.forward(in, batch);
        long double s = 0;
        for (std::size_t b = 0; b < batch; ++b) s += static_cast<long double>(u[b]) * out[b];
        return static_cast<double>(s);
    };
    for (std::size_t j = 0; j < 6; ++j) {
        auto a = x, b = x;
        a[j] += 1e-6;
        b[j] -= 1e-6;
        test::near(g.input[j], (loss(a) - loss(b)) / 2e-6, 1e-7);
    }
    // Full-batch MSE training moves every translation and log scale, and the
    // two terms of an edge, which start apart, stay distinct.
    const auto mse = [&] {
        const auto out = network.forward(x, batch);
        double s = 0;
        for (std::size_t b = 0; b < batch; ++b) s += (out[b] - y[b]) * (out[b] - y[b]);
        return s / batch;
    };
    const double start = mse();
    const auto initial = edges_of(test::layer(network, 0));
    for (int step = 0; step < 300; ++step) {
        const auto out = network.forward(x, batch);
        std::vector<double> d(batch);
        for (std::size_t b = 0; b < batch; ++b) d[b] = 2 * (out[b] - y[b]) / batch;
        network.sgd(network.backward(x, batch, d), 0.1);
    }
    REQUIRE(mse() < 0.5 * start);
    const auto& trained = edges_of(test::layer(network, 0));
    for (std::size_t j = 0; j < trained.translations.size(); ++j) {
        REQUIRE(trained.translations[j] != initial.translations[j]);
        REQUIRE(trained.log_scales[j] != initial.log_scales[j]);
    }
    for (std::size_t e = 0; e < 6; ++e) REQUIRE(trained.translations[2 * e] != trained.translations[2 * e + 1]);
}

int main() { return test::run(); }

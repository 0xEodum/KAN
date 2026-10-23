// M5 phase 1: PqcEdges through the carrier protocol on the CPU. Construction,
// the symmetric point, layer VJPs against finite differences, validation and
// atomic setters/SGD, L2, the residual branch, a mixed network, the D8
// initializers (pinned digest, draw order, moments) and deterministic training.
#include "kan/families.hpp"
#include "kan/initializers.hpp"
#include "kan/network.hpp"
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
#include <utility>
#include <vector>

namespace {
using kan::Entangler;
using kan::PqcConfig;
using kan::PqcReadout;
using kan::Rotation;
constexpr double pi = std::numbers::pi;
const double inf = std::numeric_limits<double>::infinity();
const double not_a_number = std::numeric_limits<double>::quiet_NaN();

PqcConfig config(std::size_t n, std::size_t r, std::vector<Rotation> block = {Rotation::Z, Rotation::Y, Rotation::Z},
                 Entangler entangler = Entangler::None, PqcReadout readout = PqcReadout::PauliZ,
                 Rotation encoding = Rotation::Y) {
    PqcConfig c;
    c.qubits = n;
    c.repetitions = r;
    c.encoding = encoding;
    c.block = std::move(block);
    c.entangler = entangler;
    c.readout = readout;
    return c;
}

std::vector<double> wave(std::size_t n, double scale, double frequency, double phase = 0) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * std::sin(frequency * static_cast<double>(k) + phase);
    return v;
}

const kan::PqcEdges& edges_of(const kan::Layer& layer) { return std::get<kan::PqcEdges>(layer.carrier()); }
const kan::PqcGradients& nonlinear(const kan::LayerGradients& g) { return std::get<kan::PqcGradients>(g.nonlinear); }

// A PQC layer with distinct, nonzero parameters.
kan::Layer pqc(std::size_t in, std::size_t out, const PqcConfig& c) {
    kan::Layer layer(in, out, c);
    const auto& e = edges_of(layer);
    kan::set_pqc_parameters(layer, wave(e.coefficients.size(), 0.7, 0.9, 0.3), wave(e.angles.size(), 2.5, 1.7, 0.1),
                            wave(e.encoding_weights.size(), 1.2, 2.3, 0.9),
                            wave(e.encoding_biases.size(), 0.6, 0.7, 0.2), wave(out, 0.1, 1.1));
    return layer;
}

double objective(const kan::Layer& layer, const std::vector<double>& x, std::size_t batch,
                 const std::vector<double>& u) {
    const auto y = layer.forward(x, batch);
    long double sum = 0;
    for (std::size_t j = 0; j < y.size(); ++j) sum += static_cast<long double>(u[j]) * y[j];
    return static_cast<double>(sum);
}
void against_fd(double actual, double fd, const char* what) {
    if (!(std::abs(actual - fd) <= 1e-7 * std::abs(fd) + 1e-9))
        throw std::runtime_error(std::string(what) + ": analytic=" + std::to_string(actual) +
                                 " fd=" + std::to_string(fd));
}

// Independent SplitMix64 draw `index` (0-based) of a generator started at seed.
std::uint64_t mix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
double symmetric_uniform(std::uint64_t seed, std::uint64_t index) {
    return 2 * (static_cast<double>(mix(seed + (index + 1) * 0x9E3779B97F4A7C15ull) >> 11) * 0x1p-53) - 1;
}

void fnv(std::uint64_t& h, std::span<const double> values) {
    for (double v : values) {
        const auto bits = std::bit_cast<std::uint64_t>(v);
        for (int b = 0; b < 8; ++b) h = (h ^ ((bits >> (8 * b)) & 0xff)) * 0x100000001b3ull;
    }
}
std::string digest(const kan::Layer& layer) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    const auto& e = edges_of(layer);
    fnv(h, e.coefficients);
    fnv(h, e.angles);
    fnv(h, e.encoding_weights);
    fnv(h, e.encoding_biases);
    fnv(h, layer.bias());
    if (layer.residual()) fnv(h, layer.residual()->weights);
    char buffer[24];
    std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(h));
    return buffer;
}

double mean_square(std::span<const double> v) {
    double s = 0;
    for (double x : v) s += x * x;
    return s / double(v.size());
}

// G: tape length, every rotation, encoding rotation and two-qubit gate.
std::size_t gate_count(const PqcConfig& c) {
    const std::size_t n = c.qubits, r = c.repetitions;
    const std::size_t entangling = c.entangler == Entangler::None ? 0 : c.entangler == Entangler::CnotRing ? n : n - 1;
    return (r + 1) * n * c.block.size() + r * n + r * entangling;
}

// Bound on a parameter-shift derivative d = (m(fl(p + q)) - m(fl(p - q))) / 2,
// q = fl(pi/2), at a point p where the exact derivative is 0; u = 2^-52.
// (i) In p alone m(p) = a + b cos p + c sin p (R(p) enters through cos p/2 and
//     sin p/2 on both sides), so m(p + s) - m(p - s) = 2 sin(s) m'(p) = 0 for
//     every s: the inexact q costs nothing.
// (ii) fl(p +- q) = p +- q + eta, |eta| <= (u/2)(|p| + q), and |m'| <=
//     sqrt(b^2 + c^2) <= 1 because |m| <= 1 (Z or parity readout): at most
//     (|eta+| + |eta-|) / 2 <= (u/2)(|p| + 2).
// (iii) Each computed expectation: state error |e| <= 2 G u (the per-gate
//     budget of the normalization bound (4G + 2^n) u in pqc_test), so
//     |m^ - m| <= 2|e| + |e|^2 + 2^n u (readout sum, |M| <= 1), at most
//     (4G + 2^n) u; half the difference of two such errors is no larger.
//     The 2 G u state error is an assumed first-order per-gate budget, not a
//     proven forward-error bound for the fused 2x2 products, so the total is a
//     first-order estimate calibrated by measurement (M5-pqc-cpu evidence).
// The final subtraction and halving add at most u/2 |d|. Total:
// |d| <= (4G + 2^n + |p| + 2) u.
double shift_zero_bound(const PqcConfig& c, double p) {
    return (4 * double(gate_count(c)) + double(std::size_t{1} << c.qubits) + std::abs(p) + 2) * 0x1p-52;
}
} // namespace

TEST(pqc_layer_construction) {
    const auto c = config(2, 3, {Rotation::Z, Rotation::Y}, Entangler::CnotChain);
    const kan::Layer layer(3, 4, c);
    const auto* edges = std::get_if<kan::PqcEdges>(&layer.carrier());
    REQUIRE(edges != nullptr);
    REQUIRE(edges->config == c);
    REQUIRE(layer.terms() == 2);
    REQUIRE(edges->coefficients == std::vector<double>(24, 0.0));
    REQUIRE(edges->angles == std::vector<double>(12 * 16, 0.0)); // (r+1) n B = 16 per edge
    REQUIRE(edges->encoding_weights == std::vector<double>(12 * 6, 1.0));
    REQUIRE(edges->encoding_biases == std::vector<double>(12 * 6, 0.0));
    REQUIRE(edges->shot_seed == 0);
    REQUIRE(layer.bias().size() == 4 && layer.bias()[0] == 0);
    REQUIRE(kan::Layer(3, 2, config(3, 1, {Rotation::X}, Entangler::None, PqcReadout::Parity)).terms() == 1);
    REQUIRE((kan::Layer(2, 2, PqcConfig{}).carrier() == kan::Layer(2, 2, PqcConfig{}).carrier()));
    test::throws<std::invalid_argument>([] { kan::Layer(0, 1, PqcConfig{}); });
    test::throws<std::invalid_argument>([&] { kan::Layer(1, 1, config(0, 1)); });
    auto shots = PqcConfig{};
    shots.shots = 16;
    test::throws<std::invalid_argument>([&] { kan::Layer(1, 1, shots); });
    // n = 1, Y encoding, c = 1: every edge is cos(r x).
    for (std::size_t r : {1u, 2u, 4u}) {
        kan::Layer one(2, 1, config(1, r));
        one.set_parameters(std::vector<double>{1, 1}, std::vector<double>{0.25});
        const std::vector<double> x{0.3, -1.1, 2.0, 0.45};
        const auto y = one.forward(x, 2);
        for (std::size_t b = 0; b < 2; ++b)
            test::near(y[b], 0.25 + std::cos(double(r) * x[2 * b]) + std::cos(double(r) * x[2 * b + 1]), 1e-15);
    }
    // Z encoding: the constructor circuit is diagonal and the edge constant in x.
    kan::Layer diagonal(
        1, 1, config(1, 2, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::None, PqcReadout::PauliZ, Rotation::Z));
    diagonal.set_parameters(std::vector<double>{1}, std::vector<double>{0});
    const auto y = diagonal.forward(std::vector<double>{-0.7, 0.2, 1.9}, 3);
    for (double v : y) REQUIRE(std::abs(v - 1) <= 4 * 0x1p-52); // |phase|^2 = c^2 + s^2 rounds
}

TEST(pqc_symmetric_point_derivatives) {
    // Constructor parameters, Y encoding, c = 1: a real-amplitude circuit, so
    // every Z-angle adjoint derivative is exactly 0 (the algorithm guarantees
    // it), the parameter-shift one is within shift_zero_bound (measured: exactly
    // 0 on glibc, the +-q states being bitwise conjugates; IEEE 754 does not
    // guarantee that), while input, encoding and Y-angle derivatives are not 0.
    // Shallow cases, then mid-depth (n = 3, r = 8) and the D9 limit (n = 5,
    // r = 32) with every entangler and both readouts.
    std::vector<PqcConfig> cases{config(1, 2),
                                 config(2, 3, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CnotChain),
                                 config(3, 2, {Rotation::Y, Rotation::Z}, Entangler::CnotRing, PqcReadout::Parity)};
    for (auto [n, r] : {std::pair<std::size_t, std::size_t>{3, 8}, {5, 32}})
        for (auto entangler : {Entangler::None, Entangler::CnotChain, Entangler::CzChain, Entangler::CnotRing})
            for (auto readout : {PqcReadout::PauliZ, PqcReadout::Parity})
                cases.push_back(config(n, r, {Rotation::Z, Rotation::Y, Rotation::Z}, entangler, readout));
    double worst = 0; // largest |shift Z-angle derivative| / shift_zero_bound
    for (const auto& c : cases) {
        kan::Layer layer(1, 1, c);
        layer.set_parameters(std::vector<double>(layer.terms(), 1.0), std::vector<double>{0});
        const double x = 0.37;
        const auto g = layer.backward(std::vector<double>{x}, 1, std::vector<double>{1});
        const auto& e = edges_of(layer);
        const std::size_t B = c.block.size();
        bool y_nonzero = false;
        for (std::size_t j = 0; j < e.angles.size(); ++j) {
            if (c.block[j % B] == Rotation::Z)
                REQUIRE(nonlinear(g).angles[j] == 0.0);
            else
                y_nonzero = y_nonzero || nonlinear(g).angles[j] != 0.0;
        }
        REQUIRE(y_nonzero);
        REQUIRE(g.input[0] != 0.0);
        for (std::size_t j = 0; j < e.encoding_weights.size(); ++j)
            REQUIRE(nonlinear(g).encoding_weights[j] != 0.0 && nonlinear(g).encoding_biases[j] != 0.0);
        const auto shift = kan::evaluate_pqc(c, x, e.angles, e.encoding_weights, e.encoding_biases,
                                             kan::GradientMethod::ParameterShift);
        const auto adjoint = kan::evaluate_pqc(c, x, e.angles, e.encoding_weights, e.encoding_biases);
        for (std::size_t t = 0; t < layer.terms(); ++t)
            for (std::size_t j = 0; j < e.angles.size(); ++j)
                if (c.block[j % B] == Rotation::Z) {
                    REQUIRE(adjoint.angle_derivatives[t * e.angles.size() + j] == 0.0);
                    const double d = std::abs(shift.angle_derivatives[t * e.angles.size() + j]);
                    const double bound = shift_zero_bound(c, e.angles[j]);
                    worst = std::max(worst, d / bound);
                    REQUIRE(d <= bound);
                }
    }
    std::printf("symmetric point: largest |shift Z derivative| is %.3e of (4G + 2^n + |p| + 2) 2^-52\n", worst);
}

TEST(pqc_layer_forward_sums_every_edge_term) {
    const auto c = config(2, 2, {Rotation::X, Rotation::Z}, Entangler::CzChain);
    const auto layer = pqc(3, 2, c);
    const auto& e = edges_of(layer);
    const std::vector<double> x{0.3, -0.8, 1.2, -0.1, 0.05, 2.0};
    const auto y = layer.forward(x, 2);
    const std::size_t A = kan::pqc_angles(c), E = kan::pqc_encodings(c);
    for (std::size_t b = 0; b < 2; ++b)
        for (std::size_t o = 0; o < 2; ++o) {
            double sum = layer.bias()[o];
            for (std::size_t i = 0; i < 3; ++i) {
                const std::size_t edge = o * 3 + i;
                const auto m = kan::evaluate_pqc(c, x[b * 3 + i], std::span(e.angles).subspan(edge * A, A),
                                                 std::span(e.encoding_weights).subspan(edge * E, E),
                                                 std::span(e.encoding_biases).subspan(edge * E, E))
                                   .values;
                for (std::size_t t = 0; t < 2; ++t) sum += e.coefficients[edge * 2 + t] * m[t];
            }
            REQUIRE(y[b * 2 + o] == sum); // the documented order: ascending i, then t
            REQUIRE(std::abs(sum - layer.bias()[o]) > 1e-3);
        }
    REQUIRE(layer.forward({}, 0).empty());
    const auto g = layer.backward({}, 0, {});
    REQUIRE(nonlinear(g).angles == std::vector<double>(e.angles.size(), 0.0));
    REQUIRE(nonlinear(g).encoding_weights == std::vector<double>(e.encoding_weights.size(), 0.0));
    REQUIRE(g.coefficients == std::vector<double>(e.coefficients.size(), 0.0));
}

TEST(pqc_layer_vjps_match_finite_differences) {
    for (const auto& c : {config(2, 2, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CnotChain),
                          config(3, 1, {Rotation::Y, Rotation::X}, Entangler::CnotRing, PqcReadout::Parity),
                          config(1, 3, {Rotation::X, Rotation::Z}, Entangler::None, PqcReadout::PauliZ, Rotation::X),
                          config(4, 1, {Rotation::Y, Rotation::Z}, Entangler::CzChain, PqcReadout::Parity),
                          config(5, 1, {Rotation::X}, Entangler::CnotRing, PqcReadout::PauliZ, Rotation::Z)})
        for (bool residual : {false, true}) {
            auto layer = pqc(3, 2, c);
            if (residual) layer.set_residual(kan::SiluResidual{wave(6, 0.3, 0.7, 0.5)});
            const std::size_t batch = 3;
            const auto x = wave(batch * 3, 1.1, 0.6, 0.2), u = wave(batch * 2, 0.9, 1.3, 0.4);
            const auto g = layer.backward(x, batch, u);
            const double h = 1e-6;
            for (std::size_t j = 0; j < x.size(); ++j) {
                auto a = x, b = x;
                a[j] += h;
                b[j] -= h;
                against_fd(g.input[j], (objective(layer, a, batch, u) - objective(layer, b, batch, u)) / (2 * h),
                           "d/dx");
            }
            const auto& e = edges_of(layer);
            const auto perturbed = [&](int which, std::size_t j, double delta) {
                auto next = e;
                (which == 0   ? next.coefficients
                 : which == 1 ? next.angles
                 : which == 2 ? next.encoding_weights
                              : next.encoding_biases)[j] += delta;
                auto copy = layer;
                copy.set_carrier(next);
                return objective(copy, x, batch, u);
            };
            const auto fd = [&](int which, std::size_t j) {
                return (perturbed(which, j, h) - perturbed(which, j, -h)) / (2 * h);
            };
            for (std::size_t j = 0; j < e.coefficients.size(); ++j) against_fd(g.coefficients[j], fd(0, j), "d/dc");
            for (std::size_t j = 0; j < e.angles.size(); ++j) against_fd(nonlinear(g).angles[j], fd(1, j), "d/dangle");
            for (std::size_t j = 0; j < e.encoding_weights.size(); ++j) {
                against_fd(nonlinear(g).encoding_weights[j], fd(2, j), "d/dw");
                against_fd(nonlinear(g).encoding_biases[j], fd(3, j), "d/db");
            }
            for (std::size_t o = 0; o < 2; ++o) test::near(g.bias[o], u[o] + u[2 + o] + u[4 + o], 1e-15);
            REQUIRE(g.residual.size() == (residual ? 6u : 0u));
            // Not a comparison of zeros: the edges vary with x and their parameters.
            REQUIRE(std::ranges::any_of(g.input, [](double v) { return std::abs(v) > 1e-3; }));
            REQUIRE(std::ranges::any_of(nonlinear(g).angles, [](double v) { return std::abs(v) > 1e-3; }));
            REQUIRE(std::ranges::any_of(nonlinear(g).encoding_biases, [](double v) { return std::abs(v) > 1e-3; }));
        }
}

TEST(pqc_inputs_and_parameters_must_be_finite) {
    auto layer = pqc(2, 1, config(1, 2));
    test::throws<std::invalid_argument>([&] { layer.forward(std::vector<double>{0.1, not_a_number}, 1); });
    test::throws<std::invalid_argument>(
        [&] { layer.backward(std::vector<double>{0.1, inf}, 1, std::vector<double>{1}); });
    test::throws<std::invalid_argument>(
        [&] { layer.backward(std::vector<double>{0.1, 0.2}, 1, std::vector<double>{not_a_number}); });
    // Overflowing readout (constructor circuit at x = 0: m = 1): the forward result is not finite.
    kan::Layer big(2, 1, config(1, 2));
    big.set_parameters(std::vector<double>{1.5e308, 1.5e308}, std::vector<double>{1.5e308});
    test::throws<std::overflow_error>([&] { big.forward(std::vector<double>{0.0, 0.0}, 1); });
    // An overflowing encoding angle w x + b (w = 10): forward and backward raise, the layer is unchanged.
    kan::Layer steep(2, 1, config(1, 2));
    const auto& e = edges_of(steep);
    kan::set_pqc_parameters(steep, std::vector<double>{1, 1}, e.angles, std::vector<double>(4, 10.0), e.encoding_biases,
                            std::vector<double>{0});
    const auto before = steep.carrier();
    test::throws<std::overflow_error>([&] { steep.forward(std::vector<double>{1e308, 0.5}, 1); });
    test::throws<std::overflow_error>(
        [&] { steep.backward(std::vector<double>{1e308, 0.5}, 1, std::vector<double>{1}); });
    REQUIRE(steep.carrier() == before);
}

TEST(pqc_set_carrier_and_setters_validate_atomically) {
    const auto c = config(2, 1, {Rotation::Y, Rotation::Z}, Entangler::CzChain);
    auto layer = pqc(2, 1, c);
    const auto before = layer.carrier();
    const auto bias_before = std::vector<double>(layer.bias().begin(), layer.bias().end());
    const auto& e = edges_of(layer);
    const auto C = e.coefficients, A = e.angles, W = e.encoding_weights, Bb = e.encoding_biases;
    const std::vector<double> bias{0.5};
    const auto shorter = [](std::vector<double> v) {
        v.pop_back();
        return v;
    };
    const auto with = [](std::vector<double> v, double value) {
        v[1] = value;
        return v;
    };
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, shorter(C), A, W, Bb, bias); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, shorter(A), W, Bb, bias); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, A, shorter(W), Bb, bias); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, A, W, shorter(Bb), bias); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, A, W, Bb, std::vector<double>{}); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, with(C, inf), A, W, Bb, bias); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, with(A, not_a_number), W, Bb, bias); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, A, with(W, -inf), Bb, bias); });
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, A, W, with(Bb, inf), bias); });
    test::throws<std::invalid_argument>(
        [&] { kan::set_pqc_parameters(layer, C, A, W, Bb, std::vector<double>{not_a_number}); });
    // set_carrier: configuration, shapes, finiteness, shot seed without shots.
    auto next = e;
    next.shot_seed = 7;
    test::throws<std::invalid_argument>([&] { layer.set_carrier(next); });
    next = e;
    next.config.qubits = 6;
    test::throws<std::invalid_argument>([&] { layer.set_carrier(next); });
    next = e;
    next.config.shots = 4;
    test::throws<std::invalid_argument>([&] { layer.set_carrier(next); });
    next = e;
    next.config.repetitions = 2; // parameter shapes no longer match
    test::throws<std::invalid_argument>([&] { layer.set_carrier(next); });
    next = e;
    next.angles.push_back(0);
    test::throws<std::invalid_argument>([&] { layer.set_carrier(next); });
    next = e;
    next.encoding_biases[0] = inf;
    test::throws<std::invalid_argument>([&] { layer.set_carrier(next); });
    REQUIRE(layer.carrier() == before);
    REQUIRE(std::ranges::equal(layer.bias(), bias_before));
    // Another carrier is rejected; a valid call replaces everything and keeps the seed state.
    kan::Layer chebyshev(2, 1, kan::ChebyshevConfig{2});
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(chebyshev, C, A, W, Bb, bias); });
    const auto A2 = wave(A.size(), 1.0, 0.3), W2 = wave(W.size(), 0.5, 0.2, 1.0);
    kan::set_pqc_parameters(layer, C, A2, W2, Bb, bias);
    REQUIRE(edges_of(layer).angles == A2 && edges_of(layer).encoding_weights == W2);
    REQUIRE(layer.bias()[0] == 0.5);
    // set_parameters keeps the nonlinear parameters; the setter keeps the residual branch.
    layer.set_residual(kan::SiluResidual{{0.1, 0.2}});
    layer.set_parameters(std::vector<double>(C.size(), 2.0), bias);
    REQUIRE(edges_of(layer).angles == A2 && edges_of(layer).encoding_weights == W2 &&
            edges_of(layer).encoding_biases == Bb);
    kan::set_pqc_parameters(layer, C, A, W, Bb, bias);
    REQUIRE(layer.residual().has_value());
    kan::Layer moved = std::move(layer);
    test::throws<std::invalid_argument>([&] { kan::set_pqc_parameters(layer, C, A, W, Bb, bias); });
    (void)moved;
}

TEST(pqc_sgd_updates_parameters_and_is_atomic) {
    auto layer = pqc(2, 2, config(2, 2, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CnotChain));
    const auto x = wave(6, 0.9, 0.7), u = wave(6, 0.5, 1.1);
    const auto g = layer.backward(x, 3, u);
    const auto before = edges_of(layer);
    auto next = layer;
    next.sgd(g, 0.05);
    const auto& after = edges_of(next);
    for (std::size_t j = 0; j < before.angles.size(); ++j)
        REQUIRE(after.angles[j] == before.angles[j] - 0.05 * nonlinear(g).angles[j]);
    for (std::size_t j = 0; j < before.encoding_weights.size(); ++j) {
        REQUIRE(after.encoding_weights[j] == before.encoding_weights[j] - 0.05 * nonlinear(g).encoding_weights[j]);
        REQUIRE(after.encoding_biases[j] == before.encoding_biases[j] - 0.05 * nonlinear(g).encoding_biases[j]);
    }
    for (std::size_t j = 0; j < before.coefficients.size(); ++j)
        REQUIRE(after.coefficients[j] == before.coefficients[j] - 0.05 * g.coefficients[j]);
    // A nonfinite candidate (finite gradient, overflowing step): overflow, no change.
    for (int field = 0; field < 3; ++field) {
        auto huge = g;
        auto& p = std::get<kan::PqcGradients>(huge.nonlinear);
        (field == 0 ? p.angles : field == 1 ? p.encoding_weights : p.encoding_biases)[1] = -1.7e308;
        test::throws<std::overflow_error>([&] { layer.sgd(huge, 1.5); });
        REQUIRE(edges_of(layer) == before);
    }
    // Wrong alternative, shape or nonfinite gradient: invalid_argument.
    auto wrong = g;
    wrong.nonlinear = kan::WaveletGradients{};
    test::throws<std::invalid_argument>([&] { layer.sgd(wrong, 0.05); });
    wrong = g;
    std::get<kan::PqcGradients>(wrong.nonlinear).encoding_biases.pop_back();
    test::throws<std::invalid_argument>([&] { layer.sgd(wrong, 0.05); });
    wrong = g;
    std::get<kan::PqcGradients>(wrong.nonlinear).angles[0] = not_a_number;
    test::throws<std::invalid_argument>([&] { layer.sgd(wrong, 0.05); });
    REQUIRE(edges_of(layer) == before);
    kan::Layer basis(2, 2, kan::LegendreConfig{2});
    auto b = basis.backward(x, 3, u);
    b.nonlinear = kan::PqcGradients{};
    test::throws<std::invalid_argument>([&] { basis.sgd(b, 0.1); });
}

TEST(pqc_regularization_penalizes_coefficients_only) {
    auto layer = pqc(2, 1, config(2, 1));
    layer.set_residual(kan::SiluResidual{{0.4, -0.2}});
    const auto r = layer.regularization(0.5);
    const auto& e = edges_of(layer);
    long double value = 0.5L * 0.5L * (0.4L * 0.4L + 0.2L * 0.2L);
    for (std::size_t j = 0; j < e.coefficients.size(); ++j) {
        REQUIRE(r.gradients.coefficients[j] == 0.5 * e.coefficients[j]);
        value += 0.25L * e.coefficients[j] * e.coefficients[j];
    }
    test::near(r.value, static_cast<double>(value), 1e-15);
    REQUIRE(nonlinear(r.gradients).angles == std::vector<double>(e.angles.size(), 0.0));
    REQUIRE(nonlinear(r.gradients).encoding_weights == std::vector<double>(e.encoding_weights.size(), 0.0));
    REQUIRE(nonlinear(r.gradients).encoding_biases == std::vector<double>(e.encoding_biases.size(), 0.0));
    REQUIRE(r.gradients.residual == (std::vector<double>{0.2, -0.1}));
    REQUIRE(r.gradients.bias == std::vector<double>{0});
}

TEST(pqc_in_a_mixed_network_matches_finite_differences) {
    // Chebyshev -> TanhMap -> PQC -> wavelet.
    kan::Layer first(2, 3, kan::ChebyshevConfig{3});
    first.set_parameters(wave(18, 0.5, 0.8, 0.2), wave(3, 0.1, 0.5));
    const auto middle = pqc(3, 2, config(2, 2, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CnotChain));
    kan::Layer last(2, 1, kan::WaveletConfig{2});
    kan::set_wavelet_parameters(last, wave(4, 0.6, 0.7, 0.4), wave(4, 0.5, 1.9), wave(4, 0.2, 0.3),
                                std::vector<double>{0.05});
    kan::Network network({first, kan::InputMap(3, kan::TanhMap{1.0}), middle, last});
    const std::size_t batch = 4;
    const auto x = wave(batch * 2, 0.9, 0.77, 0.1), u = wave(batch, 0.8, 1.4);
    const auto g = network.backward(x, batch, u);
    const auto loss = [&](const kan::Network& net, const std::vector<double>& in) {
        const auto out = net.forward(in, batch);
        long double s = 0;
        for (std::size_t b = 0; b < batch; ++b) s += static_cast<long double>(u[b]) * out[b];
        return static_cast<double>(s);
    };
    const double h = 1e-6;
    for (std::size_t j = 0; j < x.size(); ++j) {
        auto a = x, b = x;
        a[j] += h;
        b[j] -= h;
        against_fd(g.input[j], (loss(network, a) - loss(network, b)) / (2 * h), "network d/dx");
    }
    const auto& pg = std::get<kan::PqcGradients>(test::grad(g, 2).nonlinear);
    const auto& e = edges_of(middle);
    REQUIRE(std::ranges::any_of(pg.angles, [](double v) { return std::abs(v) > 1e-3; }));
    REQUIRE(std::ranges::any_of(g.input, [](double v) { return std::abs(v) > 1e-3; }));
    for (std::size_t j = 0; j < e.angles.size(); j += 5) {
        const auto rebuilt = [&](double delta) {
            auto changed = middle;
            auto next = e;
            next.angles[j] += delta;
            changed.set_carrier(next);
            return kan::Network({first, kan::InputMap(3, kan::TanhMap{1.0}), changed, last});
        };
        against_fd(pg.angles[j], (loss(rebuilt(h), x) - loss(rebuilt(-h), x)) / (2 * h), "network d/dangle");
    }
    for (std::size_t j = 0; j < e.encoding_weights.size(); ++j) {
        auto changed = middle, other = middle;
        auto plus = e, minus = e;
        plus.encoding_weights[j] += h;
        minus.encoding_weights[j] -= h;
        changed.set_carrier(plus);
        other.set_carrier(minus);
        const kan::Network a({first, kan::InputMap(3, kan::TanhMap{1.0}), changed, last}),
            b({first, kan::InputMap(3, kan::TanhMap{1.0}), other, last});
        against_fd(pg.encoding_weights[j], (loss(a, x) - loss(b, x)) / (2 * h), "network d/dw");
    }
}

TEST(pqc_initializer_draw_order_and_moments) {
    // D8: coefficients in layout order (sigma_t^2 = gain^2 (1/3) / (inputs terms m)),
    // then block angles pi (2u - 1) in layout order; encodings kept; bias zero.
    struct Case {
        PqcConfig config;
        double m;
    };
    for (const auto& [c, m] : {Case{config(2, 1, {Rotation::Y, Rotation::Z}), 1.0 / 3},
                               Case{config(2, 2, {Rotation::Y}, Entangler::None, PqcReadout::Parity), 1.0 / 9},
                               Case{config(3, 1, {Rotation::X}, Entangler::None, PqcReadout::Parity), 1.0 / 27},
                               Case{config(2, 1, {Rotation::Z, Rotation::Y}, Entangler::CnotChain), 1.0 / 5},
                               Case{config(3, 2, {Rotation::Y}, Entangler::CzChain, PqcReadout::Parity), 1.0 / 9}}) {
        kan::Layer layer(3, 2, c);
        auto placed = edges_of(layer);
        const auto W = wave(placed.encoding_weights.size(), 0.8, 0.3, 1.0);
        kan::set_pqc_parameters(layer, placed.coefficients, placed.angles, W, placed.encoding_biases,
                                std::vector<double>{0.3, 0.4});
        const std::uint64_t seed = 42;
        kan::initialize(layer, kan::VarianceScaling{0.9, kan::Distribution::Uniform, seed, {}});
        const auto& e = edges_of(layer);
        const double terms = double(layer.terms());
        const double sigma = std::sqrt(0.81 * (1.0 / 3) / (3 * terms * m));
        const std::size_t K = e.coefficients.size();
        for (std::size_t j = 0; j < K; ++j)
            test::near(e.coefficients[j], std::sqrt(3.0) * sigma * symmetric_uniform(seed, j), 1e-14);
        for (std::size_t j = 0; j < e.angles.size(); ++j) {
            REQUIRE(e.angles[j] == pi * symmetric_uniform(seed, K + j));
            REQUIRE(e.angles[j] >= -pi && e.angles[j] < pi);
        }
        REQUIRE(e.encoding_weights == W);
        REQUIRE(e.encoding_biases == placed.encoding_biases);
        REQUIRE(e.shot_seed == 0);
        REQUIRE(std::ranges::equal(layer.bias(), std::vector<double>{0, 0}));
        // NoiseInit: a = scale / (terms sqrt(inputs)), uniform (a/2)(2u - 1); then the angles.
        kan::initialize(layer, kan::NoiseInit{0.3, kan::Distribution::Uniform, seed + 1, {}});
        const double a = 0.3 / (terms * std::sqrt(3.0));
        for (std::size_t j = 0; j < K; ++j)
            test::near(edges_of(layer).coefficients[j], 0.5 * a * symmetric_uniform(seed + 1, j), 1e-14);
        for (std::size_t j = 0; j < e.angles.size(); ++j)
            REQUIRE(edges_of(layer).angles[j] == pi * symmetric_uniform(seed + 1, K + j));
    }
    // A pending second normal draw of the coefficients survives the uniform
    // angle draws and is the first residual draw (the denominator rule):
    // three PQC coefficients (Parity, 1 x 3) against four Chebyshev ones.
    kan::Layer quantum(1, 3,
                       config(1, 2, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::None, PqcReadout::Parity));
    quantum.set_residual(kan::SiluResidual{{0, 0, 0}});
    kan::Layer chebyshev(1, 4, kan::ChebyshevConfig{1});
    const kan::NoiseInit noise{0.3, kan::Distribution::Normal, 5, {}, 0.0, std::sqrt(3.0)};
    kan::initialize(quantum, noise);
    kan::initialize(chebyshev, noise);
    for (std::size_t j = 0; j < 3; ++j) REQUIRE(quantum.coefficients()[j] == chebyshev.coefficients()[j]);
    const double z = chebyshev.coefficients()[3] / (0.3 / std::sqrt(12.0));
    test::near(quantum.residual()->weights[0], z * (std::sqrt(3.0) / std::sqrt(3.0)), 1e-14);
}

TEST(pqc_initializer_digest_is_pinned) {
    kan::Layer layer(5, 4, config(2, 3, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CnotChain));
    layer.set_residual(kan::SiluResidual{std::vector<double>(20, 0.0)});
    auto copy = layer;
    kan::initialize(layer, kan::VarianceScaling{1, kan::Distribution::Normal, 12, {}});
    kan::initialize(copy, kan::VarianceScaling{1, kan::Distribution::Normal, 12, {}});
    REQUIRE(digest(layer) == digest(copy));
    kan::initialize(copy, kan::VarianceScaling{1, kan::Distribution::Normal, 13, {}});
    REQUIRE(digest(layer) != digest(copy));
    const auto value = digest(layer);
    std::printf("pinned %s\n", value.c_str());
    // Pinned with GCC on aarch64; MSVC 19.50 on x86-64 reproduces it
    // (docs/evidence/port-windows-3090.md).
    REQUIRE(value == "c8bec86f1c294f6f");
}

TEST(pqc_variance_scaling_preserves_the_second_moment) {
    // E[y^2] / (gain^2 / 3) for x uniform on [-1, 1], averaged over 16 seeds:
    // asserted within 0.15 for Entangler::None, r = 4 (n = 1, 2); recorded otherwise.
    std::uint64_t state = 7;
    const auto next_u = [&] {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return (double(state >> 11) + 0.5) * 0x1p-53;
    };
    const std::size_t inputs = 16, outputs = 16, batch = 256;
    std::vector<double> x(inputs * batch);
    for (auto& v : x) v = 2 * next_u() - 1;
    const auto ratio = [&](const PqcConfig& c) {
        double mean = 0;
        for (std::uint64_t seed = 0; seed < 16; ++seed) {
            kan::Layer layer(inputs, outputs, c);
            kan::initialize(layer, kan::VarianceScaling{0.8, kan::Distribution::Uniform, 31 + 100 * seed, {}});
            mean += mean_square(layer.forward(x, batch)) / (0.64 / 3) / 16;
        }
        return mean;
    };
    for (std::size_t n : {1u, 2u}) {
        const double value = ratio(config(n, 4));
        std::printf("asserted: n=%zu r=4 None PauliZ ratio %.3f\n", n, value);
        test::near(value, 1, 0.15);
    }
    for (const auto& c :
         {config(1, 1), config(1, 2), config(2, 2),
          config(2, 4, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::None, PqcReadout::Parity),
          config(2, 4, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CnotChain),
          config(3, 4, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CnotRing),
          config(3, 4, {Rotation::Z, Rotation::Y, Rotation::Z}, Entangler::CzChain, PqcReadout::Parity)})
        std::printf("recorded: n=%zu r=%zu entangler=%d readout=%d ratio %.3f\n", c.qubits, c.repetitions,
                    int(c.entangler), int(c.readout), ratio(c));
}

TEST(pqc_training_is_deterministic_and_generalizes) {
    // Toy regression t(x) = 0.6 sin(2x) + 0.3 cos(3x) - 0.1 on [-1, 1]: a PQC
    // layer (1 -> 4, one qubit, r = 3) and a linear Chebyshev readout, full-batch
    // SGD on 64 training points; an independent holdout of 50 random points.
    const auto target = [](double x) { return 0.6 * std::sin(2 * x) + 0.3 * std::cos(3 * x) - 0.1; };
    std::vector<double> train(64), holdout(50);
    for (std::size_t j = 0; j < train.size(); ++j) train[j] = -1 + 2 * (double(j) + 0.5) / double(train.size());
    for (std::size_t j = 0; j < holdout.size(); ++j) holdout[j] = symmetric_uniform(2718, j);
    const auto run = [&] {
        kan::Network network({kan::Layer(1, 4, config(1, 3)), kan::Layer(4, 1, kan::ChebyshevConfig{2})});
        kan::initialize(network, kan::VarianceScaling{1, kan::Distribution::Uniform, 3, {}});
        const auto mse = [&](const std::vector<double>& x) {
            const auto y = network.forward(x, x.size());
            double s = 0;
            for (std::size_t b = 0; b < x.size(); ++b) s += (y[b] - target(x[b])) * (y[b] - target(x[b]));
            return s / double(x.size());
        };
        const double start = mse(holdout);
        for (int step = 0; step < 3000; ++step) {
            const auto y = network.forward(train, train.size());
            std::vector<double> d(train.size());
            for (std::size_t b = 0; b < train.size(); ++b) d[b] = 2 * (y[b] - target(train[b])) / double(train.size());
            network.sgd(network.backward(train, train.size(), d), 0.05);
        }
        return std::pair{std::pair{start, mse(holdout)}, network};
    };
    const auto [first, network] = run();
    const auto [second, again] = run();
    double variance = 0, mean = 0;
    for (double x : holdout) mean += target(x) / double(holdout.size());
    for (double x : holdout) variance += (target(x) - mean) * (target(x) - mean) / double(holdout.size());
    std::printf("training: holdout MSE %.3e -> %.3e (target variance %.3e)\n", first.first, first.second, variance);
    REQUIRE(first == second);
    REQUIRE(edges_of(test::layer(network, 0)) == edges_of(test::layer(again, 0)));
    REQUIRE(first.second < 0.01 * variance); // fixed in the RED commit
    REQUIRE(first.second < 0.05 * first.first);
}

int main() { return test::run(); }

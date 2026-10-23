// M5 phase 1: the PQC circuit (kan/pqc.hpp). Validation, closed forms, an
// independent dense Kronecker-product reference, unit norm and Born
// probabilities, and the adjoint and parameter-shift gradients against each
// other and against central finite differences.
#include "kan/pqc.hpp"
#include "support/dense_qubit_reference.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>

namespace {
using kan::Entangler;
using kan::GradientMethod;
using kan::PqcConfig;
using kan::PqcReadout;
using kan::Rotation;
constexpr double pi = std::numbers::pi;
const double not_a_number = std::numeric_limits<double>::quiet_NaN();

// Test-local generator (SplitMix64), uniform on [lo, hi).
struct Random {
    std::uint64_t state;
    double operator()(double lo, double hi) {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return lo + (hi - lo) * (static_cast<double>(z >> 11) * 0x1p-53);
    }
    std::size_t below(std::size_t n) { return std::min(n - 1, static_cast<std::size_t>((*this)(0, double(n)))); }
};

PqcConfig config(std::size_t n, std::size_t r, Rotation encoding, std::vector<Rotation> block,
                 Entangler entangler = Entangler::None, PqcReadout readout = PqcReadout::PauliZ) {
    PqcConfig c;
    c.qubits = n;
    c.repetitions = r;
    c.encoding = encoding;
    c.block = std::move(block);
    c.entangler = entangler;
    c.readout = readout;
    return c;
}

dense_qubit::Circuit circuit(const PqcConfig& c) {
    dense_qubit::Circuit d;
    d.qubits = c.qubits;
    d.repetitions = c.repetitions;
    d.encoding = static_cast<int>(c.encoding);
    for (auto a : c.block) d.block.push_back(static_cast<int>(a));
    d.entangler = static_cast<int>(c.entangler);
    d.readout = static_cast<int>(c.readout);
    return d;
}

struct Parameters {
    std::vector<double> angles, weights, biases;
};
std::size_t angle_count(const PqcConfig& c) { return (c.repetitions + 1) * c.qubits * c.block.size(); }
std::size_t encoding_count(const PqcConfig& c) { return c.repetitions * c.qubits; }
Parameters random_parameters(const PqcConfig& c, Random& random) {
    Parameters p{std::vector<double>(angle_count(c)), std::vector<double>(encoding_count(c)),
                 std::vector<double>(encoding_count(c))};
    for (auto& v : p.angles) v = random(-pi, pi);
    for (auto& v : p.weights) v = random(-1.5, 1.5);
    for (auto& v : p.biases) v = random(-1, 1);
    return p;
}

// A random valid configuration (n <= max_qubits, r <= max_repetitions).
PqcConfig random_config(Random& random, std::size_t max_qubits, std::size_t max_repetitions) {
    PqcConfig c;
    c.qubits = 1 + random.below(max_qubits);
    c.repetitions = 1 + random.below(max_repetitions);
    c.encoding = static_cast<Rotation>(random.below(3));
    c.block.clear();
    const std::size_t length = 1 + random.below(3);
    while (c.block.size() < length) {
        const auto a = static_cast<Rotation>(random.below(3));
        if (c.block.empty() || c.block.back() != a) c.block.push_back(a);
    }
    if (c.qubits == 1)
        c.entangler = Entangler::None;
    else {
        c.entangler = static_cast<Entangler>(random.below(4));
        if (c.entangler == Entangler::CnotRing && c.qubits < 3) c.entangler = Entangler::CnotChain;
    }
    c.readout = static_cast<PqcReadout>(random.below(2));
    return c;
}

kan::PqcEvaluation evaluate(const PqcConfig& c, double x, const Parameters& p,
                            GradientMethod method = GradientMethod::Adjoint) {
    return kan::evaluate_pqc(c, x, p.angles, p.weights, p.biases, method);
}

void absolute(double actual, double expected, double tolerance, const char* what) {
    if (!(std::abs(actual - expected) <= tolerance))
        throw std::runtime_error(std::string(what) + ": actual=" + std::to_string(actual) + " expected=" +
                                 std::to_string(expected) + " difference=" + std::to_string(actual - expected));
}
// |actual - fd| <= relative |fd| + floor (the plan's finite-difference rule).
void against_fd(double actual, double fd, const char* what) {
    if (!(std::abs(actual - fd) <= 1e-7 * std::abs(fd) + 1e-9))
        throw std::runtime_error(std::string(what) + ": analytic=" + std::to_string(actual) +
                                 " fd=" + std::to_string(fd) + " difference=" + std::to_string(actual - fd));
}

// G: tape length, every rotation, encoding rotation and two-qubit gate.
std::size_t gate_count(const PqcConfig& c) {
    const std::size_t n = c.qubits, r = c.repetitions;
    const std::size_t entangling = c.entangler == Entangler::None ? 0 : c.entangler == Entangler::CnotRing ? n : n - 1;
    return (r + 1) * n * c.block.size() + r * n + r * entangling;
}

// Adjoint vs parameter shift, first-order error bounds (u = 2^-52, |M| <= 1,
// every derivative of one slot |d| <= 1 because |m| <= 1):
// - parameter shift at slot value p: (4G + 2^n + |p| + 2) u (the derivation in
//   pqc_layer_test, shift_zero_bound; away from a zero the shift identity
//   2 sin(q) m'(p) differs from 2 m'(p) by O(u^2));
// - adjoint: the fused forward state carries |e| <= 2G u (the normalization
//   budget), the reverse pass recovers psi and lambda through at most G more
//   inverse gates each, so both are within 4G u, and d = Im <lambda|sigma|psi>
//   summed over 2^(n-1) pairs is within (8G + 2^n) u.
// Angle and bias derivatives: their sum, (12G + 2^(n+1) + |p| + 2) u.
// Weights (x d, one rounding each side): |x| (bound + u). The input,
// sum_j w_j d_j over E slots: sum |w_j| (max bound + 2 E u).
struct Agreement {
    double difference = 0, ratio = 0; // largest |adjoint - shift| and its largest ratio to the bound
};
void adjoint_agrees_with_shift(const PqcConfig& c, double x, const Parameters& p, const kan::PqcEvaluation& adjoint,
                               const kan::PqcEvaluation& shift, Agreement& agreement) {
    const std::size_t terms = adjoint.values.size(), A = p.angles.size(), E = p.weights.size();
    const double u = 0x1p-52;
    const double base = 12 * double(gate_count(c)) + 2 * double(std::size_t{1} << c.qubits) + 2;
    const auto check = [&](double a, double s, double bound) {
        agreement.difference = std::max(agreement.difference, std::abs(a - s));
        agreement.ratio = std::max(agreement.ratio, std::abs(a - s) / bound);
        absolute(a, s, bound, "adjoint vs parameter shift");
    };
    REQUIRE(shift.angle_derivatives.size() == terms * A && shift.encoding_bias_derivatives.size() == terms * E);
    REQUIRE(shift.encoding_weight_derivatives.size() == terms * E && shift.input_derivatives.size() == terms);
    double largest_phase = 0, weights = 0;
    for (std::size_t j = 0; j < E; ++j) {
        largest_phase = std::max(largest_phase, std::abs(p.weights[j] * x + p.biases[j]));
        weights += std::abs(p.weights[j]);
    }
    for (std::size_t t = 0; t < terms; ++t) {
        for (std::size_t j = 0; j < A; ++j)
            check(adjoint.angle_derivatives[t * A + j], shift.angle_derivatives[t * A + j],
                  (base + std::abs(p.angles[j])) * u);
        for (std::size_t j = 0; j < E; ++j) {
            const double bound = (base + std::abs(p.weights[j] * x + p.biases[j])) * u;
            check(adjoint.encoding_bias_derivatives[t * E + j], shift.encoding_bias_derivatives[t * E + j], bound);
            check(adjoint.encoding_weight_derivatives[t * E + j], shift.encoding_weight_derivatives[t * E + j],
                  std::abs(x) * (bound + u) + std::numeric_limits<double>::min());
        }
        check(adjoint.input_derivatives[t], shift.input_derivatives[t],
              weights * ((base + largest_phase) * u + 2 * double(E) * u) + std::numeric_limits<double>::min());
    }
}
} // namespace

TEST(pqc_configuration_validation) {
    const auto rejects = [](PqcConfig c) { test::throws<std::invalid_argument>([&] { kan::validate_pqc(c); }); };
    kan::validate_pqc(PqcConfig{});
    REQUIRE(PqcConfig{}.qubits == 1 && PqcConfig{}.repetitions == 2 && PqcConfig{}.encoding == Rotation::Y);
    REQUIRE((PqcConfig{}.block == std::vector<Rotation>{Rotation::Z, Rotation::Y, Rotation::Z}));
    REQUIRE(PqcConfig{}.entangler == Entangler::None && PqcConfig{}.readout == PqcReadout::PauliZ);
    REQUIRE(PqcConfig{}.shots == 0);
    auto c = PqcConfig{};
    for (std::size_t n : {0u, 6u}) {
        c = PqcConfig{};
        c.qubits = n;
        rejects(c);
    }
    for (std::size_t r : {0u, 33u}) {
        c = PqcConfig{};
        c.repetitions = r;
        rejects(c);
    }
    c = PqcConfig{};
    c.repetitions = 32;
    c.qubits = 5;
    c.entangler = Entangler::CnotRing;
    kan::validate_pqc(c); // the limits themselves are valid
    c = PqcConfig{};
    c.block = {};
    rejects(c);
    c.block = {Rotation::X, Rotation::Y, Rotation::Z, Rotation::X};
    rejects(c);
    c.block = {Rotation::Y, Rotation::Y};
    rejects(c);
    c.block = {Rotation::Z, Rotation::X, Rotation::X};
    rejects(c);
    c.block = {Rotation::X, Rotation::Z, Rotation::X}; // equal, but not adjacent
    kan::validate_pqc(c);
    for (auto e : {Entangler::CnotChain, Entangler::CnotRing, Entangler::CzChain}) {
        c = PqcConfig{};
        c.entangler = e;
        rejects(c); // one qubit
    }
    c = PqcConfig{};
    c.qubits = 3;
    kan::validate_pqc(c); // None on three qubits: independent DARUAN circuits
    c.qubits = 2;
    c.entangler = Entangler::CnotRing;
    rejects(c);
    c.entangler = Entangler::CnotChain;
    kan::validate_pqc(c);
    c.entangler = Entangler::CzChain;
    kan::validate_pqc(c);
    c = PqcConfig{};
    c.encoding = static_cast<Rotation>(7);
    rejects(c);
    c = PqcConfig{};
    c.block = {Rotation::Z, static_cast<Rotation>(3)};
    rejects(c);
    c = PqcConfig{};
    c.qubits = 2;
    c.entangler = static_cast<Entangler>(4);
    rejects(c);
    c = PqcConfig{};
    c.readout = static_cast<PqcReadout>(2);
    rejects(c);
    // Finite shots arrive in a later phase (D13).
    c = PqcConfig{};
    c.shots = 1;
    try {
        kan::validate_pqc(c);
        throw std::runtime_error("shots were accepted");
    } catch (const std::invalid_argument& e) {
        REQUIRE(std::string(e.what()) == "finite shots are not available");
    }
    REQUIRE(kan::pqc_terms(config(3, 1, Rotation::Y, {Rotation::Y})) == 3);
    REQUIRE(kan::pqc_terms(config(3, 1, Rotation::Y, {Rotation::Y}, Entangler::None, PqcReadout::Parity)) == 1);
    REQUIRE(kan::pqc_angles(config(3, 4, Rotation::Y, {Rotation::Z, Rotation::Y})) == 30);
    REQUIRE(kan::pqc_encodings(config(3, 4, Rotation::Y, {Rotation::Z, Rotation::Y})) == 12);
}

TEST(pqc_evaluation_validates_parameters) {
    const auto c = config(2, 2, Rotation::Y, {Rotation::Z, Rotation::Y});
    Random random{3};
    auto p = random_parameters(c, random);
    evaluate(c, 0.3, p);
    test::throws<std::invalid_argument>([&] { evaluate(c, not_a_number, p); });
    test::throws<std::invalid_argument>(
        [&] { evaluate(c, std::numeric_limits<double>::infinity(), p, GradientMethod::ParameterShift); });
    auto bad = p;
    bad.angles.pop_back();
    test::throws<std::invalid_argument>([&] { evaluate(c, 0.3, bad); });
    bad = p;
    bad.weights.push_back(0);
    test::throws<std::invalid_argument>([&] { evaluate(c, 0.3, bad); });
    bad = p;
    bad.biases[1] = not_a_number;
    test::throws<std::invalid_argument>([&] { evaluate(c, 0.3, bad); });
    test::throws<std::invalid_argument>([&] { kan::pqc_state(c, 0.3, bad.angles, bad.weights, bad.biases); });
    // w x + b overflows: nonfinite results raise std::overflow_error.
    auto huge = p;
    huge.weights[0] = 1e300;
    test::throws<std::overflow_error>([&] { evaluate(c, 1e300, huge); });
    test::throws<std::overflow_error>([&] { evaluate(c, 1e300, huge, GradientMethod::ParameterShift); });
    test::throws<std::overflow_error>([&] { kan::pqc_state(c, 1e300, huge.angles, huge.weights, huge.biases); });
    auto invalid = c;
    invalid.qubits = 0;
    test::throws<std::invalid_argument>([&] { evaluate(invalid, 0.3, p); });
    test::throws<std::invalid_argument>(
        [&] { kan::evaluate_pqc(c, 0.3, p.angles, p.weights, p.biases, static_cast<GradientMethod>(5)); });
}

TEST(pqc_matches_closed_forms) {
    Random random{11};
    // One qubit, Y encoding, block {Y}: every rotation is about y, so
    // <Z> = cos(sum of angles + sum of (w x + b)), here in long double. 1e-15
    // up to r = 2 (5 rotations); r = 5 (11 rotations) is held to G 2^-52.
    for (std::size_t r : {1u, 2u, 5u}) {
        const auto c = config(1, r, Rotation::Y, {Rotation::Y});
        const double tolerance = r <= 2 ? 1e-15 : double(2 * r + 1) * 0x1p-52;
        for (int trial = 0; trial < 20; ++trial) {
            const auto p = random_parameters(c, random);
            const double x = random(-1.5, 1.5);
            long double phase = 0;
            for (double a : p.angles) phase += a;
            for (std::size_t j = 0; j < p.weights.size(); ++j) phase += p.weights[j] * x + p.biases[j];
            absolute(evaluate(c, x, p).values.at(0), static_cast<double>(std::cos(phase)), tolerance,
                     "cos of the summed Y rotations");
        }
    }
    // One qubit, Y encoding, ZYZ blocks, r = 1: Bloch vector rotations.
    {
        const auto c = config(1, 1, Rotation::Y, {Rotation::Z, Rotation::Y, Rotation::Z});
        for (int trial = 0; trial < 50; ++trial) {
            const auto p = random_parameters(c, random);
            const double x = random(-2, 2), phi = p.weights[0] * x + p.biases[0];
            // R_z(a): (x, y) -> (x cos a - y sin a, x sin a + y cos a);
            // R_y(a): (x, z) -> (x cos a + z sin a, -x sin a + z cos a).
            double bx = 0, by = 0, bz = 1;
            const auto rz = [&](double a) {
                const double nx = bx * std::cos(a) - by * std::sin(a), ny = bx * std::sin(a) + by * std::cos(a);
                bx = nx;
                by = ny;
            };
            const auto ry = [&](double a) {
                const double nx = bx * std::cos(a) + bz * std::sin(a), nz = -bx * std::sin(a) + bz * std::cos(a);
                bx = nx;
                bz = nz;
            };
            rz(p.angles[0]);
            ry(p.angles[1]);
            rz(p.angles[2]);
            ry(phi);
            rz(p.angles[3]);
            ry(p.angles[4]);
            rz(p.angles[5]);
            absolute(evaluate(c, x, p).values.at(0), bz, 1e-15, "ZYZ Bloch vector");
        }
    }
    // Two qubits: R_y(a) (x) R_y(b), then CNOT(0 -> 1) (encodings and the last
    // block are the identity): <Z0> = cos a, <Z1> = cos a cos b, parity cos b.
    for (int trial = 0; trial < 20; ++trial) {
        const double a = random(-pi, pi), b = random(-pi, pi);
        const Parameters p{{a, b, 0, 0}, {0, 0}, {0, 0}};
        const auto z = evaluate(config(2, 1, Rotation::Y, {Rotation::Y}, Entangler::CnotChain), 0.7, p).values;
        absolute(z.at(0), std::cos(a), 1e-15, "<Z0>");
        absolute(z.at(1), std::cos(a) * std::cos(b), 1e-15, "<Z1>");
        const auto parity =
            evaluate(config(2, 1, Rotation::Y, {Rotation::Y}, Entangler::CnotChain, PqcReadout::Parity), 0.7, p);
        REQUIRE(parity.values.size() == 1);
        absolute(parity.values[0], std::cos(b), 1e-15, "parity");
    }
    // CZ chain on computational states: R_y(pi) prepares |1>, CZ multiplies the
    // amplitude by (-1)^(sum_q b_q b_(q+1)); populations are unchanged.
    for (std::size_t n : {2u, 3u, 4u})
        for (std::size_t k = 0; k < (std::size_t{1} << n); ++k) {
            const auto c = config(n, 1, Rotation::Y, {Rotation::Y}, Entangler::CzChain);
            Parameters p{std::vector<double>(2 * n, 0.0), std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
            for (std::size_t q = 0; q < n; ++q) p.angles[q] = (k >> q) & 1 ? pi : 0;
            const auto psi = kan::pqc_state(c, 0.1, p.angles, p.weights, p.biases);
            REQUIRE(psi.size() == (std::size_t{1} << n));
            int pairs = 0;
            for (std::size_t q = 0; q + 1 < n; ++q) pairs += int((k >> q) & (k >> (q + 1)) & 1);
            for (std::size_t j = 0; j < psi.size(); ++j) {
                const double expected = j == k ? (pairs % 2 ? -1.0 : 1.0) : 0.0;
                absolute(psi[j].real(), expected, 1e-15, "CZ amplitude (real)");
                absolute(psi[j].imag(), 0.0, 1e-15, "CZ amplitude (imaginary)");
            }
            const auto z = evaluate(c, 0.1, p).values;
            for (std::size_t q = 0; q < n; ++q)
                absolute(z[q], (k >> q) & 1 ? -1.0 : 1.0, 1e-15, "CZ leaves populations unchanged");
        }
}

TEST(pqc_matches_the_dense_reference) {
    Random random{2026};
    double largest_amplitude = 0, largest_expectation = 0;
    std::size_t coverage[4] = {0, 0, 0, 0};
    for (int trial = 0; trial < 200; ++trial) {
        const auto c = random_config(random, 4, 4);
        ++coverage[static_cast<int>(c.entangler)];
        const auto p = random_parameters(c, random);
        const double x = random(-2, 2);
        const auto reference = dense_qubit::state(circuit(c), x, p.angles, p.weights, p.biases);
        const auto psi = kan::pqc_state(c, x, p.angles, p.weights, p.biases);
        REQUIRE(psi.size() == reference.size());
        for (std::size_t k = 0; k < psi.size(); ++k)
            largest_amplitude = std::max(largest_amplitude, std::abs(psi[k] - reference[k]));
        const auto m = evaluate(c, x, p).values;
        const auto expected = dense_qubit::expectations(circuit(c), reference);
        REQUIRE(m.size() == expected.size());
        for (std::size_t t = 0; t < m.size(); ++t)
            largest_expectation = std::max(largest_expectation, std::abs(m[t] - expected[t]));
    }
    std::printf("dense reference: largest amplitude difference %.3e, expectation %.3e (entanglers %zu/%zu/%zu/%zu)\n",
                largest_amplitude, largest_expectation, coverage[0], coverage[1], coverage[2], coverage[3]);
    REQUIRE(largest_amplitude <= 1e-13);
    REQUIRE(largest_expectation <= 1e-13);
    for (auto count : coverage) REQUIRE(count > 0);
}

TEST(pqc_states_are_unit_norm_with_born_probabilities) {
    Random random{77};
    double worst = 0; // largest |sum p - 1| / bound
    for (int trial = 0; trial < 120; ++trial) {
        auto c = random_config(random, 5, 32);
        if (trial < 8) c.repetitions = 32; // the limit
        const auto p = random_parameters(c, random);
        const auto psi = kan::pqc_state(c, random(-3, 3), p.angles, p.weights, p.biases);
        const double bound = double(4 * gate_count(c) + psi.size()) * 0x1p-52;
        double total = 0;
        for (const auto& a : psi) {
            const double probability = std::norm(a);
            REQUIRE(probability >= 0 && probability <= 1 + bound);
            total += probability;
        }
        worst = std::max(worst, std::abs(total - 1) / bound);
        REQUIRE(std::abs(total - 1) <= bound);
    }
    std::printf("normalization: largest |sum p - 1| is %.3f of the bound (4G + 2^n) 2^-52\n", worst);
}

TEST(pqc_gradients_agree_with_each_other_and_finite_differences) {
    Random random{404};
    const double h = 1e-6;
    double largest_derivative = 0;
    Agreement agreement;
    for (int trial = 0; trial < 40; ++trial) {
        const auto c = random_config(random, 3, 3);
        const auto p = random_parameters(c, random);
        const double x = random(-1.5, 1.5);
        const auto adjoint = evaluate(c, x, p);
        const auto shift = evaluate(c, x, p, GradientMethod::ParameterShift);
        const std::size_t terms = adjoint.values.size(), A = p.angles.size(), E = p.weights.size();
        REQUIRE(terms == kan::pqc_terms(c));
        REQUIRE(adjoint.input_derivatives.size() == terms && adjoint.angle_derivatives.size() == terms * A);
        REQUIRE(adjoint.encoding_weight_derivatives.size() == terms * E);
        REQUIRE(adjoint.encoding_bias_derivatives.size() == terms * E);
        REQUIRE(shift.values == adjoint.values);
        for (double d : adjoint.angle_derivatives) largest_derivative = std::max(largest_derivative, std::abs(d));
        adjoint_agrees_with_shift(c, x, p, adjoint, shift, agreement);
        // Central differences of the values, every parameter and the input.
        const auto fd = [&](auto&& perturbed) {
            const auto plus = perturbed(h), minus = perturbed(-h);
            std::vector<double> d(terms);
            for (std::size_t t = 0; t < terms; ++t) d[t] = (plus[t] - minus[t]) / (2 * h);
            return d;
        };
        const auto check = [&](const std::vector<double>& d, std::size_t stride, std::size_t j,
                               const std::vector<double>& numeric, const char* what) {
            for (std::size_t t = 0; t < terms; ++t) against_fd(d[t * stride + j], numeric[t], what);
        };
        const auto dx = fd([&](double e) { return evaluate(c, x + e, p).values; });
        for (std::size_t t = 0; t < terms; ++t) {
            against_fd(adjoint.input_derivatives[t], dx[t], "adjoint d/dx");
            against_fd(shift.input_derivatives[t], dx[t], "parameter-shift d/dx");
        }
        for (std::size_t j = 0; j < A; ++j) {
            const auto d = fd([&](double e) {
                auto q = p;
                q.angles[j] += e;
                return evaluate(c, x, q).values;
            });
            check(adjoint.angle_derivatives, A, j, d, "adjoint d/dangle");
            for (std::size_t t = 0; t < terms; ++t)
                against_fd(shift.angle_derivatives[t * A + j], d[t], "parameter-shift d/dangle");
        }
        for (std::size_t j = 0; j < E; ++j) {
            const auto dw = fd([&](double e) {
                auto q = p;
                q.weights[j] += e;
                return evaluate(c, x, q).values;
            });
            const auto db = fd([&](double e) {
                auto q = p;
                q.biases[j] += e;
                return evaluate(c, x, q).values;
            });
            check(adjoint.encoding_weight_derivatives, E, j, dw, "adjoint d/dw");
            check(adjoint.encoding_bias_derivatives, E, j, db, "adjoint d/db");
            for (std::size_t t = 0; t < terms; ++t) {
                against_fd(shift.encoding_weight_derivatives[t * E + j], dw[t], "parameter-shift d/dw");
                against_fd(shift.encoding_bias_derivatives[t * E + j], db[t], "parameter-shift d/db");
            }
        }
        // The input enters every encoding gate: d/dx = sum_j w_j d/db_j, d/dw_j = x d/db_j.
        for (std::size_t t = 0; t < terms; ++t) {
            double sum = 0;
            for (std::size_t j = 0; j < E; ++j) {
                sum += p.weights[j] * adjoint.encoding_bias_derivatives[t * E + j];
                absolute(adjoint.encoding_weight_derivatives[t * E + j],
                         x * adjoint.encoding_bias_derivatives[t * E + j], 1e-15, "d/dw = x d/db");
            }
            absolute(adjoint.input_derivatives[t], sum, 1e-13, "d/dx = sum w d/db");
        }
    }
    std::printf("adjoint vs parameter shift: largest difference %.3e, %.3e of the bound (largest derivative %.3f)\n",
                agreement.difference, agreement.ratio, largest_derivative);
    REQUIRE(largest_derivative > 0.5); // the comparisons are not between zeros
}

TEST(pqc_adjoint_and_parameter_shift_agree_on_deep_circuits) {
    // Random circuits up to the D9 limits (n <= 5, r <= 32; the first four at n = 5,
    // r = 32 with each entangler), every axis, entangler and readout. No finite differences here:
    // an expectation rounding of ~1e-15 over 2h = 2e-6 is ~5e-10, at the 1e-9
    // floor of the FD rule, so FD stays at the shallow depths above.
    Random random{2027};
    double largest_derivative = 0;
    std::size_t largest_gates = 0;
    Agreement agreement;
    for (int trial = 0; trial < 16; ++trial) {
        auto c = random_config(random, 5, 32);
        if (trial < 4) { // the D9 limit, each entangler once
            c.qubits = 5;
            c.repetitions = 32;
            c.entangler = static_cast<Entangler>(trial);
        }
        largest_gates = std::max(largest_gates, gate_count(c));
        const auto p = random_parameters(c, random);
        const double x = random(-1.5, 1.5);
        const auto adjoint = evaluate(c, x, p);
        const auto shift = evaluate(c, x, p, GradientMethod::ParameterShift);
        REQUIRE(shift.values == adjoint.values);
        for (double d : adjoint.angle_derivatives) largest_derivative = std::max(largest_derivative, std::abs(d));
        adjoint_agrees_with_shift(c, x, p, adjoint, shift, agreement);
    }
    std::printf("deep adjoint vs parameter shift (G up to %zu): largest difference %.3e, %.3e of the bound (largest "
                "derivative %.3f)\n",
                largest_gates, agreement.difference, agreement.ratio, largest_derivative);
    REQUIRE(largest_derivative > 0.5);
}

TEST(pqc_gradients_match_finite_differences_of_the_reference) {
    // Finite differences of the independent dense reference (not of the library).
    Random random{9};
    const double h = 1e-6;
    for (int trial = 0; trial < 12; ++trial) {
        const auto c = random_config(random, 3, 2);
        const auto p = random_parameters(c, random);
        const double x = random(-1, 1);
        const auto adjoint = evaluate(c, x, p);
        const auto reference = [&](const Parameters& q, double at) {
            return dense_qubit::expectations(circuit(c),
                                             dense_qubit::state(circuit(c), at, q.angles, q.weights, q.biases));
        };
        const std::size_t terms = adjoint.values.size(), A = p.angles.size();
        for (std::size_t j = 0; j < A; ++j) {
            auto plus = p, minus = p;
            plus.angles[j] += h;
            minus.angles[j] -= h;
            const auto a = reference(plus, x), b = reference(minus, x);
            for (std::size_t t = 0; t < terms; ++t)
                against_fd(adjoint.angle_derivatives[t * A + j], (a[t] - b[t]) / (2 * h), "reference d/dangle");
        }
        const auto a = reference(p, x + h), b = reference(p, x - h);
        for (std::size_t t = 0; t < terms; ++t)
            against_fd(adjoint.input_derivatives[t], (a[t] - b[t]) / (2 * h), "reference d/dx");
    }
}

TEST(pqc_structurally_inert_angles_have_exactly_zero_gradients) {
    // Default ZYZ block: the first Z rotations act on |0...0> (a phase) and the
    // last Z rotations commute with the Z/parity readouts.
    Random random{5};
    for (auto entangler : {Entangler::None, Entangler::CnotChain, Entangler::CzChain})
        for (auto readout : {PqcReadout::PauliZ, PqcReadout::Parity}) {
            auto c = config(2, 3, Rotation::Y, {Rotation::Z, Rotation::Y, Rotation::Z}, entangler, readout);
            const auto p = random_parameters(c, random);
            const double x = random(-1, 1), h = 1e-6;
            const std::size_t A = p.angles.size(), n = 2, B = 3, r = 3;
            for (auto method : {GradientMethod::Adjoint, GradientMethod::ParameterShift}) {
                const auto e = evaluate(c, x, p, method);
                double other = 0; // the remaining angles are not inert
                for (double d : e.angle_derivatives) other = std::max(other, std::abs(d));
                REQUIRE(other > 1e-3);
                for (std::size_t t = 0; t < e.values.size(); ++t)
                    for (std::size_t q = 0; q < n; ++q) {
                        REQUIRE(e.angle_derivatives[t * A + (0 * n + q) * B + 0] == 0.0);
                        REQUIRE(e.angle_derivatives[t * A + (r * n + q) * B + 2] == 0.0);
                    }
            }
            for (std::size_t q = 0; q < n; ++q)
                for (std::size_t j : {(0 * n + q) * B + 0, (r * n + q) * B + 2}) {
                    auto plus = p, minus = p;
                    plus.angles[j] += h;
                    minus.angles[j] -= h;
                    const auto a = evaluate(c, x, plus).values, b = evaluate(c, x, minus).values;
                    for (std::size_t t = 0; t < a.size(); ++t) against_fd(0.0, (a[t] - b[t]) / (2 * h), "inert FD");
                }
        }
}

int main() { return test::run(); }

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kan {

// Parametrized quantum circuit edges (M5 phase 1; QKAN/DARUAN, Jiang et al.
// 2025, arXiv 2509.14026, generalized to a few qubits). Every edge (o,i) is a
// data re-uploading circuit on n qubits, little-endian (bit q of a basis index
// is qubit q), started in |0...0>:
//     |psi(x)> = W_r S_r(x) E W_(r-1) ... S_1(x) E W_0 |0...0>,
//     S_l(x) = (x)_q R_enc(w[l,q] x + b[l,q]),   l = 1..r,
//     W_l = (x)_q R_(a_(B-1))(theta[l,q,B-1]) ... R_(a_0)(theta[l,q,0]) (axis 0 first),
//     R_a(p) = exp(-i p sigma_a / 2),
// E the entangler (identity for None), and its value
//     phi_{o,i}(x) = sum_t c[o,i,t] m_t(x),
//     m_q = <psi|Z_q|psi> (PauliZ, terms n) or m = <psi|Z...Z|psi> (Parity, terms 1).
// The readout coefficients c are the carrier's per-edge coefficient tensor
// (L2-penalized); angles, encoding weights and biases are nonlinear parameters.
// Expectations are exact (from amplitudes, never renormalized); finite shots
// are a later phase and are rejected.
enum class Rotation : std::uint8_t { X, Y, Z };
// CnotChain: CNOT(q -> q+1) for q ascending; CnotRing adds CNOT(n-1 -> 0)
// (n >= 3); CzChain: CZ(q, q+1) for q ascending; None: identity.
enum class Entangler : std::uint8_t { None, CnotChain, CnotRing, CzChain };
enum class PqcReadout : std::uint8_t { PauliZ, Parity };
// Adjoint differentiation (default) or the parameter-shift rule.
enum class GradientMethod : std::uint8_t { Adjoint, ParameterShift };

struct PqcConfig {
    std::size_t qubits = 1;                                             // n, 1..5
    std::size_t repetitions = 2;                                        // r, 1..32 (DARUAN's r)
    Rotation encoding = Rotation::Y;                                    // S(phi) = R_enc(phi)
    std::vector<Rotation> block{Rotation::Z, Rotation::Y, Rotation::Z}; // W, 1..3 axes
    Entangler entangler = Entangler::None;                              // must be None for n = 1
    PqcReadout readout = PqcReadout::PauliZ;
    std::size_t shots = 0; // 0: exact expectations (finite shots: not available yet)
    bool operator==(const PqcConfig&) const = default;
};

// Measured terms per edge: n (PauliZ) or 1 (Parity).
std::size_t pqc_terms(const PqcConfig& config) noexcept;
// Per-edge parameter counts: angles (r+1) n B, index (l n + q) B + k; encoding
// weights and biases r n each, index (l-1) n + q.
std::size_t pqc_angles(const PqcConfig& config) noexcept;
std::size_t pqc_encodings(const PqcConfig& config) noexcept;

// std::invalid_argument for qubits outside 1..5, repetitions outside 1..32, a
// block of 0 or more than 3 axes, adjacent equal block axes, an enumerator out
// of range, an entangler other than None on one qubit, CnotRing below three
// qubits, and shots > 0 ("finite shots are not available").
void validate_pqc(const PqcConfig& config);

// One edge circuit at x, unit coefficients: the measured terms m_t and their
// derivatives, every derivative array row-major per term t:
// input_derivatives[t], angle_derivatives[t * angles + j],
// encoding_weight_derivatives[t * encodings + j] (likewise biases).
struct PqcEvaluation {
    std::vector<double> values;
    std::vector<double> input_derivatives;
    std::vector<double> angle_derivatives;
    std::vector<double> encoding_weight_derivatives;
    std::vector<double> encoding_bias_derivatives;
};
// Validates the configuration, the parameter counts and finiteness
// (std::invalid_argument).
PqcEvaluation evaluate_pqc(const PqcConfig& config, double x, std::span<const double> angles,
                           std::span<const double> encoding_weights, std::span<const double> encoding_biases,
                           GradientMethod method = GradientMethod::Adjoint);
// The final state |psi(x)> of one edge circuit (2^n amplitudes, little-endian),
// for physical checks (unit norm, Born probabilities); same validation.
std::vector<std::complex<double>> pqc_state(const PqcConfig& config, double x, std::span<const double> angles,
                                            std::span<const double> encoding_weights,
                                            std::span<const double> encoding_biases);

} // namespace kan

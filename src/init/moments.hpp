#pragma once

#include "kan/initializers.hpp"
#include <span>
#include <vector>

namespace kan::init {

// mu_k = E[u^(2k) / Q(u)^2] for u uniform on [-1, 1], k = 0..numerator_degree,
// with S(u) = sum_k beta[k-1] u^k and Q the policy's denominator. For
// z = radius * u and b_k = beta_k / radius^k, E[z^(2k)/Q^2] = radius^(2k) mu_k.
std::vector<double> rational_moments(DenominatorPolicy policy, std::size_t numerator_degree,
                                     std::span<const double> beta);

// Per-edge wavelets (backlog M7, WaveletEdges): x uniform on
// [min(tau - s), max(tau + s)], s = exp(l) (portable_exp), the span of every
// wavelet's central lobe, and m_j = E[psi_j(x)^2] per wavelet (the
// L2-normalized Mexican hat with center tau_j and scale s_j) in closed form
// from erf, by a positive series and portable_exp (bitwise portable).
// Nonempty spans of equal length.
BasisMoments wavelet_moments(std::span<const double> translations, std::span<const double> log_scales);

// PQC edges (M5 D8), x uniform on [-1, 1]: the second moment m = E[m_t^2]
// assumed for every measured term. Entangler::None: 1/3 for PauliZ (the
// many-block limit of E[<Z>^2] for uniformly drawn ZYZ blocks), 3^-n for
// Parity (independent qubits); with an entangler 1/(2^n + 1), the Haar
// average of <P>^2 for one Pauli observable on n qubits (an assumption about
// the drawn circuits). Exact in double.
double pqc_second_moment(const PqcConfig& config);

} // namespace kan::init

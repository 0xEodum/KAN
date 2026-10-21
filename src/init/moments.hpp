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

} // namespace kan::init

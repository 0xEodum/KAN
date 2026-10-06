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

} // namespace kan::init

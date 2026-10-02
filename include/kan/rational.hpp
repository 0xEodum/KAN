#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace kan {

// Singularity policy of the rational denominator, with S(z) = sum_{k=1..n} b_k z^k:
//   Guarded  Q = 1 + S     relative pole guard; unsafe samples raise domain_error
//   Absolute Q = 1 + |S|   safe PAU (Molina et al. 2019); dQ/dS = sign(S), sign(0) = 0
//   Smooth   Q = 1 + S^2   smooth and pole free; dQ/dS = 2S
// Absolute and Smooth have Q >= 1 for every input: no pole, no guard.
enum class DenominatorPolicy { Guarded, Absolute, Smooth };

struct RationalConfig {
    std::size_t numerator_degree = 3;
    std::size_t denominator_degree = 2;
    double center = 0;
    double scale = 1;
    double epsilon = 1e-8; // relative pole guard; used by DenominatorPolicy::Guarded only
    DenominatorPolicy denominator_policy = DenominatorPolicy::Guarded;
    bool operator==(const RationalConfig&) const = default;
};

struct RationalEvaluation {
    double value = 0;
    double input_derivative = 0;
    std::vector<double> numerator_derivatives;
    std::vector<double> denominator_derivatives;
};

void validate_rational(const RationalConfig& config);
RationalEvaluation evaluate_rational(const RationalConfig& config, double x,
                                    std::span<const double> numerator,
                                    std::span<const double> denominator);

} // namespace kan

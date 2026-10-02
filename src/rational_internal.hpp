#pragma once

#include "kan/rational.hpp"
#include <array>

namespace kan::detail {

// Private result for callers that already validated the configuration, finite
// input/coefficients and exact edge parameter spans. Unused array slots are
// zeroed. Numerical guards and every derivative check still execute.
struct RationalTerms {
    double value;
    double input_derivative;
    std::array<double,17> numerator_derivatives;
    std::array<double,16> denominator_derivatives;
};
// Policy must equal config.denominator_policy; callers dispatch once per
// operation (visit_denominator_policy) and loop over edges and samples.
// Explicitly instantiated for every DenominatorPolicy in src/rational.cpp.
template<DenominatorPolicy Policy>
RationalTerms evaluate_rational_trusted(const RationalConfig& config,double x,
                                      std::span<const double> numerator,
                                      std::span<const double> denominator);

} // namespace kan::detail

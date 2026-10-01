#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace kan {

struct RationalConfig {
    std::size_t numerator_degree = 3;
    std::size_t denominator_degree = 2;
    double center = 0;
    double scale = 1;
    double epsilon = 1e-8;
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

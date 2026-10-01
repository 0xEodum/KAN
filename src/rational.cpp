#include "kan/rational.hpp"
#include <stdexcept>
namespace kan {
void validate_rational(const RationalConfig&) { throw std::logic_error("M4 RED: rational validation pending"); }
RationalEvaluation evaluate_rational(const RationalConfig&, double, std::span<const double>, std::span<const double>) {
    throw std::logic_error("M4 RED: rational evaluation pending");
}
}

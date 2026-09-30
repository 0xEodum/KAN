#include "kan/basis.hpp"
#include <stdexcept>

namespace kan {
void validate_basis(const BasisConfig&) { throw std::logic_error("basis not implemented"); }
BasisValues evaluate_basis(const BasisConfig&, double) {
    throw std::logic_error("basis not implemented");
}
}

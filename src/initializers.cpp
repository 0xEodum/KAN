// Backlog M4 RED stub: declarations only, no behaviour.
#include "kan/initializers.hpp"
#include <stdexcept>

namespace kan {

BasisMoments reference_moments(const BasisConfig&) { throw std::logic_error("initializers not implemented"); }
void initialize(Layer&, const Initializer&) { throw std::logic_error("initializers not implemented"); }
void initialize(Network&, const Initializer&) { throw std::logic_error("initializers not implemented"); }
std::uint64_t layer_seed(std::uint64_t, std::size_t) noexcept { return 0; }

} // namespace kan

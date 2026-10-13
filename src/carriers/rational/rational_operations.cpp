// Operations specific to rational edges. Like the local-family operations
// (carriers/local), they commit through Layer::set_carrier, which revalidates
// the whole layer; a failure leaves the layer unchanged.
#include "kan/families.hpp"
#include "core/checks.hpp"
#include <stdexcept>
#include <variant>

namespace kan {
namespace {
using detail::require_finite;
} // namespace

void set_rational_parameters(Layer& layer, std::span<const double> numerator,
                             std::span<const double> denominators, std::span<const double> bias) {
    const auto* edges = std::get_if<RationalEdges>(&layer.carrier());
    if (!edges || numerator.size() != edges->coefficients.size() ||
        denominators.size() != edges->denominators.size() || bias.size() != layer.bias().size())
        throw std::invalid_argument("rational parameter shape or layer type mismatch");
    require_finite(numerator);
    require_finite(denominators);
    require_finite(bias);
    layer.set_carrier(RationalEdges{edges->config, {numerator.begin(), numerator.end()},
                                    {denominators.begin(), denominators.end()}},
                      bias);
}

} // namespace kan

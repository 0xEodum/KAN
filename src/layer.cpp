#include "kan/layer.hpp"
#include <stdexcept>

namespace kan {
Layer::Layer(std::size_t inputs, std::size_t outputs, BasisConfig basis)
    : inputs_(inputs), outputs_(outputs), basis_(std::move(basis)) {
    throw std::logic_error("Layer not implemented");
}
void Layer::set_parameters(std::span<const double>, std::span<const double>) { throw std::logic_error("not implemented"); }
std::vector<double> Layer::forward(std::span<const double>, std::size_t) const { throw std::logic_error("not implemented"); }
LayerGradients Layer::backward(std::span<const double>, std::size_t, std::span<const double>) const { throw std::logic_error("not implemented"); }
void Layer::sgd(const LayerGradients&, double) { throw std::logic_error("not implemented"); }
} // namespace kan

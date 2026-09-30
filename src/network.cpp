#include "kan/network.hpp"
#include <stdexcept>
namespace kan {
Network::Network(std::vector<Layer>) { throw std::logic_error("Network not implemented"); }
std::vector<double> Network::forward(std::span<const double>, std::size_t) const { throw std::logic_error("not implemented"); }
NetworkGradients Network::backward(std::span<const double>, std::size_t, std::span<const double>) const { throw std::logic_error("not implemented"); }
void Network::sgd(const NetworkGradients&, double) { throw std::logic_error("not implemented"); }
} // namespace kan

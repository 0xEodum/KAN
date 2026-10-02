#pragma once
// Test-only accessors for heterogeneous networks: positions in
// Network::layers() hold a kan::Layer or a kan::InputMap, and positions in
// NetworkGradients::layers hold the matching gradient alternative.
#include "kan/network.hpp"
#include <cstddef>
#include <variant>
#include <vector>

namespace test {

inline const kan::Layer& layer(const kan::Network& network, std::size_t index) {
    return std::get<kan::Layer>(network.layers()[index]);
}
inline const kan::InputMap& input_map(const kan::Network& network, std::size_t index) {
    return std::get<kan::InputMap>(network.layers()[index]);
}
// Copies of the KAN layers of a network without input maps.
inline std::vector<kan::Layer> layers(const kan::Network& network) {
    std::vector<kan::Layer> result;
    for (const auto& stage : network.layers()) result.push_back(std::get<kan::Layer>(stage));
    return result;
}
inline const kan::LayerGradients& grad(const kan::NetworkGradients& gradients, std::size_t index) {
    return std::get<kan::LayerGradients>(gradients.layers[index]);
}
inline kan::LayerGradients& grad(kan::NetworkGradients& gradients, std::size_t index) {
    return std::get<kan::LayerGradients>(gradients.layers[index]);
}
inline const kan::InputMapGradients& map_grad(const kan::NetworkGradients& gradients, std::size_t index) {
    return std::get<kan::InputMapGradients>(gradients.layers[index]);
}
inline kan::InputMapGradients& map_grad(kan::NetworkGradients& gradients, std::size_t index) {
    return std::get<kan::InputMapGradients>(gradients.layers[index]);
}

} // namespace test

#include "kan/network.hpp"
#include <stdexcept>
namespace kan {
Network::Network(std::vector<Layer> layers) : layers_(std::move(layers)) {
    if (layers_.empty()) throw std::invalid_argument("network must contain at least one layer");
    for (std::size_t i = 1; i < layers_.size(); ++i)
        if (layers_[i - 1].outputs() != layers_[i].inputs())
            throw std::invalid_argument("incompatible adjacent layer dimensions");
}
std::vector<double> Network::forward(std::span<const double> input, std::size_t batch) const {
    auto output = layers_.front().forward(input, batch);
    for (std::size_t i = 1; i < layers_.size(); ++i) output = layers_[i].forward(output, batch);
    return output;
}
NetworkGradients Network::backward(std::span<const double> input, std::size_t batch,
                                   std::span<const double> output_gradient) const {
    std::vector<std::vector<double>> activations;
    activations.reserve(layers_.size());
    auto current = input;
    for (const auto& layer : layers_) {
        activations.push_back(layer.forward(current, batch));
        current = activations.back();
    }
    NetworkGradients gradient;
    gradient.layers.resize(layers_.size());
    auto upstream = output_gradient;
    for (std::size_t i = layers_.size(); i-- > 0;) {
        const auto layer_input = i == 0 ? input : std::span<const double>(activations[i - 1]);
        gradient.layers[i] = layers_[i].backward(layer_input, batch, upstream);
        upstream = gradient.layers[i].input;
    }
    gradient.input = gradient.layers.front().input;
    return gradient;
}
void Network::sgd(const NetworkGradients& gradients, double learning_rate) {
    if (gradients.layers.size() != layers_.size()) throw std::invalid_argument("network gradient shape mismatch");
    auto next = layers_;
    for (std::size_t i = 0; i < next.size(); ++i) next[i].sgd(gradients.layers[i], learning_rate);
    layers_.swap(next);
}
} // namespace kan

#include "kan/network.hpp"
#include <stdexcept>
#include <cmath>
namespace kan {
void Network::insert_knot(std::size_t index, double x) {
    validate_state();if(index>=layers_.size())throw std::invalid_argument("layer index out of range");
    layers_[index].insert_knot(x);
}
double Network::adapt_grid(std::size_t index, std::span<const double> samples) {
    validate_state();if(index>=layers_.size())throw std::invalid_argument("layer index out of range");
    return layers_[index].adapt_grid(samples);
}
NetworkRegularizationResult Network::regularization(double lambda) const {
    validate_state();NetworkRegularizationResult r;
    for(const auto& l:layers_) {
        auto q=l.regularization(lambda);r.value+=q.value;r.gradients.layers.push_back(std::move(q.gradients));
        if(!std::isfinite(r.value))throw std::overflow_error("nonfinite network penalty");
    }
    return r;
}
Network::Network(std::vector<Layer> layers) : layers_(std::move(layers)) {
    validate_state();
    for (const auto& layer : layers_) layer.validate_state();
    for (std::size_t i = 1; i < layers_.size(); ++i)
        if (layers_[i - 1].outputs() != layers_[i].inputs())
            throw std::invalid_argument("incompatible adjacent layer dimensions");
}
void Network::validate_state() const {
    if (layers_.empty()) throw std::invalid_argument("network is empty or moved from");
}
std::vector<double> Network::forward(std::span<const double> input, std::size_t batch) const {
    validate_state();
    auto output = layers_.front().forward(input, batch);
    for (std::size_t i = 1; i < layers_.size(); ++i) output = layers_[i].forward(output, batch);
    return output;
}
NetworkGradients Network::backward(std::span<const double> input, std::size_t batch,
                                   std::span<const double> output_gradient) const {
    validate_state();
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
    validate_state();
    if (gradients.layers.size() != layers_.size()) throw std::invalid_argument("network gradient shape mismatch");
    auto next = layers_;
    for (std::size_t i = 0; i < next.size(); ++i) next[i].sgd(gradients.layers[i], learning_rate);
    layers_.swap(next);
}
} // namespace kan

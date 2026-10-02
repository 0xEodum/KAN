// Network: a sequence of heterogeneous layer kinds (KAN layers and input
// maps). Each operation dispatches on a layer's kind once per call.
#include "kan/network.hpp"
#include "kan/families.hpp"
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace kan {
namespace {
std::size_t inputs_of(const NetworkLayer& layer) {
    return std::visit([](const auto& l) { return l.inputs(); }, layer);
}
std::size_t outputs_of(const NetworkLayer& layer) {
    return std::visit([](const auto& l) { return l.outputs(); }, layer);
}
const std::vector<double>& input_gradient(const NetworkLayerGradients& gradients) {
    return std::visit([](const auto& g) -> const std::vector<double>& { return g.input; }, gradients);
}
// Regularization gradients of an input map: no input gradient, zero
// trainable-parameter gradients (maps are not penalized).
InputMapGradients zero_map_gradients(const InputMap& map) {
    const auto* norm = std::get_if<LayerNormMap>(&map.map());
    const auto count = norm ? norm->gain.size() : 0;
    return {{}, std::vector<double>(count, 0.0), std::vector<double>(count, 0.0)};
}
} // namespace

Network::Network(std::vector<NetworkLayer> layers) : layers_(std::move(layers)) {
    validate_state();
    for (const auto& layer : layers_) std::visit([](const auto& l) { l.validate_state(); }, layer);
    for (std::size_t i = 1; i < layers_.size(); ++i)
        if (outputs_of(layers_[i - 1]) != inputs_of(layers_[i]))
            throw std::invalid_argument("incompatible adjacent layer dimensions");
}

void Network::validate_state() const {
    if (layers_.empty()) throw std::invalid_argument("network is empty or moved from");
}

std::size_t Network::inputs() const {
    validate_state();
    return inputs_of(layers_.front());
}

std::size_t Network::outputs() const {
    validate_state();
    return outputs_of(layers_.back());
}

Layer& Network::kan_layer(std::size_t index) {
    validate_state();
    if (index >= layers_.size()) throw std::invalid_argument("layer index out of range");
    auto* layer = std::get_if<Layer>(&layers_[index]);
    if (!layer) throw std::invalid_argument("layer index does not refer to a KAN layer");
    return *layer;
}

void Network::insert_knot(std::size_t index, double x) { kan::insert_knot(kan_layer(index), x); }

double Network::adapt_grid(std::size_t index, std::span<const double> samples) {
    return kan::adapt_grid(kan_layer(index), samples);
}

NetworkRegularizationResult Network::regularization(double lambda) const {
    validate_state();
    if (!std::isfinite(lambda) || lambda < 0)
        throw std::invalid_argument("L2 coefficient must be finite and nonnegative");
    NetworkRegularizationResult r;
    for (const auto& layer : layers_) {
        if (const auto* l = std::get_if<Layer>(&layer)) {
            auto q = l->regularization(lambda);
            r.value += q.value;
            r.gradients.layers.emplace_back(std::move(q.gradients));
        } else {
            const auto& map = std::get<InputMap>(layer);
            map.validate_state();
            r.gradients.layers.emplace_back(zero_map_gradients(map));
        }
        if (!std::isfinite(r.value)) throw std::overflow_error("nonfinite network penalty");
    }
    return r;
}

std::vector<double> Network::forward(std::span<const double> input, std::size_t batch) const {
    validate_state();
    auto output = std::visit([&](const auto& l) { return l.forward(input, batch); }, layers_.front());
    for (std::size_t i = 1; i < layers_.size(); ++i)
        output = std::visit([&](const auto& l) { return l.forward(output, batch); }, layers_[i]);
    return output;
}

NetworkGradients Network::backward(std::span<const double> input, std::size_t batch,
                                   std::span<const double> output_gradient) const {
    validate_state();
    std::vector<std::vector<double>> activations;
    activations.reserve(layers_.size());
    auto current = input;
    for (const auto& layer : layers_) {
        activations.push_back(std::visit([&](const auto& l) { return l.forward(current, batch); }, layer));
        current = activations.back();
    }
    NetworkGradients gradient;
    gradient.layers.resize(layers_.size());
    auto upstream = output_gradient;
    for (std::size_t i = layers_.size(); i-- > 0;) {
        const auto layer_input = i == 0 ? input : std::span<const double>(activations[i - 1]);
        gradient.layers[i] = std::visit([&](const auto& l) -> NetworkLayerGradients {
            return l.backward(layer_input, batch, upstream);
        }, layers_[i]);
        upstream = input_gradient(gradient.layers[i]);
    }
    gradient.input = input_gradient(gradient.layers.front());
    return gradient;
}

void Network::sgd(const NetworkGradients& gradients, double learning_rate) {
    validate_state();
    if (gradients.layers.size() != layers_.size()) throw std::invalid_argument("network gradient shape mismatch");
    auto next = layers_;
    for (std::size_t i = 0; i < next.size(); ++i) {
        std::visit([&](auto& layer) {
            using Gradients = std::conditional_t<std::is_same_v<std::decay_t<decltype(layer)>, Layer>,
                                                 LayerGradients, InputMapGradients>;
            const auto* g = std::get_if<Gradients>(&gradients.layers[i]);
            if (!g) throw std::invalid_argument("network gradient kind does not match the layer");
            layer.sgd(*g, learning_rate);
        }, next[i]);
    }
    layers_.swap(next);
}

} // namespace kan

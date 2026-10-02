#pragma once

#include "kan/input_map.hpp"
#include "kan/layer.hpp"
#include <concepts>
#include <iterator>

namespace kan {

// A network layer kind: a KAN layer or an explicit input map (backlog M1).
using NetworkLayer = std::variant<Layer, InputMap>;
// Gradients of one network layer; the alternative matches its kind.
using NetworkLayerGradients = std::variant<LayerGradients, InputMapGradients>;

struct NetworkGradients {
    std::vector<double> input;
    std::vector<NetworkLayerGradients> layers; // same order as Network::layers()
};

struct NetworkRegularizationResult {
    double value = 0;
    NetworkGradients gradients;
};

// A nonempty sequence of network layers with matching adjacent dimensions.
// Layer indices are positions in layers(), maps included. Every operation
// dispatches on each layer's kind once per call.
class Network {
public:
    explicit Network(std::vector<NetworkLayer> layers);
    // KAN layers only (unchanged pre-M1 construction).
    template<class L> requires std::same_as<L, Layer>
    explicit Network(std::vector<L> layers)
        : Network(std::vector<NetworkLayer>(std::make_move_iterator(layers.begin()),
                                            std::make_move_iterator(layers.end()))) {}
    std::span<const NetworkLayer> layers() const noexcept { return layers_; }
    std::size_t inputs() const;
    std::size_t outputs() const;
    std::vector<double> forward(std::span<const double> input, std::size_t batch) const;
    NetworkGradients backward(std::span<const double> input, std::size_t batch,
                              std::span<const double> output_gradient) const;
    // Each gradient's alternative must match its layer's kind.
    void sgd(const NetworkGradients& gradients, double learning_rate);
    // Forward to kan::insert_knot / kan::adapt_grid (kan/families.hpp) on the
    // KAN layer at layer_index; an input map there raises invalid_argument.
    void insert_knot(std::size_t layer_index, double x);
    double adapt_grid(std::size_t layer_index, std::span<const double> samples);
    // Coefficient L2 of the KAN layers; input maps contribute zero value and
    // zero gradients of their trainable parameters.
    NetworkRegularizationResult regularization(double coefficient_l2) const;
private:
    void validate_state() const;
    Layer& kan_layer(std::size_t index);
    std::vector<NetworkLayer> layers_;
};

} // namespace kan

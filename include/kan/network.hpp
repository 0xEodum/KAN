#pragma once

#include "kan/layer.hpp"

namespace kan {

struct NetworkGradients {
    std::vector<double> input;
    std::vector<LayerGradients> layers; // same order as Network::layers()
};

class Network {
public:
    explicit Network(std::vector<Layer> layers);
    std::span<const Layer> layers() const noexcept { return layers_; }
    std::vector<double> forward(std::span<const double> input, std::size_t batch) const;
    NetworkGradients backward(std::span<const double> input, std::size_t batch,
                              std::span<const double> output_gradient) const;
    void sgd(const NetworkGradients& gradients, double learning_rate);
private:
    void validate_state() const;
    std::vector<Layer> layers_;
};

} // namespace kan

#pragma once

#include "kan/basis.hpp"
#include <span>

namespace kan {

struct LayerGradients {
    std::vector<double> input;
    std::vector<double> coefficients;
    std::vector<double> bias;
};

class Layer {
public:
    Layer(std::size_t inputs, std::size_t outputs, BasisConfig basis);
    std::size_t inputs() const noexcept { return inputs_; }
    std::size_t outputs() const noexcept { return outputs_; }
    const BasisConfig& basis() const noexcept { return basis_; }
    std::span<const double> coefficients() const noexcept { return coefficients_; }
    std::span<const double> bias() const noexcept { return bias_; }
    void set_parameters(std::span<const double> coefficients, std::span<const double> bias);
    std::vector<double> forward(std::span<const double> input, std::size_t batch) const;
    LayerGradients backward(std::span<const double> input, std::size_t batch,
                            std::span<const double> output_gradient) const;
    void sgd(const LayerGradients& gradients, double learning_rate);

private:
    std::size_t inputs_, outputs_;
    BasisConfig basis_;
    std::vector<double> coefficients_, bias_;
};

} // namespace kan

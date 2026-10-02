#pragma once

#include "kan/carrier.hpp"
#include <concepts>
#include <span>
#include <type_traits>

namespace kan {

struct LayerGradients {
    std::vector<double> input;
    std::vector<double> coefficients; // per-edge coefficients (rational numerators)
    std::vector<double> bias;
    NonlinearGradients nonlinear;     // alternative matches the layer's carrier
};

struct RegularizationResult {
    double value = 0;
    LayerGradients gradients;
};

// A KAN layer: dimensions, a per-output bias and one edge carrier. Every
// operation dispatches on the carrier once per call. Family-specific
// operations (knot insertion, RBF and rational parameter setters) are free
// functions in kan/families.hpp.
class Layer {
public:
    // Zero-initialized parameters. A TrainableRbfConfig selects
    // TrainableRbfEdges; every other basis selects BasisEdges.
    Layer(std::size_t inputs, std::size_t outputs, BasisConfig basis);
    template<class Config> requires std::same_as<std::remove_cvref_t<Config>, RationalConfig>
    Layer(std::size_t inputs, std::size_t outputs, Config&& config)
        : Layer(inputs, outputs, config, RationalTag{}) {}
    std::size_t inputs() const noexcept { return inputs_; }
    std::size_t outputs() const noexcept { return outputs_; }
    const Carrier& carrier() const noexcept { return carrier_; }
    std::size_t terms() const noexcept; // coefficients per edge
    std::span<const double> coefficients() const noexcept;
    std::span<const double> bias() const noexcept { return bias_; }
    // Replaces the per-edge coefficients of any carrier and the bias.
    void set_parameters(std::span<const double> coefficients, std::span<const double> bias);
    // Validates a whole carrier (configuration, shapes for this layer's
    // dimensions, finite parameters) and replaces it, optionally with the bias.
    void set_carrier(Carrier carrier);
    void set_carrier(Carrier carrier, std::span<const double> bias);
    std::vector<double> forward(std::span<const double> input, std::size_t batch) const;
    LayerGradients backward(std::span<const double> input, std::size_t batch,
                            std::span<const double> output_gradient) const;
    void sgd(const LayerGradients& gradients, double learning_rate);
    RegularizationResult regularization(double coefficient_l2) const;

private:
    friend class Network;
    struct RationalTag {};
    Layer(std::size_t inputs, std::size_t outputs, RationalConfig config, RationalTag);
    void validate_state() const;
    void replace(Carrier carrier, std::span<const double> bias);
    std::size_t inputs_, outputs_;
    Carrier carrier_;
    std::vector<double> bias_;
};

} // namespace kan

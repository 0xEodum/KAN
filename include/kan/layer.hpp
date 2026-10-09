#pragma once

#include "kan/carrier.hpp"
#include <concepts>
#include <optional>
#include <span>
#include <type_traits>

namespace kan {

// Optional residual branch of a layer (backlog M3, the base branch of the
// original KAN): every edge adds weights[o,i] * silu(x[b,i]), with
// silu(x) = x * sigmoid(x), to its carrier's edge function. It is part of the
// layer, not of the carrier, so carrier replacements keep it unchanged.
struct SiluResidual {
    std::vector<double> weights; // (outputs, inputs)
    bool operator==(const SiluResidual&) const = default;
};

struct LayerGradients {
    std::vector<double> input;
    std::vector<double> coefficients; // per-edge coefficients (rational numerators)
    std::vector<double> bias;
    NonlinearGradients nonlinear;     // alternative matches the layer's carrier
    std::vector<double> residual;     // (outputs, inputs); empty without the residual branch
};

struct RegularizationResult {
    double value = 0;
    LayerGradients gradients;
};

// A KAN layer: dimensions, a per-output bias, one edge carrier and an
// optional SiLU residual branch:
//     y[b,o] = bias[o] + carrier_o(x_b) + sum_i weights[o,i] * silu(x[b,i]).
// Every operation dispatches on the carrier once per call. Family-specific
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
    // The residual branch; std::nullopt (the constructors' default) disables it.
    const std::optional<SiluResidual>& residual() const noexcept { return residual_; }
    // Validates the weight shape (outputs * inputs) and finite weights, then
    // enables, replaces or (std::nullopt) disables the branch atomically.
    void set_residual(std::optional<SiluResidual> residual);
    std::vector<double> forward(std::span<const double> input, std::size_t batch) const;
    LayerGradients backward(std::span<const double> input, std::size_t batch,
                            std::span<const double> output_gradient) const;
    // gradients.residual must match the branch: outputs * inputs values when
    // it is enabled, empty otherwise.
    void sgd(const LayerGradients& gradients, double learning_rate);
    // 0.5 * lambda * (sum c^2 + sum w^2) over the per-edge coefficients c and
    // the residual weights w: the linear parameters of every edge function.
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
    std::optional<SiluResidual> residual_;
};

} // namespace kan

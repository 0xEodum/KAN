#pragma once

#include "kan/basis.hpp"
#include "kan/rational.hpp"
#include <concepts>
#include <span>
#include <type_traits>

namespace kan {

struct LayerGradients {
    std::vector<double> input;
    std::vector<double> coefficients;
    std::vector<double> bias;
    std::vector<double> centers; // shared trainable RBF basis parameters
    std::vector<double> log_widths;
    std::vector<double> denominators;
};

struct RegularizationResult {
    double value = 0;
    LayerGradients gradients;
};

class Layer {
public:
    Layer(std::size_t inputs, std::size_t outputs, BasisConfig basis);
    template<class Config> requires std::same_as<std::remove_cvref_t<Config>, RationalConfig>
    Layer(std::size_t inputs, std::size_t outputs, Config&& config)
        : Layer(inputs, outputs, config, RationalTag{}) {}
    std::size_t inputs() const noexcept { return inputs_; }
    std::size_t outputs() const noexcept { return outputs_; }
    const BasisConfig& basis() const;
    bool is_rational() const noexcept { return rational_; }
    const RationalConfig& rational_config() const;
    std::span<const double> denominators() const noexcept { return denominators_; }
    std::span<const double> coefficients() const noexcept { return coefficients_; }
    std::span<const double> bias() const noexcept { return bias_; }
    void set_parameters(std::span<const double> coefficients, std::span<const double> bias);
    void set_rational_parameters(std::span<const double> numerator,
                                std::span<const double> denominator, std::span<const double> bias);
    std::vector<double> forward(std::span<const double> input, std::size_t batch) const;
    LayerGradients backward(std::span<const double> input, std::size_t batch,
                            std::span<const double> output_gradient) const;
    void sgd(const LayerGradients& gradients, double learning_rate);
    void set_rbf_parameters(std::span<const double> centers, std::span<const double> log_widths);
    void insert_knot(double x);
    double adapt_grid(std::span<const double> samples);
    RegularizationResult regularization(double coefficient_l2) const;

private:
    friend class Network;
    struct RationalTag {};
    Layer(std::size_t inputs, std::size_t outputs, RationalConfig config, RationalTag);
    void validate_state() const;
    std::size_t terms() const noexcept; // coefficients per edge
    TrainableRbfConfig* trainable_rbf() noexcept; // null unless a trainable RBF layer
    const TrainableRbfConfig* trainable_rbf() const noexcept;
    std::size_t inputs_, outputs_;
    BasisConfig basis_;
    bool rational_ = false;
    RationalConfig rational_config_;
    std::vector<double> denominators_;
    std::vector<double> coefficients_, bias_;
};

} // namespace kan

#include "kan/layer.hpp"
#include <cmath>
#include <stdexcept>

namespace kan {
namespace {
std::size_t checked_size(std::size_t left, std::size_t right) {
    const auto max = std::vector<double>().max_size();
    if (right != 0 && left > max / right) throw std::overflow_error("array size overflow");
    return left * right;
}
void require_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("data must be finite");
}
void result_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::overflow_error("nonfinite numerical result");
}
}

Layer::Layer(std::size_t inputs, std::size_t outputs, BasisConfig basis)
    : inputs_(inputs), outputs_(outputs), basis_(std::move(basis)) {
    if (inputs == 0 || outputs == 0) throw std::invalid_argument("layer dimensions must be positive");
    validate_basis(basis_);
    coefficients_.resize(checked_size(checked_size(inputs, outputs), basis_.size), 0.0);
    bias_.resize(checked_size(outputs, 1), 0.0);
}
void Layer::set_parameters(std::span<const double> coefficients, std::span<const double> bias) {
    if (coefficients.size() != coefficients_.size() || bias.size() != bias_.size())
        throw std::invalid_argument("parameter shape mismatch");
    require_finite(coefficients); require_finite(bias);
    std::vector<double> next_coefficients(coefficients.begin(), coefficients.end()), next_bias(bias.begin(), bias.end());
    coefficients_.swap(next_coefficients); bias_.swap(next_bias);
}

std::vector<double> Layer::forward(std::span<const double> input, std::size_t batch) const {
    const auto input_size = checked_size(batch, inputs_), output_size = checked_size(batch, outputs_);
    if (input.size() != input_size) throw std::invalid_argument("input shape mismatch");
    require_finite(input);
    std::vector<double> output(output_size);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t o = 0; o < outputs_; ++o) output[b * outputs_ + o] = bias_[o];
        for (std::size_t i = 0; i < inputs_; ++i) {
            const auto basis = evaluate_basis(basis_, input[b * inputs_ + i]);
            for (std::size_t o = 0; o < outputs_; ++o)
                for (std::size_t k = 0; k < basis_.size; ++k)
                    output[b * outputs_ + o] += coefficients_[(o * inputs_ + i) * basis_.size + k] * basis.values[k];
        }
    }
    result_finite(output);
    return output;
}

LayerGradients Layer::backward(std::span<const double> input, std::size_t batch,
                               std::span<const double> output_gradient) const {
    const auto input_size = checked_size(batch, inputs_), output_size = checked_size(batch, outputs_);
    if (input.size() != input_size || output_gradient.size() != output_size)
        throw std::invalid_argument("backward shape mismatch");
    require_finite(input); require_finite(output_gradient);
    LayerGradients gradient{std::vector<double>(input_size, 0.0),
                            std::vector<double>(coefficients_.size(), 0.0),
                            std::vector<double>(outputs_, 0.0)};
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t o = 0; o < outputs_; ++o) gradient.bias[o] += output_gradient[b * outputs_ + o];
        for (std::size_t i = 0; i < inputs_; ++i) {
            const auto basis = evaluate_basis(basis_, input[b * inputs_ + i]);
            for (std::size_t o = 0; o < outputs_; ++o) {
                const auto upstream = output_gradient[b * outputs_ + o];
                for (std::size_t k = 0; k < basis_.size; ++k) {
                    const auto index = (o * inputs_ + i) * basis_.size + k;
                    gradient.coefficients[index] += upstream * basis.values[k];
                    gradient.input[b * inputs_ + i] += upstream * coefficients_[index] * basis.derivatives[k];
                }
            }
        }
    }
    result_finite(gradient.input); result_finite(gradient.coefficients); result_finite(gradient.bias);
    return gradient;
}

void Layer::sgd(const LayerGradients& gradients, double learning_rate) {
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
        throw std::invalid_argument("learning rate must be finite and positive");
    if (gradients.coefficients.size() != coefficients_.size() || gradients.bias.size() != bias_.size())
        throw std::invalid_argument("parameter gradient shape mismatch");
    require_finite(gradients.coefficients); require_finite(gradients.bias);
    auto next_coefficients = coefficients_, next_bias = bias_;
    for (std::size_t i = 0; i < next_coefficients.size(); ++i)
        next_coefficients[i] -= learning_rate * gradients.coefficients[i];
    for (std::size_t i = 0; i < next_bias.size(); ++i) next_bias[i] -= learning_rate * gradients.bias[i];
    result_finite(next_coefficients); result_finite(next_bias);
    coefficients_.swap(next_coefficients); bias_.swap(next_bias);
}
} // namespace kan

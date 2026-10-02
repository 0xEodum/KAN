// Carrier-independent Layer: dimensions, bias, validation and the SGD/L2
// protocol. Each numerical operation dispatches on the carrier exactly once.
#include "kan/layer.hpp"
#include "carriers/edge_ops.hpp"
#include "detail/checks.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <variant>

namespace kan {
namespace {
using detail::checked_size;
using detail::require_finite;
using detail::result_finite;

// Every carrier has a per-edge coefficient tensor.
template<class C> auto& coefficients_of(C& carrier) noexcept {
    return std::visit([](auto& edges) -> auto& { return edges.coefficients; }, carrier);
}

Carrier basis_carrier(BasisConfig basis) {
    if (auto* rbf = std::get_if<TrainableRbfConfig>(&basis))
        return TrainableRbfEdges{std::move(*rbf), {}};
    return BasisEdges{std::move(basis), {}};
}
} // namespace

Layer::Layer(std::size_t inputs, std::size_t outputs, BasisConfig basis)
    : inputs_(inputs), outputs_(outputs), carrier_(basis_carrier(std::move(basis))) {
    if (inputs == 0 || outputs == 0) throw std::invalid_argument("layer dimensions must be positive");
    std::visit([](const auto& edges) {
        if constexpr (requires { edges.basis; }) validate_basis(edges.basis);
    }, carrier_);
    coefficients_of(carrier_).resize(checked_size(checked_size(inputs, outputs), terms()), 0.0);
    bias_.resize(checked_size(outputs, 1), 0.0);
}

Layer::Layer(std::size_t inputs, std::size_t outputs, RationalConfig config, RationalTag)
    : inputs_(inputs), outputs_(outputs), carrier_(RationalEdges{config, {}, {}}) {
    if (inputs == 0 || outputs == 0) throw std::invalid_argument("layer dimensions must be positive");
    validate_rational(config);
    const auto edges = checked_size(inputs, outputs);
    auto& rational = std::get<RationalEdges>(carrier_);
    rational.coefficients.resize(checked_size(edges, config.numerator_degree + 1));
    rational.denominators.resize(checked_size(edges, config.denominator_degree));
    bias_.resize(checked_size(outputs, 1));
}

std::size_t Layer::terms() const noexcept {
    return std::visit([](const auto& edges) { return detail::terms(edges); }, carrier_);
}

std::span<const double> Layer::coefficients() const noexcept { return coefficients_of(carrier_); }

void Layer::validate_state() const {
    if (inputs_ == 0 || outputs_ == 0 || bias_.size() != outputs_)
        throw std::invalid_argument("layer is uninitialized or moved from");
    std::visit([&](const auto& edges) { detail::validate(edges, {inputs_, outputs_}); }, carrier_);
}

void Layer::set_parameters(std::span<const double> coefficients, std::span<const double> bias) {
    validate_state();
    auto& current = coefficients_of(carrier_);
    if (coefficients.size() != current.size() || bias.size() != bias_.size())
        throw std::invalid_argument("parameter shape mismatch");
    require_finite(coefficients);
    require_finite(bias);
    std::vector<double> next_coefficients(coefficients.begin(), coefficients.end()), next_bias(bias.begin(), bias.end());
    current.swap(next_coefficients);
    bias_.swap(next_bias);
}

void Layer::set_carrier(Carrier carrier) {
    validate_state();
    replace(std::move(carrier), bias_);
}

void Layer::set_carrier(Carrier carrier, std::span<const double> bias) {
    validate_state();
    replace(std::move(carrier), bias);
}

void Layer::replace(Carrier carrier, std::span<const double> bias) {
    std::visit([&](const auto& edges) {
        detail::validate(edges, {inputs_, outputs_});
        require_finite(edges.coefficients);
        detail::require_finite_nonlinear(edges);
    }, carrier);
    if (bias.size() != outputs_) throw std::invalid_argument("parameter shape mismatch");
    require_finite(bias);
    std::vector<double> next_bias(bias.begin(), bias.end());
    carrier_ = std::move(carrier);
    bias_.swap(next_bias);
}

std::vector<double> Layer::forward(std::span<const double> input, std::size_t batch) const {
    validate_state();
    const auto input_size = checked_size(batch, inputs_), output_size = checked_size(batch, outputs_);
    if (input.size() != input_size) throw std::invalid_argument("input shape mismatch");
    require_finite(input);
    std::vector<double> output(output_size);
    // Configuration and inputs are validated above; carriers evaluate without revalidating.
    std::visit([&](const auto& edges) {
        detail::forward(edges, {inputs_, outputs_}, bias_, input, batch, output);
    }, carrier_);
    result_finite(output);
    return output;
}

LayerGradients Layer::backward(std::span<const double> input, std::size_t batch,
                               std::span<const double> output_gradient) const {
    validate_state();
    const auto input_size = checked_size(batch, inputs_), output_size = checked_size(batch, outputs_);
    if (input.size() != input_size || output_gradient.size() != output_size)
        throw std::invalid_argument("backward shape mismatch");
    require_finite(input);
    require_finite(output_gradient);
    LayerGradients gradient{std::vector<double>(input_size, 0.0),
                            std::vector<double>(coefficients().size(), 0.0),
                            std::vector<double>(outputs_, 0.0), {}};
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t o = 0; o < outputs_; ++o) gradient.bias[o] += output_gradient[b * outputs_ + o];
    std::visit([&](const auto& edges) {
        auto nonlinear = detail::zero_nonlinear(edges);
        detail::backward(edges, {inputs_, outputs_}, input, batch, output_gradient, gradient.input,
                         gradient.coefficients, nonlinear);
        detail::check_result(nonlinear);
        gradient.nonlinear = std::move(nonlinear);
    }, carrier_);
    result_finite(gradient.input);
    result_finite(gradient.coefficients);
    result_finite(gradient.bias);
    return gradient;
}

void Layer::sgd(const LayerGradients& gradients, double learning_rate) {
    validate_state();
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
        throw std::invalid_argument("learning rate must be finite and positive");
    if (gradients.coefficients.size() != coefficients().size() || gradients.bias.size() != bias_.size())
        throw std::invalid_argument("parameter gradient shape mismatch");
    require_finite(gradients.coefficients);
    require_finite(gradients.bias);
    // Validate every gradient and candidate before committing any parameter.
    auto next = std::visit([&](const auto& edges) -> Carrier {
        using Edges = std::decay_t<decltype(edges)>;
        const auto* nonlinear = std::get_if<detail::nonlinear_t<Edges>>(&gradients.nonlinear);
        if (!nonlinear) throw std::invalid_argument("nonlinear gradient type does not match the carrier");
        detail::check_gradient(edges, *nonlinear);
        auto candidate = edges;
        detail::update_nonlinear(candidate, *nonlinear, learning_rate);
        auto& coefficients = candidate.coefficients;
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            coefficients[i] -= learning_rate * gradients.coefficients[i];
        result_finite(coefficients);
        return candidate;
    }, carrier_);
    auto next_bias = bias_;
    for (std::size_t i = 0; i < next_bias.size(); ++i) next_bias[i] -= learning_rate * gradients.bias[i];
    result_finite(next_bias);
    carrier_ = std::move(next);
    bias_.swap(next_bias);
}

RegularizationResult Layer::regularization(double lambda) const {
    validate_state();
    if (!std::isfinite(lambda) || lambda < 0)
        throw std::invalid_argument("L2 coefficient must be finite and nonnegative");
    const auto coefficients = this->coefficients();
    RegularizationResult r;
    r.gradients.coefficients.resize(coefficients.size());
    r.gradients.bias.resize(outputs_);
    r.gradients.nonlinear = std::visit([](const auto& edges) -> NonlinearGradients {
        return detail::zero_nonlinear(edges);
    }, carrier_);
    for (std::size_t j = 0; j < coefficients.size(); ++j) {
        const double g = lambda * coefficients[j];
        r.gradients.coefficients[j] = g;
        r.value += (0.5 * g) * coefficients[j];
    }
    result_finite(r.gradients.coefficients);
    result_finite(std::span<const double>(&r.value, 1));
    return r;
}

} // namespace kan

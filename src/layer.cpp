// Carrier-independent Layer: dimensions, bias, the optional SiLU residual
// branch, validation and the SGD/L2 protocol. Each numerical operation
// dispatches on the carrier exactly once.
#include "kan/layer.hpp"
#include "carriers/edge_ops.hpp"
#include "detail/checks.hpp"
#include "detail/residual_formulas.hpp"
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

// Residual branch (backlog M3), weights w of layout (outputs, inputs). The
// silu values are computed once per sample and input. Intermediates are not
// guarded one by one: a product or sum of finite values that overflows stays
// nonfinite, and the callers check every result (std::overflow_error).
constexpr auto unguarded = [](double v) { return v; };

// output[b,o] += sum_i w[o,i] silu(x[b,i]): the sum in ascending i, then
// added to the carrier's output.
void residual_forward(std::span<const double> w, std::size_t inputs, std::size_t outputs,
                      std::span<const double> input, std::size_t batch, std::span<double> output) {
    std::vector<double> s(inputs);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < inputs; ++i) s[i] = detail::silu_value(input[b * inputs + i], unguarded);
        for (std::size_t o = 0; o < outputs; ++o) {
            const double* row = w.data() + o * inputs;
            double sum = 0;
            for (std::size_t i = 0; i < inputs; ++i) sum += row[i] * s[i];
            output[b * outputs + o] += sum;
        }
    }
}

// dw[o,i] = sum_b u[b,o] silu(x[b,i]) (ascending b); input_gradient[b,i] +=
// silu'(x[b,i]) * sum_o u[b,o] w[o,i] (ascending o), after the carrier's VJP.
void residual_backward(std::span<const double> w, std::size_t inputs, std::size_t outputs,
                       std::span<const double> input, std::size_t batch, std::span<const double> upstream,
                       std::span<double> input_gradient, std::span<double> weight_gradient) {
    std::vector<double> s(inputs), ds(inputs), t(inputs);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < inputs; ++i) {
            const auto f = detail::silu(input[b * inputs + i], unguarded);
            s[i] = f.value;
            ds[i] = f.derivative;
            t[i] = 0;
        }
        for (std::size_t o = 0; o < outputs; ++o) {
            const double u = upstream[b * outputs + o];
            const double* row = w.data() + o * inputs;
            double* dw = weight_gradient.data() + o * inputs;
            for (std::size_t i = 0; i < inputs; ++i) {
                dw[i] += u * s[i];
                t[i] += u * row[i];
            }
        }
        for (std::size_t i = 0; i < inputs; ++i) input_gradient[b * inputs + i] += ds[i] * t[i];
    }
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

Layer::Layer(const Layer&) = default;

std::size_t Layer::terms() const noexcept {
    return std::visit([](const auto& edges) { return detail::terms(edges); }, carrier_);
}

std::span<const double> Layer::coefficients() const noexcept { return coefficients_of(carrier_); }

void Layer::validate_state() const {
    if (inputs_ == 0 || outputs_ == 0 || bias_.size() != outputs_ ||
        (residual_ && residual_->weights.size() != inputs_ * outputs_))
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

void Layer::set_residual(std::optional<SiluResidual> residual) {
    validate_state();
    if (residual) {
        if (residual->weights.size() != inputs_ * outputs_) throw std::invalid_argument("residual weight shape mismatch");
        require_finite(residual->weights);
    }
    residual_ = std::move(residual);
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
    if (residual_) residual_forward(residual_->weights, inputs_, outputs_, input, batch, output);
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
                            std::vector<double>(outputs_, 0.0), {},
                            std::vector<double>(residual_ ? residual_->weights.size() : 0, 0.0)};
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t o = 0; o < outputs_; ++o) gradient.bias[o] += output_gradient[b * outputs_ + o];
    std::visit([&](const auto& edges) {
        auto nonlinear = detail::zero_nonlinear(edges);
        detail::backward(edges, {inputs_, outputs_}, input, batch, output_gradient, gradient.input,
                         gradient.coefficients, nonlinear);
        detail::check_result(nonlinear);
        gradient.nonlinear = std::move(nonlinear);
    }, carrier_);
    if (residual_)
        residual_backward(residual_->weights, inputs_, outputs_, input, batch, output_gradient, gradient.input,
                          gradient.residual);
    result_finite(gradient.input);
    result_finite(gradient.coefficients);
    result_finite(gradient.bias);
    result_finite(gradient.residual);
    return gradient;
}

void Layer::sgd(const LayerGradients& gradients, double learning_rate) {
    validate_state();
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
        throw std::invalid_argument("learning rate must be finite and positive");
    if (gradients.coefficients.size() != coefficients().size() || gradients.bias.size() != bias_.size())
        throw std::invalid_argument("parameter gradient shape mismatch");
    if (gradients.residual.size() != (residual_ ? residual_->weights.size() : 0))
        throw std::invalid_argument("residual gradient does not match the layer's residual branch");
    require_finite(gradients.coefficients);
    require_finite(gradients.bias);
    require_finite(gradients.residual);
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
    auto next_residual = residual_;
    if (next_residual) {
        auto& w = next_residual->weights;
        for (std::size_t i = 0; i < w.size(); ++i) w[i] -= learning_rate * gradients.residual[i];
        result_finite(w);
    }
    carrier_ = std::move(next);
    bias_.swap(next_bias);
    residual_.swap(next_residual);
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
    // The residual weights are linear parameters of the edge functions too.
    if (residual_) {
        const auto& w = residual_->weights;
        r.gradients.residual.resize(w.size());
        for (std::size_t j = 0; j < w.size(); ++j) {
            const double g = lambda * w[j];
            r.gradients.residual[j] = g;
            r.value += (0.5 * g) * w[j];
        }
    }
    result_finite(r.gradients.coefficients);
    result_finite(r.gradients.residual);
    result_finite(std::span<const double>(&r.value, 1));
    return r;
}

} // namespace kan

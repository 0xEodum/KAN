// Carriers on the expansion + contraction engine: BasisEdges (linear in all
// its parameters) and TrainableRbfEdges (adds nonlinear shared centers and
// log widths, whose VJPs ride along the contraction VJP).
#include "backends/cpu/edge_ops.hpp"
#include "backends/cpu/linear_engine.hpp"
#include <cmath>
#include <stdexcept>
#include <variant>

namespace kan::detail {
namespace {

void require_coefficient_shape(std::size_t actual, EdgeShape shape, std::size_t terms) {
    if (actual != checked_size(checked_size(shape.inputs, shape.outputs), terms))
        throw std::invalid_argument("carrier coefficient shape mismatch");
}

template<class Config>
void expansion_forward(const Config& basis, const std::vector<double>& coefficients, EdgeShape shape,
                       std::span<const double> bias, std::span<const double> input, std::size_t batch,
                       std::span<double> output) {
    if (batch == 0) return; // no scratch for shape-only calls
    Expansion expansion(basis_view(basis), shape.inputs);
    for (std::size_t b = 0; b < batch; ++b) {
        expansion.expand(input.data() + b * shape.inputs);
        contract(coefficients.data(), bias.data(), expansion.values.data(), shape.outputs, expansion.width(),
                 output.data() + b * shape.outputs);
    }
}

// Knots between the grids of consecutive inputs (backlog M6): 0 for one shared grid.
std::size_t knot_stride(const BSplineConfig& spline) {
    return spline.grids == 1 ? 0 : spline.knots.size() / spline.grids;
}

} // namespace

void check_grids(const BasisConfig& basis, std::size_t inputs) {
    if (const auto* spline = std::get_if<BSplineConfig>(&basis);
        spline && spline->grids != 1 && spline->grids != inputs)
        throw std::invalid_argument("spline grids must be 1 (shared) or the layer's input count");
}

std::size_t terms(const BasisEdges& edges) noexcept { return basis_size(edges.basis); }
std::size_t terms(const TrainableRbfEdges& edges) noexcept { return basis_size(edges.basis); }

void validate(const BasisEdges& edges, EdgeShape shape) {
    if (std::holds_alternative<TrainableRbfConfig>(edges.basis))
        throw std::invalid_argument("a trainable RBF basis requires TrainableRbfEdges");
    validate_basis(edges.basis);
    check_grids(edges.basis, shape.inputs);
    require_coefficient_shape(edges.coefficients.size(), shape, terms(edges));
}

void validate(const TrainableRbfEdges& edges, EdgeShape shape) {
    validate_basis(edges.basis);
    require_coefficient_shape(edges.coefficients.size(), shape, terms(edges));
}

// B-splines take the sparse engine (backlog C5): bitwise the dense result.
void forward(const BasisEdges& edges, EdgeShape shape, std::span<const double> bias, std::span<const double> input,
             std::size_t batch, std::span<double> output) {
    if (const auto* spline = std::get_if<BSplineConfig>(&edges.basis)) {
        if (batch == 0) return;
        SplineExpansion expansion(basis_view(*spline), shape.inputs, knot_stride(*spline));
        for (std::size_t b = 0; b < batch; ++b) {
            expansion.expand(input.data() + b * shape.inputs);
            contract_sparse(edges.coefficients.data(), bias.data(), expansion, shape.inputs, shape.outputs,
                            output.data() + b * shape.outputs);
        }
        return;
    }
    expansion_forward(edges.basis, edges.coefficients, shape, bias, input, batch, output);
}

// The forward pass evaluates the RBF partials as well, so the basis guard
// applies to them exactly as in backward.
void forward(const TrainableRbfEdges& edges, EdgeShape shape, std::span<const double> bias,
             std::span<const double> input, std::size_t batch, std::span<double> output) {
    expansion_forward(edges.basis, edges.coefficients, shape, bias, input, batch, output);
}

void backward(const BasisEdges& edges, EdgeShape shape, std::span<const double> input, std::size_t batch,
              std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, std::monostate&) {
    if (batch == 0) return;
    if (const auto* spline = std::get_if<BSplineConfig>(&edges.basis)) {
        SplineExpansion expansion(basis_view(*spline), shape.inputs, knot_stride(*spline));
        const auto magnitudes =
            edge_magnitudes(edges.coefficients.data(), shape.inputs * shape.outputs, expansion.terms());
        for (std::size_t b = 0; b < batch; ++b) {
            expansion.expand(input.data() + b * shape.inputs);
            contract_vjp_sparse(edges.coefficients.data(), magnitudes.data(), upstream.data() + b * shape.outputs,
                                expansion, shape.inputs, shape.outputs, coefficient_gradient.data(),
                                input_gradient.data() + b * shape.inputs);
        }
        return;
    }
    Expansion expansion(basis_view(edges.basis), shape.inputs);
    for (std::size_t b = 0; b < batch; ++b) {
        expansion.expand(input.data() + b * shape.inputs);
        contract_vjp(edges.coefficients.data(), upstream.data() + b * shape.outputs, expansion, shape.inputs,
                     shape.outputs, coefficient_gradient.data(), input_gradient.data() + b * shape.inputs,
                     [](std::size_t, std::size_t, double) {});
    }
}

void backward(const TrainableRbfEdges& edges, EdgeShape shape, std::span<const double> input, std::size_t batch,
              std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, TrainableRbfGradients& nonlinear) {
    if (batch == 0) return;
    Expansion expansion(basis_view(edges.basis), shape.inputs);
    double* centers = nonlinear.centers.data();
    double* log_widths = nonlinear.log_widths.data();
    const double* center_partials = expansion.center_derivatives.data();
    const double* log_width_partials = expansion.log_width_derivatives.data();
    for (std::size_t b = 0; b < batch; ++b) {
        expansion.expand(input.data() + b * shape.inputs);
        contract_vjp(edges.coefficients.data(), upstream.data() + b * shape.outputs, expansion, shape.inputs,
                     shape.outputs, coefficient_gradient.data(), input_gradient.data() + b * shape.inputs,
                     [&](std::size_t k, std::size_t j, double weighted) {
                         centers[k] += weighted * center_partials[j];
                         log_widths[k] += weighted * log_width_partials[j];
                     });
    }
}

TrainableRbfGradients zero_nonlinear(const TrainableRbfEdges& edges) {
    const auto size = edges.basis.centers.size();
    return {std::vector<double>(size, 0.0), std::vector<double>(size, 0.0)};
}

void check_gradient(const TrainableRbfEdges& edges, const TrainableRbfGradients& gradient) {
    const auto size = edges.basis.centers.size();
    if (gradient.centers.size() != size || gradient.log_widths.size() != size)
        throw std::invalid_argument("nonlinear gradient shape mismatch");
    require_finite(gradient.centers);
    require_finite(gradient.log_widths);
}

void update_nonlinear(TrainableRbfEdges& candidate, const TrainableRbfGradients& gradient, double rate) {
    auto& basis = candidate.basis;
    for (std::size_t k = 0; k < basis.centers.size(); ++k) {
        basis.centers[k] -= rate * gradient.centers[k];
        basis.log_widths[k] -= rate * gradient.log_widths[k];
    }
    result_finite(basis.centers);
    result_finite(basis.log_widths);
    for (double w : basis.log_widths)
        if (!std::isfinite(std::exp(w)) || std::exp(w) <= 0)
            throw std::overflow_error("RBF candidate width is not finite and positive");
}

void check_result(const TrainableRbfGradients& gradient) {
    result_finite(gradient.centers);
    result_finite(gradient.log_widths);
}

} // namespace kan::detail

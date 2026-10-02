// Rational edges: every edge evaluates P(z)/Q(z) with its own numerator and
// denominator; the VJPs come from the shared rational formulas.
#include "edge_ops.hpp"
#include "../rational_internal.hpp"
#include "../detail/checks.hpp"
#include <stdexcept>

namespace kan::detail {
namespace {

// Edge (o,i) of a validated carrier at x.
RationalTerms edge_terms(const RationalEdges& edges, std::size_t edge, double x) {
    const auto m = edges.config.numerator_degree + 1, n = edges.config.denominator_degree;
    return evaluate_rational_trusted(edges.config, x,
                                     std::span<const double>(edges.coefficients).subspan(edge * m, m),
                                     std::span<const double>(edges.denominators).subspan(edge * n, n));
}

} // namespace

std::size_t terms(const RationalEdges& edges) noexcept { return edges.config.numerator_degree + 1; }

void validate(const RationalEdges& edges, EdgeShape shape) {
    validate_rational(edges.config);
    const auto count = checked_size(shape.inputs, shape.outputs);
    if (edges.coefficients.size() != checked_size(count, terms(edges)))
        throw std::invalid_argument("carrier coefficient shape mismatch");
    if (edges.denominators.size() != checked_size(count, edges.config.denominator_degree))
        throw std::invalid_argument("rational denominator shape mismatch");
}

void require_finite_nonlinear(const RationalEdges& edges) { require_finite(edges.denominators); }

void forward(const RationalEdges& edges, EdgeShape shape, std::span<const double> bias,
             std::span<const double> input, std::size_t batch, std::span<double> output) {
    const auto inputs = shape.inputs, outputs = shape.outputs;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t o = 0; o < outputs; ++o) output[b * outputs + o] = bias[o];
        for (std::size_t i = 0; i < inputs; ++i)
            for (std::size_t o = 0; o < outputs; ++o) {
                output[b * outputs + o] += edge_terms(edges, o * inputs + i, input[b * inputs + i]).value;
                result_finite(output.subspan(b * outputs + o, 1));
            }
    }
}

void backward(const RationalEdges& edges, EdgeShape shape, std::span<const double> input,
              std::size_t batch, std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, RationalGradients& nonlinear) {
    const auto inputs = shape.inputs, outputs = shape.outputs;
    const auto m = edges.config.numerator_degree + 1, n = edges.config.denominator_degree;
    auto& denominators = nonlinear.denominators;
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t i = 0; i < inputs; ++i)
            for (std::size_t o = 0; o < outputs; ++o) {
                const auto edge = o * inputs + i;
                const auto r = edge_terms(edges, edge, input[b * inputs + i]);
                const auto u = upstream[b * outputs + o];
                input_gradient[b * inputs + i] += u * r.input_derivative;
                for (std::size_t k = 0; k < m; ++k) coefficient_gradient[edge * m + k] += u * r.numerator_derivatives[k];
                for (std::size_t k = 0; k < n; ++k) denominators[edge * n + k] += u * r.denominator_derivatives[k];
            }
}

RationalGradients zero_nonlinear(const RationalEdges& edges) {
    return {std::vector<double>(edges.denominators.size(), 0.0)};
}

void check_gradient(const RationalEdges& edges, const RationalGradients& gradient) {
    if (gradient.denominators.size() != edges.denominators.size())
        throw std::invalid_argument("denominator gradient shape mismatch");
    require_finite(gradient.denominators);
}

void update_nonlinear(RationalEdges& candidate, const RationalGradients& gradient, double rate) {
    auto& denominators = candidate.denominators;
    for (std::size_t k = 0; k < denominators.size(); ++k) denominators[k] -= rate * gradient.denominators[k];
    result_finite(denominators);
}

void check_result(const RationalGradients& gradient) { result_finite(gradient.denominators); }

} // namespace kan::detail

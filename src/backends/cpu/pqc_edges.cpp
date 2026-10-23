// PQC edges on the CPU (M5 phase 1). RED stub: shapes only, bias-only forward,
// zero VJPs.
#include "backends/cpu/edge_ops.hpp"
#include "core/checks.hpp"
#include <stdexcept>

namespace kan::detail {

std::size_t terms(const PqcEdges& edges) noexcept { return pqc_terms(edges.config); }

void validate(const PqcEdges& edges, EdgeShape shape) {
    const auto count = checked_size(shape.inputs, shape.outputs);
    if (edges.coefficients.size() != checked_size(count, terms(edges)))
        throw std::invalid_argument("carrier coefficient shape mismatch");
}

void require_finite_nonlinear(const PqcEdges&) {}

void forward(const PqcEdges&, EdgeShape shape, std::span<const double> bias, std::span<const double>, std::size_t batch,
             std::span<double> output) {
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t o = 0; o < shape.outputs; ++o) output[b * shape.outputs + o] = bias[o];
}

void backward(const PqcEdges&, EdgeShape, std::span<const double>, std::size_t, std::span<const double>,
              std::span<double>, std::span<double>, PqcGradients&) {}

PqcGradients zero_nonlinear(const PqcEdges& edges) {
    return {std::vector<double>(edges.angles.size(), 0.0), std::vector<double>(edges.encoding_weights.size(), 0.0),
            std::vector<double>(edges.encoding_biases.size(), 0.0)};
}

void check_gradient(const PqcEdges& edges, const PqcGradients& gradient) {
    if (gradient.angles.size() != edges.angles.size() ||
        gradient.encoding_weights.size() != edges.encoding_weights.size() ||
        gradient.encoding_biases.size() != edges.encoding_biases.size())
        throw std::invalid_argument("PQC gradient shape mismatch");
}

void update_nonlinear(PqcEdges&, const PqcGradients&, double) {}

void check_result(const PqcGradients&) {}

} // namespace kan::detail

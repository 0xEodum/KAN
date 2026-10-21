// Per-edge trainable Mexican-hat wavelets (backlog M7): every edge (o,i) sums
// its own wavelets k, each with coefficient w, translation tau and log scale l,
// from the shared formulas (wavelet_scales once per wavelet and call,
// wavelet_terms per sample). No expansion is shared between edges, so there is
// no contraction form: the loops run over samples, outputs, inputs and terms.
#include "backends/cpu/edge_ops.hpp"
#include "backends/cpu/basis_host.hpp"
#include "carriers/local/local_formulas.hpp"
#include "core/checks.hpp"
#include <stdexcept>

namespace kan::detail {
namespace {
std::vector<WaveletScales<double>> scales_of(const WaveletEdges& edges) {
    std::vector<WaveletScales<double>> scales(edges.log_scales.size());
    for (std::size_t j = 0; j < scales.size(); ++j) scales[j] = wavelet_scales(edges.log_scales[j]);
    return scales;
}
} // namespace

std::size_t terms(const WaveletEdges& edges) noexcept { return edges.config.size; }

void validate(const WaveletEdges& edges, EdgeShape shape) {
    validate_wavelet(edges.config);
    const auto count = checked_size(checked_size(shape.inputs, shape.outputs), edges.config.size);
    if (edges.coefficients.size() != count) throw std::invalid_argument("carrier coefficient shape mismatch");
    if (edges.translations.size() != count || edges.log_scales.size() != count)
        throw std::invalid_argument("wavelet translation or log-scale shape mismatch");
}

void require_finite_nonlinear(const WaveletEdges& edges) {
    require_finite(edges.translations);
    for (double l : edges.log_scales)
        if (!valid_log_scale(l)) throw std::invalid_argument("wavelet log scales must be finite with finite nonzero exp(+-l)");
}

// output[b,o] = bias[o] + sum_i sum_k w * psi, accumulated in ascending i,
// then k. Layer checks the finished output.
void forward(const WaveletEdges& edges, EdgeShape shape, std::span<const double> bias, std::span<const double> input,
             std::size_t batch, std::span<double> output) {
    const auto inputs = shape.inputs, outputs = shape.outputs, size = edges.config.size;
    const auto scales = scales_of(edges);
    const FiniteBasisGuard guard;
    const double* w = edges.coefficients.data();
    const double* tau = edges.translations.data();
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t o = 0; o < outputs; ++o) {
            double sum = bias[o];
            for (std::size_t i = 0; i < inputs; ++i) {
                const double x = input[b * inputs + i];
                for (std::size_t j = (o * inputs + i) * size, end = j + size; j < end; ++j)
                    sum += w[j] * wavelet_terms<false>(x, tau[j], scales[j], guard).value;
            }
            output[b * outputs + o] = sum;
        }
}

// With t = w * dpsi/dx per sample and wavelet: dx[b,i] += u t (ascending o,
// then k), dw += u psi, dtau -= u t, dl += u (w dpsi/dl) (ascending b).
void backward(const WaveletEdges& edges, EdgeShape shape, std::span<const double> input, std::size_t batch,
              std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, WaveletGradients& nonlinear) {
    const auto inputs = shape.inputs, outputs = shape.outputs, size = edges.config.size;
    const auto scales = scales_of(edges);
    const FiniteBasisGuard guard;
    const double* w = edges.coefficients.data();
    const double* tau = edges.translations.data();
    double* dw = coefficient_gradient.data();
    double* dtau = nonlinear.translations.data();
    double* dl = nonlinear.log_scales.data();
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t o = 0; o < outputs; ++o) {
            const double u = upstream[b * outputs + o];
            for (std::size_t i = 0; i < inputs; ++i) {
                const double x = input[b * inputs + i];
                double& dx = input_gradient[b * inputs + i];
                for (std::size_t j = (o * inputs + i) * size, end = j + size; j < end; ++j) {
                    const auto r = wavelet_terms<true>(x, tau[j], scales[j], guard);
                    const double t = guard(w[j] * r.input_derivative);
                    dx += u * t;
                    dw[j] += u * r.value;
                    dtau[j] -= u * t;
                    dl[j] += u * guard(w[j] * r.log_scale_derivative);
                }
            }
        }
}

WaveletGradients zero_nonlinear(const WaveletEdges& edges) {
    return {std::vector<double>(edges.translations.size(), 0.0), std::vector<double>(edges.log_scales.size(), 0.0)};
}

void check_gradient(const WaveletEdges& edges, const WaveletGradients& gradient) {
    if (gradient.translations.size() != edges.translations.size() || gradient.log_scales.size() != edges.log_scales.size())
        throw std::invalid_argument("wavelet gradient shape mismatch");
    require_finite(gradient.translations);
    require_finite(gradient.log_scales);
}

// A candidate log scale outside the valid range (exp(+-l) overflowing or
// vanishing) raises std::overflow_error, as an RBF width candidate does.
void update_nonlinear(WaveletEdges& candidate, const WaveletGradients& gradient, double rate) {
    for (std::size_t j = 0; j < candidate.translations.size(); ++j) {
        candidate.translations[j] -= rate * gradient.translations[j];
        candidate.log_scales[j] -= rate * gradient.log_scales[j];
    }
    result_finite(candidate.translations);
    for (double l : candidate.log_scales)
        if (!valid_log_scale(l)) throw std::overflow_error("wavelet log-scale candidate leaves the valid range");
}

void check_result(const WaveletGradients& gradient) {
    result_finite(gradient.translations);
    result_finite(gradient.log_scales);
}

} // namespace kan::detail

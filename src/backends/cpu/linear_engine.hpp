#pragma once

// CPU engine of the carriers that are linear in their coefficients
// (BasisEdges, and the coefficient part of TrainableRbfEdges):
//
//   expansion    Phi: R^I -> R^(I*K), one row per sample, Phi[i*K+k] = Phi_k(x_i)
//   contraction  Y = Phi * C^T + b, C being outputs x (I*K) row-major,
//                i.e. the coefficient layout (o*I + i)*K + k.
//
// Each output starts at its bias and accumulates j = i*K + k in ascending
// order; the VJPs keep the M1 summation order. A GEMM backend (backlog C2)
// replaces `contract`/`contract_vjp` over blocks of rows.

#include "backends/cpu/basis_host.hpp"
#include "core/checks.hpp"
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <vector>

namespace kan::detail {

// Expansion rows of one sample: values, input derivatives and, for trainable
// bases, center and log-width partials, each inputs x terms. Buffers are reused
// across samples; every evaluator writes all its outputs.
class Expansion {
public:
    Expansion(const BasisView& view, std::size_t inputs)
        : view_(view), inputs_(inputs), width_(checked_size(inputs, view.terms)), values(width_), derivatives(width_),
          center_derivatives(view.trainable ? width_ : 0), log_width_derivatives(view.trainable ? width_ : 0) {}
    std::size_t terms() const noexcept { return view_.terms; }
    std::size_t width() const noexcept { return width_; }
    // x: the sample's `inputs` values, finite.
    void expand(const double* x) {
        const auto terms = view_.terms;
        for (std::size_t i = 0; i < inputs_; ++i) {
            const auto row = i * terms;
            basis_terms(view_, x[i],
                        {values.data() + row, derivatives.data() + row,
                         view_.trainable ? center_derivatives.data() + row : nullptr,
                         view_.trainable ? log_width_derivatives.data() + row : nullptr},
                        FiniteBasisGuard{});
        }
    }

private:
    BasisView view_;
    std::size_t inputs_, width_;

public:
    std::vector<double> values, derivatives, center_derivatives, log_width_derivatives;
};

// One sample: output[o] = bias[o] + sum_j coefficients[o*width + j] * phi[j].
inline void contract(const double* coefficients, const double* bias, const double* phi, std::size_t outputs,
                     std::size_t width, double* output) {
    for (std::size_t o = 0; o < outputs; ++o) {
        const double* row = coefficients + o * width;
        double sum = bias[o];
        for (std::size_t j = 0; j < width; ++j) sum += row[j] * phi[j];
        output[o] = sum;
    }
}

// One sample's VJP of the contraction:
//   coefficient_gradient[o,i,k] += upstream[o] * Phi[i,k]
//   input_gradient[i]          += sum_o sum_k (upstream[o] * C[o,i,k]) * Phi'[i,k]
// `nonlinear(k, j, weighted)` receives weighted = upstream[o] * C[o,i,k] for
// each term, j = i*K + k, so a carrier with trainable basis parameters adds
// their VJPs inside the same loop; it is a no-op for linear carriers.
template<class Nonlinear>
void contract_vjp(const double* coefficients, const double* upstream, const Expansion& expansion, std::size_t inputs,
                  std::size_t outputs, double* coefficient_gradient, double* input_gradient, Nonlinear&& nonlinear) {
    const auto terms = expansion.terms();
    const double* phi = expansion.values.data();
    const double* slope = expansion.derivatives.data();
    for (std::size_t i = 0; i < inputs; ++i) {
        const auto row = i * terms;
        double sum = input_gradient[i];
        for (std::size_t o = 0; o < outputs; ++o) {
            const double u = upstream[o];
            const auto edge = (o * inputs + i) * terms;
            for (std::size_t k = 0; k < terms; ++k) {
                coefficient_gradient[edge + k] += u * phi[row + k];
                const double weighted = u * coefficients[edge + k];
                sum += weighted * slope[row + k];
                nonlinear(k, row + k, weighted);
            }
        }
        input_gradient[i] = sum;
    }
}

// Sparse B-spline expansion (backlog C5). Per input only the degree+1 terms
// from first[i] on can be nonzero (spline_span_terms); every other Phi and
// Phi' entry is an exact zero. The sparse contraction and VJP below skip those
// zero terms and keep the order of the remaining ones, so they compute
// bitwise what `contract`/`contract_vjp` compute on the dense rows:
// - A skipped term c*0 (or u*0, (u*c)*0 with finite u*c) is +-0, and
//   s + (+-0) == s for every s except s = -0 plus +0. Every sum here starts at
//   +0 (zero-initialized gradients) or at the bias and only becomes -0 if it
//   starts at -0 and adds only -0, so only a forward output whose bias is -0
//   can differ; such outputs take the dense loop.
// - Dense input VJP terms are (u*c)*Phi': u*c overflowing to +-inf at a zero
//   Phi' gives NaN, which the dense loop propagates (and Layer reports).
//   fl(|u|*max_k |c|) overflows exactly when some fl(|u*c_k|) does (rounding is
//   monotone), so a sample/edge whose bound overflows takes the dense loop.
// Per-input grids (backlog M6): input i reads the knot vector at
// view.knots + i * knot_stride (knot_stride 0: one grid shared by every input).
class SplineExpansion {
public:
    SplineExpansion(const BasisView& view, std::size_t inputs, std::size_t knot_stride = 0)
        : view_(view), inputs_(inputs), width_(view.degree + 1), knot_stride_(knot_stride), first(inputs),
          values(checked_size(inputs, width_)), derivatives(checked_size(inputs, width_)) {}
    std::size_t terms() const noexcept { return view_.terms; }
    std::size_t width() const noexcept { return width_; } // degree + 1
    // x: the sample's `inputs` values, finite. Guards run in the order of
    // Expansion::expand (input by input, the same formula operations).
    void expand(const double* x) {
        for (std::size_t i = 0; i < inputs_; ++i)
            first[i] =
                spline_span_terms(view_.knots + i * knot_stride_, view_.terms, view_.degree, x[i],
                                  values.data() + i * width_, derivatives.data() + i * width_, FiniteBasisGuard{});
    }
    // Dense Phi / Phi' entry k of input i (zero outside the span).
    double value(std::size_t i, std::size_t k) const noexcept { return entry(values, i, k); }
    double derivative(std::size_t i, std::size_t k) const noexcept { return entry(derivatives, i, k); }

private:
    double entry(const std::vector<double>& rows, std::size_t i, std::size_t k) const noexcept {
        return first[i] != view_.terms && k >= first[i] && k - first[i] < width_ ? rows[i * width_ + k - first[i]]
                                                                                 : 0.0;
    }
    BasisView view_;
    std::size_t inputs_, width_, knot_stride_;

public:
    std::vector<std::size_t> first; // first possibly nonzero term, `terms` outside the domain
    std::vector<double> values, derivatives;
};

// Sparse `contract` over a SplineExpansion; bitwise equal (see above).
inline void contract_sparse(const double* coefficients, const double* bias, const SplineExpansion& expansion,
                            std::size_t inputs, std::size_t outputs, double* output) {
    const auto terms = expansion.terms(), width = expansion.width();
    for (std::size_t o = 0; o < outputs; ++o) {
        const double* row = coefficients + o * inputs * terms;
        double sum = bias[o];
        if (sum == 0 && std::signbit(sum)) {
            for (std::size_t i = 0; i < inputs; ++i)
                for (std::size_t k = 0; k < terms; ++k) sum += row[i * terms + k] * expansion.value(i, k);
        } else {
            for (std::size_t i = 0; i < inputs; ++i) {
                const auto first = expansion.first[i];
                if (first == terms) continue;
                const double* c = row + i * terms + first;
                const double* phi = expansion.values.data() + i * width;
                for (std::size_t r = 0; r < width; ++r) sum += c[r] * phi[r];
            }
        }
        output[o] = sum;
    }
}

// Largest coefficient magnitude per edge, (o*I + i), for the overflow bound of
// contract_vjp_sparse.
inline std::vector<double> edge_magnitudes(const double* coefficients, std::size_t edges, std::size_t terms) {
    std::vector<double> result(edges, 0.0);
    for (std::size_t e = 0; e < edges; ++e)
        for (std::size_t k = 0; k < terms; ++k)
            result[e] = std::fmax(result[e], std::fabs(coefficients[e * terms + k]));
    return result;
}

// Sparse `contract_vjp` (no nonlinear partials) over a SplineExpansion;
// bitwise equal (see above). `magnitudes` from edge_magnitudes.
inline void contract_vjp_sparse(const double* coefficients, const double* magnitudes, const double* upstream,
                                const SplineExpansion& expansion, std::size_t inputs, std::size_t outputs,
                                double* coefficient_gradient, double* input_gradient) {
    const auto terms = expansion.terms(), width = expansion.width();
    for (std::size_t i = 0; i < inputs; ++i) {
        const auto first = expansion.first[i];
        double sum = input_gradient[i];
        for (std::size_t o = 0; o < outputs; ++o) {
            const double u = upstream[o];
            const auto edge = (o * inputs + i) * terms;
            if (!(std::fabs(u) * magnitudes[o * inputs + i] <= DBL_MAX)) {
                for (std::size_t k = 0; k < terms; ++k) {
                    coefficient_gradient[edge + k] += u * expansion.value(i, k);
                    sum += (u * coefficients[edge + k]) * expansion.derivative(i, k);
                }
                continue;
            }
            if (first == terms) continue;
            const double* phi = expansion.values.data() + i * width;
            const double* slope = expansion.derivatives.data() + i * width;
            for (std::size_t r = 0; r < width; ++r) {
                coefficient_gradient[edge + first + r] += u * phi[r];
                const double weighted = u * coefficients[edge + first + r];
                sum += weighted * slope[r];
            }
        }
        input_gradient[i] = sum;
    }
}

} // namespace kan::detail

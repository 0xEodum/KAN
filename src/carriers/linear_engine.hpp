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

#include "../detail/basis_host.hpp"
#include "../detail/checks.hpp"
#include <cstddef>
#include <vector>

namespace kan::detail {

// Expansion rows of one sample: values, input derivatives and, for trainable
// bases, center and log-width partials, each inputs x terms. Buffers are reused
// across samples; every evaluator writes all its outputs.
class Expansion {
public:
    Expansion(const BasisView& view, std::size_t inputs)
        : view_(view), inputs_(inputs), width_(checked_size(inputs, view.terms)),
          values(width_), derivatives(width_), center_derivatives(view.trainable ? width_ : 0),
          log_width_derivatives(view.trainable ? width_ : 0) {}
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
inline void contract(const double* coefficients, const double* bias, const double* phi,
                     std::size_t outputs, std::size_t width, double* output) {
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
void contract_vjp(const double* coefficients, const double* upstream, const Expansion& expansion,
                  std::size_t inputs, std::size_t outputs, double* coefficient_gradient,
                  double* input_gradient, Nonlinear&& nonlinear) {
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

} // namespace kan::detail

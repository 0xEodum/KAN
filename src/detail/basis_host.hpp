#pragma once

// CPU-only basis evaluation helpers. Not for nvcc translation units: the guard
// throws, which device instantiations of the shared formulas cannot do.

#include "basis_view.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace kan::detail {

struct FiniteBasisGuard {
    double operator()(double value) const {
        if (!std::isfinite(value)) throw std::overflow_error("basis result is not finite");
        return value;
    }
};

// Repeated evaluation of one validated configuration at finite inputs, reusing
// a single row of buffers. Layer validates once per call instead of per sample.
// Every evaluator writes all outputs, so reuse never exposes stale terms.
class TrustedBasis {
public:
    explicit TrustedBasis(const BasisConfig& config)
        : view_(basis_view(config)), values(view_.terms), derivatives(view_.terms),
          center_derivatives(view_.trainable ? view_.terms : 0),
          log_width_derivatives(view_.trainable ? view_.terms : 0) {}
    std::size_t terms() const noexcept { return view_.terms; }
    void evaluate(double x) {
        basis_terms(view_, x, {values.data(), derivatives.data(),
                               view_.trainable ? center_derivatives.data() : nullptr,
                               view_.trainable ? log_width_derivatives.data() : nullptr},
                    FiniteBasisGuard{});
    }

private:
    BasisView view_;

public:
    std::vector<double> values, derivatives, center_derivatives, log_width_derivatives;
};

} // namespace kan::detail

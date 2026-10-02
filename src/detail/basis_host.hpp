#pragma once

// CPU-only basis evaluation helpers. Not for nvcc translation units: the guard
// throws, which device instantiations of the shared formulas cannot do.

#include "basis_view.hpp"
#include <cmath>
#include <stdexcept>

namespace kan::detail {

struct FiniteBasisGuard {
    double operator()(double value) const {
        if (!std::isfinite(value)) throw std::overflow_error("basis result is not finite");
        return value;
    }
};

} // namespace kan::detail

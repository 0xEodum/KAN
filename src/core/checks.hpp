#pragma once

// CPU-only shape and finiteness checks shared by Layer, the carriers and the
// family operations.

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace kan::detail {

inline std::size_t checked_size(std::size_t left, std::size_t right) {
    const auto max = std::vector<double>().max_size();
    if (right != 0 && left > max / right) throw std::overflow_error("array size overflow");
    return left * right;
}

// Caller-supplied data: std::invalid_argument.
inline void require_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("data must be finite");
}

// Computed results: std::overflow_error.
inline void result_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::overflow_error("nonfinite numerical result");
}

} // namespace kan::detail

#pragma once

// Input-map formulas shared by the CPU InputMap and the resident CUDA
// kernels (single source, as for the basis and rational formulas). Row
// reductions (LayerNorm means) are performed by the callers: sequentially on
// the CPU, by warp shuffles on the device; parity is tolerance based.

#include "host_device.hpp"

namespace kan::detail {

KAN_HOST_DEVICE KAN_FORCE_INLINE double affine_value(double scale, double shift, double x) {
    return scale * x + shift;
}

KAN_HOST_DEVICE KAN_FORCE_INLINE double tanh_value(double scale, double x) {
    return math::tanh(scale * x);
}

// d tanh(s*x)/dx = s / cosh(s*x)^2. Evaluated through cosh instead of
// s*(1 - y*y), which cancels to zero long before the derivative underflows;
// an overflowing cosh gives the exact limit zero.
KAN_HOST_DEVICE KAN_FORCE_INLINE double tanh_derivative(double scale, double x) {
    const double c = math::cosh(scale * x);
    return scale / (c * c);
}

// 1 / sqrt(var + epsilon) from the population variance of a row.
KAN_HOST_DEVICE KAN_FORCE_INLINE double layer_norm_rstd(double variance, double epsilon) {
    return 1.0 / math::sqrt(variance + epsilon);
}

KAN_HOST_DEVICE KAN_FORCE_INLINE double layer_norm_normalized(double x, double mean, double rstd) {
    return (x - mean) * rstd;
}

// Input VJP of xhat = (x - mean) * rstd for a row of n features, with
// w = u * gain (or u without gain), mean_w = sum(w)/n and
// mean_wx = sum(w * xhat)/n:  dx = rstd * ((w - mean_w) - xhat * mean_wx).
KAN_HOST_DEVICE KAN_FORCE_INLINE double layer_norm_input_vjp(double rstd, double weighted, double mean_weighted,
                                                             double normalized, double mean_weighted_normalized) {
    return rstd * ((weighted - mean_weighted) - normalized * mean_weighted_normalized);
}

} // namespace kan::detail

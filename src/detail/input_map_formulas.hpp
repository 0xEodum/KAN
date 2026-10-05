#pragma once

// Input-map formulas shared by the CPU InputMap and the resident CUDA
// kernels (single source, as for the basis and rational formulas). Row
// reductions (LayerNorm means) are performed by the callers: sequentially on
// the CPU, by warp shuffles on the device; parity is tolerance based.
// Templates over the scalar type (backlog C1): double on the CPU and the FP64
// resident executor, float on the FP32 resident executor.

#include "host_device.hpp"

namespace kan::detail {

template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar affine_value(Scalar scale, Scalar shift, Scalar x) {
    return scale * x + shift;
}

template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar tanh_value(Scalar scale, Scalar x) {
    return math::tanh(scale * x);
}

// d tanh(s*x)/dx = s * (1 - y) * (1 + y) from the output y = tanh(s*x).
// The factored form avoids the cancellation of 1 - y*y; its relative error is
// about ulp(1)/(1-|y|), so it degrades only where the derivative is already
// below ~1e-8*s, and a saturated y = +-1 gives the exact limit zero. Using
// the forward output instead of cosh keeps the device VJP at three FP64
// operations (GA102 runs FP64 at 1/64 rate; see M1 profiling evidence).
template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar tanh_derivative(Scalar scale, Scalar y) {
    return scale * ((Scalar(1) - y) * (Scalar(1) + y));
}

// Row means are sum * (1/n) with the reciprocal computed once per map:
// one multiplication instead of a division per row (and per device lane).
template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar layer_norm_mean(Scalar sum, Scalar inverse_count) {
    return sum * inverse_count;
}

// 1 / sqrt(var + epsilon) from the population variance of a row.
template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar layer_norm_rstd(Scalar variance, Scalar epsilon) {
    return Scalar(1) / math::sqrt(variance + epsilon);
}

template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar layer_norm_normalized(Scalar x, Scalar mean, Scalar rstd) {
    return (x - mean) * rstd;
}

// Input VJP of xhat = (x - mean) * rstd for a row of n features, with
// w = u * gain (or u without gain), mean_w = mean(w) and
// mean_wx = mean(w * xhat):  dx = rstd * ((w - mean_w) - xhat * mean_wx).
template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar layer_norm_input_vjp(Scalar rstd, Scalar weighted, Scalar mean_weighted,
                                                             Scalar normalized, Scalar mean_weighted_normalized) {
    return rstd * ((weighted - mean_weighted) - normalized * mean_weighted_normalized);
}

} // namespace kan::detail

#pragma once

// Single source of the SiLU residual-branch formulas (backlog M3) for the CPU
// backend and the resident CUDA kernels:
//     silu(x)  = x * sigma(x),  sigma(x) = 1 / (1 + exp(-x)),
//     silu'(x) = sigma(x) * (1 + x * (1 - sigma(x))).
// Callers validate finite input. Both are formed from e = exp(-|x|) in (0, 1]:
// sigma = 1/(1+e) and 1-sigma = e/(1+e) for x >= 0, sigma = e/(1+e) and
// 1-sigma = 1/(1+e) for x < 0. exp never overflows, the complement is never
// formed by cancellation, |sigma| <= 1 and |x * (1-sigma)| <= |x|, so no
// intermediate is nonfinite for any finite x (Scalar double or float); the
// guard calls document the contract and never fire.
//
// Scalar (backlog C1): double for the CPU and the FP64 resident executor,
// float for the FP32 resident executor. Guard: callable `Scalar(Scalar)`
// returning its argument (see basis_formulas.hpp).

#include "host_device.hpp"

namespace kan::detail {

template<class Scalar>
struct SiluOf {
    Scalar value;      // silu(x)
    Scalar derivative; // silu'(x)
};

// silu(x) alone (forward pass); bitwise equal to silu(x, guard).value.
template<class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar silu_value(Scalar x, const Guard& guard) {
    (void)x; (void)guard;
    return Scalar(0); // RED stub
}

// silu(x) and silu'(x) (backward pass).
template<class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE SiluOf<Scalar> silu(Scalar x, const Guard& guard) {
    (void)x; (void)guard;
    return {Scalar(0), Scalar(0)}; // RED stub
}

} // namespace kan::detail

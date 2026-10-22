#pragma once

// Trigonometric family: Fourier terms cos/sin(k*frequency*x), global support,
// periodic. Shared by the CPU backend and the resident CUDA kernels; callers
// validate configuration and finite input.
//
// Scalar (backlog C1): every formula is a template over the floating-point
// type of its data. The CPU and the FP64 resident executor instantiate double,
// the FP32 resident executor float. Literals are written as Scalar(...) so a
// float instantiation never promotes to double; the double instantiation
// performs exactly the operations it performed before C1.
//
// Guard: callable `Scalar(Scalar)` returning its argument. It is invoked on
// every value the contract requires to be finite; the CPU guard throws, the
// device guard records a status bit and execution continues.

#include "core/host_device.hpp"
#include <cstddef>

namespace kan::detail {

template<class Scalar, class Guard>
KAN_HOST_DEVICE void fourier_terms(Scalar frequency, std::size_t terms, Scalar x, Scalar* v, Scalar* d,
                                   const Guard& guard) {
    v[0] = 1;
    d[0] = 0;
    for (std::size_t k = 1; k <= terms / 2; ++k) {
        const Scalar angular = guard(static_cast<Scalar>(k) * frequency);
        const Scalar phase = guard(angular * x);
        const Scalar cosine = math::cos(phase);
        const Scalar sine = math::sin(phase);
        v[2 * k - 1] = cosine;
        v[2 * k] = sine;
        d[2 * k - 1] = guard(-angular * sine);
        d[2 * k] = guard(angular * cosine);
    }
}

} // namespace kan::detail

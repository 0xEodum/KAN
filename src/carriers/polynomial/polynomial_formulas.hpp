#pragma once

// Orthogonal polynomial families (Chebyshev, Legendre, Jacobi, physicists'
// Hermite): global support, every term evaluated by a three-term recurrence.
// Shared by the CPU backend and the resident CUDA kernels; callers validate
// configuration and finite input.
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

#include "carriers/basis_types.hpp"
#include "core/host_device.hpp"
#include <cstddef>

namespace kan::detail {

// Chebyshev, Legendre, physicists' Hermite and Jacobi three-term recurrences.
template<class Scalar, class Guard>
KAN_HOST_DEVICE void polynomial_terms(BasisKind kind, Scalar alpha, Scalar beta, std::size_t terms,
                                      Scalar x, Scalar* v, Scalar* d, const Guard& guard) {
    constexpr Scalar half = Scalar(0.5);
    v[0] = 1;
    d[0] = 0;
    if (terms == 1) return;
    const bool jacobi = kind == BasisKind::Jacobi;
    // Half-sums avoid overflow in Jacobi's valid, very large parameters.
    const Scalar half_sum = jacobi ? half * alpha + half * beta : Scalar(0);
    // Preserve distance from the admissible boundary alpha,beta > -1.
    // Adding one after summing the parameters can erase that distance.
    const Scalar shifted_half_sum = jacobi ? half * (alpha + 1) + half * (beta + 1) : Scalar(0);
    const Scalar half_difference = jacobi ? half * alpha - half * beta : Scalar(0);
    if (jacobi && (x == -1 || x == 1)) {
        // DLMF 18.6.T1: endpoint rising-factorial values. The general
        // three-term recurrence can cancel a small endpoint against huge terms.
        const Scalar parameter = x == 1 ? alpha : beta;
        Scalar endpoint = 1;
        Scalar shifted_endpoint = 1;
        for (std::size_t k = 1; k < terms; ++k) {
            const Scalar n = static_cast<Scalar>(k);
            endpoint = guard(endpoint * ((parameter + n) / n));
            v[k] = x < 0 && k % 2 ? -endpoint : endpoint;
            if (k > 1) shifted_endpoint = guard(shifted_endpoint * ((parameter + n) / (n - 1)));
            // DLMF 18.9.E15: P'_n = (n+alpha+beta+1)/2 * P_(n-1)^(alpha+1,beta+1).
            // Half-sums keep a finite slope representable even when alpha+beta
            // would overflow.
            const Scalar derivative = guard((shifted_half_sum + half * (n - 1)) * shifted_endpoint);
            d[k] = x < 0 && k % 2 == 0 ? -derivative : derivative;
        }
        return;
    }
    Scalar first_slope = kind == BasisKind::Hermite ? 2 : 1;
    Scalar first_offset = 0;
    if (jacobi) {
        first_slope = shifted_half_sum;
        first_offset = half_difference;
    }
    v[1] = guard(first_slope * x + first_offset);
    d[1] = guard(first_slope);
    for (std::size_t k = 1; k < terms - 1; ++k) {
        const Scalar n = static_cast<Scalar>(k);
        Scalar a = 2, b = 0, c = 1; // Chebyshev
        if (kind == BasisKind::Legendre) {
            a = (2 * n + 1) / (n + 1);
            c = n / (n + 1);
        } else if (kind == BasisKind::Hermite) {
            c = 2 * n;
        } else if (jacobi) {
            // NIST DLMF 18.9.2, rearranged into ratios to avoid squaring
            // large parameters. P1 above handles alpha+beta = -1 or 0.
            const Scalar t = shifted_half_sum + (n - 1);
            const Scalar denominator_half = shifted_half_sum + half * (n - 1);
            a = ((t + half) / (n + 1)) * ((t + 1) / denominator_half);
            b = (half_difference / (n + 1)) * (half_sum / t) * ((t + half) / denominator_half);
            c = half * ((n + alpha) / (n + 1)) * ((n + beta) / t) * ((t + 1) / denominator_half);
        }
        const Scalar factor = a * x + b;
        v[k + 1] = guard(factor * v[k] - c * v[k - 1]);
        d[k + 1] = guard(a * v[k] + factor * d[k] - c * d[k - 1]);
    }
}

} // namespace kan::detail

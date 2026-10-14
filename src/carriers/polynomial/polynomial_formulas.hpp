#pragma once

// Orthogonal polynomial families (Chebyshev, Legendre, Jacobi, physicists'
// Hermite) and the normalized Hermite functions (backlog M5): global support,
// every term evaluated by a three-term recurrence.
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

// Normalized Hermite functions (backlog M5). Per scalar type:
// pi^(-1/4) and sqrt(2); ln 2 split for Cody-Waite reduction (fdlibm's exp/expf
// splits) and 1/ln 2; `band` = B: the recurrence runs on u_n = psi_n
// exp(x^2/2) pi^(1/4) 2^(-B-E) and rescales by 2^-B whenever |u| exceeds 1;
// `tail` is a bound on x^2 beyond which every |psi_n| and |psi_n'| is below half
// the smallest subnormal (see hermite_function_terms).
template<class Scalar> struct HermiteFunctionConstants;
template<> struct HermiteFunctionConstants<double> {
    static constexpr double seed = 0x1.8093870155910p-1, sqrt2 = 0x1.6a09e667f3bcdp+0;
    static constexpr double ln2_hi = 0x1.62e42fee00000p-1, ln2_lo = 0x1.a39ef35793c76p-33;
    static constexpr double log2e = 0x1.71547652b82fep+0, unit = 0x1p-512, tail = 3072;
    static constexpr int band = 512;
};
template<> struct HermiteFunctionConstants<float> {
    static constexpr float seed = 0x1.809388p-1f, sqrt2 = 0x1.6a09e6p+0f;
    static constexpr float ln2_hi = 0x1.62e3p-1f, ln2_lo = 0x1.2fefa2p-17f;
    static constexpr float log2e = 0x1.715476p+0f, unit = 0x1p-64f, tail = 448;
    static constexpr int band = 64;
};

// pi^(-1/4) exp(-(square + square_low)/2) 2^exponent without intermediate
// underflow: half the square is reduced to j ln2 + r, |r| <= ln2/2 (Cody-Waite,
// using the exact low part of x*x), and 2^(exponent - j) is applied exactly.
template<class Scalar>
KAN_HOST_DEVICE inline Scalar hermite_function_weight(Scalar square, Scalar square_low, long long exponent) {
    using C = HermiteFunctionConstants<Scalar>;
    const Scalar half = Scalar(0.5) * square;
    const Scalar j = math::nearbyint(half * C::log2e);
    // fma rounds j*ln2_hi - half once, without a product rounding, for every j
    // (the FP32 split alone makes the product exact only below j = 128, |x| ~ 15).
    const Scalar r = math::fma(j, C::ln2_hi, -half) + (j * C::ln2_lo - Scalar(0.5) * square_low);
    long long shift = exponent - static_cast<long long>(j);
    // Every shift below -1100 gives zero; clamping keeps the int conversion exact.
    if (shift < -1100) shift = -1100;
    return math::ldexp(C::seed * math::exp(r), static_cast<int>(shift));
}

// psi_n(x) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi)) and psi_n'(x) for
// n < terms by the stable recurrence
//   psi_(n+1) = (sqrt(2) x psi_n - sqrt(n) psi_(n-1)) * (1/sqrt(n+1)),
//   psi_n'    = sqrt(2n) psi_(n-1) - x psi_n.
// The reciprocal does not depend on x, so its division leaves the dependent
// chain (CPU training step 1.10x -> 1.03-1.07x of the physicists' cost).
// The recurrence runs on scaled values u_n (seed 2^-B) so that neither the
// weight exp(-x^2/2), which underflows from |x| = 38.6 (FP64) / 14.4 (FP32)
// while psi_n near its turning point sqrt(2n+1) is O(0.1), nor the polynomial
// growth limits the result; psi_n = u_n * weight with weight = pi^(-1/4)
// exp(-x^2/2) 2^(B+E), recomputed (one exp) only when a rescale changes E.
// No intermediate overflows and no inf*0 arises: |u| <= sqrt(2)|x| + 1 before
// a rescale, and x is bounded by the tail test below.
//
// Tail: for x^2 >= 8n, |H_n(x)| <= (2|x|)^n exp(n^2/(4x^2)) and n! >= (n/e)^n
// give |psi_n(x)| <= exp(-x^2/4), and |psi_n'| <= 2|x| exp(-x^2/4); beyond
// x^2 >= max(8 terms, tail) both are below half the smallest subnormal, so the
// row is exactly zero (FP64 |x| >= 55.4, FP32 |x| >= 21.2 for terms <= 384/56).
template<class Scalar, class Guard>
KAN_HOST_DEVICE void hermite_function_terms(std::size_t terms, Scalar x, Scalar* v, Scalar* d, const Guard& guard) {
    using C = HermiteFunctionConstants<Scalar>;
    const Scalar square = x * x;
    const Scalar reach = Scalar(8) * static_cast<Scalar>(terms);
    // Callers pass finite x only; a NaN would also take this branch (zeros).
    if (!(square < (reach > C::tail ? reach : C::tail))) {
        for (std::size_t n = 0; n < terms; ++n) {
            v[n] = 0;
            d[n] = 0;
        }
        return;
    }
    const Scalar square_low = math::fma(x, x, -square); // x*x = square + square_low exactly
    long long exponent = C::band;
    Scalar weight = hermite_function_weight(square, square_low, exponent);
    const Scalar scaled_x = C::sqrt2 * x;
    Scalar previous = 0, current = C::unit, root = 0; // u_(n-1), u_n, sqrt(n)
    for (std::size_t n = 0;; ++n) {
        v[n] = guard(current * weight);
        d[n] = guard((C::sqrt2 * root * previous - x * current) * weight);
        if (n + 1 == terms) return;
        const Scalar next_root = math::sqrt(static_cast<Scalar>(n + 1));
        const Scalar inverse = Scalar(1) / next_root;
        Scalar next = (scaled_x * current - root * previous) * inverse;
        if (math::abs(next) > 1) {
            next = math::ldexp(next, -C::band);
            current = math::ldexp(current, -C::band);
            exponent += C::band;
            weight = hermite_function_weight(square, square_low, exponent);
        }
        previous = current;
        current = next;
        root = next_root;
    }
}

} // namespace kan::detail

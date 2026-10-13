#pragma once

// Locally supported or decaying families: clamped B-splines (compact support,
// degree+1 nonzero terms per input), Gaussian RBFs and Mexican-hat wavelets
// (terms centered on the domain, decaying tails). Shared by the CPU backend and
// the resident CUDA kernels; callers validate configuration and finite input.
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

inline constexpr std::size_t max_spline_degree = 16;

// Overflow in x-center is possible only for opposite signs. Divide before
// subtracting in that case, retaining a wide kernel's tail.
template<class Scalar>
KAN_HOST_DEVICE inline Scalar normalized_distance(Scalar x, Scalar center, Scalar width) {
    const Scalar distance = x - center;
    return math::finite(distance) ? distance / width : x / width - center / width;
}

// Ratios on a huge domain remain finite even when its length overflows.
template<class Scalar>
KAN_HOST_DEVICE inline Scalar interval_ratio(Scalar numerator_right, Scalar numerator_left,
                                             Scalar right, Scalar left) {
    const Scalar denominator = right - left;
    if (denominator == 0) return 0;
    if (math::finite(denominator)) return (numerator_right - numerator_left) / denominator;
    return (Scalar(0.5) * numerator_right - Scalar(0.5) * numerator_left) /
           (Scalar(0.5) * right - Scalar(0.5) * left);
}

template<class Scalar>
KAN_HOST_DEVICE inline Scalar spline_slope_term(Scalar value, std::size_t degree, Scalar right,
                                                Scalar left) {
    if (value == 0 || right == left) return 0;
    const Scalar denominator = right - left;
    const Scalar p = static_cast<Scalar>(degree);
    return math::finite(denominator) ? (p * value) / denominator
                                     : (Scalar(0.5) * p * value) / (Scalar(0.5) * right - Scalar(0.5) * left);
}

// Clamped B-spline by Cox-de Boor on the single active span. Only degree+1
// terms can be nonzero, so fixed scratch covers every admissible degree.
//
// Dense = false is the sparse form (backlog C5): returns the index `first` =
// span - degree of the first of the degree+1 terms that can be nonzero and
// writes their values and input derivatives to v[0..degree] and d[0..degree]
// (term first + r at r); every other term is exactly zero. Inputs outside the
// domain return `terms` and write nothing: all terms are zero. Dense = true
// writes all `terms` values and derivatives (zero outside the span) in place,
// the M3 evaluation; both forms perform the same operations.
template<bool Dense, class Scalar, class Guard>
KAN_HOST_DEVICE std::size_t spline_evaluate(const Scalar* t, std::size_t terms, std::size_t degree, Scalar x,
                                            Scalar* v, Scalar* d, const Guard& guard) {
    if constexpr (Dense) {
        for (std::size_t k = 0; k < terms; ++k) {
            v[k] = 0;
            d[k] = 0;
        }
    }
    if (x < t[degree] || x > t[terms]) return terms;
    // Seed the last inward span at the upper end; the search selects the
    // right side of repeated interior knots.
    std::size_t span = terms - 1;
    if (x != t[terms]) {
        std::size_t lo = degree, hi = terms;
        while (lo < hi) {
            const auto mid = lo + (hi - lo) / 2;
            if (t[mid] <= x) lo = mid + 1;
            else hi = mid;
        }
        span = lo - 1;
    }
    Scalar lower[max_spline_degree + 2] = {};
    Scalar next[max_spline_degree + 2] = {};
    lower[degree] = 1;
    const auto start = span - degree;
    const auto out = Dense ? start : 0; // output index of term start
    if constexpr (!Dense) {
        if (degree == 0) d[0] = 0; // degree-zero derivatives are zero
    }
    for (std::size_t p = 1; p <= degree; ++p) {
        for (std::size_t r = degree - p; r <= degree; ++r) {
            const auto i = start + r;
            Scalar value = 0;
            if (lower[r] != 0) value += interval_ratio(x, t[i], t[i + p], t[i]) * lower[r];
            if (lower[r + 1] != 0)
                value += interval_ratio(t[i + p + 1], x, t[i + p + 1], t[i + 1]) * lower[r + 1];
            next[r] = guard(value);
            if (p == degree)
                d[out + r] = guard(spline_slope_term(lower[r], p, t[i + p], t[i]) -
                                   spline_slope_term(lower[r + 1], p, t[i + p + 1], t[i + 1]));
        }
        for (std::size_t r = degree - p; r <= degree; ++r) lower[r] = next[r];
    }
    for (std::size_t r = 0; r <= degree; ++r) v[out + r] = lower[r];
    return start;
}
template<class Scalar, class Guard>
KAN_HOST_DEVICE std::size_t spline_span_terms(const Scalar* t, std::size_t terms, std::size_t degree, Scalar x,
                                              Scalar* v, Scalar* d, const Guard& guard) {
    return spline_evaluate<false>(t, terms, degree, x, v, d, guard);
}
template<class Scalar, class Guard>
KAN_HOST_DEVICE void spline_terms(const Scalar* t, std::size_t terms, std::size_t degree, Scalar x,
                                  Scalar* v, Scalar* d, const Guard& guard) {
    spline_evaluate<true>(t, terms, degree, x, v, d, guard);
}

// log(2/sqrt(3)) - log(pi)/4 per scalar type, as literals so device code does
// not evaluate three transcendental functions per input.
template<class Scalar> inline constexpr Scalar mexican_hat_log_normalization = Scalar(-0x1.2383e809a67e3p-3);

// L2-normalized Mexican hat (Ricker) wavelets, evaluated in log space.
template<class Scalar, class Guard>
KAN_HOST_DEVICE void mexican_hat_terms(const Scalar* centers, const Scalar* scales,
                                       std::size_t terms, Scalar x, Scalar* v, Scalar* d,
                                       const Guard& guard) {
    constexpr Scalar log_normalization = mexican_hat_log_normalization<Scalar>;
    for (std::size_t k = 0; k < terms; ++k) {
        v[k] = 0;
        d[k] = 0;
        const Scalar scale = scales[k];
        const Scalar q = normalized_distance(x, centers[k], scale), q2 = q * q;
        // An infinite distance is an exact zero tail, not an inf*zero NaN.
        if (!math::finite(q2)) continue;
        const Scalar log_scale = math::log(scale);
        const Scalar envelope = log_normalization - Scalar(0.5) * log_scale - Scalar(0.5) * q2;
        if (q2 != 1)
            v[k] = guard(math::copysign(math::exp(envelope + math::log(math::abs(1 - q2))), 1 - q2));
        if (q != 0 && q2 != 3)
            d[k] = guard(math::copysign(
                math::exp(envelope + math::log(math::abs(q)) + math::log(math::abs(q2 - 3)) - log_scale),
                q * Scalar(q2 > 3 ? 1 : -1)));
    }
}

// Gaussian RBF exp(-((x-c)/w)^2). With trainable parameters w = exp(log_width)
// and the nonlinear center/log-width derivatives are produced when requested.
template<class Scalar, class Guard>
KAN_HOST_DEVICE void gaussian_terms(const BasisViewOf<Scalar>& basis, Scalar x, const BasisRowOf<Scalar>& row,
                                    const Guard& guard) {
    constexpr Scalar min_normal = math::constants<Scalar>::min_normal, ln2 = math::constants<Scalar>::ln2;
    for (std::size_t k = 0; k < basis.terms; ++k) {
        const Scalar width = basis.trainable ? math::exp(basis.log_widths[k]) : basis.width;
        const Scalar q = normalized_distance(x, basis.centers[k], width);
        const Scalar value = math::exp(-q * q);
        row.values[k] = value;
        Scalar derivative = 0;
        if (q != 0 && math::finite(q)) {
            if (value < min_normal) {
                // A tiny width can amplify a value that underflows to zero
                // into a representable derivative. Preserve it in log space.
                const Scalar log_magnitude = ln2 + math::log(math::abs(q)) - q * q - math::log(width);
                derivative = -math::copysign(math::exp(log_magnitude), q);
            } else {
                derivative = (-2 * q * value) / width;
            }
        }
        row.derivatives[k] = guard(derivative);
        if (row.center_derivatives) row.center_derivatives[k] = -derivative;
        if (row.log_width_derivatives) {
            // Log space avoids q*q * zero in remote tails.
            row.log_width_derivatives[k] =
                q != 0 && math::finite(q) ? guard(math::exp(ln2 + 2 * math::log(math::abs(q)) - q * q)) : Scalar(0);
        }
    }
}

} // namespace kan::detail

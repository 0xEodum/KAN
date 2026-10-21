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
#include <type_traits>

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
//
// Lower = true (sparse form only, backlog C13) writes, instead of the input
// derivatives, the degree nonzero degree-(degree-1) values the derivative is
// built from: d[r] = B_{first+1+r, degree-1}(x), r < degree (nothing for degree
// zero). They are the Cox-de Boor values at the start of the last step, so the
// span, the side taken at repeated knots and the right endpoint are those of
// the values. The derivative expressions are still evaluated and guarded (the
// status the values' derivatives report is unchanged) but not written.
template<bool Dense, bool Lower = false, class Scalar, class Guard>
KAN_HOST_DEVICE std::size_t spline_evaluate(const Scalar* t, std::size_t terms, std::size_t degree, Scalar x,
                                            Scalar* v, Scalar* d, const Guard& guard) {
    static_assert(!(Dense && Lower), "the lower-degree values have the sparse form only");
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
    if constexpr (!Dense && !Lower) {
        if (degree == 0) d[0] = 0; // degree-zero derivatives are zero
    }
    for (std::size_t p = 1; p <= degree; ++p) {
        if constexpr (Lower) {
            if (p == degree)
                for (std::size_t r = 0; r < degree; ++r) d[r] = lower[r + 1];
        }
        for (std::size_t r = degree - p; r <= degree; ++r) {
            const auto i = start + r;
            Scalar value = 0;
            if (lower[r] != 0) value += interval_ratio(x, t[i], t[i + p], t[i]) * lower[r];
            if (lower[r + 1] != 0)
                value += interval_ratio(t[i + p + 1], x, t[i + p + 1], t[i + 1]) * lower[r + 1];
            next[r] = guard(value);
            if (p == degree) {
                const Scalar derivative = guard(spline_slope_term(lower[r], p, t[i + p], t[i]) -
                                                spline_slope_term(lower[r + 1], p, t[i + p + 1], t[i + 1]));
                if constexpr (!Lower) d[out + r] = derivative;
            }
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
// The values and (backlog C13) the degree nonzero degree-(degree-1) values of
// the derivative's spline (see spline_evaluate, Lower); returns `first` as
// spline_span_terms does (`terms` outside the domain, nothing written).
template<class Scalar, class Guard>
KAN_HOST_DEVICE std::size_t spline_lower_terms(const Scalar* t, std::size_t terms, std::size_t degree, Scalar x,
                                               Scalar* v, Scalar* lower, const Guard& guard) {
    return spline_evaluate<false, true>(t, terms, degree, x, v, lower, guard);
}
// Derivative coefficients (backlog C13). The derivative of a clamped spline is
// a spline of one degree less on the same knots,
//   d/dx sum_k c_k B_{k,p}(x) = sum_{j=1}^{K-1} C*_j B_{j,p-1}(x),
//   C*_j = (c_j - c_{j-1}) s_j,  s_j = p / (t_{j+p} - t_j),
// (B_{0,p-1} and B_{K,p-1} vanish on clamped knots; a zero knot gap gives a
// vanishing B_{j,p-1} and s_j = 0), the regrouping of terms that
// spline_evaluate's derivative expressions perform per input. Its weights are
// nonnegative and sum to one at every input of the domain, so rounding of C*
// or of products with C* is not amplified, unlike the signed derivatives
// B'_{k,p}, of order p/(knot spacing), which sum to zero. The knot scales s_j
// depend on the knots only (spline_slope_term: zero gap, overflow-safe division).
template<class Scalar>
KAN_HOST_DEVICE inline Scalar spline_derivative_scale(std::size_t degree, Scalar right, Scalar left) {
    return spline_slope_term(Scalar(1), degree, right, left);
}
// A difference beyond the range (coefficients of opposite sign near the
// largest value) is halved first, so C* overflows only when its value does.
template<class Scalar>
KAN_HOST_DEVICE inline Scalar spline_derivative_coefficient(Scalar previous, Scalar current, Scalar scale) {
    const Scalar difference = current - previous;
    if (math::finite(difference)) return difference * scale;
    return (Scalar(0.5) * current - Scalar(0.5) * previous) * scale * Scalar(2);
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

// Per-edge trainable Mexican hat (backlog M7): psi = a (1 - q^2) exp(-q^2/2),
// q = (x - tau) r, with the inverse scale r = exp(-l) and the amplitude
// a = A sqrt(r) = A s^(-1/2) (A = 2/(sqrt(3) pi^(1/4)), correctly rounded per
// scalar type) of log scale l. Both depend on the wavelet's parameters only:
// callers form them once per wavelet (wavelet_scales), not per sample, so a
// term costs one exp.
template<class Scalar> inline constexpr Scalar mexican_hat_amplitude = Scalar(0x1.bc12081a51aa9p-1);
template<class Scalar> inline constexpr Scalar wavelet_root_high = Scalar(0x1.ace664ccfff80p+1); // (7 + sqrt(41))/4
template<class Scalar> inline constexpr Scalar wavelet_root_low = Scalar(0x1.3199b330007ffp-3);  // (7 - sqrt(41))/4
template<class Scalar> struct WaveletScales {
    Scalar inverse, amplitude;
};
template<class Scalar>
KAN_HOST_DEVICE inline WaveletScales<Scalar> wavelet_scales(Scalar log_scale) {
    const Scalar inverse = math::exp(-log_scale);
    return {inverse, mexican_hat_amplitude<Scalar> * math::sqrt(inverse)};
}
// One wavelet with unit coefficient: the value psi (= d phi/dw), the input
// derivative d psi/dx = a r q (q^2 - 3) E (d/dtau is its negative) and the
// log-scale derivative d psi/dl = a E (-q^4 + 3.5 q^2 - 0.5)
// = -a E (q^2 - r1)(q^2 - r2), E = exp(-q^2/2), with the roots
// r1,2 = (7 +- sqrt(41))/4 (correctly rounded): the factored form keeps the
// relative accuracy near the roots that the expanded polynomial loses.
// Derivatives = false computes the value only (forward). The products are
// ordered so that no intermediate exceeds the result's magnitude: the
// polynomial factors times E are bounded by 2, so a and r multiply last.
// Where E is below the normal range (q^2 > 1416 in FP64, > 175 in FP32; the
// signs are then known: 1 - q^2 < 0, q^2 - 3 > 0, (q^2 - r1)(q^2 - r2) > 0)
// all three are formed in log space, which keeps the derivatives of a narrow
// wavelet (large r) that an underflowing E would lose, as the M3 Mexican hat
// and Gaussian do; an infinite q^2 is the exact zero tail.
template<class Scalar> struct WaveletTerms {
    Scalar value, input_derivative, log_scale_derivative;
};
// E = exp(-q^2/2) on the direct path. FP32 (the resident executor only):
// 2^(c q^2) with c = -log2(e)/2 rounded to float, by one MUFU.EX2 that flushes
// below the normal range, exactly where the log-space branch takes over; the
// rounding of c q^2 adds at most |c q^2| 2^-24 (< 8e-6 for the normal range)
// to the exponent. FP64 (CPU, FP64 resident): exp, the parity formula.
template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar wavelet_envelope(Scalar q2) {
    if constexpr (std::is_same_v<Scalar, float>) return math::exp2_flush(q2 * -0x1.715476p-1f);
    else return math::exp(Scalar(-0.5) * q2);
}
// The rare branches of wavelet_terms, out of line (KAN_NOINLINE) so that the
// unrolled kernel loops keep only the direct path: inlined, the log-space
// code (four logarithms and three exponentials) multiplied every unrolled
// evaluation into kernels of about 130 KB that thrashed the instruction cache.
// q2 = q*q is not finite (an overflowed distance x - tau is retried as
// x r - tau r) or exp(-q^2/2) is below the normal range (log space).
template<bool Derivatives, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_NOINLINE WaveletTerms<Scalar> wavelet_tail_terms(Scalar x, Scalar translation, WaveletScales<Scalar> s,
                                                                     Scalar q, Scalar q2, const Guard& guard) {
    constexpr Scalar min_normal = math::constants<Scalar>::min_normal;
    WaveletTerms<Scalar> r{0, 0, 0};
    if (!math::finite(q2)) {
        if (math::finite(x - translation)) return r; // the exact zero tail
        q = x * s.inverse - translation * s.inverse;
        q2 = q * q;
        if (!math::finite(q2)) return r;
        const Scalar e = wavelet_envelope(q2);
        if (e >= min_normal) {
            r.value = guard(s.amplitude * ((1 - q2) * e));
            if constexpr (Derivatives) {
                r.input_derivative = guard((s.amplitude * (q * (q2 - 3) * e)) * s.inverse);
                r.log_scale_derivative = guard(-(s.amplitude * ((q2 - wavelet_root_high<Scalar>) * (q2 - wavelet_root_low<Scalar>) * e)));
            }
            return r;
        }
    }
    const Scalar envelope = math::log(s.amplitude) - Scalar(0.5) * q2;
    r.value = guard(-math::exp(envelope + math::log(q2 - 1)));
    if constexpr (Derivatives) {
        r.input_derivative = guard(math::copysign(
            math::exp(envelope + math::log(math::abs(q)) + math::log(q2 - 3) + math::log(s.inverse)), q));
        r.log_scale_derivative = guard(
            -math::exp(envelope + math::log(q2 - wavelet_root_high<Scalar>) + math::log(q2 - wavelet_root_low<Scalar>)));
    }
    return r;
}
// The direct path from q, q^2 and E = wavelet_envelope(q^2) >= min_normal.
template<bool Derivatives, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE WaveletTerms<Scalar> wavelet_direct_terms(Scalar q, Scalar q2, Scalar e, WaveletScales<Scalar> s,
                                                                           const Guard& guard) {
    WaveletTerms<Scalar> r;
    r.value = guard(s.amplitude * ((1 - q2) * e));
    r.input_derivative = r.log_scale_derivative = 0;
    if constexpr (Derivatives) {
        r.input_derivative = guard((s.amplitude * (q * (q2 - 3) * e)) * s.inverse);
        r.log_scale_derivative = guard(-(s.amplitude * ((q2 - wavelet_root_high<Scalar>) * (q2 - wavelet_root_low<Scalar>) * e)));
    }
    return r;
}
template<bool Derivatives, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE WaveletTerms<Scalar> wavelet_terms(Scalar x, Scalar translation, WaveletScales<Scalar> s,
                                                                    const Guard& guard) {
    constexpr Scalar min_normal = math::constants<Scalar>::min_normal;
    const Scalar q = (x - translation) * s.inverse, q2 = q * q;
    const Scalar e = wavelet_envelope(q2);
    // A nonfinite q2 gives e = 0: one test selects both rare branches.
    if (!(e >= min_normal)) return wavelet_tail_terms<Derivatives>(x, translation, s, q, q2, guard);
    return wavelet_direct_terms<Derivatives>(q, q2, e, s, guard);
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

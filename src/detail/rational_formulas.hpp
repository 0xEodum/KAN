#pragma once

// Single source of the rational-edge formulas for the CPU backend and the
// resident CUDA kernels: r(x) = P(z)/Q(z), z = (x-center)/scale. The
// denominator follows a DenominatorPolicy fixed at compile time, with
// S(z) = sum_{k>=1} b_k z^k and the gain g = dQ/dS:
//   Guarded  Q = 1 + S,   g = 1        (relative pole guard)
//   Absolute Q = 1 + |S|, g = sign(S)  (subgradient sign(0) = 0)
//   Smooth   Q = 1 + S^2, g = 2S
// Q' = g S', dr/da_k = z^k/Q, dr/db_k = -r g z^k/Q, dr/dx = (P' - r Q')/(Q scale).
// Callers validate configuration and finite data.
//
// Scalar (backlog C1): the formulas are templates over the floating-point
// type of the data (double: CPU and FP64 resident; float: FP32 resident) and
// over the configuration type, which provides numerator_degree,
// denominator_degree, center, scale and epsilon (RationalConfig on the CPU,
// RationalScalars<Scalar> on the device). Configuration scalars are converted
// to Scalar where they are used; for double that conversion is the identity.
//
// Guard: callable `Scalar(Scalar)` returning its argument, invoked on every
// intermediate the contract requires to be finite (see basis_formulas.hpp).

#include "host_device.hpp"
#include "kan/rational.hpp"
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace kan::detail {

template<DenominatorPolicy Policy>
using PolicyConstant = std::integral_constant<DenominatorPolicy, Policy>;

// Host-side dispatch of a validated runtime policy to a compile-time one:
// f(PolicyConstant<P>{}) is called once per operation, not per sample.
template<class F>
decltype(auto) visit_denominator_policy(DenominatorPolicy policy, F&& f) {
    switch (policy) {
    case DenominatorPolicy::Guarded: return f(PolicyConstant<DenominatorPolicy::Guarded>{});
    case DenominatorPolicy::Absolute: return f(PolicyConstant<DenominatorPolicy::Absolute>{});
    case DenominatorPolicy::Smooth: return f(PolicyConstant<DenominatorPolicy::Smooth>{});
    }
    throw std::logic_error("unvalidated rational denominator policy");
}

// Device-side rational configuration in the executor's precision.
template<class Scalar>
struct RationalScalars {
    std::size_t numerator_degree, denominator_degree;
    Scalar center, scale, epsilon;
};

template<class Scalar>
struct RationalHornerOf {
    Scalar z;
    Scalar p, dp; // numerator and its z-derivative
    Scalar q, dq; // denominator Q and its z-derivative Q' = g S'
    Scalar bound; // Guarded: sum |b_k| |z|^k + 1, the pole guard reference magnitude
    Scalar ds;    // safe policies: S'
    Scalar gain;  // safe policies: g = dQ/dS (Guarded: unused and 0; its g = 1 is implicit)
};
using RationalHorner = RationalHornerOf<double>;

template<class Scalar>
struct RationalEdgeOf {
    Scalar value;
    Scalar input_derivative;
};
using RationalEdge = RationalEdgeOf<double>;

// a*b rounded on its own in every build: an FMA (KAN_CUDA_FMA=ON) build may
// otherwise contract the guarded product into a following addition, so that
// the value used differs from the value checked (and from the CPU's).
template<class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar rounded_product(Scalar a, Scalar b) {
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<Scalar, double>) return __dmul_rn(a, b);
    else return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

template<class Config, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar rational_argument(const Config& c, Scalar x, const Guard& guard) {
    return guard(guard(x - static_cast<Scalar>(c.center)) / static_cast<Scalar>(c.scale));
}

// Horner evaluation of P, P', Q, Q' (and the guard bound or S', g) at the
// argument z = rational_argument(c, x). `numerator` holds numerator_degree+1
// coefficients, `denominator` holds denominator_degree.
template<DenominatorPolicy Policy, class Config, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE RationalHornerOf<Scalar> rational_horner_at(const Config& c, Scalar z,
                                                  const Scalar* numerator,
                                                  const Scalar* denominator, const Guard& guard) {
    RationalHornerOf<Scalar> h{};
    h.z = z;
    const auto m = c.numerator_degree, n = c.denominator_degree;
    // Each Horner chain is guarded once, at its end: x*z + c of a nonfinite x
    // is nonfinite (inf*0 and inf-inf are NaN), so a chain whose intermediate
    // overflows ends nonfinite and the guard reports exactly what guarding
    // every step would (backlog C7 profiling: a guard per step was a large
    // share of the forward pass's instructions).
    h.p = numerator[m];
    for (std::size_t k = m; k > 0; --k) {
        h.dp = h.dp * h.z + h.p;
        h.p = h.p * h.z + numerator[k - 1];
    }
    guard(h.p); guard(h.dp);
    if constexpr (Policy == DenominatorPolicy::Guarded) {
        h.q = n ? denominator[n - 1] : Scalar(1);
        h.bound = n ? math::abs(denominator[n - 1]) : Scalar(1);
        for (std::size_t k = n; k > 0; --k) {
            const Scalar next = k == 1 ? Scalar(1) : denominator[k - 2];
            h.dq = h.dq * h.z + h.q;
            h.q = h.q * h.z + next;
            h.bound = h.bound * math::abs(h.z) + math::abs(next);
        }
        guard(h.q); guard(h.dq); guard(h.bound);
    } else {
        // S and S' by Horner without the constant term, then Q = 1 + f(S) >= 1.
        Scalar s = n ? denominator[n - 1] : Scalar(0);
        for (std::size_t k = n; k > 0; --k) {
            h.ds = h.ds * h.z + s;
            s = k == 1 ? s * h.z : s * h.z + denominator[k - 2];
        }
        guard(s); guard(h.ds);
        if constexpr (Policy == DenominatorPolicy::Absolute) {
            h.gain = s > 0 ? Scalar(1) : s < 0 ? Scalar(-1) : Scalar(0);
            h.q = guard(1 + math::abs(s));
        } else {
            h.gain = guard(2 * s);
            h.q = guard(1 + guard(rounded_product(s, s)));
        }
        h.dq = guard(h.gain * h.ds);
    }
    return h;
}
// The same at the input x.
template<DenominatorPolicy Policy, class Config, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE RationalHornerOf<Scalar> rational_horner(const Config& c, Scalar x,
                                               const Scalar* numerator,
                                               const Scalar* denominator, const Guard& guard) {
    return rational_horner_at<Policy>(c, rational_argument(c, x, guard), numerator, denominator, guard);
}

// Guarded policy: |Q| <= epsilon * bound is an unsafe pole. The safe policies
// have Q >= 1 and never report one.
template<DenominatorPolicy Policy, class Config, class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_pole(const Config& c, const RationalHornerOf<Scalar>& h) {
    if constexpr (Policy == DenominatorPolicy::Guarded)
        return math::finite(h.q) && math::finite(h.bound) &&
               math::abs(h.q) <= static_cast<Scalar>(c.epsilon) * h.bound;
    else
        return false;
}

// Q' for the log-space input derivative. Guarded and Absolute form Q'
// exactly (g = 1, or g in {-1, 0, 1}); Smooth uses log|g| + log|S'| because
// Q' = 2 S S' may underflow although r Q'/Q is representable.
template<DenominatorPolicy Policy, class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_slope_nonzero(const RationalHornerOf<Scalar>& h) {
    if constexpr (Policy == DenominatorPolicy::Smooth) return h.gain != 0 && h.ds != 0;
    else return h.dq != 0;
}
template<DenominatorPolicy Policy, class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar rational_log_slope(const RationalHornerOf<Scalar>& h) {
    if constexpr (Policy == DenominatorPolicy::Smooth) return math::log(math::abs(h.gain)) + math::log(math::abs(h.ds));
    else return math::log(math::abs(h.dq));
}
template<DenominatorPolicy Policy, class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_slope_negative(const RationalHornerOf<Scalar>& h) {
    if constexpr (Policy == DenominatorPolicy::Smooth) return math::signbit(h.gain) != math::signbit(h.ds);
    else return math::signbit(h.dq);
}

template<class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar rational_signed_exp(Scalar exponent, bool negative, const Guard& guard) {
    return math::copysign(guard(math::exp(exponent)), negative ? Scalar(-1) : Scalar(1));
}

// Value and dr/dx. When an intermediate quotient or product underflows, the
// representable final derivative is restored in log space; the ordinary
// Horner/quotient path is unchanged.
template<DenominatorPolicy Policy, class Config, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE RationalEdgeOf<Scalar> rational_edge(const Config& c, const RationalHornerOf<Scalar>& h,
                                           const Guard& guard) {
    const Scalar scale = static_cast<Scalar>(c.scale);
    RationalEdgeOf<Scalar> e{};
    e.value = guard(h.p / h.q);
    const Scalar numerator_term = guard(h.dp / h.q), denominator_ratio = guard(h.dq / h.q);
    const Scalar denominator_term = guard(e.value * denominator_ratio);
    e.input_derivative = guard(guard(numerator_term - denominator_term) / scale);
    const bool tiny_numerator = h.dp != 0 && math::tiny(numerator_term);
    const bool tiny_denominator = h.p != 0 && rational_slope_nonzero<Policy>(h) &&
        (math::tiny(e.value) || math::tiny(denominator_ratio) || math::tiny(denominator_term));
    if (tiny_numerator || tiny_denominator) {
        const Scalar lq = math::log(math::abs(h.q)), ls = math::log(scale);
        const Scalar first = tiny_numerator
            ? rational_signed_exp(math::log(math::abs(h.dp)) - lq - ls,
                                  math::signbit(h.dp) != math::signbit(h.q), guard)
            : guard(numerator_term / scale);
        const Scalar second = tiny_denominator
            ? rational_signed_exp(math::log(math::abs(h.p)) + rational_log_slope<Policy>(h) - 2 * lq - ls,
                                  math::signbit(h.p) != rational_slope_negative<Policy>(h), guard)
            : guard(denominator_term / scale);
        e.input_derivative = guard(first - second);
    }
    return e;
}

// Parameter VJPs for one power k. Callers pass power = z^k and the guarded
// quotient divided = z^k/Q once, shared by dr/da_k and dr/db_k.

// dr/da_k = z^k / Q (every policy).
template<class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar rational_numerator_vjp(Scalar q, Scalar z, std::size_t k, Scalar power,
                                              Scalar divided, const Guard& guard) {
    if (z != 0 && (math::tiny(power) || math::tiny(divided)))
        return rational_signed_exp(static_cast<Scalar>(k) * math::log(math::abs(z)) - math::log(math::abs(q)),
                                   math::signbit(q) != (math::signbit(z) && k % 2 != 0), guard);
    return divided;
}

// dr/db_k = -r g z^k / Q for k >= 1, with value r = P/Q and gain g = dQ/dS
// (ignored by Guarded, where g = 1).
template<DenominatorPolicy Policy, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE Scalar rational_denominator_vjp(Scalar p, Scalar q, Scalar value, Scalar gain,
                                                Scalar z, std::size_t k, Scalar power, Scalar divided,
                                                const Guard& guard) {
    if constexpr (Policy == DenominatorPolicy::Guarded) {
        const Scalar derivative = guard(-value * divided);
        if (p != 0 && z != 0 && (math::tiny(power) || math::tiny(divided) || math::tiny(value)))
            return rational_signed_exp(math::log(math::abs(p)) + static_cast<Scalar>(k) * math::log(math::abs(z)) -
                                           2 * math::log(math::abs(q)),
                                       math::signbit(p) == (math::signbit(z) && k % 2 != 0), guard);
        return derivative;
    } else if constexpr (Policy == DenominatorPolicy::Absolute) {
        // g in {-1, 0, 1}: the guarded-form derivative times g, exactly.
        return gain * rational_denominator_vjp<DenominatorPolicy::Guarded>(p, q, value, gain, z, k, power,
                                                                           divided, guard);
    } else {
        const Scalar scaled = guard(gain * divided), derivative = guard(-value * scaled);
        if (p != 0 && z != 0 && gain != 0 &&
            (math::tiny(power) || math::tiny(divided) || math::tiny(value) || math::tiny(gain) || math::tiny(scaled)))
            return rational_signed_exp(math::log(math::abs(p)) + math::log(math::abs(gain)) +
                                           static_cast<Scalar>(k) * math::log(math::abs(z)) -
                                           2 * math::log(math::abs(q)),
                                       (math::signbit(p) != (math::signbit(z) && k % 2 != 0)) == math::signbit(gain),
                                       guard);
        return derivative;
    }
}

// Every parameter-VJP intermediate of one executed sample: z^k, z^k/Q, dr/da_k
// and dr/db_k for k <= max(m, n), each passed through the guard (value = P/Q).
// The CPU computes them as results (evaluate_rational); the resident forward
// pass checks them so that its backward pass may recompute them unguarded.
template<DenominatorPolicy Policy, class Config, class Scalar, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE void rational_parameter_vjps_check(const Config& c, const RationalHornerOf<Scalar>& h,
                                                   Scalar value, const Guard& guard) {
    const auto m = c.numerator_degree, n = c.denominator_degree;
    Scalar power = 1;
    for (std::size_t k = 0; k <= (m > n ? m : n); ++k) {
        if (k) power = guard(power * h.z);
        const Scalar divided = guard(power / h.q);
        if (k <= m) rational_numerator_vjp(h.q, h.z, k, power, divided, guard);
        if (k && k <= n) rational_denominator_vjp<Policy>(h.p, h.q, value, h.gain, h.z, k, power, divided, guard);
    }
}

// Backlog C7: a cheap sufficient condition for rational_parameter_vjps_check
// to find every intermediate finite: one power, at most one division and two
// products instead of max(m, n)+1 divisions and their log-space tests. Where
// it holds, the check cannot report anything and is skipped; elsewhere (only
// within a factor 4 of the overflow threshold) the check itself decides, so
// the reported status is exactly the check's. With IEEE round to nearest
// (monotone and sign-symmetric) and limit = max/4:
// - the check's |z^k| (repeated multiplication) is nondecreasing in k for
//   |z| >= 1 and at most 1 for |z| < 1: `power` is its maximum for k <= max(m, n);
// - |z^k/Q| <= d = fl(power/|Q|) (d = power for |Q| >= 1), |r z^k/Q| <= fl(|r| d);
//   Smooth: |g z^k/Q| <= s = fl(|g| d) and |r g z^k/Q| <= fl(|r| s); Absolute
//   evaluates the Guarded form first and multiplies by g in {-1, 0, 1};
// - the log-space paths run only when one factor is below the normal range;
//   they then equal the exact product of the same factors up to a relative
//   error far below 4, which is at most a few units: finite. The numerator
//   path is at most z^k/Q with |z^k| tiny and Q nonzero (a pole is reported
//   before): at most 2^52 (FP64) or 2^23 (FP32).
// The proof and its test cases are in docs/evidence/backlog/C6-C8.md.
template<class Scalar> inline constexpr Scalar rational_vjp_limit = Scalar(0x1.fffffffffffffp+1021); // DBL_MAX/4
template<> inline constexpr float rational_vjp_limit<float> = 0x1.fffffep+125f;                       // FLT_MAX/4
template<DenominatorPolicy Policy, class Config, class Scalar>
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_parameter_vjps_bounded(const Config& c, const RationalHornerOf<Scalar>& h,
                                                     Scalar value) {
    constexpr Scalar limit = rational_vjp_limit<Scalar>;
    const auto m = c.numerator_degree, n = c.denominator_degree, degree = m > n ? m : n;
    const Scalar magnitude = math::abs(h.z), q = math::abs(h.q);
    Scalar power = 1;
    if (magnitude > 1)
        for (std::size_t k = 0; k < degree; ++k) power *= magnitude;
    const Scalar divided = q >= 1 ? power : power / q;
    if (!(divided <= limit)) return false;
    if (n == 0) return true;
    Scalar slope = divided;
    if constexpr (Policy == DenominatorPolicy::Smooth) {
        slope = math::abs(h.gain) * divided;
        if (!(slope <= limit)) return false;
    }
    return math::abs(value) * slope <= limit;
}

} // namespace kan::detail

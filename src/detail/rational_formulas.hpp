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
// Guard: callable `double(double)` returning its argument, invoked on every
// intermediate the contract requires to be finite (see basis_formulas.hpp).

#include "host_device.hpp"
#include "kan/rational.hpp"
#include <cstddef>
#include <type_traits>

namespace kan::detail {

template<DenominatorPolicy Policy>
using PolicyConstant = std::integral_constant<DenominatorPolicy, Policy>;

// Host-side dispatch of a validated runtime policy to a compile-time one:
// f(PolicyConstant<P>{}) is called once per operation, not per sample.
template<class F>
decltype(auto) visit_denominator_policy(DenominatorPolicy policy, F&& f) {
    switch (policy) {
    case DenominatorPolicy::Absolute: return f(PolicyConstant<DenominatorPolicy::Absolute>{});
    case DenominatorPolicy::Smooth: return f(PolicyConstant<DenominatorPolicy::Smooth>{});
    default: return f(PolicyConstant<DenominatorPolicy::Guarded>{});
    }
}

struct RationalHorner {
    double z;
    double p, dp; // numerator and its z-derivative
    double q, dq; // denominator Q and its z-derivative Q' = g S'
    double bound; // Guarded: sum |b_k| |z|^k + 1, the pole guard reference magnitude
    double ds;    // safe policies: S'
    double gain;  // safe policies: g = dQ/dS (Guarded: g = 1, not stored)
};

struct RationalEdge {
    double value;
    double input_derivative;
};

template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_argument(const RationalConfig& c, double x, const Guard& guard) {
    return guard(guard(x - c.center) / c.scale);
}

// Horner evaluation of P, P', Q, Q' (and the guard bound or S', g). `numerator`
// holds numerator_degree+1 coefficients, `denominator` holds denominator_degree.
template<DenominatorPolicy Policy, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE RationalHorner rational_horner(const RationalConfig& c, double x,
                                               const double* numerator,
                                               const double* denominator, const Guard& guard) {
    RationalHorner h{};
    h.z = rational_argument(c, x, guard);
    const auto m = c.numerator_degree, n = c.denominator_degree;
    h.p = numerator[m];
    for (std::size_t k = m; k > 0; --k) {
        h.dp = guard(h.dp * h.z + h.p);
        h.p = guard(h.p * h.z + numerator[k - 1]);
    }
    if constexpr (Policy == DenominatorPolicy::Guarded) {
        h.q = n ? denominator[n - 1] : 1;
        h.bound = n ? math::abs(denominator[n - 1]) : 1;
        for (std::size_t k = n; k > 0; --k) {
            const double next = k == 1 ? 1 : denominator[k - 2];
            h.dq = guard(h.dq * h.z + h.q);
            h.q = guard(h.q * h.z + next);
            h.bound = guard(h.bound * math::abs(h.z) + math::abs(next));
        }
    } else {
        // S and S' by Horner without the constant term, then Q = 1 + f(S) >= 1.
        double s = n ? denominator[n - 1] : 0;
        for (std::size_t k = n; k > 0; --k) {
            h.ds = guard(h.ds * h.z + s);
            s = guard(k == 1 ? s * h.z : s * h.z + denominator[k - 2]);
        }
        if constexpr (Policy == DenominatorPolicy::Absolute) {
            h.gain = s > 0 ? 1.0 : s < 0 ? -1.0 : 0.0;
            h.q = guard(1 + math::abs(s));
        } else {
            h.gain = guard(2 * s);
            h.q = guard(1 + guard(s * s));
        }
        h.dq = guard(h.gain * h.ds);
    }
    return h;
}

// Guarded policy: |Q| <= epsilon * bound is an unsafe pole. The safe policies
// have Q >= 1 and never report one.
template<DenominatorPolicy Policy>
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_pole(const RationalConfig& c, const RationalHorner& h) {
    if constexpr (Policy == DenominatorPolicy::Guarded)
        return math::finite(h.q) && math::finite(h.bound) && math::abs(h.q) <= c.epsilon * h.bound;
    else
        return false;
}

// Q' for the log-space input derivative. Guarded and Absolute form Q'
// exactly (g = 1, or g in {-1, 0, 1}); Smooth uses log|g| + log|S'| because
// Q' = 2 S S' may underflow although r Q'/Q is representable.
template<DenominatorPolicy Policy>
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_slope_nonzero(const RationalHorner& h) {
    if constexpr (Policy == DenominatorPolicy::Smooth) return h.gain != 0 && h.ds != 0;
    else return h.dq != 0;
}
template<DenominatorPolicy Policy>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_log_slope(const RationalHorner& h) {
    if constexpr (Policy == DenominatorPolicy::Smooth) return math::log(math::abs(h.gain)) + math::log(math::abs(h.ds));
    else return math::log(math::abs(h.dq));
}
template<DenominatorPolicy Policy>
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_slope_negative(const RationalHorner& h) {
    if constexpr (Policy == DenominatorPolicy::Smooth) return math::signbit(h.gain) != math::signbit(h.ds);
    else return math::signbit(h.dq);
}

template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_signed_exp(double exponent, bool negative, const Guard& guard) {
    return math::copysign(guard(math::exp(exponent)), negative ? -1.0 : 1.0);
}

// Value and dr/dx. When an intermediate quotient or product underflows, the
// representable final derivative is restored in log space; the ordinary
// Horner/quotient path is unchanged.
template<DenominatorPolicy Policy, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE RationalEdge rational_edge(const RationalConfig& c, const RationalHorner& h,
                                           const Guard& guard) {
    RationalEdge e{};
    e.value = guard(h.p / h.q);
    const double numerator_term = guard(h.dp / h.q), denominator_ratio = guard(h.dq / h.q);
    const double denominator_term = guard(e.value * denominator_ratio);
    e.input_derivative = guard(guard(numerator_term - denominator_term) / c.scale);
    const bool tiny_numerator = h.dp != 0 && math::tiny(numerator_term);
    const bool tiny_denominator = h.p != 0 && rational_slope_nonzero<Policy>(h) &&
        (math::tiny(e.value) || math::tiny(denominator_ratio) || math::tiny(denominator_term));
    if (tiny_numerator || tiny_denominator) {
        const double lq = math::log(math::abs(h.q)), ls = math::log(c.scale);
        const double first = tiny_numerator
            ? rational_signed_exp(math::log(math::abs(h.dp)) - lq - ls,
                                  math::signbit(h.dp) != math::signbit(h.q), guard)
            : guard(numerator_term / c.scale);
        const double second = tiny_denominator
            ? rational_signed_exp(math::log(math::abs(h.p)) + rational_log_slope<Policy>(h) - 2 * lq - ls,
                                  math::signbit(h.p) != rational_slope_negative<Policy>(h), guard)
            : guard(denominator_term / c.scale);
        e.input_derivative = guard(first - second);
    }
    return e;
}

// Parameter VJPs for one power k. Callers pass power = z^k and the guarded
// quotient divided = z^k/Q once, shared by dr/da_k and dr/db_k.

// dr/da_k = z^k / Q (every policy).
template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_numerator_vjp(double q, double z, std::size_t k, double power,
                                              double divided, const Guard& guard) {
    if (z != 0 && (math::tiny(power) || math::tiny(divided)))
        return rational_signed_exp(static_cast<double>(k) * math::log(math::abs(z)) - math::log(math::abs(q)),
                                   math::signbit(q) != (math::signbit(z) && k % 2 != 0), guard);
    return divided;
}

// dr/db_k = -r g z^k / Q for k >= 1, with value r = P/Q and gain g = dQ/dS
// (ignored by Guarded, where g = 1).
template<DenominatorPolicy Policy, class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_denominator_vjp(double p, double q, double value, double gain,
                                                double z, std::size_t k, double power, double divided,
                                                const Guard& guard) {
    const bool odd = math::signbit(z) && k % 2 != 0; // z^k < 0
    if constexpr (Policy == DenominatorPolicy::Guarded) {
        const double derivative = guard(-value * divided);
        if (p != 0 && z != 0 && (math::tiny(power) || math::tiny(divided) || math::tiny(value)))
            return rational_signed_exp(math::log(math::abs(p)) + static_cast<double>(k) * math::log(math::abs(z)) -
                                           2 * math::log(math::abs(q)),
                                       math::signbit(p) == odd, guard);
        return derivative;
    } else if constexpr (Policy == DenominatorPolicy::Absolute) {
        // g in {-1, 0, 1}: the guarded-form derivative times g, exactly.
        return gain * rational_denominator_vjp<DenominatorPolicy::Guarded>(p, q, value, gain, z, k, power,
                                                                           divided, guard);
    } else {
        const double scaled = guard(gain * divided), derivative = guard(-value * scaled);
        if (p != 0 && z != 0 && gain != 0 &&
            (math::tiny(power) || math::tiny(divided) || math::tiny(value) || math::tiny(gain) || math::tiny(scaled)))
            return rational_signed_exp(math::log(math::abs(p)) + math::log(math::abs(gain)) +
                                           static_cast<double>(k) * math::log(math::abs(z)) -
                                           2 * math::log(math::abs(q)),
                                       (math::signbit(p) != odd) == math::signbit(gain), guard);
        return derivative;
    }
}

} // namespace kan::detail

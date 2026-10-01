#pragma once

// Single source of the rational-edge formulas for the CPU backend and the
// resident CUDA kernels: r(x) = P(z)/Q(z), z = (x-center)/scale,
// Q(z) = 1 + sum_k b_k z^k. Callers validate configuration and finite data.
//
// Guard: callable `double(double)` returning its argument, invoked on every
// intermediate the contract requires to be finite (see basis_formulas.hpp).

#include "host_device.hpp"
#include "kan/rational.hpp"
#include <cstddef>

namespace kan::detail {

struct RationalHorner {
    double z;
    double p, dp; // numerator and its z-derivative
    double q, dq; // denominator and its z-derivative
    double bound; // sum |b_k| |z|^k + 1, the pole guard reference magnitude
};

struct RationalEdge {
    double value;
    double input_derivative;
};

template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_argument(const RationalConfig& c, double x, const Guard& guard) {
    return guard(guard(x - c.center) / c.scale);
}

// Horner evaluation of P, P', Q, Q' and the guard bound. `numerator` holds
// numerator_degree+1 coefficients, `denominator` holds denominator_degree.
template<class Guard>
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
    h.q = n ? denominator[n - 1] : 1;
    h.bound = n ? math::abs(denominator[n - 1]) : 1;
    for (std::size_t k = n; k > 0; --k) {
        const double next = k == 1 ? 1 : denominator[k - 2];
        h.dq = guard(h.dq * h.z + h.q);
        h.q = guard(h.q * h.z + next);
        h.bound = guard(h.bound * math::abs(h.z) + math::abs(next));
    }
    return h;
}

// Relative singularity policy: |Q| <= epsilon * bound is an unsafe pole.
KAN_HOST_DEVICE KAN_FORCE_INLINE bool rational_pole(const RationalConfig& c, const RationalHorner& h) {
    return math::finite(h.q) && math::finite(h.bound) && math::abs(h.q) <= c.epsilon * h.bound;
}

template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_signed_exp(double exponent, bool negative, const Guard& guard) {
    return math::copysign(guard(math::exp(exponent)), negative ? -1.0 : 1.0);
}

// Value and dr/dx. When an intermediate quotient or product underflows, the
// representable final derivative is restored in log space; the ordinary
// Horner/quotient path is unchanged.
template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE RationalEdge rational_edge(const RationalConfig& c, const RationalHorner& h,
                                           const Guard& guard) {
    RationalEdge e{};
    e.value = guard(h.p / h.q);
    const double numerator_term = guard(h.dp / h.q), denominator_ratio = guard(h.dq / h.q);
    const double denominator_term = guard(e.value * denominator_ratio);
    e.input_derivative = guard(guard(numerator_term - denominator_term) / c.scale);
    const bool tiny_numerator = h.dp != 0 && math::tiny(numerator_term);
    const bool tiny_denominator = h.p != 0 && h.dq != 0 &&
        (math::tiny(e.value) || math::tiny(denominator_ratio) || math::tiny(denominator_term));
    if (tiny_numerator || tiny_denominator) {
        const double lq = math::log(math::abs(h.q)), ls = math::log(c.scale);
        const double first = tiny_numerator
            ? rational_signed_exp(math::log(math::abs(h.dp)) - lq - ls,
                                  math::signbit(h.dp) != math::signbit(h.q), guard)
            : guard(numerator_term / c.scale);
        const double second = tiny_denominator
            ? rational_signed_exp(math::log(math::abs(h.p)) + math::log(math::abs(h.dq)) - 2 * lq - ls,
                                  math::signbit(h.p) != math::signbit(h.dq), guard)
            : guard(denominator_term / c.scale);
        e.input_derivative = guard(first - second);
    }
    return e;
}

// Parameter VJPs for one power k. Callers pass power = z^k and the guarded
// quotient divided = z^k/Q once, shared by dr/da_k and dr/db_k.

// dr/da_k = z^k / Q.
template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_numerator_vjp(double q, double z, std::size_t k, double power,
                                              double divided, const Guard& guard) {
    if (z != 0 && (math::tiny(power) || math::tiny(divided)))
        return rational_signed_exp(static_cast<double>(k) * math::log(math::abs(z)) - math::log(math::abs(q)),
                                   math::signbit(q) != (math::signbit(z) && k % 2 != 0), guard);
    return divided;
}

// dr/db_k = -r z^k / Q for k >= 1, with value r = P/Q.
template<class Guard>
KAN_HOST_DEVICE KAN_FORCE_INLINE double rational_denominator_vjp(double p, double q, double value, double z, std::size_t k,
                                                double power, double divided, const Guard& guard) {
    const double derivative = guard(-value * divided);
    if (p != 0 && z != 0 && (math::tiny(power) || math::tiny(divided) || math::tiny(value)))
        return rational_signed_exp(math::log(math::abs(p)) + static_cast<double>(k) * math::log(math::abs(z)) -
                                       2 * math::log(math::abs(q)),
                                   math::signbit(p) == (math::signbit(z) && k % 2 != 0), guard);
    return derivative;
}

} // namespace kan::detail

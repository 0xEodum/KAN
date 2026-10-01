#pragma once

// Single source of the linear-carrier basis formulas for the CPU backend and
// the resident CUDA kernels. Callers validate configuration and finite input.
//
// Guard: callable `double(double)` returning its argument. It is invoked on
// every value the contract requires to be finite; the CPU guard throws, the
// device guard records a status bit and execution continues.

#include "host_device.hpp"
#include "kan/basis.hpp"
#include <cstddef>
#include <stdexcept>

namespace kan::detail {

// Family tag for kernel dispatch. The public API selects a family by its
// configuration type; trainable RBFs are GaussianRbf with BasisView::trainable.
enum class BasisKind { Chebyshev, Legendre, Jacobi, Hermite, Fourier, GaussianRbf, BSpline, MexicanHat };

inline constexpr std::size_t max_spline_degree = 16;

// Non-owning view of one validated basis. Pointer fields are read only for the
// families that use them; `log_widths` is read only when `trainable` is set.
struct BasisView {
    BasisKind kind;
    std::size_t terms;
    double alpha, beta, frequency, width;
    const double* centers;
    const double* log_widths;
    const double* scales;
    const double* knots;
    std::size_t degree;
    bool trainable;
};

// Output rows of length `terms`. The nonlinear RBF rows may be null.
struct BasisRow {
    double* values;
    double* derivatives;
    double* center_derivatives;
    double* log_width_derivatives;
};

// Overflow in x-center is possible only for opposite signs. Divide before
// subtracting in that case, retaining a wide kernel's tail.
KAN_HOST_DEVICE inline double normalized_distance(double x, double center, double width) {
    const double distance = x - center;
    return math::finite(distance) ? distance / width : x / width - center / width;
}

// Ratios on a huge domain remain finite even when its length overflows.
KAN_HOST_DEVICE inline double interval_ratio(double numerator_right, double numerator_left,
                                             double right, double left) {
    const double denominator = right - left;
    if (denominator == 0) return 0;
    if (math::finite(denominator)) return (numerator_right - numerator_left) / denominator;
    return (0.5 * numerator_right - 0.5 * numerator_left) / (0.5 * right - 0.5 * left);
}

KAN_HOST_DEVICE inline double spline_slope_term(double value, std::size_t degree, double right,
                                                double left) {
    if (value == 0 || right == left) return 0;
    const double denominator = right - left;
    const double p = static_cast<double>(degree);
    return math::finite(denominator) ? (p * value) / denominator
                                     : (0.5 * p * value) / (0.5 * right - 0.5 * left);
}

// Clamped B-spline by Cox-de Boor on the single active span. Only degree+1
// terms can be nonzero, so fixed scratch covers every admissible degree.
template<class Guard>
KAN_HOST_DEVICE void spline_terms(const double* t, std::size_t terms, std::size_t degree, double x,
                                  double* v, double* d, const Guard& guard) {
    for (std::size_t k = 0; k < terms; ++k) {
        v[k] = 0;
        d[k] = 0;
    }
    if (x < t[degree] || x > t[terms]) return;
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
    double lower[max_spline_degree + 2] = {};
    double next[max_spline_degree + 2] = {};
    lower[degree] = 1;
    const auto start = span - degree;
    for (std::size_t p = 1; p <= degree; ++p) {
        for (std::size_t r = degree - p; r <= degree; ++r) {
            const auto i = start + r;
            double value = 0;
            if (lower[r] != 0) value += interval_ratio(x, t[i], t[i + p], t[i]) * lower[r];
            if (lower[r + 1] != 0)
                value += interval_ratio(t[i + p + 1], x, t[i + p + 1], t[i + 1]) * lower[r + 1];
            next[r] = guard(value);
            if (p == degree)
                d[i] = guard(spline_slope_term(lower[r], p, t[i + p], t[i]) -
                             spline_slope_term(lower[r + 1], p, t[i + p + 1], t[i + 1]));
        }
        for (std::size_t r = degree - p; r <= degree; ++r) lower[r] = next[r];
    }
    for (std::size_t r = 0; r <= degree; ++r) v[start + r] = lower[r];
}

// L2-normalized Mexican hat (Ricker) wavelets, evaluated in log space.
template<class Guard>
KAN_HOST_DEVICE void mexican_hat_terms(const double* centers, const double* scales,
                                       std::size_t terms, double x, double* v, double* d,
                                       const Guard& guard) {
    // log(2/sqrt(3)) - log(pi)/4, as a literal so device code does not
    // evaluate three transcendental functions per input.
    constexpr double log_normalization = -0x1.2383e809a67e3p-3;
    for (std::size_t k = 0; k < terms; ++k) {
        v[k] = 0;
        d[k] = 0;
        const double scale = scales[k];
        const double q = normalized_distance(x, centers[k], scale), q2 = q * q;
        // An infinite distance is an exact zero tail, not an inf*zero NaN.
        if (!math::finite(q2)) continue;
        const double log_scale = math::log(scale);
        const double envelope = log_normalization - 0.5 * log_scale - 0.5 * q2;
        if (q2 != 1)
            v[k] = guard(math::copysign(math::exp(envelope + math::log(math::abs(1 - q2))), 1 - q2));
        if (q != 0 && q2 != 3)
            d[k] = guard(math::copysign(
                math::exp(envelope + math::log(math::abs(q)) + math::log(math::abs(q2 - 3)) - log_scale),
                q * (q2 > 3 ? 1 : -1)));
    }
}

// Gaussian RBF exp(-((x-c)/w)^2). With trainable parameters w = exp(log_width)
// and the nonlinear center/log-width derivatives are produced when requested.
template<class Guard>
KAN_HOST_DEVICE void gaussian_terms(const BasisView& basis, double x, const BasisRow& row,
                                    const Guard& guard) {
    for (std::size_t k = 0; k < basis.terms; ++k) {
        const double width = basis.trainable ? math::exp(basis.log_widths[k]) : basis.width;
        const double q = normalized_distance(x, basis.centers[k], width);
        const double value = math::exp(-q * q);
        row.values[k] = value;
        double derivative = 0;
        if (q != 0 && math::finite(q)) {
            if (value < math::min_normal) {
                // A tiny width can amplify a value that underflows to zero
                // into a representable derivative. Preserve it in log space.
                const double log_magnitude = math::ln2 + math::log(math::abs(q)) - q * q - math::log(width);
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
                q != 0 && math::finite(q) ? guard(math::exp(math::ln2 + 2 * math::log(math::abs(q)) - q * q)) : 0;
        }
    }
}

template<class Guard>
KAN_HOST_DEVICE void fourier_terms(double frequency, std::size_t terms, double x, double* v,
                                   double* d, const Guard& guard) {
    v[0] = 1;
    d[0] = 0;
    for (std::size_t k = 1; k <= terms / 2; ++k) {
        const double angular = guard(static_cast<double>(k) * frequency);
        const double phase = guard(angular * x);
        const double cosine = math::cos(phase);
        const double sine = math::sin(phase);
        v[2 * k - 1] = cosine;
        v[2 * k] = sine;
        d[2 * k - 1] = guard(-angular * sine);
        d[2 * k] = guard(angular * cosine);
    }
}

// Chebyshev, Legendre, physicists' Hermite and Jacobi three-term recurrences.
template<class Guard>
KAN_HOST_DEVICE void polynomial_terms(BasisKind kind, double alpha, double beta, std::size_t terms,
                                      double x, double* v, double* d, const Guard& guard) {
    v[0] = 1;
    d[0] = 0;
    if (terms == 1) return;
    const bool jacobi = kind == BasisKind::Jacobi;
    // Half-sums avoid overflow in Jacobi's valid, very large parameters.
    const double half_sum = jacobi ? 0.5 * alpha + 0.5 * beta : 0;
    // Preserve distance from the admissible boundary alpha,beta > -1.
    // Adding one after summing the parameters can erase that distance.
    const double shifted_half_sum = jacobi ? 0.5 * (alpha + 1) + 0.5 * (beta + 1) : 0;
    const double half_difference = jacobi ? 0.5 * alpha - 0.5 * beta : 0;
    if (jacobi && (x == -1 || x == 1)) {
        // DLMF 18.6.T1: endpoint rising-factorial values. The general
        // three-term recurrence can cancel a small endpoint against huge terms.
        const double parameter = x == 1 ? alpha : beta;
        double endpoint = 1;
        double shifted_endpoint = 1;
        for (std::size_t k = 1; k < terms; ++k) {
            const double n = static_cast<double>(k);
            endpoint = guard(endpoint * ((parameter + n) / n));
            v[k] = x < 0 && k % 2 ? -endpoint : endpoint;
            if (k > 1) shifted_endpoint = guard(shifted_endpoint * ((parameter + n) / (n - 1)));
            // DLMF 18.9.E15: P'_n = (n+alpha+beta+1)/2 * P_(n-1)^(alpha+1,beta+1).
            // Half-sums keep a finite slope representable even when alpha+beta
            // would overflow.
            const double derivative = guard((shifted_half_sum + 0.5 * (n - 1)) * shifted_endpoint);
            d[k] = x < 0 && k % 2 == 0 ? -derivative : derivative;
        }
        return;
    }
    double first_slope = kind == BasisKind::Hermite ? 2 : 1;
    double first_offset = 0;
    if (jacobi) {
        first_slope = shifted_half_sum;
        first_offset = half_difference;
    }
    v[1] = guard(first_slope * x + first_offset);
    d[1] = guard(first_slope);
    for (std::size_t k = 1; k < terms - 1; ++k) {
        const double n = static_cast<double>(k);
        double a = 2, b = 0, c = 1; // Chebyshev
        if (kind == BasisKind::Legendre) {
            a = (2 * n + 1) / (n + 1);
            c = n / (n + 1);
        } else if (kind == BasisKind::Hermite) {
            c = 2 * n;
        } else if (jacobi) {
            // NIST DLMF 18.9.2, rearranged into ratios to avoid squaring
            // large parameters. P1 above handles alpha+beta = -1 or 0.
            const double t = shifted_half_sum + (n - 1);
            const double denominator_half = shifted_half_sum + 0.5 * (n - 1);
            a = ((t + 0.5) / (n + 1)) * ((t + 1) / denominator_half);
            b = (half_difference / (n + 1)) * (half_sum / t) * ((t + 0.5) / denominator_half);
            c = 0.5 * ((n + alpha) / (n + 1)) * ((n + beta) / t) * ((t + 1) / denominator_half);
        }
        const double factor = a * x + b;
        v[k + 1] = guard(factor * v[k] - c * v[k - 1]);
        d[k + 1] = guard(a * v[k] + factor * d[k] - c * d[k - 1]);
    }
}

// Family fixed at compile time: device kernels instantiate one family each, so
// no kernel carries another family's code, registers or spline scratch.
template<BasisKind Kind, class Guard>
KAN_HOST_DEVICE void basis_terms_for(const BasisView& basis, double x, const BasisRow& row,
                                     const Guard& guard) {
    if constexpr (Kind == BasisKind::BSpline)
        spline_terms(basis.knots, basis.terms, basis.degree, x, row.values, row.derivatives, guard);
    else if constexpr (Kind == BasisKind::MexicanHat)
        mexican_hat_terms(basis.centers, basis.scales, basis.terms, x, row.values, row.derivatives, guard);
    else if constexpr (Kind == BasisKind::GaussianRbf)
        gaussian_terms(basis, x, row, guard);
    else if constexpr (Kind == BasisKind::Fourier)
        fourier_terms(basis.frequency, basis.terms, x, row.values, row.derivatives, guard);
    else {
        static_assert(Kind == BasisKind::Chebyshev || Kind == BasisKind::Legendre ||
                      Kind == BasisKind::Jacobi || Kind == BasisKind::Hermite,
                      "basis family without an evaluator");
        polynomial_terms(Kind, basis.alpha, basis.beta, basis.terms, x, row.values, row.derivatives, guard);
    }
}

// Host-side runtime family dispatch over the compile-time evaluators: CPU
// evaluation and kernel launch selection. `visit` receives a tag whose `value`
// is the family.
template<BasisKind Kind>
struct BasisFamilyTag {
    static constexpr BasisKind value = Kind;
};

template<class Visitor>
void visit_basis_family(BasisKind kind, Visitor&& visit) {
    switch (kind) {
    case BasisKind::Chebyshev: visit(BasisFamilyTag<BasisKind::Chebyshev>{}); return;
    case BasisKind::Legendre: visit(BasisFamilyTag<BasisKind::Legendre>{}); return;
    case BasisKind::Jacobi: visit(BasisFamilyTag<BasisKind::Jacobi>{}); return;
    case BasisKind::Hermite: visit(BasisFamilyTag<BasisKind::Hermite>{}); return;
    case BasisKind::Fourier: visit(BasisFamilyTag<BasisKind::Fourier>{}); return;
    case BasisKind::GaussianRbf: visit(BasisFamilyTag<BasisKind::GaussianRbf>{}); return;
    case BasisKind::BSpline: visit(BasisFamilyTag<BasisKind::BSpline>{}); return;
    case BasisKind::MexicanHat: visit(BasisFamilyTag<BasisKind::MexicanHat>{}); return;
    }
    // Validation rejects unknown kinds before any dispatch.
    throw std::logic_error("unvalidated basis kind");
}

template<class Guard>
void basis_terms(const BasisView& basis, double x, const BasisRow& row, const Guard& guard) {
    visit_basis_family(basis.kind, [&](auto family) {
        basis_terms_for<decltype(family)::value>(basis, x, row, guard);
    });
}

} // namespace kan::detail

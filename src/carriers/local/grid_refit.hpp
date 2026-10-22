#pragma once

// Quantile grid refit of clamped B-splines (backlog M6), shared by the CPU
// (kan::refit_grid) and the resident executor's refit kernels. Everything is
// double: the FP32 executor evaluates the refit from its float samples and
// knots converted exactly, so a refit's arithmetic is the same in every
// precision; only its inputs (activations) and the final rounding differ.
//
// Per input, with its n samples sorted ascending:
// - knots (refit_knot): pykan's update_grid_from_samples on the domain
//   [a, c] = [x_0 - m*r, x_(n-1) + m*r], r = x_(n-1) - x_0 (m the relative
//   margin; pykan's own grid extends k intervals beyond the samples instead):
//   interior knot j = eps*(a + h*j) + (1 - eps)*x_(floor(n/G*j)), h = (c - a)/G,
//   endpoints a and c (clamped; m = 0: the extreme samples);
// - least squares: min sum_b (s(x_b) - f(x_b))^2 + ridge*sum c^2 with f the
//   current edge function, i.e. (B^T B + ridge I) c = B^T f, B^T f = M c_old
//   with the cross Gram matrix M of the new and the current basis. Sorted
//   samples form runs with one first term g (new grid) and g' (current grid;
//   `terms` outside its domain); refit_run_sums accumulates the P x P products
//   of a run (P = degree + 1) in sample order, and runs are added in order:
//   a deterministic order shared by both backends;
// - band_cholesky / band_solve: the banded normal matrix (half bandwidth p).

#include "carriers/local/local_formulas.hpp"
#include "core/host_device.hpp"
#include <cstddef>

namespace kan::detail {

// pykan's knot j of `intervals` from n >= 1 sorted samples (sorted(k) is
// sample k) with the relative margin m >= 0. With m = 0 the endpoints are the
// extreme samples exactly.
template<class Load>
KAN_HOST_DEVICE inline double refit_knot(const Load& sorted, std::size_t n, std::size_t intervals, std::size_t j, double eps,
                                         double margin) {
    const double first = sorted(0), last = sorted(n - 1), extension = margin * (last - first);
    const double lo = first - extension, hi = last + extension;
    if (j == 0) return lo;
    if (j >= intervals) return hi;
    auto rank = static_cast<std::size_t>(static_cast<double>(n) / static_cast<double>(intervals) * static_cast<double>(j));
    if (rank > n - 1) rank = n - 1;
    const double h = (hi - lo) / static_cast<double>(intervals);
    const double uniform = math::finite(h) ? lo + h * static_cast<double>(j)
                                           : 2 * (lo / 2 + (hi / 2 - lo / 2) / static_cast<double>(intervals) * static_cast<double>(j));
    return eps * uniform + (1 - eps) * sorted(rank);
}

// First term g of the degree+1 terms that can be nonzero at x, or `terms`
// outside the domain: the span search of spline_evaluate (same comparisons),
// without the evaluation.
template<class Scalar>
KAN_HOST_DEVICE inline std::size_t spline_first_term(const Scalar* t, std::size_t terms, std::size_t degree, Scalar x) {
    if (x < t[degree] || x > t[terms]) return terms;
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
    return span - degree;
}

struct RefitGuard {
    KAN_HOST_DEVICE double operator()(double v) const { return v; }
};

// Products of one run of samples x(begin..end-1) (sorted, one first term on
// each grid): nn[r*P + q] += B_{g+r}(x) B_{g+q}(x) for q >= r and
// no[r*P + q] += B_{g+r}(x) B'_{g'+q}(x) (B' the current basis; zero outside
// its domain). P = degree + 1; Width: P at compile time, 0 for any degree.
// The accumulators are zeroed first. Returns false if a sample's first terms
// differ from the run's (not a run of the given keys).
template<unsigned Width, class Load>
KAN_HOST_DEVICE bool refit_run_sums(const double* fresh, std::size_t fresh_terms, const double* current, std::size_t current_terms,
                                    std::size_t degree, const Load& x, std::size_t begin, std::size_t end, std::size_t g,
                                    std::size_t g_current, double* nn, double* no) {
    constexpr std::size_t Most = Width ? Width : max_spline_degree + 1;
    const std::size_t P = Width ? Width : degree + 1;
    for (std::size_t k = 0; k < P * P; ++k) { nn[k] = 0; no[k] = 0; }
    bool consistent = true;
    for (std::size_t b = begin; b < end; ++b) {
        const double v = x(b);
        double a[Most], da[Most], c[Most], dc[Most];
        const auto ga = spline_span_terms(fresh, fresh_terms, degree, v, a, da, RefitGuard{});
        const auto gc = spline_span_terms(current, current_terms, degree, v, c, dc, RefitGuard{});
        consistent = consistent && ga == g && gc == g_current;
        if (gc == current_terms)
            for (std::size_t q = 0; q < P; ++q) c[q] = 0;
        for (std::size_t r = 0; r < P; ++r) {
            for (std::size_t q = r; q < P; ++q) nn[r * P + q] += a[r] * a[q];
            for (std::size_t q = 0; q < P; ++q) no[r * P + q] += a[r] * c[q];
        }
    }
    return consistent;
}

// In place: the band of a symmetric positive definite matrix, band[k*P + d] =
// A(k, k - d) for d = 0..p (P = p + 1; entries with k < d unused), with ridge
// added to the diagonal, becomes its Cholesky factor L (same layout). Returns
// false for a pivot that is not positive or not above 64 epsilon of its
// diagonal (a singular or numerically singular system).
KAN_HOST_DEVICE inline bool band_cholesky(double* band, std::size_t terms, std::size_t degree, double ridge) {
    const std::size_t P = degree + 1;
    for (std::size_t k = 0; k < terms; ++k) {
        const std::size_t first = k > degree ? k - degree : 0;
        for (std::size_t j = first; j < k; ++j) {
            double s = band[k * P + (k - j)];
            // L(j, m) = 0 below m = j - p <= first.
            for (std::size_t m = first; m < j; ++m) s -= band[k * P + (k - m)] * band[j * P + (j - m)];
            band[k * P + (k - j)] = s / band[j * P];
        }
        const double diagonal = band[k * P] + ridge;
        double s = diagonal;
        for (std::size_t m = first; m < k; ++m) s -= band[k * P + (k - m)] * band[k * P + (k - m)];
        if (!(s > 64 * 0x1p-52 * diagonal) || !math::finite(s)) return false;
        band[k * P] = math::sqrt(s);
    }
    return true;
}

// Solves L L^T x = b in place for one right-hand side x[k*stride], k < terms.
KAN_HOST_DEVICE inline void band_solve(const double* band, std::size_t terms, std::size_t degree, double* x, std::size_t stride) {
    const std::size_t P = degree + 1;
    for (std::size_t k = 0; k < terms; ++k) {
        double s = x[k * stride];
        for (std::size_t m = k > degree ? k - degree : 0; m < k; ++m) s -= band[k * P + (k - m)] * x[m * stride];
        x[k * stride] = s / band[k * P];
    }
    for (std::size_t k = terms; k-- > 0;) {
        double s = x[k * stride];
        const std::size_t last = k + degree < terms - 1 ? k + degree : terms - 1;
        for (std::size_t m = k + 1; m <= last; ++m) s -= band[m * P + (m - k)] * x[m * stride];
        x[k * stride] = s / band[k * P];
    }
}

} // namespace kan::detail

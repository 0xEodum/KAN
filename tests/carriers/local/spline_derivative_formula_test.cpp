// Backlog C13: the derivative-coefficient form of the clamped B-spline input
// derivative (src/carriers/local/local_formulas.hpp), used by the FP32/TF32
// resident input VJP: d/dx sum_k c_k B_{k,p}(x) = sum_j C*_j B_{j,p-1}(x) with
// C*_j = (c_j - c_{j-1}) p / (t_{j+p} - t_j). Instantiated for double and float
// on the host: spline_lower_terms reports the same first term and values as
// spline_span_terms, guards the same expressions, and its degree-(p-1) values
// are nonnegative, sum to one and reproduce sum_k c_k B'_{k,p} through C*, for
// uniform, nonuniform and repeated knots, endpoints, knots and huge domains.
#include "carriers/local/local_formulas.hpp"
#include "support/test.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
// Counts guard calls (the status expressions a kernel's StatusGuard sees).
struct CountingGuard {
    std::size_t* calls;
    template<class Scalar> Scalar operator()(Scalar v) const { ++*calls; return v; }
};
std::uint64_t state = 0x9e3779b97f4a7c15ULL;
double uniform01() { // SplitMix64
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<double>((z ^ (z >> 31)) >> 11) * 0x1p-53;
}
// Clamped knots on [low, high]: `intervals` interior breakpoints, uniform or
// jittered, with an optional interior knot repeated `repeat` times.
template<class Scalar>
std::vector<Scalar> knots(std::size_t degree, std::size_t intervals, Scalar low, Scalar high, bool jitter, std::size_t repeat) {
    std::vector<Scalar> t(degree + 1, low);
    for (std::size_t k = 1; k < intervals; ++k) {
        const double s = (static_cast<double>(k) + (jitter ? 0.8 * (uniform01() - 0.5) : 0.0)) / static_cast<double>(intervals);
        const Scalar v = static_cast<Scalar>(static_cast<double>(low) * (1 - s) + static_cast<double>(high) * s);
        t.push_back(v);
        if (k == intervals / 2) for (std::size_t r = 1; r < repeat; ++r) t.push_back(v);
    }
    t.insert(t.end(), degree + 1, high);
    return t;
}
template<class Scalar>
void check(std::size_t degree, const std::vector<Scalar>& t, const std::string& what) {
    const auto terms = t.size() - degree - 1;
    std::vector<Scalar> c(terms);
    for (auto& v : c) v = static_cast<Scalar>(2 * uniform01() - 1);
    std::vector<Scalar> cstar(terms - 1);
    for (std::size_t j = 1; j < terms; ++j)
        cstar[j - 1] = kan::detail::spline_derivative_coefficient(c[j - 1], c[j], kan::detail::spline_derivative_scale(degree, t[j + degree], t[j]));
    const Scalar low = t[degree], high = t[terms];
    std::vector<Scalar> xs{low, high, static_cast<Scalar>(low - (high - low)), static_cast<Scalar>(high + 1)};
    for (std::size_t k = degree; k <= terms; ++k) xs.push_back(t[k]);
    for (int k = 0; k < 64; ++k) xs.push_back(static_cast<Scalar>(low + (high - low) * static_cast<Scalar>(uniform01())));
    const double unit = std::is_same_v<Scalar, double> ? 0x1p-52 : 0x1p-23;
    for (const Scalar x : xs) {
        if (!std::isfinite(static_cast<double>(x))) continue;
        Scalar v[kan::detail::max_spline_degree + 1], d[kan::detail::max_spline_degree + 1];
        Scalar lv[kan::detail::max_spline_degree + 1], lower[kan::detail::max_spline_degree + 1];
        std::size_t calls = 0, lower_calls = 0;
        const auto first = kan::detail::spline_span_terms(t.data(), terms, degree, x, v, d, CountingGuard{&calls});
        const auto lower_first = kan::detail::spline_lower_terms(t.data(), terms, degree, x, lv, lower, CountingGuard{&lower_calls});
        REQUIRE(first == lower_first);
        REQUIRE(calls == lower_calls);
        if (first == terms) continue;
        double derivative = 0, scale = 0, formula = 0, formula_scale = 0, sum = 0;
        for (std::size_t r = 0; r <= degree; ++r) {
            REQUIRE(v[r] == lv[r]);
            derivative += static_cast<double>(c[first + r]) * static_cast<double>(d[r]);
            scale += std::abs(static_cast<double>(c[first + r]) * static_cast<double>(d[r]));
        }
        for (std::size_t r = 0; r < degree; ++r) {
            REQUIRE(lower[r] >= 0);
            sum += static_cast<double>(lower[r]);
            formula += static_cast<double>(lower[r]) * static_cast<double>(cstar[first + r]);
            formula_scale += std::abs(static_cast<double>(lower[r]) * static_cast<double>(cstar[first + r]));
        }
        if (std::abs(sum - 1) > 4 * static_cast<double>(degree + 1) * unit)
            throw std::runtime_error(what + ": degree-(p-1) values sum to " + std::to_string(sum));
        // Both sides carry rounding of their own terms (products, slopes, C*).
        const double allowance = 8 * static_cast<double>(degree + 2) * unit * (scale + formula_scale);
        if (!(std::abs(derivative - formula) <= allowance))
            throw std::runtime_error(what + ": x=" + std::to_string(static_cast<double>(x)) + " sum c*B'=" + std::to_string(derivative) +
                                     " sum C*B_{p-1}=" + std::to_string(formula));
    }
}
template<class Scalar> void all_cases(Scalar huge) {
    for (std::size_t degree = 1; degree <= kan::detail::max_spline_degree; ++degree) {
        const auto name = std::string(std::is_same_v<Scalar, double> ? "double" : "float") + " degree " + std::to_string(degree);
        check<Scalar>(degree, knots<Scalar>(degree, 1, -1, 1, false, 1), name + " single interval");
        check<Scalar>(degree, knots<Scalar>(degree, 32, -1, 1, false, 1), name + " uniform");
        check<Scalar>(degree, knots<Scalar>(degree, 37, Scalar(-2.5), Scalar(4), true, 1), name + " nonuniform");
        check<Scalar>(degree, knots<Scalar>(degree, 20, -1, 1, true, degree), name + " repeated interior knot");
        check<Scalar>(degree, knots<Scalar>(degree, 12, -huge, huge, false, 1), name + " huge domain");
    }
}
} // namespace

TEST(lower_terms_reproduce_the_spline_derivative_through_difference_coefficients) {
    all_cases<double>(1e308);
    all_cases<float>(3e38f);
}

// Degree zero has no degree-(p-1) values (the derivative is zero); outside the
// domain nothing is written and `terms` is returned.
TEST(lower_terms_degree_zero_and_outside_the_domain) {
    const std::vector<double> t{-1, -0.5, 0, 0.5, 1};
    double v[2], lower[1] = {42};
    std::size_t calls = 0;
    REQUIRE(kan::detail::spline_lower_terms(t.data(), 4, 0, 0.25, v, lower, CountingGuard{&calls}) == 2);
    REQUIRE(lower[0] == 42 && v[0] == 1);
    const std::vector<double> cubic{0, 0, 0, 0, 0.5, 1, 1, 1, 1};
    double cv[4], cl[3] = {7, 7, 7};
    REQUIRE(kan::detail::spline_lower_terms(cubic.data(), 5, 3, 1.5, cv, cl, CountingGuard{&calls}) == 5);
    REQUIRE(cl[0] == 7 && cl[1] == 7 && cl[2] == 7);
}

// Knot scales p/(t_{j+p} - t_j): zero for a zero gap (whose degree-(p-1) term
// vanishes), finite for a gap beyond the range; equal neighbouring
// coefficients give C* = 0.
TEST(knot_scales_and_difference_coefficients_of_zero_gaps_and_equal_coefficients) {
    using kan::detail::spline_derivative_coefficient;
    using kan::detail::spline_derivative_scale;
    REQUIRE(spline_derivative_scale(3, 0.5, 0.5) == 0);
    REQUIRE(spline_derivative_coefficient(1.0, 3.0, spline_derivative_scale(3, 0.5, 0.5)) == 0);
    REQUIRE(spline_derivative_coefficient(2.0, 2.0, spline_derivative_scale(3, 0.5, 0.0)) == 0);
    REQUIRE(spline_derivative_coefficient(1.0, 3.0, spline_derivative_scale(2, 0.5, 0.0)) == 8);
    const float huge = spline_derivative_scale(3, 3e38f, -3e38f); // the gap overflows FP32
    REQUIRE(huge > 0 && std::isfinite(huge));
}

int main() { return test::run(); }

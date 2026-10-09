// Backlog M3: the shared host/device SiLU formulas (src/detail/residual_formulas.hpp),
// instantiated for double and float on the host, against an independent
// reference, at ordinary and extreme arguments, with a guard that throws.
#include "detail/residual_formulas.hpp"
#include "support/test.hpp"
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
// Throws on any nonfinite intermediate, as the CPU guard does.
struct ThrowingGuard {
    template<class Scalar> Scalar operator()(Scalar v) const {
        if (!std::isfinite(v)) throw std::overflow_error("nonfinite silu intermediate");
        return v;
    }
};

// Independent reference in long double: x / (1 + exp(-x)) and s + x s (1 - s).
long double reference_value(long double x) { return x / (1 + std::exp(-x)); }
long double reference_derivative(long double x) {
    const long double s = 1 / (1 + std::exp(-x));
    return s + x * s * (1 - s);
}

template<class Scalar> void bitwise_value_matches(Scalar x) {
    const auto both = kan::detail::silu(x, ThrowingGuard{});
    const Scalar alone = kan::detail::silu_value(x, ThrowingGuard{});
    REQUIRE(std::isfinite(both.value) && std::isfinite(both.derivative));
    REQUIRE(std::signbit(alone) == std::signbit(both.value) && (alone == both.value));
}
} // namespace

TEST(silu_matches_reference_at_ordinary_arguments_double) {
    for (double x : {-30.0, -20.0, -5.0, -1.2784645427610738, -1.0, -0.1, -1e-300, 0.0, 1e-300, 0.1, 1.0, 2.5, 5.0, 20.0, 30.0}) {
        const auto s = kan::detail::silu(x, ThrowingGuard{});
        test::near(s.value, double(reference_value(x)), 4e-16);
        test::near(s.derivative, double(reference_derivative(x)), 4e-16);
        bitwise_value_matches(x);
        // The derivative is the derivative of the value.
        const double h = 1e-6;
        const double fd = (kan::detail::silu_value(x + h, ThrowingGuard{}) -
                           kan::detail::silu_value(x - h, ThrowingGuard{})) / (2 * h);
        test::near(s.derivative, fd, 1e-8);
    }
    // Known values: silu(0) = 0, silu'(0) = 1/2, silu(1) = 1/(1+e^-1).
    REQUIRE(kan::detail::silu(0.0, ThrowingGuard{}).value == 0);
    REQUIRE(kan::detail::silu(0.0, ThrowingGuard{}).derivative == 0.5);
    test::near(kan::detail::silu(1.0, ThrowingGuard{}).value, 0.7310585786300049, 2e-16);
}

TEST(silu_matches_reference_at_ordinary_arguments_float) {
    for (float x : {-30.f, -5.f, -1.f, -0.1f, 0.f, 0.1f, 1.f, 5.f, 30.f}) {
        const auto s = kan::detail::silu(x, ThrowingGuard{});
        test::near(double(s.value), double(reference_value(x)), 4e-7);
        test::near(double(s.derivative), double(reference_derivative(x)), 4e-7);
        bitwise_value_matches(x);
    }
}

TEST(silu_is_finite_at_extreme_double_arguments) {
    const double max = std::numeric_limits<double>::max(), tiny = std::numeric_limits<double>::denorm_min();
    for (double x : {1e3, -1e3, 709.0, -709.0, 746.0, -746.0, 1e300, -1e300, max, -max, DBL_MIN, -DBL_MIN, tiny, -tiny, -0.0})
        bitwise_value_matches(x);
    const auto positive = kan::detail::silu(max, ThrowingGuard{});
    REQUIRE(positive.value == max);
    REQUIRE(positive.derivative == 1);
    const auto negative = kan::detail::silu(-max, ThrowingGuard{});
    REQUIRE(negative.value == 0 && negative.derivative == 0);
    REQUIRE(kan::detail::silu(1e3, ThrowingGuard{}).value == 1e3);
    REQUIRE(kan::detail::silu(1e3, ThrowingGuard{}).derivative == 1);
    REQUIRE(kan::detail::silu(-1e3, ThrowingGuard{}).value == 0);
    // Large negative arguments keep the representable tail: x e^x.
    test::near(kan::detail::silu(-700.0, ThrowingGuard{}).value, -700 * std::exp(-700.0), 1e-14);
}

TEST(silu_is_finite_at_extreme_float_arguments) {
    const float max = std::numeric_limits<float>::max();
    for (float x : {100.f, -100.f, 88.f, -88.f, 104.f, -104.f, 1e30f, -1e30f, max, -max, FLT_MIN, -FLT_MIN, -0.f})
        bitwise_value_matches(x);
    const auto positive = kan::detail::silu(max, ThrowingGuard{});
    REQUIRE(positive.value == max && positive.derivative == 1);
    const auto negative = kan::detail::silu(-max, ThrowingGuard{});
    REQUIRE(negative.value == 0 && negative.derivative == 0);
    REQUIRE(kan::detail::silu(100.f, ThrowingGuard{}).value == 100.f);
}

int main() { return test::run(); }

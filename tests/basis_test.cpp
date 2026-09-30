#include "kan/basis.hpp"
#include "support/test.hpp"

#include <limits>

namespace {
using kan::BasisConfig;
using kan::BasisKind;

BasisConfig polynomial(BasisKind kind, std::size_t size = 8) {
    BasisConfig config;
    config.kind = kind;
    config.size = size;
    return config;
}

double binomial(double top, std::size_t count) {
    double result = 1.0;
    for (std::size_t k = 1; k <= count; ++k)
        result *= (top - static_cast<double>(k) + 1.0) / static_cast<double>(k);
    return result;
}

// Independent finite-sum definition, not the production three-term recurrence.
double jacobi_sum(std::size_t degree, double alpha, double beta, double x) {
    double result = 0.0;
    for (std::size_t k = 0; k <= degree; ++k)
        result += binomial(static_cast<double>(degree) + alpha, k) *
                  binomial(static_cast<double>(degree) + beta, degree - k) *
                  std::pow((x - 1.0) / 2.0, static_cast<double>(degree - k)) *
                  std::pow((x + 1.0) / 2.0, static_cast<double>(k));
    return result;
}
}

TEST(polynomial_closed_forms) {
    for (const double x : {-1.5, -1.0, -0.3, 0.0, 0.7, 1.0, 1.4}) {
        auto cheb = kan::evaluate_basis(polynomial(BasisKind::Chebyshev, 5), x);
        auto leg = kan::evaluate_basis(polynomial(BasisKind::Legendre, 5), x);
        auto herm = kan::evaluate_basis(polynomial(BasisKind::Hermite, 5), x);
        test::near(cheb.values[4], 8*x*x*x*x - 8*x*x + 1);
        test::near(cheb.derivatives[4], 32*x*x*x - 16*x);
        test::near(leg.values[4], (35*x*x*x*x - 30*x*x + 3)/8);
        test::near(leg.derivatives[4], (140*x*x*x - 60*x)/8);
        test::near(herm.values[4], 16*x*x*x*x - 48*x*x + 12);
        test::near(herm.derivatives[4], 64*x*x*x - 96*x);
        test::near(herm.values[1], 2*x);
    }
}

TEST(polynomial_endpoint_identities) {
    for (const auto kind : {BasisKind::Chebyshev, BasisKind::Legendre}) {
        for (const double x : {-1.0, 1.0}) {
            const auto result = kan::evaluate_basis(polynomial(kind, 24), x);
            for (std::size_t k = 0; k < result.values.size(); ++k) {
                const double n = static_cast<double>(k);
                const double value = x < 0 && k % 2 ? -1.0 : 1.0;
                const double derivative_sign = x < 0 && k % 2 == 0 ? -1.0 : 1.0;
                test::near(result.values[k], value);
                test::near(result.derivatives[k], derivative_sign *
                    (kind == BasisKind::Chebyshev ? n*n : n*(n+1)/2));
            }
        }
    }
}

TEST(jacobi_independent_sum_and_derivative_identity) {
    for (const auto params : {std::pair{0.0,0.0}, std::pair{-0.5,-0.5},
                             std::pair{0.25,-0.25}, std::pair{-0.9,0.2}, std::pair{1.2,2.3}}) {
        auto config = polynomial(BasisKind::Jacobi);
        config.alpha = params.first;
        config.beta = params.second;
        for (const double x : {-1.4, -1.0, -0.2, 0.6, 1.0, 1.3}) {
            const auto result = kan::evaluate_basis(config, x);
            for (std::size_t k = 0; k < config.size; ++k) {
                test::near(result.values[k], jacobi_sum(k, config.alpha, config.beta, x));
                const double derivative = k == 0 ? 0.0 :
                    (static_cast<double>(k) + config.alpha + config.beta + 1)/2 *
                    jacobi_sum(k-1, config.alpha+1, config.beta+1, x);
                test::near(result.derivatives[k], derivative);
            }
        }
    }
}

TEST(jacobi_large_parameters_avoid_spurious_coefficient_overflow) {
    auto config = polynomial(BasisKind::Jacobi, 3);
    config.alpha = config.beta = 1e200;
    const auto result = kan::evaluate_basis(config, 0);
    test::near(result.values[1], 0);
    test::near(result.derivatives[1] / config.alpha, 1);
    test::near(result.values[2] / config.alpha, -0.25);
    test::near(result.derivatives[2], 0);
}

TEST(fourier_order_and_angular_frequency) {
    auto config = polynomial(BasisKind::Fourier, 7);
    config.frequency = 2.7;
    const double x = 0.31;
    const auto result = kan::evaluate_basis(config, x);
    test::near(result.values[0], 1);
    test::near(result.derivatives[0], 0);
    for (std::size_t k = 1; k <= 3; ++k) {
        const double angular = static_cast<double>(k)*config.frequency;
        test::near(result.values[2*k-1], std::cos(angular*x));
        test::near(result.values[2*k], std::sin(angular*x));
        test::near(result.derivatives[2*k-1], -angular*std::sin(angular*x));
        test::near(result.derivatives[2*k], angular*std::cos(angular*x));
    }
}

TEST(gaussian_values_centers_and_derivatives) {
    auto config = polynomial(BasisKind::GaussianRbf, 3);
    config.centers = {-0.4, 0.2, 1.1};
    config.width = 0.7;
    for (const double x : {-1.2, 0.2, 0.9}) {
        const auto result = kan::evaluate_basis(config, x);
        for (std::size_t k = 0; k < config.size; ++k) {
            const double delta = x-config.centers[k];
            const double expected = std::exp(-delta*delta/(config.width*config.width));
            test::near(result.values[k], expected);
            test::near(result.derivatives[k], -2*delta/(config.width*config.width)*expected);
        }
    }
}

TEST(gaussian_extreme_finite_parameters_and_underflow_tails) {
    const double max = std::numeric_limits<double>::max();
    auto config = polynomial(BasisKind::GaussianRbf, 1);
    config.centers = {-max};
    config.width = max;
    const auto wide = kan::evaluate_basis(config, max);
    test::near(wide.values[0], std::exp(-4));
    test::near(wide.derivatives[0]*max, -4*std::exp(-4));
    config.width = 1;
    const auto tail = kan::evaluate_basis(config, max);
    test::near(tail.values[0], 0);
    test::near(tail.derivatives[0], 0);
    config.centers = {0};
    config.width = std::numeric_limits<double>::denorm_min();
    const auto tiny_width = kan::evaluate_basis(config, 1);
    test::near(tiny_width.values[0], 0);
    test::near(tiny_width.derivatives[0], 0);
    const auto center = kan::evaluate_basis(config, 0);
    test::near(center.values[0], 1);
    test::near(center.derivatives[0], 0);
}

TEST(gaussian_underflow_value_can_have_representable_derivative) {
    auto config = polynomial(BasisKind::GaussianRbf, 1);
    config.centers = {0};
    config.width = std::numeric_limits<double>::denorm_min();
    const auto result = kan::evaluate_basis(config, 30*config.width);
    test::near(result.values[0], 0);
    const double expected = static_cast<double>(-60.0L*std::exp(-900.0L) /
                                                static_cast<long double>(config.width));
    REQUIRE(expected != 0);
    REQUIRE(result.derivatives[0] != 0);
    test::near(result.derivatives[0]/expected, 1, 1e-12);
}

TEST(analytic_derivatives_match_central_differences) {
    for (const auto kind : {BasisKind::Chebyshev, BasisKind::Legendre, BasisKind::Jacobi,
                           BasisKind::Hermite, BasisKind::Fourier, BasisKind::GaussianRbf}) {
        auto config = polynomial(kind, 7);
        config.alpha = -0.4;
        config.beta = 0.8;
        config.frequency = 1.7;
        config.centers = {-2, -1, -0.5, 0, 0.5, 1, 2};
        config.width = 0.8;
        for (const double x : {-1.1, -0.35, 0.4, 1.2}) {
            constexpr double h = 1e-6;
            const auto value = kan::evaluate_basis(config, x);
            const auto plus = kan::evaluate_basis(config, x+h);
            const auto minus = kan::evaluate_basis(config, x-h);
            for (std::size_t k = 0; k < config.size; ++k)
                test::near(value.derivatives[k], (plus.values[k]-minus.values[k])/(2*h), 2e-7);
        }
    }
}

TEST(one_term_bases_and_relevant_parameters_only) {
    for (const auto kind : {BasisKind::Chebyshev, BasisKind::Legendre, BasisKind::Jacobi,
                           BasisKind::Hermite, BasisKind::Fourier}) {
        auto config = polynomial(kind, 1);
        config.width = -1;
        config.centers = {std::numeric_limits<double>::quiet_NaN()};
        if (kind != BasisKind::Jacobi) config.alpha = config.beta = -2;
        if (kind != BasisKind::Fourier) config.frequency = -1;
        const auto result = kan::evaluate_basis(config, 1e100);
        REQUIRE(result.values.size() == 1);
        REQUIRE(result.derivatives.size() == 1);
        test::near(result.values[0], 1);
        test::near(result.derivatives[0], 0);
    }
    auto config = polynomial(BasisKind::GaussianRbf, 1);
    config.centers = {0};
    config.alpha = config.beta = config.frequency = std::numeric_limits<double>::quiet_NaN();
    test::near(kan::evaluate_basis(config, 0).values[0], 1);
}

TEST(invalid_kind_and_sizes) {
    auto config = polynomial(static_cast<BasisKind>(999));
    test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
    config = polynomial(BasisKind::Chebyshev, 0);
    test::throws<std::invalid_argument>([&] { kan::evaluate_basis(config, 0); });
    config = polynomial(BasisKind::Fourier, 2);
    test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
    config = polynomial(BasisKind::Chebyshev, std::numeric_limits<std::size_t>::max());
    test::throws<std::overflow_error>([&] { kan::evaluate_basis(config, 0); });
}

TEST(invalid_family_parameters) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    auto config = polynomial(BasisKind::Jacobi);
    for (double invalid : {-1.0, -2.0, nan, inf}) {
        config.alpha = invalid;
        test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
        config.alpha = 0;
        config.beta = invalid;
        test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
        config.beta = 0;
    }
    config = polynomial(BasisKind::Fourier, 3);
    for (double invalid : {0.0, -1.0, nan, inf}) {
        config.frequency = invalid;
        test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
    }
    config = polynomial(BasisKind::GaussianRbf, 2);
    test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
    config.centers = {0, nan};
    test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
    config.centers = {inf, 0};
    test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
    config.centers = {0, 1};
    for (double invalid : {0.0, -1.0, nan, inf}) {
        config.width = invalid;
        test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
    }
}

TEST(nonfinite_input_is_rejected_for_every_family) {
    for (const auto kind : {BasisKind::Chebyshev, BasisKind::Legendre, BasisKind::Jacobi,
                           BasisKind::Hermite, BasisKind::Fourier, BasisKind::GaussianRbf}) {
        auto config = polynomial(kind, 1);
        config.centers = {0};
        for (double x : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()})
            test::throws<std::invalid_argument>([&] { kan::evaluate_basis(config, x); });
    }
}

TEST(numeric_overflow_is_explicit) {
    const double max = std::numeric_limits<double>::max();
    for (const auto kind : {BasisKind::Chebyshev, BasisKind::Legendre,
                           BasisKind::Jacobi, BasisKind::Hermite}) {
        const auto config = polynomial(kind, 4);
        test::throws<std::overflow_error>([&] { kan::evaluate_basis(config, max); });
    }
    auto config = polynomial(BasisKind::Fourier, 3);
    config.frequency = 2;
    test::throws<std::overflow_error>([&] { kan::evaluate_basis(config, max); });
    config = polynomial(BasisKind::GaussianRbf, 1);
    config.centers = {0};
    config.width = std::numeric_limits<double>::denorm_min();
    test::throws<std::overflow_error>([&] { kan::evaluate_basis(config, config.width); });
}

int main() { return test::run(); }

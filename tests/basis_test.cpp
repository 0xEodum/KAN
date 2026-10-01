#include "kan/basis.hpp"
#include "support/test.hpp"

#include <limits>
#include <variant>

namespace {
using kan::BasisConfig;

// One fixture per global family with shared parameters; optionally a Gaussian
// RBF with `size` centers.
std::vector<BasisConfig> global_families(std::size_t size, bool gaussian = true) {
    std::vector<BasisConfig> result{kan::ChebyshevConfig{size}, kan::LegendreConfig{size},
                                    kan::JacobiConfig{size, -0.4, 0.8}, kan::HermiteConfig{size},
                                    kan::FourierConfig{size, 1.7}};
    if (gaussian) {
        std::vector<double> centers(size);
        for (std::size_t k = 0; k < size; ++k) centers[k] = -1.5 + 0.5 * static_cast<double>(k);
        result.push_back(kan::GaussianRbfConfig{centers, 0.8});
    }
    return result;
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
        auto cheb = kan::evaluate_basis(kan::ChebyshevConfig{5}, x);
        auto leg = kan::evaluate_basis(kan::LegendreConfig{5}, x);
        auto herm = kan::evaluate_basis(kan::HermiteConfig{5}, x);
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
    for (const bool chebyshev : {true, false}) {
        for (const double x : {-1.0, 1.0}) {
            const auto result = chebyshev ? kan::evaluate_basis(kan::ChebyshevConfig{24}, x)
                                          : kan::evaluate_basis(kan::LegendreConfig{24}, x);
            for (std::size_t k = 0; k < result.values.size(); ++k) {
                const double n = static_cast<double>(k);
                const double value = x < 0 && k % 2 ? -1.0 : 1.0;
                const double derivative_sign = x < 0 && k % 2 == 0 ? -1.0 : 1.0;
                test::near(result.values[k], value);
                test::near(result.derivatives[k], derivative_sign *
                    (chebyshev ? n*n : n*(n+1)/2));
            }
        }
    }
}

TEST(jacobi_independent_sum_and_derivative_identity) {
    for (const auto params : {std::pair{0.0,0.0}, std::pair{-0.5,-0.5},
                             std::pair{0.25,-0.25}, std::pair{-0.9,0.2}, std::pair{1.2,2.3}}) {
        const kan::JacobiConfig config{8, params.first, params.second};
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
    const kan::JacobiConfig config{3, 1e200, 1e200};
    const auto result = kan::evaluate_basis(config, 0);
    test::near(result.values[1], 0);
    test::near(result.derivatives[1] / config.alpha, 1);
    test::near(result.values[2] / config.alpha, -0.25);
    test::near(result.derivatives[2], 0);
}

TEST(jacobi_asymmetric_large_parameter_endpoint_identities) {
    for (const auto params : {std::pair{1e17,0.0}, std::pair{0.0,1e17},
                             std::pair{1e70,0.2}, std::pair{-0.7,1e70}}) {
        const kan::JacobiConfig config{5, params.first, params.second};
        for (double x : {-1.0, 1.0}) {
            const auto result = kan::evaluate_basis(config, x);
            const double endpoint_parameter = x == 1 ? config.alpha : config.beta;
            for (std::size_t k = 0; k < config.size; ++k) {
                // DLMF 18.6.T1: endpoint values depend on only one parameter.
                double expected_value = 1;
                for (std::size_t j = 1; j <= k; ++j)
                    expected_value *= (endpoint_parameter + static_cast<double>(j))/static_cast<double>(j);
                if (x < 0 && k % 2) expected_value = -expected_value;
                test::near(result.values[k]/expected_value, 1);
                if (k == 0) {
                    test::near(result.derivatives[k], 0);
                    continue;
                }
                // DLMF 18.9.E15 with the shifted polynomial's endpoint identity.
                double expected_derivative = 0.5*config.alpha + 0.5*config.beta +
                                             0.5*(static_cast<double>(k)+1);
                for (std::size_t j = 1; j < k; ++j)
                    expected_derivative *= (endpoint_parameter + static_cast<double>(j)+1)/static_cast<double>(j);
                if (x < 0 && k % 2 == 0) expected_derivative = -expected_derivative;
                test::near(result.derivatives[k]/expected_derivative, 1);
            }
        }
    }
}

TEST(jacobi_endpoint_extreme_derivative_finiteness_and_overflow) {
    const double max = std::numeric_limits<double>::max();
    kan::JacobiConfig config{2, max, max};
    for (double x : {-1.0, 1.0}) {
        const auto result = kan::evaluate_basis(config, x);
        test::near(result.values[1]/max, x);
        test::near(result.derivatives[1]/max, 1);
    }
    config.beta = 0;
    config.size = 3;
    const auto finite = kan::evaluate_basis(config, -1);
    test::near(finite.values[2], 1);
    test::near(finite.derivatives[2]/max, -1);
    test::throws<std::overflow_error>([&] { kan::evaluate_basis(config, 1); });
    config.size = 4;
    test::throws<std::overflow_error>([&] { kan::evaluate_basis(config, -1); });
}

TEST(jacobi_parameters_adjacent_to_minus_one_match_independent_sum) {
    const double first = std::nextafter(-1.0, 0.0);
    const double second = std::nextafter(first, 0.0);
    const double third = std::nextafter(second, 0.0);
    const double fourth = std::nextafter(third, 0.0);
    double eighth = fourth;
    for (int step = 0; step < 4; ++step) eighth = std::nextafter(eighth, 0.0);
    for (const auto params : {std::pair{first,first}, std::pair{first,second},
                             std::pair{second,first}, std::pair{second,second},
                             std::pair{first,fourth}, std::pair{fourth,first},
                             std::pair{fourth,fourth}, std::pair{first,eighth},
                             std::pair{eighth,first}, std::pair{eighth,eighth}}) {
        const kan::JacobiConfig config{8, params.first, params.second};
        for (double x : {-1.0, -0.8, -0.3, 0.0, 0.4, 0.9, 1.0}) {
            const auto result = kan::evaluate_basis(config, x);
            const double first_slope = 0.5*(config.alpha+1)+0.5*(config.beta+1);
            test::near(result.derivatives[1]/first_slope, 1);
            for (std::size_t k = 0; k < config.size; ++k) {
                test::near(result.values[k], jacobi_sum(k, config.alpha, config.beta, x));
                const double derivative = k == 0 ? 0 :
                    (0.5*(config.alpha+1) + 0.5*(config.beta+1) +
                     0.5*(static_cast<double>(k)-1)) *
                    jacobi_sum(k-1, config.alpha+1, config.beta+1, x);
                test::near(result.derivatives[k], derivative);
            }
        }
    }
}

TEST(fourier_order_and_angular_frequency) {
    const kan::FourierConfig config{7, 2.7};
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
    const kan::GaussianRbfConfig config{{-0.4, 0.2, 1.1}, 0.7};
    for (const double x : {-1.2, 0.2, 0.9}) {
        const auto result = kan::evaluate_basis(config, x);
        for (std::size_t k = 0; k < config.centers.size(); ++k) {
            const double delta = x-config.centers[k];
            const double expected = std::exp(-delta*delta/(config.width*config.width));
            test::near(result.values[k], expected);
            test::near(result.derivatives[k], -2*delta/(config.width*config.width)*expected);
        }
    }
}

TEST(gaussian_extreme_finite_parameters_and_underflow_tails) {
    const double max = std::numeric_limits<double>::max();
    kan::GaussianRbfConfig config{{-max}, max};
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
    const kan::GaussianRbfConfig config{{0}, std::numeric_limits<double>::denorm_min()};
    const auto result = kan::evaluate_basis(config, 30*config.width);
    test::near(result.values[0], 0);
    // Independent Python Decimal oracle (100-digit precision):
    // -60 * exp(-900) / (2 ** -1074). MSVC long double has double precision.
    constexpr double expected = -1.657039574215751920267356912558436470529179262842470867148420005915681625311244551783798700124454242E-66;
    REQUIRE(expected != 0);
    REQUIRE(result.derivatives[0] != 0);
    test::near(result.derivatives[0]/expected, 1, 1e-12);
}

TEST(analytic_derivatives_match_central_differences) {
    for (const auto& config : global_families(7)) {
        for (const double x : {-1.1, -0.35, 0.4, 1.2}) {
            constexpr double h = 1e-6;
            const auto value = kan::evaluate_basis(config, x);
            const auto plus = kan::evaluate_basis(config, x+h);
            const auto minus = kan::evaluate_basis(config, x-h);
            for (std::size_t k = 0; k < kan::basis_size(config); ++k)
                test::near(value.derivatives[k], (plus.values[k]-minus.values[k])/(2*h), 2e-7);
        }
    }
}

// Irrelevant parameters cannot exist: each configuration type holds only its
// family's fields, so only the one-term behaviour remains to check.
TEST(one_term_bases) {
    for (const auto& config : global_families(1, false)) {
        const auto result = kan::evaluate_basis(config, 1e100);
        REQUIRE(result.values.size() == 1);
        REQUIRE(result.derivatives.size() == 1);
        test::near(result.values[0], 1);
        test::near(result.derivatives[0], 0);
    }
    test::near(kan::evaluate_basis(kan::GaussianRbfConfig{{0}, 1}, 0).values[0], 1);
}

TEST(invalid_sizes) {
    test::throws<std::invalid_argument>([&] { kan::evaluate_basis(kan::ChebyshevConfig{0}, 0); });
    test::throws<std::invalid_argument>([&] { kan::validate_basis(kan::FourierConfig{2, 1}); });
    test::throws<std::overflow_error>([&] {
        kan::evaluate_basis(kan::ChebyshevConfig{std::numeric_limits<std::size_t>::max()}, 0);
    });
}

TEST(invalid_family_parameters) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    kan::JacobiConfig config;
    for (double invalid : {-1.0, -2.0, nan, inf}) {
        config.alpha = invalid;
        test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
        config.alpha = 0;
        config.beta = invalid;
        test::throws<std::invalid_argument>([&] { kan::validate_basis(config); });
        config.beta = 0;
    }
    for (double invalid : {0.0, -1.0, nan, inf})
        test::throws<std::invalid_argument>([&] { kan::validate_basis(kan::FourierConfig{3, invalid}); });
    kan::GaussianRbfConfig rbf;
    test::throws<std::invalid_argument>([&] { kan::validate_basis(rbf); });
    rbf.centers = {0, nan};
    test::throws<std::invalid_argument>([&] { kan::validate_basis(rbf); });
    rbf.centers = {inf, 0};
    test::throws<std::invalid_argument>([&] { kan::validate_basis(rbf); });
    rbf.centers = {0, 1};
    for (double invalid : {0.0, -1.0, nan, inf}) {
        rbf.width = invalid;
        test::throws<std::invalid_argument>([&] { kan::validate_basis(rbf); });
    }
}

TEST(nonfinite_input_is_rejected_for_every_family) {
    for (const auto& config : global_families(1)) {
        for (double x : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()})
            test::throws<std::invalid_argument>([&] { kan::evaluate_basis(config, x); });
    }
}

TEST(numeric_overflow_is_explicit) {
    const double max = std::numeric_limits<double>::max();
    for (const auto& config : global_families(4, false))
        if (!std::holds_alternative<kan::FourierConfig>(config))
            test::throws<std::overflow_error>([&] { kan::evaluate_basis(config, max); });
    test::throws<std::overflow_error>([&] { kan::evaluate_basis(kan::FourierConfig{3, 2}, max); });
    const double tiny = std::numeric_limits<double>::denorm_min();
    test::throws<std::overflow_error>([&] { kan::evaluate_basis(kan::GaussianRbfConfig{{0}, tiny}, tiny); });
}

int main() { return test::run(); }

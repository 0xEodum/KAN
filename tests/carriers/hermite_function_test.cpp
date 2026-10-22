// Normalized Hermite functions (backlog M5): HermiteConfig::normalization =
// Function evaluates psi_n(x) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi)).
#include "kan/families.hpp"
#include "support/test.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <variant>

namespace {
using kan::HermiteConfig;
using kan::HermiteNormalization;

HermiteConfig function(std::size_t size) { return HermiteConfig{size, HermiteNormalization::Function}; }

// |actual - expected| <= relative*|expected| + 4 smallest subnormals.
void close(double actual, double expected, double relative, const char* what) {
    const double floor = 4 * std::numeric_limits<double>::denorm_min();
    if (!std::isfinite(actual) || std::abs(actual - expected) > relative * std::abs(expected) + floor)
        throw std::runtime_error(std::string(what) + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
}

// Independent reference (docs/evidence/backlog/M5-bench/hermite_reference.py
// table): the physicists' recurrence in 700-digit decimal arithmetic at the
// exact binary x, normalized afterwards, rounded once to double.
struct Reference {
    double x;
    std::size_t terms, n;
    double value, derivative;
};
const Reference references[] = {
    {0.0, 6, 0, 0x1.8093870155910p-1, 0x0.0p+0},
    {0.0, 6, 1, 0x0.0p+0, 0x1.0fefbe1c2b6c9p+0},
    {0.0, 6, 2, -0x1.0fefbe1c2b6c9p-1, 0x0.0p+0},
    {0.0, 6, 4, 0x1.d70200d3044e3p-2, 0x0.0p+0},
    {0.0, 6, 5, 0x0.0p+0, 0x1.745d480e946afp+0},
    {0.37, 6, 0, 0x1.6721fd3a87aa8p-1, -0x1.09c21cab4fe9bp-2},
    {0.37, 6, 1, 0x1.77d6c38d90e8fp-2, 0x1.b65c384967ce2p-1},
    {0.37, 6, 2, -0x1.70d47d11c698dp-2, 0x1.bc128de40df87p-1},
    {0.37, 6, 3, -0x1.a24bced859b1bp-2, -0x1.7656812f7fc45p-1},
    {0.37, 6, 5, 0x1.a745a01328c85p-2, 0x1.fb65910ee881cp-2},
    {-1.25, 6, 0, 0x1.6024c998e66bfp-2, 0x1.b82dfbff2006fp-2},
    {-1.25, 6, 1, -0x1.3741061056a0ep-1, -0x1.1820ebdb812a7p-2},
    {-1.25, 6, 3, -0x1.fc46911a790b9p-6, 0x1.3a193544f59a3p+0},
    {-1.25, 6, 5, 0x1.707cbab92e49cp-2, -0x1.c1d7ce0cc89f9p-1},
    {2.5, 40, 0, 0x1.0e5a8814c1a98p-5, -0x1.51f12a19f213ep-4},
    {2.5, 40, 7, -0x1.9605910baf4ecp-3, 0x1.37b20079ee8e4p+0},
    {2.5, 40, 20, -0x1.4e49d8752cd24p-2, -0x1.da53c163eab30p-3},
    {2.5, 40, 39, -0x1.2845be85202fdp-6, 0x1.299571ff786c8p+1},
    {-6.0, 64, 0, 0x1.8910058f4d9ccp-27, 0x1.26cc042b7a359p-24},
    {-6.0, 64, 17, -0x1.25152b9ca0492p-2, -0x1.174fe0bec0ea7p-1},
    {-6.0, 64, 63, 0x1.06442d5d731e2p-2, -0x1.51b6806fceedep-2},
    {12.0, 128, 0, 0x1.a3a965c1c22eep-105, -0x1.3abf0c5151a32p-101},
    {12.0, 128, 40, 0x1.eb4306251e796p-26, -0x1.ed1eef19796d2p-23},
    {12.0, 128, 72, 0x1.491086363bd7ep-2, -0x1.3c64be8c33b96p-1},
    {12.0, 128, 127, -0x1.cd80bc93425c4p-3, 0x1.05ba1575db211p+0},
    {13.5, 128, 0, 0x1.168056a6d2ea4p-132, -0x1.d5f8923983eb5p-129},
    {13.5, 128, 90, 0x1.0af306fcd4ca0p-2, -0x1.4180880fdf627p-1},
    {13.5, 128, 127, 0x1.55ba9c2dcaff1p-3, 0x1.dc5608ffbefbbp+0},
    {14.75, 128, 0, 0x1.916a9eb0049dep-158, -0x1.720e4a4a44419p-154},
    {14.75, 128, 100, 0x1.4cc3bd85ed6f3p-5, -0x1.71369b9e74faap-3},
    {14.75, 128, 127, -0x1.3ccbeba833ca5p-3, -0x1.c373a248f6a0ap+0},
    {20.0, 256, 0, 0x1.08aea9c17304dp-289, -0x1.4ada5431cfc60p-285},
    {20.0, 256, 150, 0x1.019e48491802bp-28, -0x1.43974d1783411p-25},
    {20.0, 256, 199, 0x1.04bb9f2c76509p-2, -0x1.5987310eca40dp-1},
    {20.0, 256, 255, 0x1.b8b6e2e978c73p-4, -0x1.28d6b4e87ac70p+1},
    // FP64: exp(-x^2/2) is subnormal from |x| = 37.6 and zero from 38.6,
    // while psi_n near its turning point sqrt(2n+1) is O(0.1).
    {35.0, 1024, 0, 0x1.e9ec7af038df3p-885, -0x1.0bed533b5f1a1p-879},
    {35.0, 1024, 600, 0x1.ca342380a1578p-5, -0x1.3958594302db7p-2},
    {35.0, 1024, 1023, 0x1.fe3893fec4bafp-5, -0x1.f09f3342c7526p+1},
    {37.75, 1024, 0, 0x0.0313e04d8d04bp-1022, -0x0.742e536f8b70ap-1022},
    {37.75, 1024, 700, 0x1.eded35d2f8e8ep-5, -0x1.5459e7d5dae5fp-2},
    {37.75, 1024, 1023, 0x1.31fdff09ca313p-3, 0x1.6a8820da4cf81p+0},
    {38.5, 1024, 0, 0x0.0000000000015p-1022, -0x0.000000000031cp-1022},
    {38.5, 1024, 741, 0x1.00c9cfea2c79dp-2, -0x1.826af324c55abp-1},
    {38.5, 1024, 1023, 0x1.89cd9431f30c7p-4, 0x1.9356a023ce6fdp+1},
    {41.0, 1024, 0, 0x0.0p+0, -0x0.0p+0},
    {41.0, 1024, 840, 0x1.ed54fdf302453p-3, -0x1.86eadfc9c4cc4p-1},
    {41.0, 1024, 1023, -0x1.4e9e424d13f96p-3, -0x1.8fa34b93514acp+0},
    {45.0, 1024, 0, 0x0.0p+0, -0x0.0p+0},
    {45.0, 1024, 1012, 0x1.e5bca11304e6cp-3, -0x1.8cfe855207bd5p-1},
    {45.0, 1024, 1023, 0x1.6d393153aef0dp-2, 0x1.02b364e630666p-3},
    {-44.0, 1024, 1, -0x0.0p+0, -0x0.0p+0},
    {-44.0, 1024, 1000, -0x1.1f281e68d84d8p-2, 0x1.78b607f6f76fap-4},
    {-44.0, 1024, 1022, 0x1.e88e52d68bceep-6, -0x1.487f2cbd3f2ddp+1},
    {60.0, 1024, 0, 0x0.0p+0, -0x0.0p+0},
    {60.0, 1024, 1023, 0x1.ab20ae844b179p-548, -0x1.07222a483a91ep-542},
};
} // namespace

TEST(normalization_is_typed_configuration_with_physicists_default) {
    REQUIRE(HermiteConfig{}.normalization == HermiteNormalization::Physicists);
    REQUIRE(HermiteConfig{6} == (HermiteConfig{6, HermiteNormalization::Physicists}));
    REQUIRE(!(HermiteConfig{6} == function(6)));
    REQUIRE(kan::basis_size(kan::BasisConfig{function(9)}) == 9);
    kan::validate_basis(function(1));
    test::throws<std::invalid_argument>([] { kan::validate_basis(function(0)); });
    const HermiteConfig unknown{4, static_cast<HermiteNormalization>(7)};
    test::throws<std::invalid_argument>([&] { kan::validate_basis(unknown); });
    test::throws<std::invalid_argument>([&] { kan::Layer(1, 1, unknown); });
    // The physicists' option is unchanged: H_4 = 16x^4 - 48x^2 + 12.
    const double x = 0.7;
    kan::BasisValues physicists = kan::evaluate_basis(HermiteConfig{5}, x);
    test::near(physicists.values[4], 16 * x * x * x * x - 48 * x * x + 12);
}

TEST(hermite_functions_match_closed_forms) {
    const double r2 = std::sqrt(2.0), r3 = std::sqrt(3.0), r6 = std::sqrt(6.0);
    for (const double x : {-3.1, -1.0, -0.25, 0.0, 0.4, 1.0, 2.2}) {
        const auto r = kan::evaluate_basis(function(5), x);
        const double g = std::exp(-x * x / 2) / std::sqrt(std::sqrt(std::numbers::pi));
        const double expected[5] = {g, r2 * x * g, (2 * x * x - 1) / r2 * g, (2 * x * x * x - 3 * x) / r3 * g,
                                    (4 * x * x * x * x - 12 * x * x + 3) / (2 * r6) * g};
        // d/dx [p(x) g] = (p' - x p) g
        const double slopes[5] = {-x * g, r2 * (1 - x * x) * g, (4 * x - x * (2 * x * x - 1)) / r2 * g,
                                  (6 * x * x - 3 - x * (2 * x * x * x - 3 * x)) / r3 * g,
                                  (16 * x * x * x - 24 * x - x * (4 * x * x * x * x - 12 * x * x + 3)) / (2 * r6) * g};
        for (std::size_t n = 0; n < 5; ++n) {
            test::near(r.values[n], expected[n], 1e-14);
            test::near(r.derivatives[n], slopes[n], 1e-14);
        }
    }
}

TEST(hermite_functions_match_independent_high_precision_reference) {
    for (const auto& c : references) {
        const auto r = kan::evaluate_basis(function(c.terms), c.x);
        REQUIRE(r.values.size() == c.terms);
        const std::string where = "x=" + std::to_string(c.x) + " n=" + std::to_string(c.n);
        close(r.values[c.n], c.value, 2e-12, ("value " + where).c_str());
        close(r.derivatives[c.n], c.derivative, 2e-12, ("derivative " + where).c_str());
    }
}

// The trapezoid rule is spectrally accurate for these rapidly decaying entire
// functions: h = 1/32 on [-24, 24] resolves psi_m psi_n, m, n < 48, far below 1e-13.
TEST(hermite_functions_are_orthonormal) {
    constexpr std::size_t terms = 48;
    constexpr double h = 1.0 / 32, half = 24;
    std::vector<double> gram(terms * terms, 0.0);
    for (double x = -half; x <= half; x += h) {
        const auto v = kan::evaluate_basis(function(terms), x).values;
        for (std::size_t m = 0; m < terms; ++m)
            for (std::size_t n = 0; n < terms; ++n) gram[m * terms + n] += h * v[m] * v[n];
    }
    for (std::size_t m = 0; m < terms; ++m)
        for (std::size_t n = 0; n < terms; ++n)
            if (std::abs(gram[m * terms + n] - (m == n ? 1.0 : 0.0)) > 1e-13)
                throw std::runtime_error("gram(" + std::to_string(m) + "," + std::to_string(n) +
                                         ")=" + std::to_string(gram[m * terms + n]));
}

TEST(hermite_function_derivatives_match_finite_differences) {
    constexpr std::size_t terms = 40;
    for (const double x : {-7.5, -2.0, -0.3, 0.0, 0.9, 3.3, 8.0}) {
        const auto r = kan::evaluate_basis(function(terms), x);
        const double h = 1e-5;
        const auto plus = kan::evaluate_basis(function(terms), x + h).values;
        const auto minus = kan::evaluate_basis(function(terms), x - h).values;
        for (std::size_t n = 0; n < terms; ++n) test::near(r.derivatives[n], (plus[n] - minus[n]) / (2 * h), 1e-8);
    }
}

// psi_n is bounded by pi^(-1/4) and decays; the evaluation never produces a
// nonfinite value (no exp underflow times polynomial overflow) and returns
// exact zeros in the remote tail.
TEST(hermite_functions_are_bounded_and_finite_for_every_input) {
    const double bound = (1 + 1e-15) / std::sqrt(std::sqrt(std::numbers::pi));
    for (const std::size_t terms : {std::size_t{1}, std::size_t{2}, std::size_t{17}, std::size_t{300}}) {
        for (const double x : {0.0, 5e-324, -1e-300, 3.0, -25.0, 39.0, 75.0, -1e10, 1e154, 1e300,
                               std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()}) {
            const auto r = kan::evaluate_basis(function(terms), x);
            for (std::size_t n = 0; n < terms; ++n) {
                REQUIRE(std::isfinite(r.values[n]) && std::isfinite(r.derivatives[n]));
                REQUIRE(std::abs(r.values[n]) <= bound);
            }
            if (std::abs(x) >= 1e10) {
                for (std::size_t n = 0; n < terms; ++n) REQUIRE(r.values[n] == 0 && r.derivatives[n] == 0);
            }
        }
    }
    // Parity: psi_n(-x) = (-1)^n psi_n(x), psi_n'(-x) = (-1)^(n+1) psi_n'(x), bitwise,
    // without (x = 4.25) and with (x = 41, size 1024) rescaling.
    for (const auto& [x, terms] : {std::pair{4.25, std::size_t{64}}, std::pair{41.0, std::size_t{1024}}}) {
        const auto a = kan::evaluate_basis(function(terms), x), b = kan::evaluate_basis(function(terms), -x);
        for (std::size_t n = 0; n < terms; ++n) {
            REQUIRE(b.values[n] == (n % 2 ? -a.values[n] : a.values[n]));
            REQUIRE(b.derivatives[n] == (n % 2 ? a.derivatives[n] : -a.derivatives[n]));
        }
    }
}

TEST(hermite_function_layers_train_like_other_basis_layers) {
    kan::Layer layer(2, 3, function(6));
    std::vector<double> c(layer.coefficients().size());
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = std::sin(0.37 * double(i) + 0.2);
    const std::vector<double> bias{0.1, -0.2, 0.3};
    layer.set_parameters(c, bias);
    const std::vector<double> x{-0.6, 1.7, 0.7, -2.4};
    const std::vector<double> g{0.4, -0.7, 0.3, 0.2, 0.5, -0.2};
    const auto y = layer.forward(x, 2);
    for (std::size_t b = 0; b < 2; ++b)
        for (std::size_t o = 0; o < 3; ++o) {
            double expected = layer.bias()[o];
            for (std::size_t i = 0; i < 2; ++i) {
                const auto v = kan::evaluate_basis(function(6), x[b * 2 + i]).values;
                for (std::size_t k = 0; k < 6; ++k) expected += c[(o * 2 + i) * 6 + k] * v[k];
            }
            test::near(y[b * 3 + o], expected, 1e-14);
        }
    const auto grad = layer.backward(x, 2, g);
    const auto objective = [&](const kan::Layer& l, const std::vector<double>& in) {
        const auto out = l.forward(in, 2);
        double s = 0;
        for (std::size_t i = 0; i < out.size(); ++i) s += out[i] * g[i];
        return s;
    };
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto plus = x, minus = x;
        plus[i] += 1e-6;
        minus[i] -= 1e-6;
        test::near(grad.input[i], (objective(layer, plus) - objective(layer, minus)) / 2e-6, 2e-7);
    }
}

int main() { return test::run(); }

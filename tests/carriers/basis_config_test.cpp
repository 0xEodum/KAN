// Typed per-family basis configuration (backlog R1).
#include "kan/families.hpp"
#include "support/test.hpp"

#include <limits>
#include <variant>

TEST(basis_size_is_explicit_for_global_and_derived_for_local_families) {
    const std::vector<std::pair<kan::BasisConfig, std::size_t>> cases{
        {kan::ChebyshevConfig{5}, 5},
        {kan::LegendreConfig{2}, 2},
        {kan::HermiteConfig{3}, 3},
        {kan::JacobiConfig{4, 0.5, -0.5}, 4},
        {kan::FourierConfig{7, 2.0}, 7},
        {kan::GaussianRbfConfig{{-1, 0, 1}, 0.5}, 3},
        {kan::TrainableRbfConfig{{0, 1}, {0, -0.5}}, 2},
        {kan::BSplineConfig{2, {0, 0, 0, 0.5, 1, 1, 1}}, 4},
        {kan::MexicanHatConfig{{0}, {1}}, 1},
    };
    for (const auto& [config, size] : cases) {
        REQUIRE(kan::basis_size(config) == size);
        kan::validate_basis(config);
        const auto result = kan::evaluate_basis(config, 0.25);
        REQUIRE(result.values.size() == size);
        REQUIRE(result.derivatives.size() == size);
    }
}

TEST(default_configurations_of_global_families_are_valid) {
    for (const kan::BasisConfig config :
         {kan::BasisConfig{kan::ChebyshevConfig{}}, kan::BasisConfig{kan::LegendreConfig{}},
          kan::BasisConfig{kan::HermiteConfig{}}, kan::BasisConfig{kan::JacobiConfig{}},
          kan::BasisConfig{kan::FourierConfig{}}}) {
        kan::validate_basis(config);
        REQUIRE(kan::basis_size(config) > 0);
    }
    REQUIRE(std::holds_alternative<kan::ChebyshevConfig>(kan::BasisConfig{}));
}

TEST(localized_families_reject_empty_and_mismatched_vectors) {
    for (const kan::BasisConfig bad :
         {kan::BasisConfig{kan::GaussianRbfConfig{}}, kan::BasisConfig{kan::TrainableRbfConfig{}},
          kan::BasisConfig{kan::MexicanHatConfig{}},
          kan::BasisConfig{kan::TrainableRbfConfig{{0, 1}, {0}}},
          kan::BasisConfig{kan::MexicanHatConfig{{0, 1}, {1}}}})
        test::throws<std::invalid_argument>([&] { kan::validate_basis(bad); });
}

TEST(spline_with_too_few_knots_has_no_size_and_is_rejected) {
    const kan::BasisConfig short_spline = kan::BSplineConfig{3, {0, 1}};
    REQUIRE(kan::basis_size(short_spline) == 0);
    test::throws<std::invalid_argument>([&] { kan::validate_basis(short_spline); });
    test::throws<std::invalid_argument>([&] { kan::Layer(1, 1, short_spline); });
}

TEST(only_trainable_rbf_returns_nonlinear_derivatives) {
    const auto fixed = kan::evaluate_basis(kan::GaussianRbfConfig{{0, 1}, 0.7}, 0.2);
    REQUIRE(fixed.center_derivatives.empty());
    REQUIRE(fixed.log_width_derivatives.empty());
    const auto trainable = kan::evaluate_basis(kan::TrainableRbfConfig{{0, 1}, {0, 0}}, 0.2);
    REQUIRE(trainable.center_derivatives.size() == 2);
    REQUIRE(trainable.log_width_derivatives.size() == 2);
    // Same Gaussian with width exp(0) = 1.
    const auto same = kan::evaluate_basis(kan::GaussianRbfConfig{{0, 1}, 1.0}, 0.2);
    for (std::size_t k = 0; k < 2; ++k) {
        test::near(trainable.values[k], same.values[k], 0);
        test::near(trainable.derivatives[k], same.derivatives[k], 0);
        test::near(trainable.center_derivatives[k], -same.derivatives[k], 0);
    }
}

TEST(layer_exposes_its_typed_configuration) {
    kan::Layer spline(2, 1, kan::BSplineConfig{1, {-1, -1, 0, 1, 1}});
    const auto basis = [](const kan::Layer& layer) { return std::get<kan::BasisEdges>(layer.carrier()).basis; };
    REQUIRE(std::holds_alternative<kan::BSplineConfig>(basis(spline)));
    REQUIRE(spline.coefficients().size() == 2 * 3);
    kan::insert_knot(spline, 0.5);
    const auto refined = std::get<kan::BSplineConfig>(basis(spline));
    REQUIRE(refined.knots.size() == 6);
    REQUIRE(kan::basis_size(basis(spline)) == 4);
    REQUIRE(spline.coefficients().size() == 2 * 4);

    kan::Layer fixed(1, 1, kan::GaussianRbfConfig{{0, 1}, 0.7});
    test::throws<std::invalid_argument>([&] {
        kan::set_rbf_parameters(fixed, std::vector<double>{0, 1}, std::vector<double>{0, 0});
    });
    kan::Layer trainable(1, 1, kan::TrainableRbfConfig{{0, 1}, {0, 0}});
    kan::set_rbf_parameters(trainable, std::vector<double>{0.1, 0.9}, std::vector<double>{-0.1, 0.1});
    const auto rbf_basis = [](const kan::Layer& layer) { return std::get<kan::TrainableRbfEdges>(layer.carrier()).basis; };
    const auto before = rbf_basis(trainable);
    // The derived term count cannot change through the parameter setter.
    test::throws<std::invalid_argument>([&] {
        kan::set_rbf_parameters(trainable, std::vector<double>{0, 1, 2}, std::vector<double>{0, 0, 0});
    });
    REQUIRE(rbf_basis(trainable) == before);
    const auto rbf = rbf_basis(trainable);
    REQUIRE(rbf.centers == (std::vector<double>{0.1, 0.9}));
    REQUIRE(rbf.log_widths == (std::vector<double>{-0.1, 0.1}));
    const auto gradient = trainable.backward(std::vector<double>{0.3}, 1, std::vector<double>{1});
    const auto& nonlinear = std::get<kan::TrainableRbfGradients>(gradient.nonlinear);
    REQUIRE(nonlinear.centers.size() == 2);
    REQUIRE(nonlinear.log_widths.size() == 2);
}

int main() { return test::run(); }

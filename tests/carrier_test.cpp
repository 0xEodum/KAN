// R2: a Layer holds one typed carrier. Linear-in-parameters bases, trainable
// RBFs and rational edges are distinct carrier types with their own nonlinear
// gradients; family-specific operations are free functions, not Layer members.
#include "kan/families.hpp"
#include "kan/network.hpp"
#include "support/test.hpp"
#include <limits>
#include <variant>

namespace {
const double inf = std::numeric_limits<double>::infinity();

kan::BSplineConfig cubic() { return {3, {-1, -1, -1, -1, 0, 1, 1, 1, 1}}; }
kan::TrainableRbfConfig rbf() { return {{-0.5, 0.5}, {-0.2, 0.1}}; }
kan::RationalConfig pade() { return {2, 1, 0.1, 1.3, 1e-8}; }

std::vector<double> ramp(std::size_t n, double scale) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * (static_cast<double>(k % 5) - 2);
    return v;
}
} // namespace

TEST(construction_selects_the_carrier_type) {
    const kan::Layer basis(2, 3, kan::ChebyshevConfig{4});
    const auto* linear = std::get_if<kan::BasisEdges>(&basis.carrier());
    REQUIRE(linear != nullptr);
    REQUIRE(linear->basis == kan::BasisConfig{kan::ChebyshevConfig{4}});
    REQUIRE(linear->coefficients == std::vector<double>(24, 0.0));
    REQUIRE(basis.terms() == 4);

    const kan::Layer trainable(1, 2, rbf());
    const auto* nonlinear = std::get_if<kan::TrainableRbfEdges>(&trainable.carrier());
    REQUIRE(nonlinear != nullptr);
    REQUIRE(nonlinear->basis == rbf());
    REQUIRE(nonlinear->coefficients.size() == 4);

    const kan::Layer rational(2, 1, pade());
    const auto* edges = std::get_if<kan::RationalEdges>(&rational.carrier());
    REQUIRE(edges != nullptr);
    REQUIRE(edges->config == pade());
    REQUIRE(edges->coefficients.size() == 6);
    REQUIRE(edges->denominators.size() == 2);
    REQUIRE(rational.terms() == 3);
    REQUIRE(rational.coefficients().size() == 6);
}

TEST(gradients_mirror_the_carrier) {
    const std::vector<double> x{0.2, -0.4}, u{0.3, 0.7};
    kan::Layer basis(1, 2, kan::LegendreConfig{3});
    REQUIRE(std::holds_alternative<std::monostate>(basis.backward(x, 2, std::vector<double>{1, 1, 1, 1}).nonlinear));

    kan::Layer trainable(1, 1, rbf());
    trainable.set_parameters(std::vector<double>{0.4, -0.3}, std::vector<double>{0});
    const auto g = trainable.backward(x, 2, u);
    const auto* rbf_gradient = std::get_if<kan::TrainableRbfGradients>(&g.nonlinear);
    REQUIRE(rbf_gradient != nullptr);
    REQUIRE(rbf_gradient->centers.size() == 2 && rbf_gradient->log_widths.size() == 2);
    const auto penalty = trainable.regularization(0.1).gradients;
    REQUIRE(std::get<kan::TrainableRbfGradients>(penalty.nonlinear).centers == std::vector<double>(2, 0.0));

    kan::Layer rational(1, 1, pade());
    const auto r = rational.backward(x, 2, u);
    REQUIRE(std::get<kan::RationalGradients>(r.nonlinear).denominators.size() == 1);
    REQUIRE(std::get<kan::RationalGradients>(rational.backward({}, 0, {}).nonlinear).denominators ==
            std::vector<double>{0});
}

TEST(set_parameters_is_generic_over_carriers) {
    kan::Layer rational(1, 1, pade());
    rational.set_parameters(std::vector<double>{1, 2, 0}, std::vector<double>{0.5});
    const auto& edges = std::get<kan::RationalEdges>(rational.carrier());
    REQUIRE(edges.coefficients == (std::vector<double>{1, 2, 0}));
    REQUIRE(edges.denominators == std::vector<double>{0});
    const double z = (0.3 - 0.1) / 1.3;
    test::near(rational.forward(std::vector<double>{0.3}, 1)[0], 0.5 + 1 + 2 * z);
    test::throws<std::invalid_argument>([&] { rational.set_parameters(std::vector<double>{1, 2}, std::vector<double>{0}); });
}

TEST(set_carrier_validates_and_is_atomic) {
    kan::Layer layer(2, 1, cubic());
    layer.set_parameters(ramp(10, 0.1), std::vector<double>{0.2});
    const auto before = layer.carrier();

    test::throws<std::invalid_argument>([&] { layer.set_carrier(kan::BasisEdges{kan::ChebyshevConfig{4}, ramp(7, 1)}); });
    test::throws<std::invalid_argument>([&] { layer.set_carrier(kan::BasisEdges{kan::ChebyshevConfig{0}, {}}); });
    test::throws<std::invalid_argument>([&] {
        layer.set_carrier(kan::BasisEdges{kan::ChebyshevConfig{1}, std::vector<double>{1, inf}});
    });
    // A trainable RBF configuration needs its own carrier type.
    test::throws<std::invalid_argument>([&] { layer.set_carrier(kan::BasisEdges{rbf(), ramp(4, 1)}); });
    test::throws<std::invalid_argument>([&] {
        layer.set_carrier(kan::RationalEdges{pade(), ramp(6, 1), std::vector<double>{0}});
    });
    test::throws<std::invalid_argument>([&] {
        layer.set_carrier(kan::BasisEdges{kan::ChebyshevConfig{2}, ramp(4, 1)}, std::vector<double>{inf});
    });
    REQUIRE(layer.carrier() == before);
    REQUIRE(layer.bias()[0] == 0.2);

    layer.set_carrier(kan::BasisEdges{kan::ChebyshevConfig{2}, std::vector<double>{1, 2, 3, 4}});
    test::near(layer.forward(std::vector<double>{0.5, -0.5}, 1)[0], 0.2 + (1 + 2 * 0.5) + (3 - 4 * 0.5));
    layer.set_carrier(kan::RationalEdges{pade(), ramp(6, 0.1), std::vector<double>{0.01, -0.02}},
                      std::vector<double>{-1});
    REQUIRE(std::holds_alternative<kan::RationalEdges>(layer.carrier()));
    REQUIRE(layer.bias()[0] == -1);
    REQUIRE(layer.terms() == 3);
}

TEST(family_operations_are_free_functions) {
    kan::Layer spline(1, 1, cubic());
    spline.set_parameters(std::vector<double>{0.1, -0.2, 0.3, 0.4, -0.5}, std::vector<double>{0});
    const auto y = spline.forward(std::vector<double>{0.3}, 1);
    kan::insert_knot(spline, 0.5);
    REQUIRE(spline.terms() == 6);
    test::near(spline.forward(std::vector<double>{0.3}, 1)[0], y[0], 1e-12);
    test::near(kan::adapt_grid(spline, std::vector<double>{-0.6, -0.4, -0.5}), -0.5);
    REQUIRE(kan::basis_size(std::get<kan::BasisEdges>(spline.carrier()).basis) == 7);

    kan::Layer chebyshev(1, 1, kan::ChebyshevConfig{3});
    test::throws<std::invalid_argument>([&] { kan::insert_knot(chebyshev, 0.5); });
    test::throws<std::invalid_argument>([&] { kan::adapt_grid(chebyshev, std::vector<double>{0.5}); });
    test::throws<std::invalid_argument>([&] {
        kan::set_rbf_parameters(chebyshev, std::vector<double>{0}, std::vector<double>{0});
    });
    test::throws<std::invalid_argument>([&] {
        kan::set_rational_parameters(chebyshev, std::vector<double>{0, 0, 0}, std::vector<double>{}, std::vector<double>{0});
    });

    kan::Layer trainable(1, 1, rbf());
    kan::set_rbf_parameters(trainable, std::vector<double>{0.1, 0.2}, std::vector<double>{0.3, 0.4});
    REQUIRE((std::get<kan::TrainableRbfEdges>(trainable.carrier()).basis ==
             kan::TrainableRbfConfig{{0.1, 0.2}, {0.3, 0.4}}));

    kan::Layer rational(1, 1, pade());
    kan::set_rational_parameters(rational, std::vector<double>{1, 0, 0}, std::vector<double>{0.5}, std::vector<double>{2});
    REQUIRE(std::get<kan::RationalEdges>(rational.carrier()).denominators == std::vector<double>{0.5});
    REQUIRE(rational.bias()[0] == 2);
    test::throws<std::invalid_argument>([&] {
        kan::set_rational_parameters(rational, std::vector<double>{1, 0, 0}, std::vector<double>{inf}, std::vector<double>{2});
    });
    REQUIRE(std::get<kan::RationalEdges>(rational.carrier()).denominators == std::vector<double>{0.5});

    kan::Network network({kan::Layer(1, 1, cubic())});
    network.insert_knot(0, 0.25);
    REQUIRE(network.layers()[0].terms() == 6);
}

TEST(sgd_rejects_gradients_of_another_carrier) {
    const std::vector<double> x{0.2}, u{1};
    kan::Layer rational(1, 1, pade());
    auto g = rational.backward(x, 1, u);
    auto wrong = g;
    wrong.nonlinear = std::monostate{};
    test::throws<std::invalid_argument>([&] { rational.sgd(wrong, 0.1); });
    wrong.nonlinear = kan::TrainableRbfGradients{{0}, {0}};
    test::throws<std::invalid_argument>([&] { rational.sgd(wrong, 0.1); });
    rational.sgd(g, 0.1);

    kan::Layer trainable(1, 1, rbf());
    auto t = trainable.backward(x, 1, u);
    t.nonlinear = kan::RationalGradients{{0}};
    test::throws<std::invalid_argument>([&] { trainable.sgd(t, 0.1); });

    t = trainable.backward(x, 1, u);
    std::get<kan::TrainableRbfGradients>(t.nonlinear).log_widths.pop_back();
    test::throws<std::invalid_argument>([&] { trainable.sgd(t, 0.1); });

    kan::Layer basis(1, 1, kan::HermiteConfig{3});
    auto b = basis.backward(x, 1, u);
    b.nonlinear = kan::RationalGradients{};
    test::throws<std::invalid_argument>([&] { basis.sgd(b, 0.1); });
}

TEST(invalid_and_moved_from_carriers_are_rejected) {
    kan::Layer rational(1, 1, pade());
    test::throws<std::invalid_argument>([&] {
        rational.set_carrier(kan::RationalEdges{pade(), ramp(2, 1), std::vector<double>{0}});
    });
    kan::Layer spline(1, 1, cubic());
    auto moved = std::move(spline);
    test::throws<std::invalid_argument>([&] { kan::insert_knot(spline, 0.5); });
    test::throws<std::invalid_argument>([&] { kan::adapt_grid(spline, std::vector<double>{0.5}); });
    test::throws<std::invalid_argument>([&] { spline.set_carrier(moved.carrier()); });
    kan::insert_knot(moved, 0.5);
    REQUIRE(moved.terms() == 6);
}

int main() { return test::run(); }

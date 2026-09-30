#include "kan/layer.hpp"
#include "support/test.hpp"
#include <limits>
#include <numeric>

namespace {
kan::Layer fixture(kan::BasisConfig config = {}) {
    kan::Layer layer(2, 3, config);
    std::vector<double> c(layer.coefficients().size());
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = 0.02 * (static_cast<double>(i % 7) - 3.0);
    const std::vector<double> b{0.1, -0.2, 0.3};
    layer.set_parameters(c, b);
    return layer;
}
double objective(const kan::Layer& layer, const std::vector<double>& x, const std::vector<double>& g) {
    const auto y = layer.forward(x, 2);
    return std::inner_product(y.begin(), y.end(), g.begin(), 0.0);
}
}

TEST(zero_initialization) {
    kan::Layer layer(2, 3, {});
    const auto y = layer.forward(std::vector<double>{0.2, -0.4, 1.0, -1.0}, 2);
    REQUIRE(y.size() == 6);
    for (double v : y) test::near(v, 0.0);
}
TEST(known_forward_layout_and_batch) {
    kan::BasisConfig basis; basis.size = 3;
    kan::Layer layer(2, 2, basis);
    // y0 = 1 + 2*x0 + 3*T2(x1); y1 = -2 + 4*x1
    const std::vector<double> c{0,2,0, 0,0,3, 0,0,0, 0,4,0}, b{1,-2};
    layer.set_parameters(c,b);
    const auto y = layer.forward(std::vector<double>{0.5, 0.0, -1.0, 1.0},2);
    test::near(y[0],-1); test::near(y[1],-2); test::near(y[2],2); test::near(y[3],2);
}
TEST(all_basis_layer_gradients_match_finite_differences) {
    for (auto kind : {kan::BasisKind::Chebyshev,kan::BasisKind::Legendre,kan::BasisKind::Jacobi,
                      kan::BasisKind::Hermite,kan::BasisKind::Fourier,kan::BasisKind::GaussianRbf}) {
        kan::BasisConfig config; config.kind=kind; config.size=5; config.alpha=0.3; config.beta=0.7;
        config.frequency=1.7; config.width=0.8; config.centers={-1,-0.5,0,0.5,1};
        auto layer=fixture(config);
        std::vector<double> x{-0.6,0.2,0.7,-0.1}, g{0.4,-0.7,0.3,0.2,0.5,-0.2};
        const auto grad=layer.backward(x,2,g);
        const double h=1e-6;
        for (std::size_t i=0;i<x.size();++i) {
            auto plus=x,minus=x; plus[i]+=h;minus[i]-=h;
            test::near(grad.input[i],(objective(layer,plus,g)-objective(layer,minus,g))/(2*h),2e-7);
        }
        std::vector<double> c(layer.coefficients().begin(),layer.coefficients().end()), b(layer.bias().begin(),layer.bias().end());
        for (std::size_t i=0;i<c.size();++i) {
            auto plus=layer,minus=layer; auto pc=c,mc=c;pc[i]+=h;mc[i]-=h;
            plus.set_parameters(pc,b);minus.set_parameters(mc,b);
            test::near(grad.coefficients[i],(objective(plus,x,g)-objective(minus,x,g))/(2*h),2e-7);
        }
        for (std::size_t i=0;i<b.size();++i) {
            auto plus=layer,minus=layer;auto pb=b,mb=b;pb[i]+=h;mb[i]-=h;
            plus.set_parameters(c,pb);minus.set_parameters(c,mb);
            test::near(grad.bias[i],(objective(plus,x,g)-objective(minus,x,g))/(2*h),2e-7);
        }
    }
}
TEST(backward_sums_without_averaging_or_mutation) {
    kan::BasisConfig basis; basis.size=1; kan::Layer layer(1,1,basis);
    const auto grad=layer.backward(std::vector<double>{0.2,0.4},2,std::vector<double>{2,3});
    test::near(grad.coefficients[0],5);test::near(grad.bias[0],5);
    for(double v:grad.input)test::near(v,0);
    test::near(layer.coefficients()[0],0);
}
TEST(empty_batch_is_well_defined) {
    auto layer=fixture(); REQUIRE(layer.forward({},0).empty());
    const auto grad=layer.backward({},0,{});REQUIRE(grad.input.empty());
    REQUIRE(grad.coefficients.size()==layer.coefficients().size());
    for(double v:grad.coefficients)test::near(v,0);for(double v:grad.bias)test::near(v,0);
}
TEST(invalid_dimensions_and_shape_overflow) {
    test::throws<std::invalid_argument>([]{kan::Layer layer(0,1,{});});
    test::throws<std::invalid_argument>([]{kan::Layer layer(1,0,{});});
    test::throws<std::overflow_error>([]{kan::Layer layer(std::numeric_limits<std::size_t>::max(),2,{});});
    auto layer=fixture();
    test::throws<std::overflow_error>([&]{layer.forward({},std::numeric_limits<std::size_t>::max());});
    test::throws<std::overflow_error>([&]{layer.backward({},std::numeric_limits<std::size_t>::max(),{});});
    kan::BasisConfig bad;bad.size=0;
    test::throws<std::invalid_argument>([&]{kan::Layer layer(1,1,bad);});
}
TEST(invalid_shapes_and_nonfinite_data) {
    auto layer=fixture();std::vector<double> x{0.2,0.3},g{1,2,3};
    test::throws<std::invalid_argument>([&]{layer.forward(x,2);});
    test::throws<std::invalid_argument>([&]{layer.backward(x,1,x);});
    test::throws<std::invalid_argument>([&]{layer.backward(x,2,g);});
    test::throws<std::invalid_argument>([&]{layer.forward(x,0);});
    x[0]=std::numeric_limits<double>::quiet_NaN();
    test::throws<std::invalid_argument>([&]{layer.forward(x,1);});
    test::throws<std::invalid_argument>([&]{layer.backward(x,1,g);});
    x[0]=0.2;g[0]=std::numeric_limits<double>::infinity();
    test::throws<std::invalid_argument>([&]{layer.backward(x,1,g);});
}
TEST(parameter_set_is_validated_before_mutation) {
    auto layer=fixture();const auto before=layer.forward(std::vector<double>{0.1,0.2},1);
    std::vector<double> c(layer.coefficients().begin(),layer.coefficients().end()),b(layer.bias().begin(),layer.bias().end());
    test::throws<std::invalid_argument>([&]{layer.set_parameters({},b);});
    test::throws<std::invalid_argument>([&]{layer.set_parameters(c,{});});
    c[0]=std::numeric_limits<double>::infinity();
    test::throws<std::invalid_argument>([&]{layer.set_parameters(c,b);});
    c[0]=0;b[1]=std::numeric_limits<double>::quiet_NaN();
    test::throws<std::invalid_argument>([&]{layer.set_parameters(c,b);});
    REQUIRE(layer.forward(std::vector<double>{0.1,0.2},1)==before);
}
TEST(sgd_updates_with_explicit_rate) {
    auto layer=fixture();auto grad=layer.backward(std::vector<double>{0.3,0.1},1,std::vector<double>{1,2,3});
    std::vector<double> c(layer.coefficients().begin(),layer.coefficients().end()),b(layer.bias().begin(),layer.bias().end());
    layer.sgd(grad,0.1);
    for(std::size_t i=0;i<c.size();++i)test::near(layer.coefficients()[i],c[i]-0.1*grad.coefficients[i]);
    for(std::size_t i=0;i<b.size();++i)test::near(layer.bias()[i],b[i]-0.1*grad.bias[i]);
}
TEST(sgd_invalid_update_is_atomic) {
    auto layer=fixture();auto grad=layer.backward(std::vector<double>{0.3,0.1},1,std::vector<double>{1,2,3});
    const std::vector<double> c(layer.coefficients().begin(),layer.coefficients().end());
    for(double rate:{0.0,-1.0,std::numeric_limits<double>::infinity()})
        test::throws<std::invalid_argument>([&]{layer.sgd(grad,rate);});
    auto bad=grad;bad.coefficients.pop_back();test::throws<std::invalid_argument>([&]{layer.sgd(bad,0.1);});
    bad=grad;bad.bias.clear();test::throws<std::invalid_argument>([&]{layer.sgd(bad,0.1);});
    bad=grad;bad.bias.back()=std::numeric_limits<double>::quiet_NaN();
    test::throws<std::invalid_argument>([&]{layer.sgd(bad,0.1);});
    bad=grad;bad.coefficients.back()=std::numeric_limits<double>::max();
    test::throws<std::overflow_error>([&]{layer.sgd(bad,2.0);});
    REQUIRE(std::vector<double>(layer.coefficients().begin(),layer.coefficients().end())==c);
}
TEST(nonfinite_contractions_raise_overflow) {
    kan::BasisConfig config;config.size=1;kan::Layer layer(2,1,config);
    const double huge=std::numeric_limits<double>::max();
    layer.set_parameters(std::vector<double>{huge,huge},std::vector<double>{0});
    test::throws<std::overflow_error>([&]{layer.forward(std::vector<double>{0,0},1);});
    test::throws<std::overflow_error>([&]{layer.backward(std::vector<double>{0,0,0,0},2,std::vector<double>{huge,huge});});
}

TEST(moved_from_layer_rejects_numerical_and_update_operations) {
    auto source=fixture(); auto destination=std::move(source);
    REQUIRE(destination.forward(std::vector<double>{0.1,0.2},1).size()==3);
    test::throws<std::invalid_argument>([&]{source.forward(std::vector<double>{0.1,0.2},1);});
    test::throws<std::invalid_argument>([&]{source.backward(std::vector<double>{0.1,0.2},1,std::vector<double>{1,1,1});});
    test::throws<std::invalid_argument>([&]{source.set_parameters({},{});});
    test::throws<std::invalid_argument>([&]{source.sgd({},0.1);});
    source=fixture();REQUIRE(source.forward(std::vector<double>{0.1,0.2},1).size()==3);
}

int main(){return test::run();}

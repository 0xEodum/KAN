#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include "support/test.hpp"
#include <cmath>
#include <limits>

namespace {
void compare(std::span<const double> a, std::span<const double> b) {
    REQUIRE(a.size()==b.size());for(std::size_t i=0;i<a.size();++i)test::near(a[i],b[i],5e-10);
}
kan::Layer rational(std::size_t in,std::size_t out,std::size_t m=3,std::size_t n=2) {
    kan::RationalConfig r; r.numerator_degree=m;r.denominator_degree=n;r.center=0.1;r.scale=1.3;
    kan::Layer l(in,out,r);std::vector<double>a(l.coefficients().size()),b(l.denominators().size());
    for(std::size_t k=0;k<a.size();++k)a[k]=0.04*std::sin(static_cast<double>(k+1));
    for(std::size_t k=0;k<b.size();++k)b[k]=0.03*std::cos(static_cast<double>(k+1));
    l.set_rational_parameters(a,b,std::vector<double>(out,0.01));return l;
}
void gradients(const kan::NetworkGradients& a,const kan::NetworkGradients& b) {
    compare(a.input,b.input);REQUIRE(a.layers.size()==b.layers.size());
    for(std::size_t j=0;j<a.layers.size();++j) {
        compare(a.layers[j].coefficients,b.layers[j].coefficients);compare(a.layers[j].denominators,b.layers[j].denominators);
        compare(a.layers[j].bias,b.layers[j].bias);compare(a.layers[j].input,b.layers[j].input);
    }
}
}
TEST(m4_resident_independent_pade_identity_and_vjp) {
    kan::RationalConfig r;r.numerator_degree=1;r.denominator_degree=1;kan::Layer l(1,1,r);
    l.set_rational_parameters(std::vector<double>{1,0.5},std::vector<double>{-0.5},std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({l}),3);const std::vector<double>x{-0.5,0,0.5};
    gpu.upload_input(x,3);gpu.upload_output_gradient(std::vector<double>{1,1,1});gpu.forward();gpu.backward();
    compare(gpu.download_output(),std::vector<double>{0.6,1,5.0/3.0});const auto g=gpu.download_gradients();
    compare(g.input,std::vector<double>{0.64,1,16.0/9.0});
    compare(g.layers[0].coefficients,std::vector<double>{0.8+1+4.0/3.0,-0.4+2.0/3.0});
    compare(g.layers[0].denominators,std::vector<double>{0.24-10.0/9.0});
}
TEST(m4_resident_mixed_network_all_vjps_and_trajectory) {
    for(const auto orders:{std::pair<std::size_t,std::size_t>{0,0},{0,3},{4,1},{16,16}}) {
        kan::Layer basis(3,2,{kan::BasisKind::Chebyshev,3});basis.set_parameters(std::vector<double>(18,0.02),std::vector<double>(2,0));
        kan::Network cpu({rational(2,3,orders.first,orders.second),basis,rational(2,1)});kan::cuda::ResidentNetwork gpu(cpu,8);
        const auto count=gpu.workspace_allocations();const std::vector<double>x{-0.5,0.2,0.8,-0.2,0.1,0.4},dy{0.2,-0.1,0.3};
        gpu.upload_input(x,3);gpu.upload_output_gradient(dy);
        for(int step=0;step<3;++step) {
            gpu.forward();compare(gpu.download_output(),cpu.forward(x,3));gpu.backward(0.1);
            auto expected=cpu.backward(x,3,dy);const auto reg=cpu.regularization(0.1).gradients;
            for(std::size_t j=0;j<expected.layers.size();++j)for(std::size_t k=0;k<expected.layers[j].coefficients.size();++k)expected.layers[j].coefficients[k]+=reg.layers[j].coefficients[k];
            gradients(gpu.download_gradients(),expected);gpu.sgd(0.03);cpu.sgd(expected,0.03);
        }
        const auto actual=gpu.download_parameters();
        for(std::size_t j=0;j<cpu.layers().size();++j){compare(actual.layers()[j].coefficients(),cpu.layers()[j].coefficients());compare(actual.layers()[j].denominators(),cpu.layers()[j].denominators());}
        REQUIRE(count==gpu.workspace_allocations());
    }
}
TEST(m4_resident_guard_invalidation_recovery_and_overflow) {
    kan::RationalConfig r;r.numerator_degree=1;r.denominator_degree=1;kan::Layer l(1,1,r);
    l.set_rational_parameters(std::vector<double>{1,-1},std::vector<double>{-1},std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({l}),1);
    for(double pole:{1.0,1.0-1e-9}) {
        gpu.upload_input(std::vector<double>{pole},1);gpu.upload_output_gradient(std::vector<double>{0});
        test::throws<std::domain_error>([&]{gpu.forward();});test::throws<std::logic_error>([&]{gpu.download_output();});
        test::throws<std::logic_error>([&]{gpu.backward();});test::throws<std::logic_error>([&]{gpu.sgd(0.1);});
    }
    gpu.upload_input(std::vector<double>{0.2},1);gpu.upload_output_gradient(std::vector<double>{1});gpu.forward();gpu.backward();compare(gpu.download_output(),std::vector<double>{1});
    auto huge=rational(1,1,16,0);kan::cuda::ResidentNetwork over(kan::Network({huge}),1);over.upload_input(std::vector<double>{1e100},1);
    test::throws<std::overflow_error>([&]{over.forward();});test::throws<std::logic_error>([&]{over.download_output();});
}
TEST(m4_resident_zero_batch_l2_and_atomic_sgd) {
    auto l=rational(1,1);kan::Network cpu({l});kan::cuda::ResidentNetwork empty(cpu,0);
    empty.upload_input({},0);empty.upload_output_gradient({});empty.forward();empty.backward(0.3);gradients(empty.download_gradients(),cpu.regularization(0.3).gradients);
    auto g=cpu.regularization(0.3).gradients;empty.sgd(0.1);cpu.sgd(g,0.1);compare(empty.download_parameters().layers()[0].denominators(),l.denominators());
    kan::RationalConfig r;r.numerator_degree=0;r.denominator_degree=1;kan::Layer first(1,1,r),last(1,1,r);
    first.set_rational_parameters(std::vector<double>{1},std::vector<double>{0},std::vector<double>{0});
    last.set_rational_parameters(std::vector<double>{1e100},std::vector<double>{0},std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({first,last}),1);gpu.upload_input(std::vector<double>{0},1);gpu.upload_output_gradient(std::vector<double>{1e100});gpu.forward();gpu.backward();
    test::throws<std::overflow_error>([&]{gpu.sgd(1e200);});auto same=gpu.download_parameters();compare(same.layers()[0].coefficients(),first.coefficients());compare(same.layers()[1].denominators(),last.denominators());
    gpu.sgd(1e-201);REQUIRE(std::isfinite(gpu.download_parameters().layers()[1].denominators()[0]));
}
TEST(m4_resident_representable_extreme_denominator_derivatives) {
    kan::RationalConfig r;r.numerator_degree=0;r.denominator_degree=2;kan::Layer l(1,1,r);
    l.set_rational_parameters(std::vector<double>{1e300},std::vector<double>{0,0},std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({l}),1);gpu.upload_input(std::vector<double>{1e-200},1);
    gpu.upload_output_gradient(std::vector<double>{1});gpu.forward();gpu.backward();
    const auto g=gpu.download_gradients();REQUIRE(g.layers[0].denominators[1]!=0);
    test::near(g.layers[0].denominators[1]/-1e-100,1,1e-12);
    r.denominator_degree=1;kan::Layer huge(1,1,r);
    huge.set_rational_parameters(std::vector<double>{1e300},std::vector<double>{1e200},std::vector<double>{0});
    kan::cuda::ResidentNetwork other(kan::Network({huge}),1);other.upload_input(std::vector<double>{1},1);
    other.upload_output_gradient(std::vector<double>{1});other.forward();other.backward();
    const auto d=other.download_gradients();test::near(d.layers[0].denominators[0]/-1e-100,1,1e-12);test::near(d.input[0]/-1e100,1,1e-12);
    kan::Layer tiny(1,1,r);tiny.set_rational_parameters(std::vector<double>{1e-300},std::vector<double>{1e-270},std::vector<double>{0});
    kan::cuda::ResidentNetwork final(kan::Network({tiny}),1);final.upload_input(std::vector<double>{1e300},1);
    final.upload_output_gradient(std::vector<double>{1});final.forward();final.backward();
    test::near(final.download_gradients().layers[0].denominators[0]/-1e-60,1,1e-12);
    r.denominator_degree=16;kan::Layer high(1,1,r);high.set_rational_parameters(std::vector<double>{1e300},std::vector<double>(16,0),std::vector<double>{0});
    kan::cuda::ResidentNetwork subnormal(kan::Network({high}),1);subnormal.upload_input(std::vector<double>{-1e-20},1);
    subnormal.upload_output_gradient(std::vector<double>{1});subnormal.forward();subnormal.backward();
    const auto d16=subnormal.download_gradients().layers[0].denominators;test::near(d16[15]/-1e-20,1,1e-12);test::near(d16[14],1,1e-12);
    r.denominator_degree=1;r.scale=1e-320;kan::Layer scaled(1,1,r);
    scaled.set_rational_parameters(std::vector<double>{1e-300},std::vector<double>{1e-270},std::vector<double>{0});
    kan::cuda::ResidentNetwork smallscale(kan::Network({scaled}),1);smallscale.upload_input(std::vector<double>{1e-20},1);
    smallscale.upload_output_gradient(std::vector<double>{1});smallscale.forward();smallscale.backward();
    test::near(smallscale.download_gradients().input[0]/(-r.scale*1e10),1,1e-8);
}
int main(){if(!kan::cuda::available())return 1;return test::run();}

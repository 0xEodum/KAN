#include "kan/families.hpp"
#include "kan/network.hpp"
#include "support/families.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <limits>
#include <numeric>

namespace {
kan::BSplineConfig spline(std::size_t degree=3) {
    kan::BSplineConfig b{degree,{}};
    b.knots.assign(degree+1,-1); b.knots.push_back(0);
    b.knots.insert(b.knots.end(),degree+1,1); return b;
}
kan::TrainableRbfConfig rbf() {
    return {{-0.6,0.1,0.8},{-0.3,0.2,-0.1}};
}
const std::vector<double>& centers(const kan::Layer& l) { return test::trainable(l).centers; }
const std::vector<double>& log_widths(const kan::Layer& l) { return test::trainable(l).log_widths; }
kan::Layer fixture(kan::BasisConfig b) {
    kan::Layer l(2,2,b); std::vector<double> c(l.coefficients().size());
    for (std::size_t i=0;i<c.size();++i) c[i]=0.07*(static_cast<double>(i%9)-4);
    l.set_parameters(c,std::vector<double>{0.2,-0.1});return l;
}
double loss(const kan::Layer& l, const std::vector<double>& x, const std::vector<double>& u) {
    auto y=l.forward(x,2);return std::inner_product(y.begin(),y.end(),u.begin(),0.0);
}
void unchanged(const kan::Layer& a,const kan::Layer& b) {
    REQUIRE(std::equal(a.coefficients().begin(),a.coefficients().end(),b.coefficients().begin(),b.coefficients().end()));
    REQUIRE(a.carrier()==b.carrier());
}
}
TEST(trainable_shared_rbf_vjp_and_sgd) {
    auto l=fixture(rbf());const std::vector<double> x{-0.5,0.2,0.7,-0.1},u{0.4,-0.8,0.3,0.2};
    auto g=l.backward(x,2,u);REQUIRE(test::trainable(g).centers.size()==3);REQUIRE(test::trainable(g).log_widths.size()==3);
    for(std::size_t k=0;k<3;++k) {
        auto p=l,m=l;auto pc=centers(l),mc=pc,w=log_widths(l);
        pc[k]+=1e-6;mc[k]-=1e-6;kan::set_rbf_parameters(p,pc,w);kan::set_rbf_parameters(m,mc,w);
        test::near(test::trainable(g).centers[k],(loss(p,x,u)-loss(m,x,u))/2e-6,2e-7);
        p=l;m=l;auto pw=w,mw=w;pw[k]+=1e-6;mw[k]-=1e-6;
        kan::set_rbf_parameters(p,centers(l),pw);kan::set_rbf_parameters(m,centers(l),mw);
        test::near(test::trainable(g).log_widths[k],(loss(p,x,u)-loss(m,x,u))/2e-6,2e-7);
    }
    auto before=l;l.sgd(g,0.03);
    for(std::size_t k=0;k<3;++k) {
        test::near(centers(l)[k],centers(before)[k]-0.03*test::trainable(g).centers[k]);
        test::near(log_widths(l)[k],log_widths(before)[k]-0.03*test::trainable(g).log_widths[k]);
    }
    auto z=l.backward({},0,{});for(double v:test::trainable(z).centers)test::near(v,0);for(double v:test::trainable(z).log_widths)test::near(v,0);
}
TEST(rbf_invalid_updates_are_atomic_across_network) {
    auto l=fixture(rbf());const auto before=l;
    test::throws<std::invalid_argument>([&]{kan::set_rbf_parameters(l,std::vector<double>{1},std::vector<double>{1});});
    unchanged(l,before);
    auto g=l.backward(std::vector<double>{0.2,0.4},1,std::vector<double>{1,1});
    test::trainable(g).log_widths[1]=-1000;test::throws<std::overflow_error>([&]{l.sgd(g,1);});unchanged(l,before);
    test::trainable(g).log_widths[1]=1000;test::throws<std::overflow_error>([&]{l.sgd(g,1);});unchanged(l,before);
    kan::Network n({l,l});auto ng=n.backward(std::vector<double>{0.2,0.4},1,std::vector<double>{1,1});
    test::trainable(ng.layers[1]).centers[0]=std::numeric_limits<double>::infinity();
    test::throws<std::invalid_argument>([&]{n.sgd(ng,0.1);});for(auto& q:n.layers())unchanged(q,before);
    kan::Layer fixed(1,1,{});test::throws<std::invalid_argument>([&]{kan::set_rbf_parameters(fixed,{},{});});
}
TEST(exact_knot_insertion_preserves_values_and_derivatives) {
    for(std::size_t p=0;p<=4;++p) {
        auto l=fixture(spline(p)),before=l;
        kan::insert_knot(l,0.3); REQUIRE(l.terms()==before.terms()+1);
        for(double x:{-2.,-1.,-0.7,-0.001,0.,0.15,0.3,0.65,1.,2.}) {
            const std::vector<double> a{x,x};auto y=l.forward(a,1),ref=before.forward(a,1);
            for(std::size_t k=0;k<y.size();++k)test::near(y[k],ref[k],2e-12);
            // Exclude new degree-zero jump conventions (no derivative everywhere).
            auto g=l.backward(a,1,std::vector<double>{0.3,0.7}),r=before.backward(a,1,std::vector<double>{0.3,0.7});
            for(std::size_t k=0;k<g.input.size();++k)test::near(g.input[k],r.input[k],2e-11);
        }
        const auto snapshot=l;
        test::throws<std::invalid_argument>([&]{kan::insert_knot(l,1);});unchanged(l,snapshot);
        for(std::size_t count=0;count<p;++count)kan::insert_knot(l,0.3);
        const auto full=l;
        test::throws<std::invalid_argument>([&]{kan::insert_knot(l,0.3);});unchanged(l,full);
        auto stale=before.backward(std::vector<double>{0.1,0.2},1,std::vector<double>{1,1});
        test::throws<std::invalid_argument>([&]{l.sgd(stale,0.1);});
    }
}
TEST(data_adaptation_and_indexed_network_refinement) {
    auto l=fixture(spline()),before=l;
    const double knot=kan::adapt_grid(l,std::vector<double>{-0.8,-0.6,-0.4,0.7,9});test::near(knot,-0.6);
    auto y=l.forward(std::vector<double>{-0.7,0.4},1),ref=before.forward(std::vector<double>{-0.7,0.4},1);
    for(std::size_t i=0;i<y.size();++i)test::near(y[i],ref[i]);
    const auto snapshot=l;
    test::throws<std::invalid_argument>([&]{kan::adapt_grid(l,{});});
    test::throws<std::invalid_argument>([&]{kan::adapt_grid(l,std::vector<double>{4});});
    test::throws<std::invalid_argument>([&]{kan::adapt_grid(l,std::vector<double>{std::numeric_limits<double>::quiet_NaN()});});unchanged(l,snapshot);
    kan::Network n({before,before});auto output=n.forward(std::vector<double>{0.3,0.4},1);
    n.insert_knot(1,0.4);n.adapt_grid(0,std::vector<double>{-0.5});auto after=n.forward(std::vector<double>{0.3,0.4},1);
    for(std::size_t i=0;i<after.size();++i)test::near(after[i],output[i]);
    test::throws<std::invalid_argument>([&]{n.insert_knot(8,0.2);});
}
TEST(l2_objective_vjp_and_zero_defaults) {
    auto l=fixture(rbf());auto r=l.regularization(0.3);double sum=0;
    for(std::size_t k=0;k<l.coefficients().size();++k) {
        sum+=0.15*l.coefficients()[k]*l.coefficients()[k];test::near(r.gradients.coefficients[k],0.3*l.coefficients()[k]);
    }
    test::near(r.value,sum);for(double v:r.gradients.bias)test::near(v,0);
    for(double v:test::trainable(r.gradients).centers)test::near(v,0);for(double v:test::trainable(r.gradients).log_widths)test::near(v,0);
    test::near(l.regularization(0).value,0);auto before=l;
    test::throws<std::invalid_argument>([&]{l.regularization(-1);});unchanged(l,before);
    kan::Network n({l,l});auto nr=n.regularization(0.3);test::near(nr.value,2*sum);n.sgd(nr.gradients,0.01);
    for(const auto& q:n.layers())for(std::size_t k=0;k<q.coefficients().size();++k)test::near(q.coefficients()[k],0.997*l.coefficients()[k]);
}
int main(){return test::run();}

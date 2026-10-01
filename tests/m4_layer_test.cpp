#include "kan/network.hpp"
#include "support/test.hpp"
#include <numeric>
#include <limits>
#include <iostream>

namespace {
kan::Layer rational(size_t inputs=2,size_t outputs=2) {
    kan::RationalConfig c;c.numerator_degree=2;c.denominator_degree=2;c.center=0.1;c.scale=1.3;
    kan::Layer l(inputs,outputs,c);std::vector<double>a(l.coefficients().size()),b(l.denominators().size()),bias(outputs);
    for(size_t i=0;i<a.size();++i)a[i]=0.03*(double(i%7)-3);
    for(size_t i=0;i<b.size();++i)b[i]=0.02*(double(i%3)-1);
    for(size_t i=0;i<outputs;++i)bias[i]=0.04*(i+1);
    l.set_rational_parameters(a,b,bias);return l;
}
double objective(const kan::Network& n,const std::vector<double>& x,const std::vector<double>& g) {
    auto y=n.forward(x,2);return std::inner_product(y.begin(),y.end(),g.begin(),0.0);
}
void check_fd(kan::Network n) {
    std::vector<double>x{-0.6,0.2,0.5,-0.1},g{0.3,-0.4};const double h=1e-6;
    auto grad=n.backward(x,2,g);
    for(size_t j=0;j<x.size();++j){auto p=x,m=x;p[j]+=h;m[j]-=h;test::near(grad.input[j],(objective(n,p,g)-objective(n,m,g))/(2*h),3e-7);}
    auto base=n.layers();
    for(size_t k=0;k<base.size();++k) {
        auto l=base[k];std::vector<double>a(l.coefficients().begin(),l.coefficients().end()),b(l.denominators().begin(),l.denominators().end()),v(l.bias().begin(),l.bias().end());
        auto set=[&](kan::Layer& target,const auto& ca,const auto& cb,const auto& cv){if(target.is_rational())target.set_rational_parameters(ca,cb,cv);else target.set_parameters(ca,cv);};
        for(int family=0;family<3;++family) {
            const auto& values=family==0?a:(family==1?b:v);const auto& analytic=family==0?grad.layers[k].coefficients:(family==1?grad.layers[k].denominators:grad.layers[k].bias);
            for(size_t j=0;j<values.size();++j){auto pa=a,ma=a,pb=b,mb=b,pv=v,mv=v;
                auto& p=family==0?pa:(family==1?pb:pv);auto& m=family==0?ma:(family==1?mb:mv);p[j]+=h;m[j]-=h;
                std::vector<kan::Layer> plus(base.begin(),base.end()),minus=plus;set(plus[k],pa,pb,pv);set(minus[k],ma,mb,mv);
                test::near(analytic[j],(objective(kan::Network(plus),x,g)-objective(kan::Network(minus),x,g))/(2*h),3e-7);
            }
        }
    }
}
}
TEST(rational_identity_shape_layout_and_empty_batch) {
    kan::RationalConfig c;c.numerator_degree=1;c.denominator_degree=1;kan::Layer l(2,2,c);
    REQUIRE(l.is_rational());REQUIRE(l.coefficients().size()==8);REQUIRE(l.denominators().size()==4);
    REQUIRE(l.forward(std::vector<double>{1,2},1)==std::vector<double>({0,0}));
    l.set_rational_parameters(std::vector<double>{1,2,3,4,5,6,7,8},std::vector<double>{0.1,0.2,0.3,0.4},std::vector<double>{0.5,-0.5});
    auto y=l.forward(std::vector<double>{1,2},1);
    test::near(y[0],0.5+3/1.1+11/1.4);test::near(y[1],-0.5+11/1.3+23/1.8);
    REQUIRE(l.forward({},0).empty());auto g=l.backward({},0,{});REQUIRE(g.denominators.size()==4);
    for(double v:g.denominators)test::near(v,0);
    test::throws<std::invalid_argument>([&]{l.basis();});
    kan::Layer basis(1,1,{});test::throws<std::invalid_argument>([&]{basis.rational_config();});
}
TEST(layer_and_mixed_network_all_vjps) {
    check_fd(kan::Network({rational(2,1)}));
    kan::BasisConfig c;c.size=3;kan::Layer b(2,1,c);b.set_parameters(std::vector<double>{0.1,0.2,-0.03,-0.2,0.15,0.07},std::vector<double>{0.04});
    check_fd(kan::Network({rational(),b,rational(1,1)}));
}
TEST(invalid_setters_sgd_and_network_atomicity) {
    auto l=rational();std::vector<double>a(l.coefficients().begin(),l.coefficients().end()),b(l.denominators().begin(),l.denominators().end()),v(l.bias().begin(),l.bias().end());
    test::throws<std::invalid_argument>([&]{l.set_parameters(a,v);});
    test::throws<std::invalid_argument>([&]{l.set_rbf_parameters({},{});});
    test::throws<std::invalid_argument>([&]{l.insert_knot(0);});test::throws<std::invalid_argument>([&]{l.adapt_grid(std::vector<double>{0});});
    test::throws<std::invalid_argument>([&]{l.set_rational_parameters({},b,v);});
    test::throws<std::invalid_argument>([&]{l.set_rational_parameters(a,{},v);});
    auto bad=b;bad.back()=std::numeric_limits<double>::infinity();test::throws<std::invalid_argument>([&]{l.set_rational_parameters(a,bad,v);});
    REQUIRE(std::vector<double>(l.denominators().begin(),l.denominators().end())==b);
    auto g=l.backward(std::vector<double>{0.1,0.2},1,std::vector<double>{0.3,0.4});
    auto wrong=g;wrong.denominators.clear();test::throws<std::invalid_argument>([&]{l.sgd(wrong,0.1);});
    wrong=g;wrong.centers={1};test::throws<std::invalid_argument>([&]{l.sgd(wrong,0.1);});
    wrong=g;wrong.denominators.back()=std::numeric_limits<double>::max();test::throws<std::overflow_error>([&]{l.sgd(wrong,2);});
    REQUIRE(std::vector<double>(l.coefficients().begin(),l.coefficients().end())==a);
    REQUIRE(std::vector<double>(l.denominators().begin(),l.denominators().end())==b);
    kan::Network n({l,rational()});auto before=n.forward(std::vector<double>{0.1,0.2},1);
    auto ng=n.backward(std::vector<double>{0.1,0.2},1,std::vector<double>{1,1});ng.layers.back().denominators.back()=std::numeric_limits<double>::max();
    test::throws<std::overflow_error>([&]{n.sgd(ng,2);});REQUIRE(n.forward(std::vector<double>{0.1,0.2},1)==before);
    auto penalty=l.regularization(0.2);REQUIRE(penalty.gradients.denominators.size()==b.size());
    for(double d:penalty.gradients.denominators)test::near(d,0);
    for(size_t j=0;j<a.size();++j)test::near(penalty.gradients.coefficients[j],0.2*a[j]);
}
TEST(pole_guards_even_zero_upstream_and_explicit_update) {
    kan::RationalConfig c;c.numerator_degree=0;c.denominator_degree=1;kan::Layer l(1,1,c);
    l.set_rational_parameters(std::vector<double>{0},std::vector<double>{-1},std::vector<double>{0});
    test::throws<std::domain_error>([&]{l.forward(std::vector<double>{1},1);});
    test::throws<std::domain_error>([&]{l.backward(std::vector<double>{1},1,std::vector<double>{0});});
    l.set_rational_parameters(std::vector<double>{2},std::vector<double>{0.2},std::vector<double>{0.1});
    auto g=l.backward(std::vector<double>{0.4},1,std::vector<double>{0.3});l.sgd(g,0.1);
    test::near(l.coefficients()[0],2-0.1*g.coefficients[0]);test::near(l.denominators()[0],0.2-0.1*g.denominators[0]);
    test::near(l.bias()[0],0.1-0.1*g.bias[0]);
    // SGD admits finite coefficients; safety is checked on subsequent execution.
    auto candidate=l.regularization(0).gradients;candidate.denominators[0]=l.denominators()[0]+1;
    l.sgd(candidate,1);
    test::throws<std::domain_error>([&]{l.forward(std::vector<double>{1},1);});
}
TEST(rational_dimensions_shapes_nonfinite_and_moved_state) {
    kan::RationalConfig c;c.numerator_degree=0;c.denominator_degree=0;
    test::throws<std::invalid_argument>([&]{kan::Layer l(0,1,c);});
    test::throws<std::invalid_argument>([&]{kan::Layer l(1,0,c);});
    test::throws<std::overflow_error>([&]{kan::Layer l(std::numeric_limits<size_t>::max(),2,c);});
    kan::Layer l(1,1,c);l.set_rational_parameters(std::vector<double>{2},{},std::vector<double>{1});
    test::near(l.forward(std::vector<double>{3},1)[0],3);
    test::near(l.backward(std::vector<double>{3},1,std::vector<double>{4}).input[0],0);
    test::throws<std::invalid_argument>([&]{l.forward({},1);});
    test::throws<std::invalid_argument>([&]{l.backward(std::vector<double>{0},1,{});});
    test::throws<std::invalid_argument>([&]{l.forward(std::vector<double>{std::numeric_limits<double>::infinity()},1);});
    test::throws<std::invalid_argument>([&]{l.backward(std::vector<double>{0},1,std::vector<double>{std::numeric_limits<double>::quiet_NaN()});});
    test::throws<std::invalid_argument>([&]{l.set_rational_parameters(std::vector<double>{2},{},{});});
    test::throws<std::invalid_argument>([&]{l.set_rational_parameters(std::vector<double>{2},{},std::vector<double>{std::numeric_limits<double>::quiet_NaN()});});
    auto moved=std::move(l);test::near(moved.forward(std::vector<double>{0},1)[0],3);
    test::throws<std::invalid_argument>([&]{l.forward(std::vector<double>{0},1);});
    kan::Layer basis(1,1,{});test::throws<std::invalid_argument>([&]{basis.set_rational_parameters(std::vector<double>{2},{},std::vector<double>{1});});
    kan::Layer huge(2,1,c);const auto max=std::numeric_limits<double>::max();
    huge.set_rational_parameters(std::vector<double>{max,max},{},std::vector<double>{0});
    test::throws<std::overflow_error>([&]{huge.forward(std::vector<double>{0,0},1);});
    test::throws<std::overflow_error>([&]{huge.backward(std::vector<double>{0,0,0,0},2,std::vector<double>{max,max});});
}
TEST(deterministic_rational_learning_with_independent_holdout) {
    kan::RationalConfig c;c.numerator_degree=1;c.denominator_degree=1;kan::Network n({kan::Layer(1,1,c)});
    std::vector<double>x(41),target(41);for(size_t j=0;j<x.size();++j){x[j]=-0.8+1.6*j/40;target[j]=(0.4+0.7*x[j])/(1+0.35*x[j]);}
    double initial=0,final=0;
    for(size_t step=0;step<6000;++step){auto y=n.forward(x,x.size());std::vector<double>g(y.size());double loss=0;
        for(size_t j=0;j<y.size();++j){double e=y[j]-target[j];loss+=e*e/y.size();g[j]=2*e/y.size();}
        if(step==0)initial=loss;final=loss;n.sgd(n.backward(x,x.size(),g),0.04);
    }
    double holdout=0;for(double h:{-0.77,-0.43,-0.07,0.19,0.53,0.79}) {double e=n.forward(std::vector<double>{h},1)[0]-(0.4+0.7*h)/(1+0.35*h);holdout+=e*e/6;}
    std::cout<<"rational training initial="<<initial<<" final="<<final<<" holdout="<<holdout<<'\n';
    REQUIRE(final<1e-7);REQUIRE(holdout<1e-7);REQUIRE(final<initial*1e-4);
}
int main(){return test::run();}

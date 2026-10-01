#include "kan/rational.hpp"
#include "support/test.hpp"
#include <cmath>
#include <limits>

TEST(independent_pade_and_scaled_quotient_identities) {
    kan::RationalConfig c; c.numerator_degree=1;c.denominator_degree=1;
    for(double x:{-0.8,-0.1,0.0,0.6}) {
        auto r=kan::evaluate_rational(c,x,std::vector<double>{1,0.5},std::vector<double>{-0.5});
        test::near(r.value,(2+x)/(2-x));test::near(r.input_derivative,4/((2-x)*(2-x)));
        test::near(r.numerator_derivatives[0],1/(1-x/2));
        test::near(r.denominator_derivatives[0],-(1+x/2)*x/((1-x/2)*(1-x/2)));
    }
    c.center=2;c.scale=3;c.numerator_degree=2;
    auto r=kan::evaluate_rational(c,5,std::vector<double>{2,3,4},std::vector<double>{0.5});
    test::near(r.value,6);test::near(r.input_derivative,16.0/9.0);
    c.numerator_degree=0;c.denominator_degree=0;
    r=kan::evaluate_rational(c,7,std::vector<double>{2.5},{});
    test::near(r.value,2.5);test::near(r.input_derivative,0);REQUIRE(r.denominator_derivatives.empty());
}
TEST(all_orders_input_and_nonlinear_parameter_derivatives) {
    for(auto orders:{std::pair{0u,3u},std::pair{4u,1u},std::pair{3u,5u},std::pair{16u,16u}}) {
        kan::RationalConfig c;c.numerator_degree=orders.first;c.denominator_degree=orders.second;c.center=0.1;c.scale=1.4;
        std::vector<double> a(c.numerator_degree+1),b(c.denominator_degree);
        for(size_t i=0;i<a.size();++i)a[i]=0.07*(i+1);
        for(size_t i=0;i<b.size();++i)b[i]=0.01*(i+1);
        for(double x:{-0.7,0.0,0.8}) {
            const double h=1e-6;auto r=kan::evaluate_rational(c,x,a,b);
            test::near(r.input_derivative,(kan::evaluate_rational(c,x+h,a,b).value-kan::evaluate_rational(c,x-h,a,b).value)/(2*h),3e-7);
            for(size_t i=0;i<a.size();++i) {auto p=a,m=a;p[i]+=h;m[i]-=h;
                test::near(r.numerator_derivatives[i],(kan::evaluate_rational(c,x,p,b).value-kan::evaluate_rational(c,x,m,b).value)/(2*h),3e-7);}
            for(size_t i=0;i<b.size();++i) {auto p=b,m=b;p[i]+=h;m[i]-=h;
                test::near(r.denominator_derivatives[i],(kan::evaluate_rational(c,x,a,p).value-kan::evaluate_rational(c,x,a,m).value)/(2*h),3e-7);}
        }
    }
}
TEST(relative_guard_boundary_poles_and_conditioning) {
    kan::RationalConfig c;c.numerator_degree=1;c.denominator_degree=1;c.epsilon=0.125;
    // At x=1, b=-7/9 gives Q=2/9 and epsilon*(1+|b|)=2/9.
    for(double b:{-1.0,-0.8,-7.0/9.0})
        test::throws<std::domain_error>([&]{kan::evaluate_rational(c,1,std::vector<double>{1,-1},std::vector<double>{b});});
    auto safe=kan::evaluate_rational(c,1,std::vector<double>{0,0},std::vector<double>{-0.7});test::near(safe.value,0);
    c.epsilon=1e-8;c.denominator_degree=2;
    test::throws<std::domain_error>([&]{kan::evaluate_rational(c,1,std::vector<double>{0,0},std::vector<double>{1e9,-1e9});});
    auto well=kan::evaluate_rational(c,1,std::vector<double>{1,0},std::vector<double>{1e9,-1e9+100});
    test::near(well.value,1.0/101.0);
}
TEST(large_finite_denominator_preserves_small_nonlinear_vjp) {
    kan::RationalConfig c;c.numerator_degree=0;c.denominator_degree=1;
    auto r=kan::evaluate_rational(c,1,std::vector<double>{1e300},std::vector<double>{1e200});
    test::near(r.value/1e100,1);test::near(r.input_derivative/(-1e100),1);
    test::near(r.denominator_derivatives[0]/(-1e-100),1);
    test::near(r.numerator_derivatives[0]/1e-200,1);
}
TEST(representable_vjp_survives_underflowing_power_and_quotient) {
    kan::RationalConfig c;c.numerator_degree=0;c.denominator_degree=2;
    auto r=kan::evaluate_rational(c,1e-200,std::vector<double>{1e300},std::vector<double>{0,0});
    test::near(r.denominator_derivatives[0]/(-1e100),1);
    test::near(r.denominator_derivatives[1]/(-1e-100),1);
    c.denominator_degree=16;
    r=kan::evaluate_rational(c,1e-20,std::vector<double>{1e300},std::vector<double>(16));
    test::near(r.denominator_derivatives[15]/(-1e-20),1,1e-10);
    // P/Q itself is below binary64 range, yet P*z/Q^2 is representable.
    c.denominator_degree=1;
    r=kan::evaluate_rational(c,1e300,std::vector<double>{1e-300},std::vector<double>{1e-270});
    test::near(r.denominator_derivatives[0]/(-1e-60),1);
}
TEST(invalid_configuration_shapes_data_and_intermediates) {
    kan::RationalConfig c;
    for(double v:{0.0,-1.0,std::numeric_limits<double>::infinity()}) {auto bad=c;bad.scale=v;test::throws<std::invalid_argument>([&]{kan::validate_rational(bad);});}
    for(double v:{0.0,1.0,-0.1,std::numeric_limits<double>::quiet_NaN()}) {auto bad=c;bad.epsilon=v;test::throws<std::invalid_argument>([&]{kan::validate_rational(bad);});}
    auto bad=c;bad.center=std::numeric_limits<double>::infinity();test::throws<std::invalid_argument>([&]{kan::validate_rational(bad);});
    bad=c;bad.numerator_degree=17;test::throws<std::invalid_argument>([&]{kan::validate_rational(bad);});
    bad=c;bad.denominator_degree=17;test::throws<std::invalid_argument>([&]{kan::validate_rational(bad);});
    test::throws<std::invalid_argument>([&]{kan::evaluate_rational(c,0,{},{});});
    std::vector<double>a(4),b(2);a[0]=std::numeric_limits<double>::infinity();
    test::throws<std::invalid_argument>([&]{kan::evaluate_rational(c,0,a,b);});a[0]=1;
    test::throws<std::invalid_argument>([&]{kan::evaluate_rational(c,std::numeric_limits<double>::quiet_NaN(),a,b);});
    b[0]=std::numeric_limits<double>::quiet_NaN();test::throws<std::invalid_argument>([&]{kan::evaluate_rational(c,0,a,b);});b[0]=0;
    test::throws<std::overflow_error>([&]{kan::evaluate_rational(c,1e200,a,b);});
    c.center=-1e308;test::throws<std::overflow_error>([&]{kan::evaluate_rational(c,1e308,a,b);});
}
int main(){return test::run();}

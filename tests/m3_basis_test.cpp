#include "kan/basis.hpp"
#include "support/test.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace {
kan::BasisConfig spline(std::size_t degree, std::vector<double> knots) {
    kan::BasisConfig c;
    c.kind = kan::BasisKind::BSpline;
    c.degree = degree;
    c.size = knots.size()-degree-1;
    c.knots = std::move(knots);
    return c;
}
kan::BasisConfig wavelet() {
    kan::BasisConfig c;
    c.kind = kan::BasisKind::MexicanHat;
    c.size = 3;
    c.centers = {-0.7, 0.2, 1.1};
    c.scales = {0.4, 0.8, 1.3};
    return c;
}
kan::BasisConfig rbf() {
    auto c = wavelet();
    c.kind = kan::BasisKind::GaussianRbf;
    c.trainable_rbf = true;
    c.log_widths = {std::log(0.4), std::log(0.8), std::log(1.3)};
    return c;
}
void input_differences(const kan::BasisConfig& c, double x) {
    constexpr double h = 1e-6;
    auto a = kan::evaluate_basis(c, x), p = kan::evaluate_basis(c, x+h), m = kan::evaluate_basis(c, x-h);
    for (std::size_t k=0; k<c.size; ++k)
        test::near(a.derivatives[k], (p.values[k]-m.values[k])/(2*h), 2e-7);
}
}

TEST(clamped_cubic_matches_bernstein_closed_form) {
    auto c = spline(3, {0,0,0,0,1,1,1,1});
    for (double x : {0.,0.17,0.5,0.83,1.}) {
        auto a = kan::evaluate_basis(c,x);
        const double y=1-x;
        std::vector<double> values{y*y*y,3*x*y*y,3*x*x*y,x*x*x};
        std::vector<double> slopes{-3*y*y,3*y*y-6*x*y,6*x*y-3*x*x,3*x*x};
        for (std::size_t k=0;k<4;++k) {
            test::near(a.values[k],values[k]); test::near(a.derivatives[k],slopes[k]);
        }
        REQUIRE(a.center_derivatives.empty()); REQUIRE(a.log_width_derivatives.empty());
    }
}

TEST(spline_right_knots_upper_inward_and_zero_extension) {
    auto c = spline(1,{0,0,0.5,1,1});
    auto a=kan::evaluate_basis(c,0.5);
    test::near(a.values[1],1); test::near(a.derivatives[0],0);
    test::near(a.derivatives[1],-2); test::near(a.derivatives[2],2);
    auto repeated=spline(1,{0,0,0.5,0.5,1,1});
    a=kan::evaluate_basis(repeated,0.5);
    test::near(a.values[2],1); test::near(a.values[1],0);
    test::near(a.derivatives[2],-2); test::near(a.derivatives[3],2);
    a=kan::evaluate_basis(repeated,1);
    test::near(a.values[3],1); test::near(a.derivatives[2],-2); test::near(a.derivatives[3],2);
    for(double x : {-1.,std::nextafter(0.,-1.),std::nextafter(1.,2.),2.}) {
        a=kan::evaluate_basis(repeated,x);
        for(double v:a.values) test::near(v,0);
        for(double v:a.derivatives) test::near(v,0);
    }
    auto zero=spline(0,{0,0.5,1});
    a=kan::evaluate_basis(zero,0.5); test::near(a.values[0],0); test::near(a.values[1],1);
    a=kan::evaluate_basis(zero,1); test::near(a.values[1],1);
    for(double v:a.derivatives) test::near(v,0);
}

TEST(spline_partition_nonuniform_repeated_and_derivatives) {
    for(auto c : {spline(2,{0,0,0,0.2,0.2,0.7,1,1,1}),
                  spline(3,{-2,-2,-2,-2,-0.4,0.2,0.2,0.2,1,1,1,1}),
                  spline(16,[] { std::vector<double> t(17,0);t.insert(t.end(),17,1);return t;}())}) {
        for(double x:{0.01,0.13,0.31,0.53,0.91}) {
            auto a=kan::evaluate_basis(c,x);
            test::near(std::accumulate(a.values.begin(),a.values.end(),0.),1);
            test::near(std::accumulate(a.derivatives.begin(),a.derivatives.end(),0.),0);
            for(double v:a.values) REQUIRE(v>=0);
            input_differences(c,x);
        }
    }
}

TEST(spline_contract_validation) {
    auto good=spline(2,{0,0,0,0.5,1,1,1});
    kan::validate_basis(good);
    auto bad=good; bad.degree=17;
    test::throws<std::invalid_argument>([&]{kan::validate_basis(bad);});
    bad=good; bad.size=2;
    test::throws<std::invalid_argument>([&]{kan::validate_basis(bad);});
    for(auto knots : {std::vector<double>{0,0,0,1,1,1}, std::vector<double>{0,0,0,0.8,0.7,1,1},
                      std::vector<double>{0,0,0,0,1,1,1}, std::vector<double>{0,0,0,1,1,1,1},
                      std::vector<double>{0,0,0,0,0,0,0},
                      std::vector<double>{0,0,0,std::numeric_limits<double>::quiet_NaN(),1,1,1}}) {
        bad=good;bad.knots=knots;
        test::throws<std::invalid_argument>([&]{kan::validate_basis(bad);});
    }
    bad=spline(1,{0,0,0.5,0.5,0.5,1,1});
    test::throws<std::invalid_argument>([&]{kan::validate_basis(bad);});
}

TEST(spline_extreme_domains_preserve_partition_and_slopes) {
    const double max=std::numeric_limits<double>::max();
    auto c=spline(1,{-max,-max,max,max});
    auto a=kan::evaluate_basis(c,0);
    test::near(a.values[0],0.5);test::near(a.values[1],0.5);
    test::near(a.derivatives[0]*max,-0.5);test::near(a.derivatives[1]*max,0.5);
    const double tiny=std::numeric_limits<double>::denorm_min();
    c=spline(1,{0,0,tiny,tiny});
    test::throws<std::overflow_error>([&]{kan::evaluate_basis(c,0);});
}

TEST(wavelet_closed_forms_and_translation_scale_convention) {
    auto c=wavelet();
    const double norm=2/(std::sqrt(3.)*std::pow(std::acos(-1.),0.25));
    for(double x:{-1.3,0.2,1.1,2.3}) {
        auto a=kan::evaluate_basis(c,x);
        for(std::size_t k=0;k<c.size;++k) {
            const double q=(x-c.centers[k])/c.scales[k];
            const double n=norm/std::sqrt(c.scales[k]),e=std::exp(-q*q/2);
            test::near(a.values[k],n*(1-q*q)*e);
            test::near(a.derivatives[k],n*(q*q*q-3*q)*e/c.scales[k]);
        }
        input_differences(c,x);
        REQUIRE(a.center_derivatives.empty());REQUIRE(a.log_width_derivatives.empty());
    }
}

TEST(wavelet_unit_energy_and_zero_mean) {
    auto c=wavelet();c.size=1;c.centers={0};c.scales={1};
    // Composite Simpson quadrature independently checks the normalization.
    constexpr std::size_t intervals=12000;
    constexpr double lo=-12,step=24./intervals;
    double energy=0,mean=0;
    for(std::size_t j=0;j<=intervals;++j) {
        const double v=kan::evaluate_basis(c,lo+step*static_cast<double>(j)).values[0];
        const double weight=j==0 || j==intervals ? 1 : (j%2 ? 4 : 2);
        energy+=weight*v*v;mean+=weight*v;
    }
    test::near(energy*step/3,1,1e-11);test::near(mean*step/3,0,1e-11);
}

TEST(trainable_rbf_input_and_parameter_derivatives) {
    auto c=rbf(); c.width=-1; // Scalar width is irrelevant in the opt-in mode.
    for(double x:{-1.2,0.2,1.7}) {
        auto a=kan::evaluate_basis(c,x);
        REQUIRE(a.center_derivatives.size()==c.size);REQUIRE(a.log_width_derivatives.size()==c.size);
        for(std::size_t k=0;k<c.size;++k) {
            const double width=std::exp(c.log_widths[k]),q=(x-c.centers[k])/width;
            test::near(a.values[k],std::exp(-q*q));
            test::near(a.center_derivatives[k],2*q*a.values[k]/width);
            test::near(a.log_width_derivatives[k],2*q*q*a.values[k]);
            constexpr double h=1e-6;
            auto p=c,m=c;p.centers[k]+=h;m.centers[k]-=h;
            test::near(a.center_derivatives[k],(kan::evaluate_basis(p,x).values[k]-kan::evaluate_basis(m,x).values[k])/(2*h),2e-7);
            p=c;m=c;p.log_widths[k]+=h;m.log_widths[k]-=h;
            test::near(a.log_width_derivatives[k],(kan::evaluate_basis(p,x).values[k]-kan::evaluate_basis(m,x).values[k])/(2*h),2e-7);
        }
        input_differences(c,x);
    }
    c.trainable_rbf=false;c.width=1;
    auto fixed=kan::evaluate_basis(c,0.3);
    REQUIRE(fixed.center_derivatives.empty());REQUIRE(fixed.log_width_derivatives.empty());
}

TEST(localized_parameter_and_input_validation) {
    for(auto good:{wavelet(),rbf()}) {
        kan::validate_basis(good);
        auto bad=good;bad.centers.pop_back();
        test::throws<std::invalid_argument>([&]{kan::validate_basis(bad);});
        bad=good;bad.centers[0]=std::numeric_limits<double>::infinity();
        test::throws<std::invalid_argument>([&]{kan::validate_basis(bad);});
        for(double x:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
            test::throws<std::invalid_argument>([&]{kan::evaluate_basis(good,x);});
    }
    auto c=wavelet();c.scales.pop_back();
    test::throws<std::invalid_argument>([&]{kan::validate_basis(c);});
    for(double scale:{0.,-1.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}) {
        c=wavelet();c.scales[0]=scale;
        test::throws<std::invalid_argument>([&]{kan::validate_basis(c);});
    }
    c=rbf();c.log_widths.pop_back();
    test::throws<std::invalid_argument>([&]{kan::validate_basis(c);});
    for(double log_width:{-1000.,1000.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}) {
        c=rbf();c.log_widths[0]=log_width;
        test::throws<std::invalid_argument>([&]{kan::validate_basis(c);});
    }
    c=wavelet();c.trainable_rbf=true;
    test::throws<std::invalid_argument>([&]{kan::validate_basis(c);});
}

TEST(localized_extreme_tails_and_explicit_overflow) {
    const double max=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();
    auto c=rbf();c.size=1;c.centers={-max};c.log_widths={std::log(max)};
    auto a=kan::evaluate_basis(c,max);
    const double width=std::exp(c.log_widths[0]),q=max/width+max/width;
    test::near(a.values[0],std::exp(-q*q));
    test::near(a.log_width_derivatives[0],2*q*q*std::exp(-q*q));
    c.centers={0};c.log_widths={std::log(tiny)};
    a=kan::evaluate_basis(c,1);
    test::near(a.values[0],0);test::near(a.derivatives[0],0);test::near(a.center_derivatives[0],0);test::near(a.log_width_derivatives[0],0);
    a=kan::evaluate_basis(c,30*tiny);
    constexpr double oracle=-1.65703957421575192026735691255843647e-66;
    REQUIRE(a.derivatives[0]!=0);test::near(a.derivatives[0]/oracle,1,1e-12);
    test::near(a.center_derivatives[0]/-oracle,1,1e-12);
    test::throws<std::overflow_error>([&]{kan::evaluate_basis(c,tiny);});
    c=wavelet();c.size=1;c.centers={-max};c.scales={1};
    a=kan::evaluate_basis(c,max);test::near(a.values[0],0);test::near(a.derivatives[0],0);
    c.centers={0};c.scales={tiny};
    a=kan::evaluate_basis(c,1);test::near(a.values[0],0);test::near(a.derivatives[0],0);
    a=kan::evaluate_basis(c,0);REQUIRE(std::isfinite(a.values[0]));test::near(a.derivatives[0],0);
    a=kan::evaluate_basis(c,55*tiny);
    // Python Decimal, 100 digits; value is -1.59025760021617e-492.
    constexpr double wavelet_oracle=1.76912363925034805752500849835501785e-167;
    test::near(a.values[0],0);REQUIRE(a.derivatives[0]!=0);
    test::near(a.derivatives[0]/wavelet_oracle,1,2e-12);
    test::throws<std::overflow_error>([&]{kan::evaluate_basis(c,tiny);});
}

int main(){return test::run();}

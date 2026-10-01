#include "kan/rational.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace kan {
namespace {
double finite(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("nonfinite rational intermediate or result");
    return value;
}
void data_finite(std::span<const double> values) {
    for(double v:values)if(!std::isfinite(v))throw std::invalid_argument("rational data must be finite");
}
}
void validate_rational(const RationalConfig& c) {
    if(c.numerator_degree>16 || c.denominator_degree>16 || !std::isfinite(c.center) ||
       !std::isfinite(c.scale) || c.scale<=0 || !std::isfinite(c.epsilon) || c.epsilon<=0 || c.epsilon>=1)
        throw std::invalid_argument("invalid rational configuration");
}
RationalEvaluation evaluate_rational(const RationalConfig& c, double x, std::span<const double> a, std::span<const double> b) {
    validate_rational(c);
    if(a.size()!=c.numerator_degree+1 || b.size()!=c.denominator_degree || !std::isfinite(x))
        throw std::invalid_argument("rational shape or input mismatch");
    data_finite(a);data_finite(b);
    const double z=finite(finite(x-c.center)/c.scale);
    double p=a.back(),dp=0;
    for(std::size_t k=c.numerator_degree;k>0;--k) {
        dp=finite(finite(dp*z)+p);
        p=finite(finite(p*z)+a[k-1]);
    }
    double q=c.denominator_degree ? b.back() : 1,dq=0;
    double bound=c.denominator_degree ? std::abs(b.back()) : 1;
    for(std::size_t k=c.denominator_degree;k>0;--k) {
        dq=finite(finite(dq*z)+q);
        const double next=k==1 ? 1 : b[k-2];
        q=finite(finite(q*z)+next);
        bound=finite(finite(bound*std::abs(z))+std::abs(next));
    }
    if(std::abs(q)<=finite(c.epsilon*bound))throw std::domain_error("unsafe rational denominator");
    RationalEvaluation r;r.value=finite(p/q);
    r.input_derivative=finite(finite(finite(dp/q)-finite(r.value*finite(dq/q)))/c.scale);
    r.numerator_derivatives.resize(a.size());r.denominator_derivatives.resize(b.size());
    double power=1;
    const auto degree=std::max(c.numerator_degree,c.denominator_degree);
    for(std::size_t k=0;k<=degree;++k) {
        const double divided=finite(power/q);
        if(k<a.size())r.numerator_derivatives[k]=divided;
        if(k>0 && k<=b.size())r.denominator_derivatives[k-1]=finite(-r.value*divided);
        if(k<degree)power=finite(power*z);
    }
    return r;
}
}

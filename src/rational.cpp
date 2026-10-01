#include "kan/rational.hpp"
#include "rational_internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
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
bool tiny(double value) { return std::abs(value)<std::numeric_limits<double>::min(); }
double signed_exp(double exponent, bool negative) {
    return std::copysign(finite(std::exp(exponent)),negative ? -1.0 : 1.0);
}
}
void validate_rational(const RationalConfig& c) {
    if(c.numerator_degree>16 || c.denominator_degree>16 || !std::isfinite(c.center) ||
       !std::isfinite(c.scale) || c.scale<=0 || !std::isfinite(c.epsilon) || c.epsilon<=0 || c.epsilon>=1)
        throw std::invalid_argument("invalid rational configuration");
}
detail::RationalTerms detail::evaluate_rational_trusted(const RationalConfig& c, double x, std::span<const double> a, std::span<const double> b) {
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
    RationalTerms r{};r.value=finite(p/q);
    const double numerator_term=finite(dp/q),denominator_ratio=finite(dq/q);
    const double denominator_term=finite(r.value*denominator_ratio);
    r.input_derivative=finite(finite(numerator_term-denominator_term)/c.scale);
    // Restore representable final derivatives when an intermediate quotient or
    // product has underflowed. The ordinary Horner/quotient path is unchanged.
    const bool tiny_numerator=dp!=0 && tiny(numerator_term);
    const bool tiny_denominator=p!=0 && dq!=0 &&
        (tiny(r.value) || tiny(denominator_ratio) || tiny(denominator_term));
    if(tiny_numerator || tiny_denominator) {
        const double lq=std::log(std::abs(q)),ls=std::log(c.scale);
        const double first=tiny_numerator ? signed_exp(std::log(std::abs(dp))-lq-ls,
            std::signbit(dp)!=std::signbit(q)) : finite(numerator_term/c.scale);
        const double second=tiny_denominator ? signed_exp(std::log(std::abs(p))+std::log(std::abs(dq))-2*lq-ls,
            std::signbit(p)!=std::signbit(dq)) : finite(denominator_term/c.scale);
        r.input_derivative=finite(first-second);
    }
    double power=1;
    const auto degree=std::max(c.numerator_degree,c.denominator_degree);
    for(std::size_t k=0;k<=degree;++k) {
        const double divided=finite(power/q);
        if(k<a.size()) {
            r.numerator_derivatives[k]=divided;
            if(z!=0 && (tiny(power) || tiny(divided)))
                r.numerator_derivatives[k]=signed_exp(k*std::log(std::abs(z))-std::log(std::abs(q)),
                    std::signbit(q)!=(std::signbit(z) && k%2!=0));
        }
        if(k>0 && k<=b.size()) {
            r.denominator_derivatives[k-1]=finite(-r.value*divided);
            if(p!=0 && z!=0 && (tiny(power) || tiny(divided) || tiny(r.value)))
                r.denominator_derivatives[k-1]=signed_exp(std::log(std::abs(p))+k*std::log(std::abs(z))-2*std::log(std::abs(q)),
                    !(std::signbit(p)!=(std::signbit(z) && k%2!=0)));
        }
        if(k<degree)power=finite(power*z);
    }
    return r;
}
RationalEvaluation evaluate_rational(const RationalConfig& c, double x, std::span<const double> a, std::span<const double> b) {
    validate_rational(c);
    if(a.size()!=c.numerator_degree+1 || b.size()!=c.denominator_degree || !std::isfinite(x))
        throw std::invalid_argument("rational shape or input mismatch");
    data_finite(a);data_finite(b);
    const auto terms=detail::evaluate_rational_trusted(c,x,a,b);
    RationalEvaluation result;result.value=terms.value;result.input_derivative=terms.input_derivative;
    result.numerator_derivatives.assign(terms.numerator_derivatives.begin(),terms.numerator_derivatives.begin()+a.size());
    result.denominator_derivatives.assign(terms.denominator_derivatives.begin(),terms.denominator_derivatives.begin()+b.size());
    return result;
}
}

#include "kan/rational.hpp"
#include "rational_internal.hpp"
#include "detail/rational_formulas.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace kan {
namespace {
struct FiniteRationalGuard {
    double operator()(double value) const {
        if (!std::isfinite(value)) throw std::overflow_error("nonfinite rational intermediate or result");
        return value;
    }
};
void data_finite(std::span<const double> values) {
    for(double v:values)if(!std::isfinite(v))throw std::invalid_argument("rational data must be finite");
}
}
void validate_rational(const RationalConfig& c) {
    if(c.numerator_degree>16 || c.denominator_degree>16 || !std::isfinite(c.center) ||
       !std::isfinite(c.scale) || c.scale<=0 || !std::isfinite(c.epsilon) || c.epsilon<=0 || c.epsilon>=1)
        throw std::invalid_argument("invalid rational configuration");
}
detail::RationalTerms detail::evaluate_rational_trusted(const RationalConfig& c, double x, std::span<const double> a, std::span<const double> b) {
    const FiniteRationalGuard guard;
    const auto h=rational_horner(c,x,a.data(),b.data(),guard);
    if(rational_pole(c,h))throw std::domain_error("unsafe rational denominator");
    const auto edge=rational_edge(c,h,guard);
    RationalTerms r{};r.value=edge.value;r.input_derivative=edge.input_derivative;
    double power=1;
    const auto degree=std::max(c.numerator_degree,c.denominator_degree);
    for(std::size_t k=0;k<=degree;++k) {
        const double divided=guard(power/h.q);
        if(k<a.size())r.numerator_derivatives[k]=rational_numerator_vjp(h.q,h.z,k,power,divided,guard);
        if(k>0 && k<=b.size())
            r.denominator_derivatives[k-1]=rational_denominator_vjp(h.p,h.q,edge.value,h.z,k,power,divided,guard);
        if(k<degree)power=guard(power*h.z);
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

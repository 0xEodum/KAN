#include "kan/layer.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace kan {
const BasisConfig& Layer::basis() const {
    if (rational_) throw std::invalid_argument("rational layers have no basis");
    return basis_;
}
const RationalConfig& Layer::rational_config() const {
    if (!rational_) throw std::invalid_argument("basis layers have no rational configuration");
    return rational_config_;
}
namespace {
std::size_t checked_size(std::size_t left, std::size_t right) {
    const auto max = std::vector<double>().max_size();
    if (right != 0 && left > max / right) throw std::overflow_error("array size overflow");
    return left * right;
}
void require_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("data must be finite");
}
void result_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::overflow_error("nonfinite numerical result");
}
}

Layer::Layer(std::size_t inputs, std::size_t outputs, RationalConfig config, RationalTag)
    : inputs_(inputs), outputs_(outputs), rational_(true), rational_config_(config) {
    if(inputs==0 || outputs==0)throw std::invalid_argument("layer dimensions must be positive");
    validate_rational(config);
    const auto edges=checked_size(inputs,outputs);
    coefficients_.resize(checked_size(edges,config.numerator_degree+1));
    denominators_.resize(checked_size(edges,config.denominator_degree));
    bias_.resize(checked_size(outputs,1));
}
void Layer::set_rational_parameters(std::span<const double> a, std::span<const double> b, std::span<const double> bias) {
    validate_state();
    if(!rational_ || a.size()!=coefficients_.size() || b.size()!=denominators_.size() || bias.size()!=bias_.size())
        throw std::invalid_argument("rational parameter shape or layer type mismatch");
    require_finite(a);require_finite(b);require_finite(bias);
    std::vector<double> next_a(a.begin(),a.end()),next_b(b.begin(),b.end()),next_bias(bias.begin(),bias.end());
    coefficients_.swap(next_a);denominators_.swap(next_b);bias_.swap(next_bias);
}
Layer::Layer(std::size_t inputs, std::size_t outputs, BasisConfig basis)
    : inputs_(inputs), outputs_(outputs), basis_(std::move(basis)) {
    if (inputs == 0 || outputs == 0) throw std::invalid_argument("layer dimensions must be positive");
    validate_basis(basis_);
    coefficients_.resize(checked_size(checked_size(inputs, outputs), basis_.size), 0.0);
    bias_.resize(checked_size(outputs, 1), 0.0);
}
void Layer::validate_state() const {
    const auto terms=rational_ ? rational_config_.numerator_degree+1 : basis_.size;
    if (inputs_ == 0 || outputs_ == 0 ||
        coefficients_.size() != checked_size(checked_size(inputs_, outputs_), terms) ||
        bias_.size() != outputs_)
        throw std::invalid_argument("layer is uninitialized or moved from");
    if(rational_) {
        validate_rational(rational_config_);
        if(denominators_.size()!=checked_size(checked_size(inputs_,outputs_),rational_config_.denominator_degree))
            throw std::invalid_argument("rational denominator shape mismatch");
    } else validate_basis(basis_);
}
void Layer::set_parameters(std::span<const double> coefficients, std::span<const double> bias) {
    validate_state();
    if(rational_)throw std::invalid_argument("basis setter rejects rational layers");
    if (coefficients.size() != coefficients_.size() || bias.size() != bias_.size())
        throw std::invalid_argument("parameter shape mismatch");
    require_finite(coefficients); require_finite(bias);
    std::vector<double> next_coefficients(coefficients.begin(), coefficients.end()), next_bias(bias.begin(), bias.end());
    coefficients_.swap(next_coefficients); bias_.swap(next_bias);
}

std::vector<double> Layer::forward(std::span<const double> input, std::size_t batch) const {
    validate_state();
    const auto input_size = checked_size(batch, inputs_), output_size = checked_size(batch, outputs_);
    if (input.size() != input_size) throw std::invalid_argument("input shape mismatch");
    require_finite(input);
    std::vector<double> output(output_size);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t o = 0; o < outputs_; ++o) output[b * outputs_ + o] = bias_[o];
        for (std::size_t i = 0; i < inputs_; ++i) {
            if(rational_) {
                const auto m=rational_config_.numerator_degree+1,n=rational_config_.denominator_degree;
                for(std::size_t o=0;o<outputs_;++o) {
                    const auto edge=o*inputs_+i;
                    const auto r=evaluate_rational(rational_config_,input[b*inputs_+i],
                        std::span<const double>(coefficients_).subspan(edge*m,m),
                        std::span<const double>(denominators_).subspan(edge*n,n));
                    output[b*outputs_+o]+=r.value;
                    result_finite(std::span<const double>(&output[b*outputs_+o],1));
                }
                continue;
            }
            const auto basis = evaluate_basis(basis_, input[b * inputs_ + i]);
            for (std::size_t o = 0; o < outputs_; ++o)
                for (std::size_t k = 0; k < basis_.size; ++k)
                    output[b * outputs_ + o] += coefficients_[(o * inputs_ + i) * basis_.size + k] * basis.values[k];
        }
    }
    result_finite(output);
    return output;
}

LayerGradients Layer::backward(std::span<const double> input, std::size_t batch,
                               std::span<const double> output_gradient) const {
    validate_state();
    const auto input_size = checked_size(batch, inputs_), output_size = checked_size(batch, outputs_);
    if (input.size() != input_size || output_gradient.size() != output_size)
        throw std::invalid_argument("backward shape mismatch");
    require_finite(input); require_finite(output_gradient);
    LayerGradients gradient{std::vector<double>(input_size, 0.0),
                            std::vector<double>(coefficients_.size(), 0.0),
                            std::vector<double>(outputs_, 0.0), {}, {}, {}};
    if(rational_)gradient.denominators.resize(denominators_.size());
    if (!rational_ && basis_.kind == BasisKind::GaussianRbf && basis_.trainable_rbf) {
        gradient.centers.resize(basis_.size); gradient.log_widths.resize(basis_.size);
    }
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t o = 0; o < outputs_; ++o) gradient.bias[o] += output_gradient[b * outputs_ + o];
        for (std::size_t i = 0; i < inputs_; ++i) {
            if(rational_) {
                const auto m=rational_config_.numerator_degree+1,n=rational_config_.denominator_degree;
                for(std::size_t o=0;o<outputs_;++o) {
                    const auto edge=o*inputs_+i;
                    const auto r=evaluate_rational(rational_config_,input[b*inputs_+i],
                        std::span<const double>(coefficients_).subspan(edge*m,m),
                        std::span<const double>(denominators_).subspan(edge*n,n));
                    const auto upstream=output_gradient[b*outputs_+o];
                    gradient.input[b*inputs_+i]+=upstream*r.input_derivative;
                    for(std::size_t k=0;k<m;++k)gradient.coefficients[edge*m+k]+=upstream*r.numerator_derivatives[k];
                    for(std::size_t k=0;k<n;++k)gradient.denominators[edge*n+k]+=upstream*r.denominator_derivatives[k];
                }
                continue;
            }
            const auto basis = evaluate_basis(basis_, input[b * inputs_ + i]);
            for (std::size_t o = 0; o < outputs_; ++o) {
                const auto upstream = output_gradient[b * outputs_ + o];
                for (std::size_t k = 0; k < basis_.size; ++k) {
                    const auto index = (o * inputs_ + i) * basis_.size + k;
                    gradient.coefficients[index] += upstream * basis.values[k];
                    gradient.input[b * inputs_ + i] += upstream * coefficients_[index] * basis.derivatives[k];
                    if (!gradient.centers.empty()) {
                        gradient.centers[k] += upstream * coefficients_[index] * basis.center_derivatives[k];
                        gradient.log_widths[k] += upstream * coefficients_[index] * basis.log_width_derivatives[k];
                    }
                }
            }
        }
    }
    result_finite(gradient.input); result_finite(gradient.coefficients); result_finite(gradient.bias);
    result_finite(gradient.centers); result_finite(gradient.log_widths);
    result_finite(gradient.denominators);
    return gradient;
}

void Layer::sgd(const LayerGradients& gradients, double learning_rate) {
    validate_state();
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
        throw std::invalid_argument("learning rate must be finite and positive");
    if (gradients.coefficients.size() != coefficients_.size() || gradients.bias.size() != bias_.size())
        throw std::invalid_argument("parameter gradient shape mismatch");
    require_finite(gradients.coefficients); require_finite(gradients.bias);
    if(gradients.denominators.size()!=denominators_.size())throw std::invalid_argument("denominator gradient shape mismatch");
    require_finite(gradients.denominators);
    const bool trainable = !rational_ && basis_.kind == BasisKind::GaussianRbf && basis_.trainable_rbf;
    const std::size_t nonlinear_size = trainable ? basis_.size : 0;
    if (gradients.centers.size() != nonlinear_size || gradients.log_widths.size() != nonlinear_size)
        throw std::invalid_argument("nonlinear gradient shape mismatch");
    require_finite(gradients.centers); require_finite(gradients.log_widths);
    auto next_basis = basis_;
    if (trainable) {
        for (std::size_t k=0;k<basis_.size;++k) {
            next_basis.centers[k] -= learning_rate*gradients.centers[k];
            next_basis.log_widths[k] -= learning_rate*gradients.log_widths[k];
        }
        result_finite(next_basis.centers); result_finite(next_basis.log_widths);
        for(double w:next_basis.log_widths)
            if (!std::isfinite(std::exp(w)) || std::exp(w)<=0)
                throw std::overflow_error("RBF candidate width is not finite and positive");
    }
    auto next_coefficients = coefficients_, next_bias = bias_, next_denominators=denominators_;
    for (std::size_t i = 0; i < next_coefficients.size(); ++i)
        next_coefficients[i] -= learning_rate * gradients.coefficients[i];
    for (std::size_t i = 0; i < next_bias.size(); ++i) next_bias[i] -= learning_rate * gradients.bias[i];
    for(std::size_t i=0;i<next_denominators.size();++i)next_denominators[i]-=learning_rate*gradients.denominators[i];
    result_finite(next_coefficients); result_finite(next_bias);result_finite(next_denominators);
    coefficients_.swap(next_coefficients); bias_.swap(next_bias);
    denominators_.swap(next_denominators);
    std::swap(basis_,next_basis);
}

void Layer::set_rbf_parameters(std::span<const double> centers, std::span<const double> log_widths) {
    validate_state();
    if (rational_ || basis_.kind != BasisKind::GaussianRbf || !basis_.trainable_rbf)
        throw std::invalid_argument("RBF parameters require a trainable Gaussian basis");
    auto next=basis_; next.centers.assign(centers.begin(),centers.end());
    next.log_widths.assign(log_widths.begin(),log_widths.end()); validate_basis(next);
    std::swap(basis_,next);
}

void Layer::insert_knot(double x) {
    validate_state();
    if (rational_ || basis_.kind != BasisKind::BSpline || !std::isfinite(x) ||
        x<=basis_.knots[basis_.degree] || x>=basis_.knots[basis_.size])
        throw std::invalid_argument("knot must be strictly inside a spline domain");
    const auto& t=basis_.knots; const auto p=basis_.degree;
    const auto multiplicity=static_cast<std::size_t>(std::count(t.begin(),t.end(),x));
    if (multiplicity>=p+1) throw std::invalid_argument("knot multiplicity exceeds degree+1");
    const auto span=static_cast<std::size_t>(std::upper_bound(t.begin(),t.end(),x)-t.begin()-1);
    auto next_basis=basis_; ++next_basis.size;
    next_basis.knots.insert(next_basis.knots.begin()+span+1,x);validate_basis(next_basis);
    std::vector<double> next(checked_size(checked_size(inputs_,outputs_),next_basis.size));
    for(std::size_t edge=0;edge<inputs_*outputs_;++edge) {
        const auto* c=coefficients_.data()+edge*basis_.size;auto* q=next.data()+edge*next_basis.size;
        for(std::size_t j=0;j<next_basis.size;++j) {
            if(j<=span-p) q[j]=c[j];
            else if(j>=span-multiplicity+1) q[j]=c[j-1];
            else {
                const double denominator=t[j+p]-t[j], numerator=x-t[j];
                const double alpha=std::isfinite(denominator) ? numerator/denominator :
                    (x/2-t[j]/2)/(t[j+p]/2-t[j]/2);
                q[j]=(1-alpha)*c[j-1]+alpha*c[j];
            }
        }
    }
    result_finite(next); coefficients_.swap(next);std::swap(basis_,next_basis);
}

double Layer::adapt_grid(std::span<const double> samples) {
    validate_state();require_finite(samples);
    if(rational_ || basis_.kind!=BasisKind::BSpline)throw std::invalid_argument("adaptation requires splines");
    const auto& t=basis_.knots;
    std::vector<std::vector<double>> spans(basis_.size);
    for(double x:samples) {
        if(x<t[basis_.degree] || x>t[basis_.size])continue;
        auto index=x==t[basis_.size] ? basis_.size-1 :
            static_cast<std::size_t>(std::upper_bound(t.begin(),t.end(),x)-t.begin()-1);
        spans[index].push_back(x);
    }
    std::size_t best=basis_.degree;
    for(std::size_t k=basis_.degree;k<basis_.size;++k)
        if(spans[k].size()>spans[best].size())best=k;
    auto values=std::move(spans[best]);
    if(values.empty())throw std::invalid_argument("no in-domain samples to adapt");
    std::sort(values.begin(),values.end());
    double x=values[values.size()/2];
    if(values.size()%2==0)x=std::midpoint(values[values.size()/2-1],x);
    if(x<=t[best] || x>=t[best+1])x=std::midpoint(t[best],t[best+1]);
    if(x<=t[best] || x>=t[best+1])throw std::invalid_argument("span has no representable interior knot");
    insert_knot(x);return x;
}

RegularizationResult Layer::regularization(double lambda) const {
    validate_state();
    if(!std::isfinite(lambda)||lambda<0)throw std::invalid_argument("L2 coefficient must be finite and nonnegative");
    RegularizationResult r; r.gradients.coefficients.resize(coefficients_.size());
    r.gradients.bias.resize(outputs_);
    r.gradients.denominators.resize(denominators_.size());
    if(!rational_&&basis_.kind==BasisKind::GaussianRbf&&basis_.trainable_rbf) {
        r.gradients.centers.resize(basis_.size); r.gradients.log_widths.resize(basis_.size);
    }
    for(std::size_t j=0;j<coefficients_.size();++j) {
        const double g=lambda*coefficients_[j];r.gradients.coefficients[j]=g;
        r.value+=(0.5*g)*coefficients_[j];
    }
    result_finite(r.gradients.coefficients);result_finite(std::span<const double>(&r.value,1));return r;
}
} // namespace kan

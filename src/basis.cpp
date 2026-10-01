#include "kan/basis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kan {

namespace {
double finite_result(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("basis result is not finite");
    return value;
}

void positive_finite(double value, const char* message) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(message);
}

double normalized_distance(double x, double center, double width) {
    const double distance = x-center;
    return std::isfinite(distance) ? distance/width : x/width-center/width;
}

// Ratios on a huge domain remain finite even when its length overflows.
double interval_ratio(double numerator_right, double numerator_left,
                      double right, double left) {
    const double denominator = right-left;
    if (denominator == 0) return 0;
    if (std::isfinite(denominator)) return (numerator_right-numerator_left)/denominator;
    return (0.5*numerator_right-0.5*numerator_left)/(0.5*right-0.5*left);
}

double spline_slope_term(double value, std::size_t degree, double right, double left) {
    if (value == 0 || right == left) return 0;
    const double denominator = right-left;
    const double p = static_cast<double>(degree);
    return std::isfinite(denominator) ? (p*value)/denominator :
           (0.5*p*value)/(0.5*right-0.5*left);
}
}

void validate_basis(const BasisConfig& config) {
    if (config.size == 0) throw std::invalid_argument("basis size must be positive");
    if (config.size > std::vector<double>{}.max_size())
        throw std::overflow_error("basis size exceeds vector capacity");
    if (config.trainable_rbf && config.kind != BasisKind::GaussianRbf)
        throw std::invalid_argument("trainable RBF parameters require GaussianRbf");
    switch (config.kind) {
    case BasisKind::Chebyshev:
    case BasisKind::Legendre:
    case BasisKind::Hermite:
        break;
    case BasisKind::Jacobi:
        if (!std::isfinite(config.alpha) || !std::isfinite(config.beta) ||
            config.alpha <= -1 || config.beta <= -1)
            throw std::invalid_argument("Jacobi alpha and beta must be finite and greater than -1");
        break;
    case BasisKind::Fourier:
        if (config.size % 2 == 0) throw std::invalid_argument("Fourier size must be odd");
        positive_finite(config.frequency, "Fourier frequency must be finite and positive");
        break;
    case BasisKind::GaussianRbf:
        if (config.centers.size() != config.size)
            throw std::invalid_argument("Gaussian center count must match basis size");
        for (double center : config.centers)
            if (!std::isfinite(center)) throw std::invalid_argument("Gaussian centers must be finite");
        if (config.trainable_rbf) {
            if (config.log_widths.size() != config.size)
                throw std::invalid_argument("Gaussian log width count must match basis size");
            for (double log_width : config.log_widths) {
                if (!std::isfinite(log_width))
                    throw std::invalid_argument("Gaussian log widths must be finite");
                positive_finite(std::exp(log_width), "Gaussian exponentiated widths must be finite and positive");
            }
        } else positive_finite(config.width, "Gaussian width must be finite and positive");
        break;
    case BasisKind::BSpline: {
        if (config.degree > 16 || config.size < config.degree+1)
            throw std::invalid_argument("spline degree must be in [0,16] and size at least degree+1");
        if (config.size > std::vector<double>{}.max_size()-config.degree-1 ||
            config.knots.size() != config.size+config.degree+1)
            throw std::invalid_argument("spline knot count must be size+degree+1");
        for (std::size_t j=0; j<config.knots.size(); ++j)
            if (!std::isfinite(config.knots[j]) || (j && config.knots[j]<config.knots[j-1]))
                throw std::invalid_argument("spline knots must be finite and nondecreasing");
        const auto& t = config.knots;
        if (!(t[config.degree]<t[config.size]))
            throw std::invalid_argument("spline domain must have positive length");
        for (std::size_t j=0; j<=config.degree; ++j)
            if (t[j] != t[config.degree] || t[config.size+j] != t[config.size])
                throw std::invalid_argument("spline endpoints must be clamped");
        if (t[config.degree+1]==t[config.degree] || t[config.size-1]==t[config.size])
            throw std::invalid_argument("spline endpoint multiplicity must be exactly degree+1");
        std::size_t multiplicity = 1;
        for (std::size_t j=config.degree+2; j<config.size; ++j) {
            multiplicity = t[j]==t[j-1] ? multiplicity+1 : 1;
            if (multiplicity>config.degree+1)
                throw std::invalid_argument("spline interior multiplicity exceeds degree+1");
        }
        break;
    }
    case BasisKind::MexicanHat:
        if (config.centers.size()!=config.size || config.scales.size()!=config.size)
            throw std::invalid_argument("wavelet translations and scales must match basis size");
        for (double center : config.centers)
            if (!std::isfinite(center)) throw std::invalid_argument("wavelet translations must be finite");
        for (double scale : config.scales)
            positive_finite(scale,"wavelet scales must be finite and positive");
        break;
    default:
        throw std::invalid_argument("unknown basis kind");
    }
}

BasisValues evaluate_basis(const BasisConfig& config, double x) {
    validate_basis(config);
    if (!std::isfinite(x)) throw std::invalid_argument("basis input must be finite");
    BasisValues result{std::vector<double>(config.size), std::vector<double>(config.size), {}, {}};

    if (config.kind == BasisKind::BSpline) {
        const auto& t = config.knots;
        if (x<t[config.degree] || x>t[config.size]) return result;
        // One interval is active. Seed the last inward span at the upper end;
        // upper_bound selects the right side of repeated interior knots.
        const std::size_t span = x==t[config.size] ? config.size-1 :
            static_cast<std::size_t>(std::upper_bound(t.begin(),t.end(),x)-t.begin()-1);
        std::vector<double> lower(t.size()-1);
        lower[span]=1;
        for (std::size_t p=1; p<=config.degree; ++p) {
            std::vector<double> next(t.size()-p-1);
            for (std::size_t i=0; i<next.size(); ++i) {
                if (lower[i]!=0)
                    next[i] += interval_ratio(x,t[i],t[i+p],t[i])*lower[i];
                if (lower[i+1]!=0)
                    next[i] += interval_ratio(t[i+p+1],x,t[i+p+1],t[i+1])*lower[i+1];
                next[i]=finite_result(next[i]);
            }
            if (p==config.degree)
                for (std::size_t i=0; i<config.size; ++i)
                    result.derivatives[i]=finite_result(
                        spline_slope_term(lower[i],p,t[i+p],t[i])-
                        spline_slope_term(lower[i+1],p,t[i+p+1],t[i+1]));
            lower=std::move(next);
        }
        std::copy_n(lower.begin(),config.size,result.values.begin());
        return result;
    }

    if (config.kind == BasisKind::MexicanHat) {
        const double log_normalization = std::log(2/std::sqrt(3.))-
                                         0.25*std::log(std::acos(-1.));
        for (std::size_t k=0; k<config.size; ++k) {
            const double scale=config.scales[k];
            const double q=normalized_distance(x,config.centers[k],scale), q2=q*q;
            // An infinite distance is an exact zero tail, not an inf*zero NaN.
            if (!std::isfinite(q2)) continue;
            const double log_scale=std::log(scale);
            const double envelope=log_normalization-0.5*log_scale-0.5*q2;
            if (q2!=1)
                result.values[k]=finite_result(std::copysign(
                    std::exp(envelope+std::log(std::abs(1-q2))),1-q2));
            if (q!=0 && q2!=3)
                result.derivatives[k]=finite_result(std::copysign(
                    std::exp(envelope+std::log(std::abs(q))+std::log(std::abs(q2-3))-log_scale),
                    q*(q2>3 ? 1 : -1)));
        }
        return result;
    }

    if (config.kind == BasisKind::GaussianRbf) {
        if (config.trainable_rbf) {
            result.center_derivatives.resize(config.size);
            result.log_width_derivatives.resize(config.size);
        }
        for (std::size_t k = 0; k < config.size; ++k) {
            const double width = config.trainable_rbf ? std::exp(config.log_widths[k]) : config.width;
            // Overflow in subtraction is possible only for opposite signs. Divide
            // before subtracting in that case, retaining a wide Gaussian's tail.
            const double q = normalized_distance(x,config.centers[k],width);
            const double value = std::exp(-q*q);
            result.values[k] = value;
            double derivative = 0;
            if (q != 0 && std::isfinite(q)) {
                if (value < std::numeric_limits<double>::min()) {
                    // A tiny width can amplify a value that underflows to zero
                    // into a representable derivative. Preserve it in log space.
                    const double log_magnitude = std::log(2.0) + std::log(std::abs(q)) -
                                                 q*q - std::log(width);
                    derivative = -std::copysign(std::exp(log_magnitude), q);
                } else {
                    derivative = (-2*q*value) / width;
                }
            }
            result.derivatives[k] = finite_result(derivative);
            if (config.trainable_rbf) {
                result.center_derivatives[k]=-derivative;
                // Log-space avoids q*q * zero in remote tails.
                if (q!=0 && std::isfinite(q))
                    result.log_width_derivatives[k]=finite_result(std::exp(
                        std::log(2.)+2*std::log(std::abs(q))-q*q));
            }
        }
        return result;
    }

    result.values[0] = 1;
    if (config.kind == BasisKind::Fourier) {
        for (std::size_t k = 1; k <= config.size / 2; ++k) {
            const double angular = finite_result(static_cast<double>(k) * config.frequency);
            const double phase = finite_result(angular*x);
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
            result.values[2*k-1] = cosine;
            result.values[2*k] = sine;
            result.derivatives[2*k-1] = finite_result(-angular*sine);
            result.derivatives[2*k] = finite_result(angular*cosine);
        }
        return result;
    }
    if (config.size == 1) return result;

    // Half-sums avoid overflow in Jacobi's valid, very large parameters.
    const double half_sum = config.kind == BasisKind::Jacobi ?
                            0.5*config.alpha + 0.5*config.beta : 0;
    // Preserve distance from the admissible boundary alpha,beta > -1.
    // Adding one after summing the parameters can erase that distance.
    const double shifted_half_sum = config.kind == BasisKind::Jacobi ?
                                   0.5*(config.alpha+1) + 0.5*(config.beta+1) : 0;
    const double half_difference = config.kind == BasisKind::Jacobi ?
                                   0.5*config.alpha - 0.5*config.beta : 0;
    if (config.kind == BasisKind::Jacobi && (x == -1 || x == 1)) {
        // DLMF 18.6.T1: endpoint rising-factorial values. The general
        // three-term recurrence can cancel a small endpoint against huge terms.
        const double parameter = x == 1 ? config.alpha : config.beta;
        double endpoint_value = 1;
        double shifted_endpoint_value = 1;
        for (std::size_t k = 1; k < config.size; ++k) {
            const double n = static_cast<double>(k);
            endpoint_value = finite_result(endpoint_value*((parameter+n)/n));
            result.values[k] = x < 0 && k % 2 ? -endpoint_value : endpoint_value;
            if (k > 1)
                shifted_endpoint_value = finite_result(shifted_endpoint_value*((parameter+n)/(n-1)));
            // DLMF 18.9.E15: P'_n = (n+alpha+beta+1)/2 *
            // P_(n-1)^(alpha+1,beta+1). Half-sums keep a finite slope
            // representable even when alpha+beta would overflow.
            const double derivative = finite_result((shifted_half_sum+0.5*(n-1))*shifted_endpoint_value);
            result.derivatives[k] = x < 0 && k % 2 == 0 ? -derivative : derivative;
        }
        return result;
    }
    double first_slope = 1;
    double first_offset = 0;
    if (config.kind == BasisKind::Hermite) first_slope = 2;
    if (config.kind == BasisKind::Jacobi) {
        first_slope = shifted_half_sum;
        first_offset = half_difference;
    }
    result.values[1] = finite_result(first_slope*x + first_offset);
    result.derivatives[1] = finite_result(first_slope);

    for (std::size_t k = 1; k < config.size-1; ++k) {
        const double n = static_cast<double>(k);
        double a = 0;
        double b = 0;
        double c = 0;
        switch (config.kind) {
        case BasisKind::Chebyshev:
            a = 2;
            c = 1;
            break;
        case BasisKind::Legendre:
            a = (2*n+1)/(n+1);
            c = n/(n+1);
            break;
        case BasisKind::Hermite:
            a = 2;
            c = 2*n;
            break;
        case BasisKind::Jacobi: {
            // NIST DLMF 18.9.2, rearranged into ratios to avoid squaring
            // large parameters. P1 above handles alpha+beta = -1 or 0.
            const double t = shifted_half_sum + (n-1);
            const double denominator_half = shifted_half_sum + 0.5*(n-1);
            a = ((t+0.5)/(n+1))*((t+1)/denominator_half);
            b = (half_difference/(n+1))*(half_sum/t)*((t+0.5)/denominator_half);
            c = 0.5*((n+config.alpha)/(n+1))*((n+config.beta)/t)*((t+1)/denominator_half);
            break;
        }
        default:
            throw std::logic_error("validated polynomial kind expected");
        }
        const double factor = a*x+b;
        result.values[k+1] = finite_result(factor*result.values[k] - c*result.values[k-1]);
        result.derivatives[k+1] = finite_result(a*result.values[k] +
            factor*result.derivatives[k] - c*result.derivatives[k-1]);
    }
    return result;
}
}

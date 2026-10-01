#include "kan/basis.hpp"
#include "detail/basis_formulas.hpp"

#include <cmath>
#include <stdexcept>

namespace kan {

namespace {
struct FiniteBasisGuard {
    double operator()(double value) const {
        if (!std::isfinite(value)) throw std::overflow_error("basis result is not finite");
        return value;
    }
};

void positive_finite(double value, const char* message) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(message);
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
        if (config.degree > detail::max_spline_degree || config.size < config.degree+1)
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
    if (config.trainable_rbf) {
        result.center_derivatives.resize(config.size);
        result.log_width_derivatives.resize(config.size);
    }
    const detail::BasisView view{config.kind, config.size, config.alpha, config.beta,
                                 config.frequency, config.width, config.centers.data(),
                                 config.log_widths.data(), config.scales.data(),
                                 config.knots.data(), config.degree, config.trainable_rbf};
    const detail::BasisRow row{result.values.data(), result.derivatives.data(),
                               config.trainable_rbf ? result.center_derivatives.data() : nullptr,
                               config.trainable_rbf ? result.log_width_derivatives.data() : nullptr};
    detail::basis_terms(view, x, row, FiniteBasisGuard{});
    return result;
}
}

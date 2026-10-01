#include "kan/basis.hpp"
#include "detail/basis_host.hpp"

#include <cmath>
#include <stdexcept>

namespace kan {

namespace {
void positive_finite(double value, const char* message) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(message);
}

void require_terms(std::size_t size) {
    if (size == 0) throw std::invalid_argument("basis size must be positive");
    if (size > std::vector<double>{}.max_size())
        throw std::overflow_error("basis size exceeds vector capacity");
}

void require_finite(const std::vector<double>& values, const char* message) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument(message);
}

void validate(const ChebyshevConfig& c) { require_terms(c.size); }
void validate(const LegendreConfig& c) { require_terms(c.size); }
void validate(const HermiteConfig& c) { require_terms(c.size); }

void validate(const JacobiConfig& c) {
    require_terms(c.size);
    if (!std::isfinite(c.alpha) || !std::isfinite(c.beta) || c.alpha <= -1 || c.beta <= -1)
        throw std::invalid_argument("Jacobi alpha and beta must be finite and greater than -1");
}

void validate(const FourierConfig& c) {
    require_terms(c.size);
    if (c.size % 2 == 0) throw std::invalid_argument("Fourier size must be odd");
    positive_finite(c.frequency, "Fourier frequency must be finite and positive");
}

void validate(const GaussianRbfConfig& c) {
    require_terms(c.centers.size());
    require_finite(c.centers, "Gaussian centers must be finite");
    positive_finite(c.width, "Gaussian width must be finite and positive");
}

void validate(const TrainableRbfConfig& c) {
    require_terms(c.centers.size());
    if (c.log_widths.size() != c.centers.size())
        throw std::invalid_argument("Gaussian log width count must match basis size");
    require_finite(c.centers, "Gaussian centers must be finite");
    for (double log_width : c.log_widths) {
        if (!std::isfinite(log_width)) throw std::invalid_argument("Gaussian log widths must be finite");
        positive_finite(std::exp(log_width), "Gaussian exponentiated widths must be finite and positive");
    }
}

void validate(const BSplineConfig& c) {
    if (c.degree > detail::max_spline_degree) throw std::invalid_argument("spline degree must be in [0,16]");
    const auto size = basis_size(c);
    if (size < c.degree + 1)
        throw std::invalid_argument("spline needs at least 2*(degree+1) knots (degree+1 terms)");
    const auto& t = c.knots;
    for (std::size_t j = 0; j < t.size(); ++j)
        if (!std::isfinite(t[j]) || (j && t[j] < t[j - 1]))
            throw std::invalid_argument("spline knots must be finite and nondecreasing");
    if (!(t[c.degree] < t[size])) throw std::invalid_argument("spline domain must have positive length");
    for (std::size_t j = 0; j <= c.degree; ++j)
        if (t[j] != t[c.degree] || t[size + j] != t[size])
            throw std::invalid_argument("spline endpoints must be clamped");
    if (t[c.degree + 1] == t[c.degree] || t[size - 1] == t[size])
        throw std::invalid_argument("spline endpoint multiplicity must be exactly degree+1");
    std::size_t multiplicity = 1;
    for (std::size_t j = c.degree + 2; j < size; ++j) {
        multiplicity = t[j] == t[j - 1] ? multiplicity + 1 : 1;
        if (multiplicity > c.degree + 1)
            throw std::invalid_argument("spline interior multiplicity exceeds degree+1");
    }
}

void validate(const MexicanHatConfig& c) {
    require_terms(c.centers.size());
    if (c.scales.size() != c.centers.size())
        throw std::invalid_argument("wavelet translations and scales must match basis size");
    require_finite(c.centers, "wavelet translations must be finite");
    for (double scale : c.scales) positive_finite(scale, "wavelet scales must be finite and positive");
}
}

std::size_t basis_size(const BasisConfig& config) noexcept {
    return std::visit([](const auto& c) { return basis_size(c); }, config);
}

void validate_basis(const BasisConfig& config) {
    std::visit([](const auto& c) { validate(c); }, config);
}

BasisValues evaluate_basis(const BasisConfig& config, double x) {
    validate_basis(config);
    if (!std::isfinite(x)) throw std::invalid_argument("basis input must be finite");
    const auto view = detail::basis_view(config);
    BasisValues result{std::vector<double>(view.terms), std::vector<double>(view.terms), {}, {}};
    if (view.trainable) {
        result.center_derivatives.resize(view.terms);
        result.log_width_derivatives.resize(view.terms);
    }
    const detail::BasisRow row{result.values.data(), result.derivatives.data(),
                               view.trainable ? result.center_derivatives.data() : nullptr,
                               view.trainable ? result.log_width_derivatives.data() : nullptr};
    detail::basis_terms(view, x, row, detail::FiniteBasisGuard{});
    return result;
}
}

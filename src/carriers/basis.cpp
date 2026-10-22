#include "kan/basis.hpp"
#include "backends/cpu/basis_host.hpp"

#include <cmath>
#include <stdexcept>

namespace kan {

namespace {
void positive_finite(double value, const char* message) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(message);
}

void require_terms(std::size_t size) {
    if (size == 0) throw std::invalid_argument("basis size must be positive");
    if (size > std::vector<double>{}.max_size()) throw std::overflow_error("basis size exceeds vector capacity");
}

void require_finite(const std::vector<double>& values, const char* message) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument(message);
}

void validate(const ChebyshevConfig& c) { require_terms(c.size); }
void validate(const LegendreConfig& c) { require_terms(c.size); }
void validate(const HermiteConfig& c) {
    require_terms(c.size);
    if (c.normalization != HermiteNormalization::Physicists && c.normalization != HermiteNormalization::Function)
        throw std::invalid_argument("unknown Hermite normalization");
}

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

// One clamped knot vector t[0..n) of degree p.
void validate_grid(std::size_t degree, const double* t, std::size_t n) {
    const auto size = n > degree && n - degree > 1 ? n - degree - 1 : 0;
    if (size < degree + 1) throw std::invalid_argument("spline needs at least 2*(degree+1) knots (degree+1 terms)");
    for (std::size_t j = 0; j < n; ++j)
        if (!std::isfinite(t[j]) || (j && t[j] < t[j - 1]))
            throw std::invalid_argument("spline knots must be finite and nondecreasing");
    if (!(t[degree] < t[size])) throw std::invalid_argument("spline domain must have positive length");
    for (std::size_t j = 0; j <= degree; ++j)
        if (t[j] != t[degree] || t[size + j] != t[size])
            throw std::invalid_argument("spline endpoints must be clamped");
    if (t[degree + 1] == t[degree] || t[size - 1] == t[size])
        throw std::invalid_argument("spline endpoint multiplicity must be exactly degree+1");
    std::size_t multiplicity = 1;
    for (std::size_t j = degree + 2; j < size; ++j) {
        multiplicity = t[j] == t[j - 1] ? multiplicity + 1 : 1;
        if (multiplicity > degree + 1) throw std::invalid_argument("spline interior multiplicity exceeds degree+1");
    }
}

// Backlog M6: `grids` knot vectors of equal length, each validated as before.
void validate(const BSplineConfig& c) {
    if (c.degree > detail::max_spline_degree) throw std::invalid_argument("spline degree must be in [0,16]");
    if (c.grids == 0 || c.knots.size() % c.grids != 0)
        throw std::invalid_argument("spline knots must hold `grids` knot vectors of equal length");
    const auto n = c.knots.size() / c.grids;
    for (std::size_t g = 0; g < c.grids; ++g) validate_grid(c.degree, c.knots.data() + g * n, n);
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

BSplineConfig spline_grid(const BSplineConfig& config, std::size_t grid) {
    validate(config);
    if (grid >= config.grids) throw std::invalid_argument("spline grid index out of range");
    const auto n = config.knots.size() / config.grids;
    const auto first = config.knots.begin() + static_cast<std::ptrdiff_t>(grid * n);
    return {config.degree, std::vector<double>(first, first + static_cast<std::ptrdiff_t>(n)), 1};
}

BSplineConfig per_input_grids(const BSplineConfig& config, std::size_t inputs) {
    validate(config);
    if (config.grids != 1) throw std::invalid_argument("per_input_grids requires a single-grid configuration");
    if (inputs == 0) throw std::invalid_argument("per_input_grids requires a positive input count");
    if (config.knots.size() > std::vector<double>{}.max_size() / inputs)
        throw std::overflow_error("spline knot count exceeds vector capacity");
    BSplineConfig result{config.degree, {}, inputs};
    result.knots.reserve(config.knots.size() * inputs);
    for (std::size_t i = 0; i < inputs; ++i)
        result.knots.insert(result.knots.end(), config.knots.begin(), config.knots.end());
    return result;
}

BasisValues evaluate_basis(const BasisConfig& config, double x) {
    validate_basis(config);
    if (const auto* spline = std::get_if<BSplineConfig>(&config); spline && spline->grids != 1)
        throw std::invalid_argument("a spline with several grids has no single evaluation; use spline_grid");
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

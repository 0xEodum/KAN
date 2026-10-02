#pragma once
// Test-only family parameterization: suites that sweep every family with one
// shared parameter set build the typed configuration through `basis`.
#include "kan/layer.hpp"
#include <cstddef>
#include <span>
#include <variant>
#include <vector>

namespace test {

enum class Family { Chebyshev, Legendre, Jacobi, Hermite, Fourier, GaussianRbf, TrainableRbf, BSpline, MexicanHat };

struct FamilyParameters {
    std::size_t size = 4; // global families only; local families use their vectors
    double alpha = 0, beta = 0, frequency = 1, width = 1;
    std::vector<double> centers, log_widths, scales;
    std::size_t degree = 3;
    std::vector<double> knots;
};

inline kan::BasisConfig basis(Family family, const FamilyParameters& p) {
    switch (family) {
    case Family::Chebyshev: return kan::ChebyshevConfig{p.size};
    case Family::Legendre: return kan::LegendreConfig{p.size};
    case Family::Jacobi: return kan::JacobiConfig{p.size, p.alpha, p.beta};
    case Family::Hermite: return kan::HermiteConfig{p.size};
    case Family::Fourier: return kan::FourierConfig{p.size, p.frequency};
    case Family::GaussianRbf: return kan::GaussianRbfConfig{p.centers, p.width};
    case Family::TrainableRbf: return kan::TrainableRbfConfig{p.centers, p.log_widths};
    case Family::BSpline: return kan::BSplineConfig{p.degree, p.knots};
    case Family::MexicanHat: return kan::MexicanHatConfig{p.centers, p.scales};
    }
    return kan::ChebyshevConfig{p.size};
}

// Carrier accessors for test assertions.
inline const kan::TrainableRbfConfig& trainable(const kan::Layer& layer) {
    return std::get<kan::TrainableRbfEdges>(layer.carrier()).basis;
}
inline const kan::TrainableRbfGradients& trainable(const kan::LayerGradients& gradients) {
    return std::get<kan::TrainableRbfGradients>(gradients.nonlinear);
}
inline kan::TrainableRbfGradients& trainable(kan::LayerGradients& gradients) {
    return std::get<kan::TrainableRbfGradients>(gradients.nonlinear);
}
inline const kan::BasisConfig& basis_of(const kan::Layer& layer) {
    return std::get<kan::BasisEdges>(layer.carrier()).basis;
}

// Nonlinear parameter and gradient views; empty for other carriers.
inline std::span<const double> centers(const kan::LayerGradients& g) {
    const auto* r = std::get_if<kan::TrainableRbfGradients>(&g.nonlinear);
    return r ? std::span<const double>(r->centers) : std::span<const double>();
}
inline std::span<const double> log_widths(const kan::LayerGradients& g) {
    const auto* r = std::get_if<kan::TrainableRbfGradients>(&g.nonlinear);
    return r ? std::span<const double>(r->log_widths) : std::span<const double>();
}
inline std::span<const double> denominators(const kan::LayerGradients& g) {
    const auto* r = std::get_if<kan::RationalGradients>(&g.nonlinear);
    return r ? std::span<const double>(r->denominators) : std::span<const double>();
}
inline kan::RationalGradients& rational(kan::LayerGradients& g) {
    return std::get<kan::RationalGradients>(g.nonlinear);
}
inline std::span<const double> denominators(const kan::Layer& layer) {
    const auto* r = std::get_if<kan::RationalEdges>(&layer.carrier());
    return r ? std::span<const double>(r->denominators) : std::span<const double>();
}

} // namespace test

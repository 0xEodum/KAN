#pragma once
// Test-only family parameterization: suites that sweep every family with one
// shared parameter set build the typed configuration through `basis`.
#include "kan/basis.hpp"
#include <cstddef>
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

inline const kan::TrainableRbfConfig& trainable(const kan::BasisConfig& config) {
    return std::get<kan::TrainableRbfConfig>(config);
}

} // namespace test

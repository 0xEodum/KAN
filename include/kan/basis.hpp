#pragma once

#include <cstddef>
#include <vector>

namespace kan {

enum class BasisKind { Chebyshev, Legendre, Jacobi, Hermite, Fourier, GaussianRbf, BSpline, MexicanHat };

struct BasisConfig {
    BasisKind kind = BasisKind::Chebyshev;
    std::size_t size = 4; // polynomial degree + 1; Fourier: 1 + 2 * harmonics
    double alpha = 0.0; // Jacobi alpha > -1
    double beta = 0.0; // Jacobi beta > -1
    double frequency = 1.0; // Fourier angular frequency > 0
    std::vector<double> centers; // RBF: exactly size centers
    double width = 1.0; // RBF: exp(-((x-center)/width)^2), width > 0
    std::size_t degree = 3; // BSpline degree, in [0,16]
    std::vector<double> knots; // BSpline: size + degree + 1 clamped knots
    std::vector<double> scales; // MexicanHat: exactly size positive scales
    bool trainable_rbf = false; // GaussianRbf nonlinear center/log-width parameters
    std::vector<double> log_widths; // Trainable GaussianRbf: exactly size log widths
};

struct BasisValues {
    std::vector<double> values;
    std::vector<double> derivatives;
    std::vector<double> center_derivatives; // Populated only for trainable RBFs
    std::vector<double> log_width_derivatives;
};

void validate_basis(const BasisConfig& config);
BasisValues evaluate_basis(const BasisConfig& config, double x);

} // namespace kan

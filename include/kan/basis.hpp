#pragma once

#include <cstddef>
#include <vector>

namespace kan {

enum class BasisKind { Chebyshev, Legendre, Jacobi, Hermite, Fourier, GaussianRbf };

struct BasisConfig {
    BasisKind kind = BasisKind::Chebyshev;
    std::size_t size = 4; // polynomial degree + 1; Fourier: 1 + 2 * harmonics
    double alpha = 0.0; // Jacobi alpha > -1
    double beta = 0.0; // Jacobi beta > -1
    double frequency = 1.0; // Fourier angular frequency > 0
    std::vector<double> centers; // RBF: exactly size centers
    double width = 1.0; // RBF: exp(-((x-center)/width)^2), width > 0
};

struct BasisValues {
    std::vector<double> values;
    std::vector<double> derivatives;
};

void validate_basis(const BasisConfig& config);
BasisValues evaluate_basis(const BasisConfig& config, double x);

} // namespace kan

#pragma once

#include <cstddef>
#include <variant>
#include <vector>

namespace kan {

// Typed per-family basis configurations. Each type holds only its family's
// parameters. Global families (polynomials, Fourier) state their term count;
// localized families derive it from their parameter vectors (see basis_size).

struct ChebyshevConfig {
    std::size_t size = 4; // terms T_0..T_(size-1), never a degree
    bool operator==(const ChebyshevConfig&) const = default;
};

struct LegendreConfig {
    std::size_t size = 4;
    bool operator==(const LegendreConfig&) const = default;
};

struct JacobiConfig {
    std::size_t size = 4;
    double alpha = 0.0; // finite, > -1
    double beta = 0.0;  // finite, > -1
    bool operator==(const JacobiConfig&) const = default;
};

struct HermiteConfig {
    std::size_t size = 4; // physicists' H_n
    bool operator==(const HermiteConfig&) const = default;
};

struct FourierConfig {
    std::size_t size = 3;   // 1 + 2 * harmonics, odd
    double frequency = 1.0; // angular frequency, finite and > 0
    bool operator==(const FourierConfig&) const = default;
};

// Fixed Gaussian RBF exp(-((x-center)/width)^2); one term per center.
struct GaussianRbfConfig {
    std::vector<double> centers;
    double width = 1.0; // finite, > 0
    bool operator==(const GaussianRbfConfig&) const = default;
};

// Gaussian RBF with trainable centers and log widths shared by a layer's edges
// (nonlinear parameters); one term per center, log_widths of equal length.
struct TrainableRbfConfig {
    std::vector<double> centers;
    std::vector<double> log_widths;
    bool operator==(const TrainableRbfConfig&) const = default;
};

// Clamped B-spline of degree 0..16; knots.size() - degree - 1 terms.
struct BSplineConfig {
    std::size_t degree = 3;
    std::vector<double> knots;
    bool operator==(const BSplineConfig&) const = default;
};

// L2-normalized Mexican hat; one term per translation, scales of equal length.
struct MexicanHatConfig {
    std::vector<double> centers; // translations
    std::vector<double> scales;  // finite, > 0
    bool operator==(const MexicanHatConfig&) const = default;
};

// A value-initialized BasisConfig is ChebyshevConfig{4}.
using BasisConfig = std::variant<ChebyshevConfig, LegendreConfig, JacobiConfig, HermiteConfig,
                                 FourierConfig, GaussianRbfConfig, TrainableRbfConfig,
                                 BSplineConfig, MexicanHatConfig>;

struct BasisValues {
    std::vector<double> values;
    std::vector<double> derivatives;
    std::vector<double> center_derivatives; // Populated only for TrainableRbfConfig
    std::vector<double> log_width_derivatives;
};

// Number of terms. It is meaningful only for a configuration that passes
// validate_basis; a spline with too few knots reports zero.
constexpr std::size_t basis_size(const ChebyshevConfig& c) noexcept { return c.size; }
constexpr std::size_t basis_size(const LegendreConfig& c) noexcept { return c.size; }
constexpr std::size_t basis_size(const JacobiConfig& c) noexcept { return c.size; }
constexpr std::size_t basis_size(const HermiteConfig& c) noexcept { return c.size; }
constexpr std::size_t basis_size(const FourierConfig& c) noexcept { return c.size; }
inline std::size_t basis_size(const GaussianRbfConfig& c) noexcept { return c.centers.size(); }
inline std::size_t basis_size(const TrainableRbfConfig& c) noexcept { return c.centers.size(); }
inline std::size_t basis_size(const MexicanHatConfig& c) noexcept { return c.centers.size(); }
inline std::size_t basis_size(const BSplineConfig& c) noexcept {
    return c.knots.size() > c.degree && c.knots.size() - c.degree > 1 ? c.knots.size() - c.degree - 1 : 0;
}
std::size_t basis_size(const BasisConfig& config) noexcept;

void validate_basis(const BasisConfig& config);
BasisValues evaluate_basis(const BasisConfig& config, double x);

} // namespace kan

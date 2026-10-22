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

// Hermite normalization (backlog M5). Physicists: H_n, the default; its terms
// grow like 2^n n! and condition poorly at high degree. Function: the
// normalized Hermite functions psi_n(x) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi)),
// orthonormal in L2(R) and bounded by pi^(-1/4) for every n and x.
enum class HermiteNormalization { Physicists, Function };

struct HermiteConfig {
    std::size_t size = 4; // terms of index 0..size-1
    HermiteNormalization normalization = HermiteNormalization::Physicists;
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

// Clamped B-spline of degree 0..16. `knots` holds `grids` knot vectors of
// equal length n one after another (backlog M6): grids = 1 is one grid shared
// by every input of a layer; a layer with grids = inputs gives input i its own
// grid knots[i*n, (i+1)*n) (pykan's per-input grid). Every grid has the same
// degree and n - degree - 1 terms (basis_size).
struct BSplineConfig {
    std::size_t degree = 3;
    std::vector<double> knots;
    std::size_t grids = 1;
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
    if (c.grids == 0 || c.knots.size() % c.grids != 0) return 0;
    const auto n = c.knots.size() / c.grids;
    return n > c.degree && n - c.degree > 1 ? n - c.degree - 1 : 0;
}
std::size_t basis_size(const BasisConfig& config) noexcept;

void validate_basis(const BasisConfig& config);
// A spline with several grids (backlog M6) has no single evaluation:
// evaluate_basis raises std::invalid_argument; evaluate spline_grid(config, i).
BasisValues evaluate_basis(const BasisConfig& config, double x);

// Per-input spline grids (backlog M6). spline_grid returns grid `grid` of a
// validated configuration as a single-grid configuration (std::invalid_argument
// for an invalid configuration or grid index); per_input_grids repeats a
// single-grid configuration's knots for `inputs` inputs (inputs > 0).
BSplineConfig spline_grid(const BSplineConfig& config, std::size_t grid);
BSplineConfig per_input_grids(const BSplineConfig& config, std::size_t inputs);

} // namespace kan

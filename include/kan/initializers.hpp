#pragma once

#include "kan/network.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace kan {

// Explicit, typed parameter initializers (backlog M4). Constructors keep
// zero initialization; an initializer is applied only on request. Every
// initializer is deterministic: the same configuration and seed give
// bitwise-identical parameters on every platform (SplitMix64 and a transform
// built from IEEE basic operations and sqrt only; see docs/CONTRACT.md).

// Distribution of the random draws. Uniform is symmetric on an interval,
// Normal is Gaussian (Marsaglia polar method); both have the stated variance.
enum class Distribution { Uniform, Normal };

// Rational denominators b are drawn so that |S(z)| = |sum_k b_k z^k| <= bound
// for every |z| <= radius: |b_k| <= bound / (n * radius^k), with
// |b_k| >= bound / (2 n radius^k), so no b_k is zero. Hence on that interval
// Q >= 1 - bound for Guarded, Q in [1, 1 + bound] for Absolute and
// Q in [1, 1 + bound^2] for Smooth. bound is in (0, 1), radius finite > 0.
struct DenominatorInit {
    double bound = 0.5;
    double radius = 1.0;
    bool operator==(const DenominatorInit&) const = default;
};

// Variance-preserving initialization. Each coefficient c[o,i,k] is drawn
// independently with zero mean and variance
//     gain^2 * variance / (inputs * terms * second_moments[k])
// from the layer's reference measure (reference_moments), so that each term
// contributes equally and an output has E[y^2] = gain^2 * variance when the
// inputs follow the reference measure. Rational numerators use the measure
// z uniform on [-radius, radius] and the drawn denominator of their edge.
// Bias is zero.
struct VarianceScaling {
    double gain = 1.0; // finite, > 0
    Distribution distribution = Distribution::Uniform;
    std::uint64_t seed = 0;
    DenominatorInit denominators;
    bool operator==(const VarianceScaling&) const = default;
};

// Small noise on the coefficients, as pykan's KANLayer: amplitude
// a = scale / (G * sqrt(inputs)), G the grid intervals of a B-spline
// (terms - degree) and the term count otherwise (rational: numerator
// degree + 1). Uniform draws are U(-a/2, a/2) (pykan), Normal draws have the
// same variance a^2/12. Rational denominators follow `denominators`. Bias is
// zero. scale is finite and > 0 (pykan's MultKAN default is 0.3).
struct NoiseInit {
    double scale = 0.3;
    Distribution distribution = Distribution::Uniform;
    std::uint64_t seed = 0;
    DenominatorInit denominators;
    bool operator==(const NoiseInit&) const = default;
};

using Initializer = std::variant<VarianceScaling, NoiseInit>;

// Reference measure of a basis family for VarianceScaling: the variance of
// the measure and E[phi_k(x)^2] per term. Polynomials use their
// orthogonality measure (Chebyshev arcsine on [-1,1], Legendre uniform,
// Jacobi the normalized weight (1-x)^alpha (1+x)^beta, Hermite N(0, 1/2));
// Fourier is uniform on one period [-pi/w, pi/w]; B-splines are uniform on
// the domain [t_p, t_K]; Gaussian RBFs and Mexican hats are uniform on
// [min center, max center], or [c - h, c + h] for coincident centers with h
// the largest width or scale.
struct BasisMoments {
    double variance = 0;
    std::vector<double> second_moments; // one per term
    bool operator==(const BasisMoments&) const = default;
};
BasisMoments reference_moments(const BasisConfig& config);

// Validates the initializer for the layer and replaces the layer's trainable
// parameters atomically (coefficients, bias, rational denominators). Trainable
// RBF centers and log widths are configuration of the reference measure and
// stay unchanged. Draw order: coefficients in layout order, then denominators.
void initialize(Layer& layer, const Initializer& initializer);

// Initializes every KAN layer of the network, the layer at position p with
// the initializer's seed replaced by layer_seed(seed, p), and resets trainable
// LayerNorm maps to gain 1 and bias 0; other maps are fixed and unchanged.
// All layers are validated before the network changes.
void initialize(Network& network, const Initializer& initializer);

// The (position + 1)-th output of SplitMix64 started at seed.
std::uint64_t layer_seed(std::uint64_t seed, std::size_t position) noexcept;

} // namespace kan

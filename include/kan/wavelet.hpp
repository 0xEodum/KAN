#pragma once

#include <cstddef>

namespace kan {

// Per-edge trainable Mexican-hat wavelets (backlog M7; Wav-KAN, Bozorgasl and
// Chen 2024). Every edge (o,i) holds `size` wavelets k, each with its own
// coefficient w, translation tau and log scale l (scale s = exp(l) > 0):
//     phi_{o,i}(x) = sum_k w[o,i,k] * psi(x; tau[o,i,k], l[o,i,k]),
//     psi(x; tau, l) = A * s^(-1/2) * (1 - q^2) * exp(-q^2/2),  q = (x - tau) / s,
// A = 2/(sqrt(3) pi^(1/4)): the L2-normalized Mexican hat of MexicanHatConfig,
// with tau and l trainable per edge and term. size = 1 is Wav-KAN's edge
// w * psi((x - tau)/s) up to the s^(-1/2) factor, a reparametrization of w;
// size > 1 is a per-edge trainable wavelet dictionary. The coefficients w
// are the carrier's per-edge coefficient tensor (L2-penalized); tau and l
// are nonlinear parameters (unpenalized).
struct WaveletConfig {
    std::size_t size = 1; // wavelets per edge, >= 1
    bool operator==(const WaveletConfig&) const = default;
};

// One wavelet with unit coefficient at x: psi and its derivatives with respect
// to x, the translation and the log scale (d/dtau = -d/dx).
struct WaveletEvaluation {
    double value = 0;
    double input_derivative = 0;
    double translation_derivative = 0;
    double log_scale_derivative = 0;
};

// size >= 1 (std::invalid_argument).
void validate_wavelet(const WaveletConfig& config);
// Log scales l are valid when l is finite and exp(l) and exp(-l) are finite
// and nonzero (|l| below about 709.78); std::invalid_argument otherwise, as
// for a nonfinite x or translation.
bool valid_log_scale(double log_scale) noexcept;
WaveletEvaluation evaluate_wavelet(double x, double translation, double log_scale);

} // namespace kan

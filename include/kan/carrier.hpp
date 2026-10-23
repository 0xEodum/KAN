#pragma once

#include "kan/basis.hpp"
#include "kan/pqc.hpp"
#include "kan/rational.hpp"
#include "kan/wavelet.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace kan {

// Edge carriers. A Layer holds exactly one carrier: the configuration and the
// trainable parameters of all its edges. Every carrier has a per-edge
// coefficient tensor of layout (outputs, inputs, terms), which the coefficient
// L2 penalty acts on (together with the layer's residual-branch weights,
// kan/layer.hpp); a carrier may add nonlinear parameters.

// Linear in its parameters: each edge is sum_k coefficients[o,i,k] * Phi_k(x).
// A layer is the expansion Phi: R^I -> R^(I*K) followed by the dense
// contraction Y = Phi * C^T + bias, C being outputs x (I*K). Holds any fixed
// family; a TrainableRbfConfig belongs to TrainableRbfEdges.
struct BasisEdges {
    BasisConfig basis;
    std::vector<double> coefficients; // (outputs, inputs, basis_size(basis))
    bool operator==(const BasisEdges&) const = default;
};

// Gaussian RBF expansion whose centers and log widths, shared by all edges of
// the layer, are nonlinear trainable parameters.
struct TrainableRbfEdges {
    TrainableRbfConfig basis;
    std::vector<double> coefficients; // (outputs, inputs, basis_size(basis))
    bool operator==(const TrainableRbfEdges&) const = default;
};

// Rational edges P(z)/Q(z), nonlinear in the per-edge denominator parameters.
struct RationalEdges {
    RationalConfig config;
    std::vector<double> coefficients; // numerator a, (outputs, inputs, numerator_degree+1)
    std::vector<double> denominators; // b, (outputs, inputs, denominator_degree)
    bool operator==(const RationalEdges&) const = default;
};

// Per-edge trainable Mexican-hat wavelets (backlog M7, kan/wavelet.hpp),
// nonlinear in the per-edge-term translations and log scales.
struct WaveletEdges {
    WaveletConfig config;
    std::vector<double> coefficients; // w, (outputs, inputs, size)
    std::vector<double> translations; // tau, (outputs, inputs, size)
    std::vector<double> log_scales;   // l = log s, (outputs, inputs, size)
    bool operator==(const WaveletEdges&) const = default;
};

// Parametrized quantum circuit edges (M5, kan/pqc.hpp): a linear readout of
// the measured expectations of one re-uploading circuit per edge, nonlinear in
// the per-edge rotation angles, encoding weights and encoding biases.
struct PqcEdges {
    PqcConfig config;
    std::vector<double> coefficients;     // c, (outputs, inputs, terms)
    std::vector<double> angles;           // theta, (outputs, inputs, (r+1) n B), index (l n + q) B + k
    std::vector<double> encoding_weights; // w, (outputs, inputs, r n), index (l-1) n + q
    std::vector<double> encoding_biases;  // b, same layout
    std::uint64_t shot_seed = 0;          // state (uploadable), used only when shots > 0
    bool operator==(const PqcEdges&) const = default;
};

using Carrier = std::variant<BasisEdges, TrainableRbfEdges, RationalEdges, WaveletEdges, PqcEdges>;

// Vector-Jacobian products of a carrier's nonlinear parameters. The active
// alternative corresponds to the carrier: std::monostate for BasisEdges.
struct TrainableRbfGradients {
    std::vector<double> centers;
    std::vector<double> log_widths;
    bool operator==(const TrainableRbfGradients&) const = default;
};

struct RationalGradients {
    std::vector<double> denominators; // (outputs, inputs, denominator_degree)
    bool operator==(const RationalGradients&) const = default;
};

struct WaveletGradients {
    std::vector<double> translations; // (outputs, inputs, size)
    std::vector<double> log_scales;   // (outputs, inputs, size)
    bool operator==(const WaveletGradients&) const = default;
};

struct PqcGradients {
    std::vector<double> angles;           // (outputs, inputs, (r+1) n B)
    std::vector<double> encoding_weights; // (outputs, inputs, r n)
    std::vector<double> encoding_biases;  // (outputs, inputs, r n)
    bool operator==(const PqcGradients&) const = default;
};

using NonlinearGradients =
    std::variant<std::monostate, TrainableRbfGradients, RationalGradients, WaveletGradients, PqcGradients>;

} // namespace kan

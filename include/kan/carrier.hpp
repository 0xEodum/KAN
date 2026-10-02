#pragma once

#include "kan/basis.hpp"
#include "kan/rational.hpp"
#include <variant>
#include <vector>

namespace kan {

// Edge carriers. A Layer holds exactly one carrier: the configuration and the
// trainable parameters of all its edges. Every carrier has a per-edge
// coefficient tensor of layout (outputs, inputs, terms), the tensor that the
// coefficient L2 penalty acts on; a carrier may add nonlinear parameters.

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

using Carrier = std::variant<BasisEdges, TrainableRbfEdges, RationalEdges>;

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

using NonlinearGradients = std::variant<std::monostate, TrainableRbfGradients, RationalGradients>;

} // namespace kan

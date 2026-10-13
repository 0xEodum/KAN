#pragma once

// CPU operations of each edge carrier. Layer dispatches on its carrier once
// per call (std::visit) and the hot loops below are compiled per carrier type,
// without per-element dispatch. A new carrier adds an alternative to
// kan::Carrier and kan::NonlinearGradients plus these overloads; Layer,
// Network and the family-independent checks stay unchanged.

#include "kan/carrier.hpp"
#include <cstddef>
#include <span>
#include <variant>

namespace kan::detail {

struct EdgeShape {
    std::size_t inputs, outputs;
};

// Nonlinear gradient type of each carrier (the NonlinearGradients alternative).
template<class Edges> struct nonlinear_of;
template<> struct nonlinear_of<BasisEdges> { using type = std::monostate; };
template<> struct nonlinear_of<TrainableRbfEdges> { using type = TrainableRbfGradients; };
template<> struct nonlinear_of<RationalEdges> { using type = RationalGradients; };
template<class Edges> using nonlinear_t = typename nonlinear_of<Edges>::type;

// Coefficients per edge.
std::size_t terms(const BasisEdges& edges) noexcept;
std::size_t terms(const TrainableRbfEdges& edges) noexcept;
std::size_t terms(const RationalEdges& edges) noexcept;

// Configuration and parameter shapes for the given dimensions
// (std::invalid_argument). Finiteness is checked separately.
void validate(const BasisEdges& edges, EdgeShape shape);
void validate(const TrainableRbfEdges& edges, EdgeShape shape);
void validate(const RationalEdges& edges, EdgeShape shape);

// Finite nonlinear parameters beyond the configuration (std::invalid_argument).
inline void require_finite_nonlinear(const BasisEdges&) {}
inline void require_finite_nonlinear(const TrainableRbfEdges&) {} // part of its configuration
void require_finite_nonlinear(const RationalEdges& edges);

// output[b,o] = bias[o] + sum_i phi_{o,i}(input[b,i]) for validated state and
// finite input. Nonfinite intermediate results throw std::overflow_error.
void forward(const BasisEdges& edges, EdgeShape shape, std::span<const double> bias,
             std::span<const double> input, std::size_t batch, std::span<double> output);
void forward(const TrainableRbfEdges& edges, EdgeShape shape, std::span<const double> bias,
             std::span<const double> input, std::size_t batch, std::span<double> output);
void forward(const RationalEdges& edges, EdgeShape shape, std::span<const double> bias,
             std::span<const double> input, std::size_t batch, std::span<double> output);

// Accumulates the input, coefficient and nonlinear VJPs, summed over the batch,
// into zero-initialized gradients. The bias VJP is carrier independent.
void backward(const BasisEdges& edges, EdgeShape shape, std::span<const double> input,
              std::size_t batch, std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, std::monostate& nonlinear);
void backward(const TrainableRbfEdges& edges, EdgeShape shape, std::span<const double> input,
              std::size_t batch, std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, TrainableRbfGradients& nonlinear);
void backward(const RationalEdges& edges, EdgeShape shape, std::span<const double> input,
              std::size_t batch, std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, RationalGradients& nonlinear);

// Correctly shaped zero nonlinear gradients.
inline std::monostate zero_nonlinear(const BasisEdges&) { return {}; }
TrainableRbfGradients zero_nonlinear(const TrainableRbfEdges& edges);
RationalGradients zero_nonlinear(const RationalEdges& edges);

// Shape and finiteness of caller-supplied nonlinear gradients (std::invalid_argument).
inline void check_gradient(const BasisEdges&, const std::monostate&) {}
void check_gradient(const TrainableRbfEdges& edges, const TrainableRbfGradients& gradient);
void check_gradient(const RationalEdges& edges, const RationalGradients& gradient);

// candidate.nonlinear -= rate * gradient, validating the candidate
// (std::overflow_error). Coefficients are updated by Layer.
inline void update_nonlinear(BasisEdges&, const std::monostate&, double) {}
void update_nonlinear(TrainableRbfEdges& candidate, const TrainableRbfGradients& gradient, double rate);
void update_nonlinear(RationalEdges& candidate, const RationalGradients& gradient, double rate);

// Computed nonlinear gradients are finite (std::overflow_error).
inline void check_result(const std::monostate&) {}
void check_result(const TrainableRbfGradients& gradient);
void check_result(const RationalGradients& gradient);

} // namespace kan::detail

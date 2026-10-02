#pragma once

#include "kan/layer.hpp"

namespace kan {

// Family-specific operations on a Layer's carrier. Each requires one carrier
// (and basis) type, raises std::invalid_argument for any other, and replaces
// the carrier atomically through Layer::set_carrier.

// B-spline grid refinement (BasisEdges holding a BSplineConfig). Boehm
// insertion of a strictly interior knot transforms every edge's coefficients.
void insert_knot(Layer& layer, double x);
// Inserts one knot in the most populated span and returns it.
double adapt_grid(Layer& layer, std::span<const double> samples);

// Shared trainable RBF centers and log widths (TrainableRbfEdges), of the
// layer's current term count.
void set_rbf_parameters(Layer& layer, std::span<const double> centers,
                        std::span<const double> log_widths);

// Rational numerator coefficients, denominators and bias (RationalEdges).
void set_rational_parameters(Layer& layer, std::span<const double> numerator,
                             std::span<const double> denominators, std::span<const double> bias);

} // namespace kan

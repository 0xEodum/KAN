#pragma once

#include "kan/layer.hpp"

namespace kan {

// Family-specific operations on a Layer's carrier. Each requires one carrier
// (and basis) type, raises std::invalid_argument for any other, and replaces
// the carrier atomically through Layer::set_carrier.

// B-spline grid refinement (BasisEdges holding a BSplineConfig). Boehm
// insertion of a strictly interior knot transforms every edge's coefficients.
// Both single-knot forms require a layer with one shared grid.
void insert_knot(Layer& layer, double x);
// Inserts one knot in the most populated span and returns it.
double adapt_grid(Layer& layer, std::span<const double> samples);

// Per-input grids (backlog M6). insert_knots inserts knots[g] into grid g, one
// knot per grid (each grid keeps the common term count), by Boehm insertion.
void insert_knots(Layer& layer, std::span<const double> knots);
// adapt_grid over layer inputs (batch x inputs): every grid takes the median
// of its samples in its most populated span (adapt_grid's rule; a shared grid
// pools all values of the batch, grid i of a per-input layer column i) and
// inserts it; returns the inserted knots, one per grid.
std::vector<double> adapt_grid(Layer& layer, std::span<const double> input, std::size_t batch);

// Quantile grid refit (backlog M6, pykan's update_grid_from_samples). Each new
// grid has `intervals` knot intervals on [min, max] of its samples: interior
// knot j = grid_eps*(uniform knot j) + (1 - grid_eps)*(the sample of rank
// floor(n*j/intervals) of the n sorted samples), clamped endpoints repeated
// degree+1 times, strictly increasing interior knots. The coefficients are
// refit by least squares to the current edge functions at the samples of
// their input, sum (s(x) - f(x))^2 + ridge*sum c^2 (banded normal equations).
struct GridRefit {
    std::size_t intervals = 0; // knot intervals G (terms = G + degree); 0 keeps the layer's
    double grid_eps = 0.02;    // in [0, 1]: 1 uniform knots, 0 quantile knots (pykan default 0.02)
    double ridge = 1e-8;       // finite, >= 0 (pykan's regularized least squares)
    bool per_input = true;     // one grid per input (pykan); false: one shared grid from all samples
    bool operator==(const GridRefit&) const = default;
};
// Refits the B-spline layer's grids from layer inputs (batch x inputs) and
// replaces its carrier atomically; bias and residual branch are unchanged.
void refit_grid(Layer& layer, std::span<const double> input, std::size_t batch, const GridRefit& options = {});

// Shared trainable RBF centers and log widths (TrainableRbfEdges), of the
// layer's current term count.
void set_rbf_parameters(Layer& layer, std::span<const double> centers,
                        std::span<const double> log_widths);

// Rational numerator coefficients, denominators and bias (RationalEdges).
void set_rational_parameters(Layer& layer, std::span<const double> numerator,
                             std::span<const double> denominators, std::span<const double> bias);

// Per-edge wavelet coefficients, translations, log scales and bias
// (WaveletEdges, backlog M7); log scales must satisfy valid_log_scale.
void set_wavelet_parameters(Layer& layer, std::span<const double> coefficients,
                            std::span<const double> translations, std::span<const double> log_scales,
                            std::span<const double> bias);

} // namespace kan

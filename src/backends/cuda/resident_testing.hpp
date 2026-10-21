#pragma once

// Test and benchmark hooks of the resident executor (backlogs C5, C12, M7). Private: not
// installed, not part of the public API; tests and evidence harnesses include it
// relative to src/. Nothing here changes results unless a scope forces a path
// (forcing a C12 reduction on an FP32 layer of at most 85 terms replaces the
// staged kernel and its summation order).

#include <cstddef>
#include <vector>

namespace kan::cuda::testing {

// Execution path of a B-spline BasisEdges layer. Automatic is the executor's
// selection rule (docs/CONTRACT.md, "Sparse B-spline path"); Dense and Sparse
// force a path for every B-spline layer the sparse kernels support (Sparse
// still keeps the dense rows for a layer they cannot run: unsupported degree,
// shared-memory or index limits).
enum class SplinePath { Automatic, Dense, Sparse };

// Center/log-width reduction of a TrainableRbfEdges layer (backlog C12).
// Automatic is the executor's choice (docs/CONTRACT.md); Column and Tile force
// nonlinear_partial_kernel or nonlinear_tile_kernel for every such layer in
// every precision (also where FP32 would take the staged kernel). Both give
// bitwise the same results.
enum class NonlinearReduction { Automatic, Column, Tile };

// Launch geometry of every WaveletEdges layer (backlog M7): samples per thread
// of the forward and backward kernels (1, 2, 4 or 8; FP64 forward 1 and
// backward at most 4), the backward partial slots and the output splits. Zero
// keeps the executor's choice (docs/CONTRACT.md); a forced value is used as
// given (slots and splits bound the workspace, so they are maxima as usual).
// Results change only in summation order.
struct WaveletLaunch {
    unsigned forward_samples = 0, backward_samples = 0, slots = 0, splits = 0;
};

// What an executor reserved at construction: per network layer whether it runs
// the sparse B-spline path, and the arena size in bytes.
struct ResidentLayout {
    std::vector<bool> sparse;
    std::size_t arena_bytes = 0;
    // Per network layer: terms per nonlinear_tile_kernel block of a trainable-RBF
    // layer (backlog C12), 0 when the tile kernel is unavailable or for other layers.
    std::vector<unsigned> reduction_groups;
};

// While alive, every ResidentNetwork constructed on this thread uses `path`
// for its B-spline layers and, if `layout` is not null, writes its layout
// there; the second form instead forces the trainable-RBF reduction (B-spline
// layers automatic), the third the wavelet launch geometry, with the same
// optional layout. Scopes do not nest.
class ResidentTestScope {
public:
    explicit ResidentTestScope(SplinePath path, ResidentLayout* layout = nullptr);
    explicit ResidentTestScope(NonlinearReduction reduction, ResidentLayout* layout = nullptr);
    explicit ResidentTestScope(WaveletLaunch launch, ResidentLayout* layout = nullptr);
    ~ResidentTestScope();
    ResidentTestScope(const ResidentTestScope&) = delete;
    ResidentTestScope& operator=(const ResidentTestScope&) = delete;
};

} // namespace kan::cuda::testing

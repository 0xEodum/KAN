#pragma once

// Test and benchmark hooks of the resident executor (backlog C5). Private: not
// installed, not part of the public API; tests and evidence harnesses include it
// relative to src/. Nothing here changes results unless a scope forces a path.

#include <cstddef>
#include <vector>

namespace kan::cuda::testing {

// Execution path of a B-spline BasisEdges layer. Automatic is the executor's
// selection rule (docs/CONTRACT.md, "Sparse B-spline path"); Dense and Sparse
// force a path for every B-spline layer the sparse kernels support (Sparse
// still keeps the dense rows for a layer they cannot run: unsupported degree,
// shared-memory or index limits).
enum class SplinePath { Automatic, Dense, Sparse };

// What an executor reserved at construction: per network layer whether it runs
// the sparse B-spline path, and the arena size in bytes.
struct ResidentLayout {
    std::vector<bool> sparse;
    std::size_t arena_bytes = 0;
};

// While alive, every ResidentNetwork constructed on this thread uses `path`
// for its B-spline layers and, if `layout` is not null, writes its layout
// there. Scopes do not nest.
class ResidentTestScope {
public:
    explicit ResidentTestScope(SplinePath path, ResidentLayout* layout = nullptr);
    ~ResidentTestScope();
    ResidentTestScope(const ResidentTestScope&) = delete;
    ResidentTestScope& operator=(const ResidentTestScope&) = delete;
};

} // namespace kan::cuda::testing

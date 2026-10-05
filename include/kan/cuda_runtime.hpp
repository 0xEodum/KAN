#pragma once

// CUDA device query shared by every kan::cuda API (backlog R8). Both the
// resident executor (kan/resident.hpp) and the deprecated legacy layer API
// (kan/cuda.hpp) include this header. It has no CUDA dependency; the
// definition lives in the optional kan::cuda library.

namespace kan::cuda {

// Whether a CUDA device is usable. Not deprecated. Without a device, CUDA
// operations fail explicitly with std::runtime_error.
bool available() noexcept;

} // namespace kan::cuda

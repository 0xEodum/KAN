// CUDA device query (backlog R8), separate from the deprecated legacy adapter
// in src/cuda.cu.
#include "kan/cuda_runtime.hpp"
#include <cuda_runtime.h>

namespace kan::cuda {

bool available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace kan::cuda

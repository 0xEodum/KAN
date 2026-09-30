#include "kan/cuda.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace kan::cuda {
bool available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
std::vector<double> forward(const Layer&, std::span<const double>, std::size_t) {
    throw std::runtime_error("CUDA forward is not implemented");
}
LayerGradients backward(const Layer&, std::span<const double>, std::size_t, std::span<const double>) {
    throw std::runtime_error("CUDA backward is not implemented");
}
}

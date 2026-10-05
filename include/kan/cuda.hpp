#pragma once

#include "kan/cuda_runtime.hpp" // kan::cuda::available(), not deprecated
#include "kan/layer.hpp"

namespace kan::cuda {

// Legacy M1 synchronous single-layer API (deprecated by backlog R7). Each call
// builds a kan::cuda::ResidentNetwork for the layer with capacity `batch`,
// runs it once and releases it, so it accepts every carrier the resident
// executor supports and pays its construction (allocation, upload, cuBLAS
// handle) per call. Results match the CPU within the resident tolerance.
[[deprecated("kan::cuda::forward/backward are deprecated: build a kan::cuda::ResidentNetwork "
             "(kan/resident.hpp) once and reuse it")]]
std::vector<double> forward(const Layer& layer, std::span<const double> input, std::size_t batch);
[[deprecated("kan::cuda::forward/backward are deprecated: build a kan::cuda::ResidentNetwork "
             "(kan/resident.hpp) once and reuse it")]]
LayerGradients backward(const Layer& layer, std::span<const double> input, std::size_t batch,
                        std::span<const double> output_gradient);

} // namespace kan::cuda

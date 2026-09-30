#pragma once

#include "kan/layer.hpp"

namespace kan::cuda {

// Synchronous host API; own device buffers per call. Only Chebyshev in M1.
bool available() noexcept;
std::vector<double> forward(const Layer& layer, std::span<const double> input, std::size_t batch);
LayerGradients backward(const Layer& layer, std::span<const double> input, std::size_t batch,
                        std::span<const double> output_gradient);

} // namespace kan::cuda

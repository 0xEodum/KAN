// Legacy M1 synchronous API as a thin adapter over the resident executor
// (backlog R7). It owns no kernels: each call builds a one-layer
// ResidentNetwork with capacity `batch`, runs it once and releases it.
#include "kan/cuda.hpp"
#include "kan/resident.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kan::cuda {
namespace {
std::size_t checked_size(std::size_t left, std::size_t right) {
    const auto max = std::vector<double>().max_size();
    if (right != 0 && left > max / right) throw std::overflow_error("CUDA array size overflow");
    return left * right;
}
void require_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("CUDA data must be finite");
}
// Host-side checks precede any device allocation, so invalid calls never
// reach the executor and keep the M1 exception types.
ResidentNetwork prepare(const Layer& layer, std::span<const double> input, std::size_t batch,
                        std::span<const double> output_gradient, bool has_gradient) {
    if (!available()) throw std::runtime_error("no CUDA device available");
    if (input.size() != checked_size(batch, layer.inputs()))
        throw std::invalid_argument("CUDA input shape mismatch");
    if (has_gradient && output_gradient.size() != checked_size(batch, layer.outputs()))
        throw std::invalid_argument("CUDA backward shape mismatch");
    require_finite(input);
    require_finite(output_gradient);
    ResidentNetwork gpu(Network(std::vector<NetworkLayer>{NetworkLayer(layer)}), batch);
    gpu.upload_input(input, batch);
    return gpu;
}
}

std::vector<double> forward(const Layer& layer, std::span<const double> input, std::size_t batch) {
    auto gpu = prepare(layer, input, batch, {}, false);
    gpu.forward();
    return gpu.download_output();
}

LayerGradients backward(const Layer& layer, std::span<const double> input, std::size_t batch,
                        std::span<const double> output_gradient) {
    auto gpu = prepare(layer, input, batch, output_gradient, true);
    gpu.upload_output_gradient(output_gradient);
    gpu.forward();
    gpu.backward();
    auto gradients = gpu.download_gradients();
    return std::get<LayerGradients>(std::move(gradients.layers.front()));
}
} // namespace kan::cuda

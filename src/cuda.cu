#include "kan/cuda.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kan::cuda {
namespace {
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::size_t checked_size(std::size_t left, std::size_t right) {
    const auto max = std::vector<double>().max_size();
    if (right != 0 && left > max / right) throw std::overflow_error("CUDA array size overflow");
    return left * right;
}
void require_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("CUDA data must be finite");
}
void result_finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::overflow_error("nonfinite CUDA numerical result");
}
void validate_layer(const Layer& layer) {
    if (!std::holds_alternative<ChebyshevConfig>(layer.basis()))
        throw std::invalid_argument("CUDA M1 supports Chebyshev only");
    validate_basis(layer.basis());
    if (layer.inputs() == 0 || layer.outputs() == 0 ||
        layer.coefficients().size() != checked_size(checked_size(layer.inputs(), layer.outputs()), basis_size(layer.basis())) ||
        layer.bias().size() != layer.outputs())
        throw std::invalid_argument("invalid CUDA layer shape");
    require_finite(layer.coefficients());
    require_finite(layer.bias());
}
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(checked_size(count, 1)) {
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(double))
            throw std::overflow_error("CUDA buffer byte size overflow");
        if (count_) check(cudaMalloc(&data_, count_ * sizeof(double)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    double* data() const { return data_; }
    void upload(std::span<const double> values, cudaStream_t stream) {
        if (count_) check(cudaMemcpyAsync(data_, values.data(), count_ * sizeof(double),
                                         cudaMemcpyHostToDevice, stream), "CUDA upload");
    }
    void download(std::span<double> values, cudaStream_t stream) {
        if (count_) check(cudaMemcpyAsync(values.data(), data_, count_ * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream), "CUDA download");
    }
private:
    double* data_ = nullptr;
    std::size_t count_;
};
class Stream {
public:
    Stream() { check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() {
        // Declared after buffers: queued operations finish before buffers are freed,
        // including when a later copy/launch throws.
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }
    void synchronize() const { check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }
private:
    cudaStream_t stream_ = nullptr;
};

// Recurrence derivative avoids divisions at x = +/-1. Forward checks both
// values and derivatives to match the CPU evaluator's overflow contract.
struct Chebyshev {
    double x, previous = 1.0, current, previous_derivative = 0.0, current_derivative = 1.0;
    __device__ explicit Chebyshev(double argument) : x(argument), current(argument) {}
    __device__ bool next(std::size_t k, double& value, double& derivative) {
        if (k == 0) { value = 1.0; derivative = 0.0; }
        else if (k == 1) { value = x; derivative = 1.0; }
        else {
            value = 2.0 * x * current - previous;
            derivative = 2.0 * current + 2.0 * x * current_derivative - previous_derivative;
            previous = current; current = value;
            previous_derivative = current_derivative; current_derivative = derivative;
        }
        return isfinite(value) && isfinite(derivative);
    }
};

__global__ void forward_kernel(const double* input, const double* coefficients, const double* bias,
                               double* output, std::size_t count, std::size_t inputs,
                               std::size_t outputs, std::size_t terms) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const auto b = index / outputs, o = index % outputs;
        double sum = bias[o];
        bool valid = true;
        for (std::size_t i = 0; i < inputs && valid; ++i) {
            Chebyshev basis(input[b * inputs + i]);
            for (std::size_t k = 0; k < terms; ++k) {
                double value, derivative;
                if (!basis.next(k, value, derivative)) { valid = false; break; }
                sum += coefficients[(o * inputs + i) * terms + k] * value;
            }
        }
        output[index] = valid ? sum : CUDART_NAN;
    }
}

__global__ void input_gradient_kernel(const double* input, const double* coefficients,
                                      const double* upstream, double* gradient, std::size_t count,
                                      std::size_t inputs, std::size_t outputs, std::size_t terms) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const auto b = index / inputs, i = index % inputs;
        double sum = 0.0;
        bool valid = true;
        for (std::size_t o = 0; o < outputs && valid; ++o) {
            Chebyshev basis(input[index]);
            for (std::size_t k = 0; k < terms; ++k) {
                double value, derivative;
                if (!basis.next(k, value, derivative)) { valid = false; break; }
                sum += upstream[b * outputs + o] * coefficients[(o * inputs + i) * terms + k] * derivative;
            }
        }
        gradient[index] = valid ? sum : CUDART_NAN;
    }
}

__global__ void coefficient_gradient_kernel(const double* input, const double* upstream,
                                            double* gradient, std::size_t count, std::size_t batch,
                                            std::size_t inputs, std::size_t outputs, std::size_t terms) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const auto k = index % terms, edge = index / terms, i = edge % inputs, o = edge / inputs;
        double sum = 0.0;
        bool valid = true;
        for (std::size_t b = 0; b < batch && valid; ++b) {
            Chebyshev basis(input[b * inputs + i]);
            double value = 0.0, derivative = 0.0;
            for (std::size_t term = 0; term <= k; ++term)
                if (!basis.next(term, value, derivative)) { valid = false; break; }
            if (valid) sum += upstream[b * outputs + o] * value;
        }
        gradient[index] = valid ? sum : CUDART_NAN;
    }
}

__global__ void bias_gradient_kernel(const double* upstream, double* gradient,
                                     std::size_t batch, std::size_t outputs) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto o = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         o < outputs; o += stride) {
        double sum = 0.0;
        for (std::size_t b = 0; b < batch; ++b) sum += upstream[b * outputs + o];
        gradient[o] = sum;
    }
}
unsigned blocks(std::size_t count) {
    return static_cast<unsigned>(std::min<std::size_t>((count - 1) / 256 + 1, 65535));
}
}

bool available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
std::vector<double> forward(const Layer& layer, std::span<const double> input, std::size_t batch) {
    if (!available()) throw std::runtime_error("no CUDA device available");
    validate_layer(layer);
    const auto input_size = checked_size(batch, layer.inputs());
    const auto output_size = checked_size(batch, layer.outputs());
    if (input.size() != input_size) throw std::invalid_argument("CUDA input shape mismatch");
    require_finite(input);
    std::vector<double> result(output_size);
    if (batch == 0) return result;
    DeviceBuffer x(input_size), coefficients(layer.coefficients().size()), bias(layer.bias().size()), y(output_size);
    Stream stream;
    x.upload(input, stream.get());
    coefficients.upload(layer.coefficients(), stream.get());
    bias.upload(layer.bias(), stream.get());
    forward_kernel<<<blocks(output_size), 256, 0, stream.get()>>>(x.data(), coefficients.data(), bias.data(),
        y.data(), output_size, layer.inputs(), layer.outputs(), basis_size(layer.basis()));
    check(cudaGetLastError(), "forward kernel launch");
    y.download(result, stream.get());
    stream.synchronize();
    result_finite(result);
    return result;
}
LayerGradients backward(const Layer& layer, std::span<const double> input, std::size_t batch,
                        std::span<const double> output_gradient) {
    if (!available()) throw std::runtime_error("no CUDA device available");
    validate_layer(layer);
    const auto input_size = checked_size(batch, layer.inputs());
    const auto output_size = checked_size(batch, layer.outputs());
    if (input.size() != input_size || output_gradient.size() != output_size)
        throw std::invalid_argument("CUDA backward shape mismatch");
    require_finite(input);
    require_finite(output_gradient);
    LayerGradients result{std::vector<double>(input_size, 0.0),
                          std::vector<double>(layer.coefficients().size(), 0.0),
                          std::vector<double>(layer.outputs(), 0.0)};
    if (batch == 0) return result;
    DeviceBuffer x(input_size), coefficients(layer.coefficients().size()), upstream(output_size),
                 dx(input_size), dc(result.coefficients.size()), db(result.bias.size());
    Stream stream;
    x.upload(input, stream.get());
    coefficients.upload(layer.coefficients(), stream.get());
    upstream.upload(output_gradient, stream.get());
    input_gradient_kernel<<<blocks(input_size), 256, 0, stream.get()>>>(x.data(), coefficients.data(),
        upstream.data(), dx.data(), input_size, layer.inputs(), layer.outputs(), basis_size(layer.basis()));
    check(cudaGetLastError(), "input gradient kernel launch");
    coefficient_gradient_kernel<<<blocks(result.coefficients.size()), 256, 0, stream.get()>>>(x.data(),
        upstream.data(), dc.data(), result.coefficients.size(), batch, layer.inputs(), layer.outputs(), basis_size(layer.basis()));
    check(cudaGetLastError(), "coefficient gradient kernel launch");
    bias_gradient_kernel<<<blocks(layer.outputs()), 256, 0, stream.get()>>>(upstream.data(), db.data(), batch, layer.outputs());
    check(cudaGetLastError(), "bias gradient kernel launch");
    dx.download(result.input, stream.get());
    dc.download(result.coefficients, stream.get());
    db.download(result.bias, stream.get());
    stream.synchronize();
    result_finite(result.input);
    result_finite(result.coefficients);
    result_finite(result.bias);
    return result;
}
}

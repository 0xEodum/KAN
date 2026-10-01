#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kan::cuda {
namespace {
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::size_t product(std::size_t a, std::size_t b) {
    const auto maximum = std::vector<double>().max_size();
    if (b && a > maximum / b) throw std::overflow_error("resident array size overflow");
    return a * b;
}
void finite(std::span<const double> values) {
    for (double x : values) if (!std::isfinite(x)) throw std::invalid_argument("resident data must be finite");
}
unsigned blocks(std::size_t count) { return static_cast<unsigned>(std::min<std::size_t>((count - 1) / 256 + 1, 65535)); }
struct Basis {
    BasisKind kind;
    std::size_t terms;
    double alpha, beta, frequency, width;
    const double* centers;
};
__device__ void report(double value, int* status) { if (!isfinite(value)) atomicExch(status, 1); }
__global__ void basis_kernel(const double* input, double* values, double* derivatives,
                             std::size_t count, Basis basis, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const double x = input[index];
        auto* v = values + index * basis.terms;
        auto* d = derivatives + index * basis.terms;
        if (basis.kind == BasisKind::GaussianRbf) {
            for (std::size_t k = 0; k < basis.terms; ++k) {
                const double distance = x - basis.centers[k];
                const double q = isfinite(distance) ? distance / basis.width : x / basis.width - basis.centers[k] / basis.width;
                v[k] = exp(-q*q);
                d[k] = 0;
                if (q != 0 && isfinite(q)) {
                    if (v[k] < 2.2250738585072014e-308) {
                        const double log_magnitude = log(2.0) + log(fabs(q)) - q*q - log(basis.width);
                        d[k] = -copysign(exp(log_magnitude), q);
                    } else d[k] = (-2*q*v[k]) / basis.width;
                }
                report(d[k], status);
            }
            continue;
        }
        v[0] = 1; d[0] = 0;
        if (basis.kind == BasisKind::Fourier) {
            for (std::size_t k = 1; k <= basis.terms / 2; ++k) {
                const double angular = static_cast<double>(k) * basis.frequency;
                const double phase = angular*x;
                report(angular, status); report(phase, status);
                v[2*k-1] = cos(phase); v[2*k] = sin(phase);
                d[2*k-1] = -angular*v[2*k]; d[2*k] = angular*v[2*k-1];
                report(d[2*k-1], status); report(d[2*k], status);
            }
            continue;
        }
        if (basis.terms == 1) continue;
        const double half_sum = 0.5*basis.alpha + 0.5*basis.beta;
        const double shifted_half_sum = 0.5*(basis.alpha+1) + 0.5*(basis.beta+1);
        const double half_difference = 0.5*basis.alpha - 0.5*basis.beta;
        if (basis.kind == BasisKind::Jacobi && (x == -1 || x == 1)) {
            const double parameter = x == 1 ? basis.alpha : basis.beta;
            double endpoint = 1, shifted = 1;
            for (std::size_t k = 1; k < basis.terms; ++k) {
                const double n = static_cast<double>(k);
                endpoint *= (parameter+n)/n;
                v[k] = x < 0 && k % 2 ? -endpoint : endpoint;
                if (k > 1) shifted *= (parameter+n)/(n-1);
                const double derivative = (shifted_half_sum+0.5*(n-1))*shifted;
                d[k] = x < 0 && k % 2 == 0 ? -derivative : derivative;
                report(v[k], status); report(shifted, status); report(d[k], status);
            }
            continue;
        }
        double slope = basis.kind == BasisKind::Hermite ? 2 : 1;
        double offset = 0;
        if (basis.kind == BasisKind::Jacobi) { slope = shifted_half_sum; offset = half_difference; }
        v[1] = slope*x + offset; d[1] = slope;
        report(v[1], status); report(d[1], status);
        for (std::size_t k = 1; k < basis.terms - 1; ++k) {
            const double n = static_cast<double>(k);
            double a = 2, b = 0, c = 1;
            if (basis.kind == BasisKind::Legendre) { a = (2*n+1)/(n+1); c = n/(n+1); }
            if (basis.kind == BasisKind::Hermite) c = 2*n;
            if (basis.kind == BasisKind::Jacobi) {
                const double t = shifted_half_sum+(n-1), denominator = shifted_half_sum+0.5*(n-1);
                a = ((t+0.5)/(n+1))*((t+1)/denominator);
                b = (half_difference/(n+1))*(half_sum/t)*((t+0.5)/denominator);
                c = 0.5*((n+basis.alpha)/(n+1))*((n+basis.beta)/t)*((t+1)/denominator);
            }
            const double factor = a*x+b;
            v[k+1] = factor*v[k] - c*v[k-1];
            d[k+1] = a*v[k] + factor*d[k] - c*d[k-1];
            report(v[k+1], status); report(d[k+1], status);
        }
    }
}
__global__ void forward_kernel(const double* v, const double* c, const double* bias, double* output,
                               std::size_t count, std::size_t inputs, std::size_t outputs, std::size_t terms, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const auto batch = index/outputs, o = index%outputs;
        double sum = bias[o];
        for (std::size_t i = 0; i < inputs; ++i)
            for (std::size_t k = 0; k < terms; ++k) sum += c[(o*inputs+i)*terms+k]*v[(batch*inputs+i)*terms+k];
        output[index] = sum; report(sum, status);
    }
}
__global__ void input_kernel(const double* d, const double* c, const double* upstream, double* dx,
                             std::size_t count, std::size_t inputs, std::size_t outputs, std::size_t terms, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        const auto batch = index/inputs, i = index%inputs;
        double sum = 0;
        for (std::size_t o = 0; o < outputs; ++o)
            for (std::size_t k = 0; k < terms; ++k) sum += upstream[batch*outputs+o]*c[(o*inputs+i)*terms+k]*d[index*terms+k];
        dx[index] = sum; report(sum, status);
    }
}
__global__ void parameter_kernel(const double* v, const double* upstream, double* gradient,
                                 std::size_t batch, std::size_t inputs, std::size_t outputs, std::size_t terms, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    const auto coefficients = inputs*outputs*terms;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < coefficients+outputs; index += stride) {
        double sum = 0;
        if (index < coefficients) {
            const auto k = index%terms, i = (index/terms)%inputs, o = index/(terms*inputs);
            for (std::size_t b = 0; b < batch; ++b) sum += upstream[b*outputs+o]*v[(b*inputs+i)*terms+k];
        } else {
            const auto o = index-coefficients;
            for (std::size_t b = 0; b < batch; ++b) sum += upstream[b*outputs+o];
        }
        gradient[index] = sum; report(sum, status);
    }
}
__global__ void candidate_kernel(const double* parameters, const double* gradients, double* next,
                                 std::size_t count, double rate, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += stride) {
        next[i] = parameters[i] - rate*gradients[i]; report(next[i], status);
    }
}
struct Layout {
    std::size_t inputs, outputs, terms, coefficients, parameter_offset;
    std::size_t values, derivatives, centers;
};
}

struct ResidentNetwork::Impl {
    Network model;
    std::size_t capacity, batch = 0, allocations = 0;
    std::size_t parameter_count = 0, parameters = 0, gradients = 0, candidates = 0;
    std::vector<Layout> layers;
    std::vector<std::size_t> activation, upstream;
    double* arena = nullptr;
    int* status = nullptr;
    cudaStream_t stream = nullptr;
    bool has_input = false, has_upstream = false, has_forward = false, has_backward = false;
    explicit Impl(const Network& source, std::size_t maximum) : model(source), capacity(maximum) {
        if (model.layers().empty()) throw std::invalid_argument("resident network is empty or moved from");
        // Validate the copied CPU state and all shape arithmetic before CUDA allocation.
        model.forward({}, 0);
        std::size_t total = 0;
        auto reserve = [&](std::size_t count) {
            const auto limit = std::vector<double>().max_size();
            if (count > limit - total) throw std::overflow_error("resident workspace size overflow");
            const auto offset = total; total += count; return offset;
        };
        for (const auto& layer : model.layers()) {
            finite(layer.coefficients()); finite(layer.bias());
            if (layer.coefficients().size() > std::vector<double>().max_size() - parameter_count - layer.outputs())
                throw std::overflow_error("resident parameter size overflow");
            layers.push_back({layer.inputs(), layer.outputs(), layer.basis().size, layer.coefficients().size(), parameter_count, 0, 0, 0});
            parameter_count += layer.coefficients().size()+layer.outputs();
        }
        parameters = reserve(parameter_count); gradients = reserve(parameter_count); candidates = reserve(parameter_count);
        activation.push_back(reserve(product(capacity, layers.front().inputs)));
        upstream.push_back(reserve(product(capacity, layers.front().inputs)));
        for (std::size_t j = 0; j < layers.size(); ++j) {
            auto& layout = layers[j];
            activation.push_back(reserve(product(capacity, layout.outputs)));
            upstream.push_back(reserve(product(capacity, layout.outputs)));
            layout.values = reserve(product(product(capacity, layout.inputs), layout.terms));
            layout.derivatives = reserve(product(product(capacity, layout.inputs), layout.terms));
            if (model.layers()[j].basis().kind == BasisKind::GaussianRbf) layout.centers = reserve(layout.terms);
        }
        const auto bytes = product(total, sizeof(double));
        try {
            if (!available()) throw std::runtime_error("no CUDA device available");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "resident stream create");
            check(cudaMalloc(&arena, bytes), "resident arena allocation"); ++allocations;
            check(cudaMalloc(&status, sizeof(int)), "resident status allocation"); ++allocations;
            for (std::size_t j = 0; j < layers.size(); ++j) {
                const auto& layout = layers[j]; const auto& layer = model.layers()[j];
                upload(ptr(parameters+layout.parameter_offset), layer.coefficients());
                upload(ptr(parameters+layout.parameter_offset+layout.coefficients), layer.bias());
                if (layer.basis().kind == BasisKind::GaussianRbf) upload(ptr(layout.centers), layer.basis().centers);
            }
            sync();
        } catch (...) { cleanup(); throw; }
    }
    ~Impl() { cleanup(); }
    void cleanup() noexcept {
        if (stream) cudaStreamSynchronize(stream);
        if (arena) cudaFree(arena);
        if (status) cudaFree(status);
        if (stream) cudaStreamDestroy(stream);
        arena = nullptr; status = nullptr; stream = nullptr;
    }
    double* ptr(std::size_t offset) const { return arena+offset; }
    void sync() { check(cudaStreamSynchronize(stream), "resident synchronize"); }
    void upload(double* destination, std::span<const double> data) {
        if (!data.empty()) check(cudaMemcpyAsync(destination, data.data(), data.size_bytes(), cudaMemcpyHostToDevice, stream), "resident upload");
    }
    void download(std::span<double> destination, const double* data) {
        if (!destination.empty()) check(cudaMemcpyAsync(destination.data(), data, destination.size_bytes(), cudaMemcpyDeviceToHost, stream), "resident download");
    }
    void reset_status() { check(cudaMemsetAsync(status, 0, sizeof(int), stream), "resident status reset"); }
    void result() {
        check(cudaGetLastError(), "resident kernel launch");
        int value = 0;
        check(cudaMemcpyAsync(&value, status, sizeof(int), cudaMemcpyDeviceToHost, stream), "resident status download");
        sync();
        if (value) throw std::overflow_error("nonfinite resident numerical result");
    }
};
ResidentNetwork::ResidentNetwork(const Network& network, std::size_t capacity) : impl_(std::make_unique<Impl>(network, capacity)) {}
ResidentNetwork::~ResidentNetwork() = default;
ResidentNetwork::ResidentNetwork(ResidentNetwork&&) noexcept = default;
ResidentNetwork& ResidentNetwork::operator=(ResidentNetwork&&) noexcept = default;
ResidentNetwork::Impl& ResidentNetwork::state() const {
    if (!impl_) throw std::logic_error("resident network is moved from");
    return *impl_;
}
void ResidentNetwork::upload_input(std::span<const double> input, std::size_t batch) {
    auto& s = state(); const auto count = product(batch, s.layers.front().inputs);
    if (batch > s.capacity || input.size() != count) throw std::invalid_argument("resident input shape or capacity mismatch");
    finite(input);
    s.upload(s.ptr(s.activation.front()), input); s.sync();
    s.batch = batch; s.has_input = true; s.has_upstream = s.has_forward = s.has_backward = false;
}
void ResidentNetwork::upload_output_gradient(std::span<const double> gradient) {
    auto& s = state();
    if (!s.has_input) throw std::logic_error("resident input must be uploaded first");
    if (gradient.size() != product(s.batch, s.layers.back().outputs)) throw std::invalid_argument("resident upstream shape mismatch");
    finite(gradient); s.upload(s.ptr(s.upstream.back()), gradient); s.sync();
    s.has_upstream = true; s.has_backward = false;
}
void ResidentNetwork::forward() {
    auto& s = state();
    if (!s.has_input) throw std::logic_error("resident input has not been uploaded");
    s.has_forward = s.has_backward = false; s.reset_status();
    for (std::size_t j = 0; j < s.layers.size() && s.batch; ++j) {
        const auto& l = s.layers[j]; const auto& b = s.model.layers()[j].basis();
        Basis basis{b.kind, b.size, b.alpha, b.beta, b.frequency, b.width, s.ptr(l.centers)};
        basis_kernel<<<blocks(s.batch*l.inputs), 256, 0, s.stream>>>(s.ptr(s.activation[j]), s.ptr(l.values), s.ptr(l.derivatives), s.batch*l.inputs, basis, s.status);
        check(cudaGetLastError(), "resident basis launch");
        forward_kernel<<<blocks(s.batch*l.outputs), 256, 0, s.stream>>>(s.ptr(l.values), s.ptr(s.parameters+l.parameter_offset), s.ptr(s.parameters+l.parameter_offset+l.coefficients), s.ptr(s.activation[j+1]), s.batch*l.outputs, l.inputs, l.outputs, l.terms, s.status);
        check(cudaGetLastError(), "resident forward launch");
    }
    s.result(); s.has_forward = true;
}
void ResidentNetwork::backward() {
    auto& s = state();
    if (!s.has_forward || !s.has_upstream) throw std::logic_error("resident backward requires current forward and upstream");
    s.has_backward = false; s.reset_status();
    for (std::size_t j = s.layers.size(); j-- > 0;) {
        const auto& l = s.layers[j];
        if (s.batch) {
            input_kernel<<<blocks(s.batch*l.inputs), 256, 0, s.stream>>>(s.ptr(l.derivatives), s.ptr(s.parameters+l.parameter_offset), s.ptr(s.upstream[j+1]), s.ptr(s.upstream[j]), s.batch*l.inputs, l.inputs, l.outputs, l.terms, s.status);
            check(cudaGetLastError(), "resident input gradient launch");
        }
        parameter_kernel<<<blocks(l.coefficients+l.outputs), 256, 0, s.stream>>>(s.ptr(l.values), s.ptr(s.upstream[j+1]), s.ptr(s.gradients+l.parameter_offset), s.batch, l.inputs, l.outputs, l.terms, s.status);
        check(cudaGetLastError(), "resident parameter gradient launch");
    }
    s.result(); s.has_backward = true;
}
void ResidentNetwork::sgd(double learning_rate) {
    auto& s = state();
    if (!std::isfinite(learning_rate) || learning_rate <= 0) throw std::invalid_argument("learning rate must be finite and positive");
    if (!s.has_backward) throw std::logic_error("resident SGD requires current gradients");
    s.reset_status();
    candidate_kernel<<<blocks(s.parameter_count), 256, 0, s.stream>>>(s.ptr(s.parameters), s.ptr(s.gradients), s.ptr(s.candidates), s.parameter_count, learning_rate, s.status);
    s.result(); // All layers validated before any parameter mutation.
    // Both regions are permanently reserved and candidate execution is complete.
    // Changing the active region commits the whole network without a tensor copy.
    std::swap(s.parameters, s.candidates);
    s.has_forward = s.has_backward = false;
}
std::vector<double> ResidentNetwork::download_output() {
    auto& s = state();
    if (!s.has_forward) throw std::logic_error("resident output requires current forward");
    std::vector<double> result(product(s.batch, s.layers.back().outputs));
    s.download(result, s.ptr(s.activation.back())); s.sync(); return result;
}
NetworkGradients ResidentNetwork::download_gradients() {
    auto& s = state();
    if (!s.has_backward) throw std::logic_error("resident gradients require current backward");
    NetworkGradients result; result.layers.resize(s.layers.size());
    for (std::size_t j = 0; j < s.layers.size(); ++j) {
        const auto& l = s.layers[j]; auto& g = result.layers[j];
        g.input.resize(product(s.batch, l.inputs)); g.coefficients.resize(l.coefficients); g.bias.resize(l.outputs);
        s.download(g.input, s.ptr(s.upstream[j])); s.download(g.coefficients, s.ptr(s.gradients+l.parameter_offset));
        s.download(g.bias, s.ptr(s.gradients+l.parameter_offset+l.coefficients));
    }
    s.sync(); result.input = result.layers.front().input; return result;
}
Network ResidentNetwork::download_parameters() {
    auto& s = state(); std::vector<Layer> layers(s.model.layers().begin(), s.model.layers().end());
    for (std::size_t j = 0; j < layers.size(); ++j) {
        const auto& l = s.layers[j]; std::vector<double> coefficients(l.coefficients), bias(l.outputs);
        s.download(coefficients, s.ptr(s.parameters+l.parameter_offset)); s.download(bias, s.ptr(s.parameters+l.parameter_offset+l.coefficients));
        s.sync(); layers[j].set_parameters(coefficients, bias);
    }
    return Network(std::move(layers));
}
void ResidentNetwork::synchronize() { state().sync(); }
std::size_t ResidentNetwork::capacity() const { return state().capacity; }
std::size_t ResidentNetwork::batch() const { return state().batch; }
std::size_t ResidentNetwork::workspace_allocations() const { return state().allocations; }
} // namespace kan::cuda

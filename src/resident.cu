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
    const double* log_widths;
    const double* scales;
    const double* knots;
    std::size_t degree;
    bool trainable;
};
__device__ void report(double value, int* status) { if (!isfinite(value)) atomicExch(status, 1); }
__device__ double interval_ratio(double nr, double nl, double right, double left) {
    const double denominator = right-left;
    if (denominator == 0) return 0;
    return isfinite(denominator) ? (nr-nl)/denominator : (0.5*nr-0.5*nl)/(0.5*right-0.5*left);
}
__device__ double slope_term(double value, std::size_t degree, double right, double left) {
    if (value == 0 || right == left) return 0;
    const double denominator = right-left, p = static_cast<double>(degree);
    return isfinite(denominator) ? (p*value)/denominator : (0.5*p*value)/(0.5*right-0.5*left);
}
__global__ void basis_kernel(const double* input, double* values, double* derivatives, double* log_derivatives,
                             std::size_t count, Basis basis, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const double x = input[index];
        auto* v = values + index * basis.terms;
        auto* d = derivatives + index * basis.terms;
        if (basis.kind == BasisKind::BSpline) {
            for (std::size_t k=0;k<basis.terms;++k) {v[k]=0;d[k]=0;}
            const auto* t=basis.knots;
            if (x<t[basis.degree] || x>t[basis.terms]) continue;
            std::size_t span=basis.terms-1;
            if (x!=t[basis.terms]) {
                std::size_t lo=basis.degree, hi=basis.terms;
                while(lo<hi) {const auto mid=lo+(hi-lo)/2;if(t[mid]<=x)lo=mid+1;else hi=mid;}
                span=lo-1;
            }
            // Only degree+1 terms can be nonzero. Fixed local scratch has no
            // size-dependent device allocation, including repeated knots.
            double lower[18]={}, next[18]={};lower[basis.degree]=1;
            const auto start=span-basis.degree;
            for(std::size_t p=1;p<=basis.degree;++p) {
                for(std::size_t r=basis.degree-p;r<=basis.degree;++r) {
                    const auto i=start+r;
                    double value=0;
                    if(lower[r]!=0)value+=interval_ratio(x,t[i],t[i+p],t[i])*lower[r];
                    if(lower[r+1]!=0)value+=interval_ratio(t[i+p+1],x,t[i+p+1],t[i+1])*lower[r+1];
                    next[r]=value;report(value,status);
                    if(p==basis.degree) {
                        d[i]=slope_term(lower[r],p,t[i+p],t[i])-slope_term(lower[r+1],p,t[i+p+1],t[i+1]);
                        report(d[i],status);
                    }
                }
                for(std::size_t r=basis.degree-p;r<=basis.degree;++r)lower[r]=next[r];
            }
            for(std::size_t r=0;r<=basis.degree;++r)v[start+r]=lower[r];
            continue;
        }
        if (basis.kind == BasisKind::MexicanHat) {
            const double log_normalization=log(2.0/sqrt(3.0))-0.25*log(acos(-1.0));
            for(std::size_t k=0;k<basis.terms;++k) {
                const double scale=basis.scales[k], distance=x-basis.centers[k];
                const double q=isfinite(distance)?distance/scale:x/scale-basis.centers[k]/scale, q2=q*q;
                v[k]=0;d[k]=0;
                if(!isfinite(q2))continue;
                const double log_scale=log(scale), envelope=log_normalization-0.5*log_scale-0.5*q2;
                if(q2!=1)v[k]=copysign(exp(envelope+log(fabs(1-q2))),1-q2);
                if(q!=0 && q2!=3)d[k]=copysign(exp(envelope+log(fabs(q))+log(fabs(q2-3))-log_scale),q*(q2>3?1:-1));
                report(v[k],status);report(d[k],status);
            }
            continue;
        }
        if (basis.kind == BasisKind::GaussianRbf) {
            for (std::size_t k = 0; k < basis.terms; ++k) {
                const double width=basis.trainable?exp(basis.log_widths[k]):basis.width;
                const double distance = x - basis.centers[k];
                const double q = isfinite(distance) ? distance / width : x / width - basis.centers[k] / width;
                v[k] = exp(-q*q);
                d[k] = 0;
                if (q != 0 && isfinite(q)) {
                    if (v[k] < 2.2250738585072014e-308) {
                        const double log_magnitude = log(2.0) + log(fabs(q)) - q*q - log(width);
                        d[k] = -copysign(exp(log_magnitude), q);
                    } else d[k] = (-2*q*v[k]) / width;
                }
                report(d[k], status);
                if(basis.trainable) {
                    const double dw=q!=0 && isfinite(q)?exp(log(2.0)+2*log(fabs(q))-q*q):0;
                    log_derivatives[index*basis.terms+k]=dw;report(dw,status);
                }
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
__global__ void parameter_kernel(const double* v, const double* upstream, const double* parameters, double* gradient,
                                 std::size_t batch, std::size_t inputs, std::size_t outputs, std::size_t terms, double lambda, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    const auto coefficients = inputs*outputs*terms;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < coefficients+outputs; index += stride) {
        double sum = 0;
        if (index < coefficients) {
            const auto k = index%terms, i = (index/terms)%inputs, o = index/(terms*inputs);
            for (std::size_t b = 0; b < batch; ++b) sum += upstream[b*outputs+o]*v[(b*inputs+i)*terms+k];
            sum += lambda*parameters[index];
        } else {
            const auto o = index-coefficients;
            for (std::size_t b = 0; b < batch; ++b) sum += upstream[b*outputs+o];
        }
        gradient[index] = sum; report(sum, status);
    }
}
__global__ void nonlinear_kernel(const double* dx, const double* dw, const double* c, const double* upstream,
                                 double* gradient, std::size_t batch, std::size_t inputs,
                                 std::size_t outputs, std::size_t terms, int* status) {
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto k=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;k<terms;k+=stride) {
        double center=0,width=0;
        for(std::size_t b=0;b<batch;++b)
            for(std::size_t o=0;o<outputs;++o)
                for(std::size_t i=0;i<inputs;++i) {
                    const double factor=upstream[b*outputs+o]*c[(o*inputs+i)*terms+k];
                    center+=factor*(-dx[(b*inputs+i)*terms+k]);
                    width+=factor*dw[(b*inputs+i)*terms+k];
                }
        gradient[k]=center;gradient[terms+k]=width;report(center,status);report(width,status);
    }
}
__global__ void validate_width_kernel(const double* next, std::size_t count, int* status) {
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto k=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=stride) {
        const double width=exp(next[k]);if(!isfinite(width)||width<=0)atomicExch(status,1);
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
    std::size_t values, derivatives, centers, log_derivatives=0, scales=0, knots=0;
    bool trainable=false;
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
            const auto extra=layer.basis().trainable_rbf?product(layer.basis().size,2):0;
            const auto count=layer.coefficients().size()+layer.outputs();
            if (count>std::vector<double>().max_size() || extra>std::vector<double>().max_size()-count || count+extra>std::vector<double>().max_size()-parameter_count)
                throw std::overflow_error("resident parameter size overflow");
            layers.push_back({layer.inputs(), layer.outputs(), layer.basis().size, layer.coefficients().size(), parameter_count, 0, 0, 0});
            layers.back().trainable=layer.basis().trainable_rbf;
            parameter_count += count+extra;
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
            const auto& basis=model.layers()[j].basis();
            if ((basis.kind == BasisKind::GaussianRbf && !layout.trainable) || basis.kind==BasisKind::MexicanHat)
                layout.centers = reserve(layout.terms);
            if(layout.trainable)layout.log_derivatives=reserve(product(product(capacity,layout.inputs),layout.terms));
            if(basis.kind==BasisKind::MexicanHat)layout.scales=reserve(layout.terms);
            if(basis.kind==BasisKind::BSpline)layout.knots=reserve(basis.knots.size());
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
                const auto& basis=layer.basis();
                if(layout.trainable) {
                    upload(ptr(parameters+layout.parameter_offset+layout.coefficients+layout.outputs),basis.centers);
                    upload(ptr(parameters+layout.parameter_offset+layout.coefficients+layout.outputs+layout.terms),basis.log_widths);
                } else if(basis.kind==BasisKind::GaussianRbf || basis.kind==BasisKind::MexicanHat)upload(ptr(layout.centers),basis.centers);
                if(basis.kind==BasisKind::MexicanHat)upload(ptr(layout.scales),basis.scales);
                if(basis.kind==BasisKind::BSpline)upload(ptr(layout.knots),basis.knots);
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
        const auto nonlinear=s.parameters+l.parameter_offset+l.coefficients+l.outputs;
        Basis basis{b.kind,b.size,b.alpha,b.beta,b.frequency,b.width,
                    s.ptr(l.trainable?nonlinear:l.centers),s.ptr(nonlinear+l.terms),
                    s.ptr(l.scales),s.ptr(l.knots),b.degree,l.trainable};
        basis_kernel<<<blocks(s.batch*l.inputs), 256, 0, s.stream>>>(s.ptr(s.activation[j]), s.ptr(l.values), s.ptr(l.derivatives), s.ptr(l.log_derivatives), s.batch*l.inputs, basis, s.status);
        check(cudaGetLastError(), "resident basis launch");
        forward_kernel<<<blocks(s.batch*l.outputs), 256, 0, s.stream>>>(s.ptr(l.values), s.ptr(s.parameters+l.parameter_offset), s.ptr(s.parameters+l.parameter_offset+l.coefficients), s.ptr(s.activation[j+1]), s.batch*l.outputs, l.inputs, l.outputs, l.terms, s.status);
        check(cudaGetLastError(), "resident forward launch");
    }
    s.result(); s.has_forward = true;
}
void ResidentNetwork::backward(double coefficient_l2) {
    auto& s = state();
    if(!std::isfinite(coefficient_l2)||coefficient_l2<0)throw std::invalid_argument("coefficient L2 must be finite and nonnegative");
    if (!s.has_forward || !s.has_upstream) throw std::logic_error("resident backward requires current forward and upstream");
    s.has_backward = false; s.reset_status();
    for (std::size_t j = s.layers.size(); j-- > 0;) {
        const auto& l = s.layers[j];
        if (s.batch) {
            input_kernel<<<blocks(s.batch*l.inputs), 256, 0, s.stream>>>(s.ptr(l.derivatives), s.ptr(s.parameters+l.parameter_offset), s.ptr(s.upstream[j+1]), s.ptr(s.upstream[j]), s.batch*l.inputs, l.inputs, l.outputs, l.terms, s.status);
            check(cudaGetLastError(), "resident input gradient launch");
        }
        parameter_kernel<<<blocks(l.coefficients+l.outputs), 256, 0, s.stream>>>(s.ptr(l.values), s.ptr(s.upstream[j+1]), s.ptr(s.parameters+l.parameter_offset), s.ptr(s.gradients+l.parameter_offset), s.batch, l.inputs, l.outputs, l.terms, coefficient_l2, s.status);
        check(cudaGetLastError(), "resident parameter gradient launch");
        if(l.trainable) {
            nonlinear_kernel<<<blocks(l.terms),256,0,s.stream>>>(s.ptr(l.derivatives),s.ptr(l.log_derivatives),s.ptr(s.parameters+l.parameter_offset),
                s.ptr(s.upstream[j+1]),s.ptr(s.gradients+l.parameter_offset+l.coefficients+l.outputs),s.batch,l.inputs,l.outputs,l.terms,s.status);
            check(cudaGetLastError(),"resident nonlinear gradient launch");
        }
    }
    s.result(); s.has_backward = true;
}
void ResidentNetwork::sgd(double learning_rate) {
    auto& s = state();
    if (!std::isfinite(learning_rate) || learning_rate <= 0) throw std::invalid_argument("learning rate must be finite and positive");
    if (!s.has_backward) throw std::logic_error("resident SGD requires current gradients");
    s.reset_status();
    candidate_kernel<<<blocks(s.parameter_count), 256, 0, s.stream>>>(s.ptr(s.parameters), s.ptr(s.gradients), s.ptr(s.candidates), s.parameter_count, learning_rate, s.status);
    check(cudaGetLastError(),"resident candidate launch");
    for(const auto& l:s.layers)if(l.trainable) {
        validate_width_kernel<<<blocks(l.terms),256,0,s.stream>>>(s.ptr(s.candidates+l.parameter_offset+l.coefficients+l.outputs+l.terms),l.terms,s.status);
        check(cudaGetLastError(),"resident width validation launch");
    }
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
        if(l.trainable) {
            g.centers.resize(l.terms);g.log_widths.resize(l.terms);
            s.download(g.centers,s.ptr(s.gradients+l.parameter_offset+l.coefficients+l.outputs));
            s.download(g.log_widths,s.ptr(s.gradients+l.parameter_offset+l.coefficients+l.outputs+l.terms));
        }
    }
    s.sync(); result.input = result.layers.front().input; return result;
}
Network ResidentNetwork::download_parameters() {
    auto& s = state(); std::vector<Layer> layers(s.model.layers().begin(), s.model.layers().end());
    for (std::size_t j = 0; j < layers.size(); ++j) {
        const auto& l = s.layers[j]; std::vector<double> coefficients(l.coefficients), bias(l.outputs);
        s.download(coefficients, s.ptr(s.parameters+l.parameter_offset)); s.download(bias, s.ptr(s.parameters+l.parameter_offset+l.coefficients));
        s.sync(); layers[j].set_parameters(coefficients, bias);
        if(l.trainable) {
            std::vector<double> centers(l.terms),widths(l.terms);
            s.download(centers,s.ptr(s.parameters+l.parameter_offset+l.coefficients+l.outputs));
            s.download(widths,s.ptr(s.parameters+l.parameter_offset+l.coefficients+l.outputs+l.terms));
            s.sync();layers[j].set_rbf_parameters(centers,widths);
        }
    }
    return Network(std::move(layers));
}
void ResidentNetwork::synchronize() { state().sync(); }
std::size_t ResidentNetwork::capacity() const { return state().capacity; }
std::size_t ResidentNetwork::batch() const { return state().batch; }
std::size_t ResidentNetwork::workspace_allocations() const { return state().allocations; }
} // namespace kan::cuda

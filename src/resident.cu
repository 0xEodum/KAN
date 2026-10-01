#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include "detail/basis_formulas.hpp"
#include "detail/rational_formulas.hpp"
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
unsigned blocks(std::size_t count,std::size_t work_per_block=256) {
    return static_cast<unsigned>(std::min<std::size_t>((count-1)/work_per_block+1,65535));
}
__device__ void report(double value, int* status) { if (!isfinite(value)) atomicOr(status, 1); }
// Device guard for the shared formulas: record a nonfinite status bit and
// continue; the host raises after the launch sequence completes.
struct StatusGuard {
    int* status;
    __device__ double operator()(double value) const { report(value, status); return value; }
};
// Identity guard for recomputing values that an earlier kernel of the same
// step already checked with StatusGuard (keeps hot reduction loops lean).
struct CheckedEarlier {
    __device__ double operator()(double value) const { return value; }
};
template<BasisKind Kind>
__global__ void basis_kernel(const double* input, double* values, double* derivatives, double* log_derivatives,
                             std::size_t count, detail::BasisView basis, int* status) {
    const StatusGuard guard{status};
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
        const auto row = index * basis.terms;
        detail::basis_terms_for<Kind>(basis, input[index],
                                      {values + row, derivatives + row, nullptr, basis.trainable ? log_derivatives + row : nullptr}, guard);
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
constexpr unsigned nonlinear_tiles=64;
__global__ void nonlinear_partial_kernel(const double* dx, const double* dw, const double* c, const double* upstream,
                                         double* partial, std::size_t count, std::size_t inputs,
                                         std::size_t outputs, std::size_t terms, unsigned tiles, int* status) {
    __shared__ double centers[256], widths[256];
    const auto k=static_cast<std::size_t>(blockIdx.x)/tiles;
    const auto tile=blockIdx.x%tiles, lane=threadIdx.x;
    double center=0,width=0;
    const auto stride=static_cast<std::size_t>(tiles)*blockDim.x;
    for(auto index=static_cast<std::size_t>(tile)*blockDim.x+lane;index<count;index+=stride) {
        const auto i=index%inputs, o=(index/inputs)%outputs, b=index/(inputs*outputs);
        const double factor=upstream[b*outputs+o]*c[(o*inputs+i)*terms+k];
        center+=factor*(-dx[(b*inputs+i)*terms+k]);width+=factor*dw[(b*inputs+i)*terms+k];
    }
    report(center,status);report(width,status);
    centers[lane]=center;widths[lane]=width;__syncthreads();
    for(unsigned step=blockDim.x/2;step;step/=2) {
        if(lane<step) {centers[lane]+=centers[lane+step];widths[lane]+=widths[lane+step];}
        __syncthreads();
    }
    if(lane==0) {
        partial[k*tiles+tile]=centers[0];partial[(terms+k)*tiles+tile]=widths[0];
        report(centers[0],status);report(widths[0],status);
    }
}
__global__ void nonlinear_finish_kernel(const double* partial, double* gradient, std::size_t terms, unsigned tiles, int* status) {
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto k=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;k<2*terms;k+=stride) {
        double sum=0;for(unsigned tile=0;tile<tiles;++tile)sum+=partial[k*tiles+tile];
        gradient[k]=sum;report(sum,status);
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
// Distinct rational execution: caches are edge-major to expose contiguous
// samples to nonlinear parameter reductions. All caches live in the arena.
__global__ void rational_forward_kernel(const double* input,const double* a,const double* b,const double* bias,
                                        double* values,double* denominator_values,double* derivatives,double* output,
                                        std::size_t batch,std::size_t capacity,std::size_t inputs,std::size_t outputs,
                                        RationalConfig config,int* status) {
    const StatusGuard guard{status};
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    const auto m=config.numerator_degree,n=config.denominator_degree;
    for(auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<batch*outputs;index+=stride) {
        const auto sample=index/outputs,o=index%outputs;double sum=bias[o];
        for(std::size_t i=0;i<inputs;++i) {
            const auto edge=o*inputs+i,cache=edge*capacity+sample;
            const auto h=detail::rational_horner(config,input[sample*inputs+i],a+edge*(m+1),b+edge*n,guard);
            if(detail::rational_pole(config,h)) {
                atomicOr(status,2);values[cache]=denominator_values[cache]=derivatives[cache]=0;continue;
            }
            const auto e=detail::rational_edge(config,h,guard);
            // Derivative powers are part of the nonlinear contract, including
            // zero upstream. Detect unusable parameter VJPs during forward.
            double power=1;
            for(std::size_t k=0;k<=(m>n?m:n);++k) {
                if(k)power=guard(power*h.z);
                const double divided=guard(power/h.q);
                if(k<=m)detail::rational_numerator_vjp(h.q,h.z,k,power,divided,guard);
                if(k&&k<=n)detail::rational_denominator_vjp(h.p,h.q,e.value,h.z,k,power,divided,guard);
            }
            values[cache]=h.p;denominator_values[cache]=h.q;derivatives[cache]=e.input_derivative;sum+=e.value;report(sum,status);
        }
        output[index]=sum;report(sum,status);
    }
}
__global__ void rational_input_kernel(const double* derivatives,const double* upstream,double* input_gradient,
                                      std::size_t batch,std::size_t capacity,std::size_t inputs,std::size_t outputs,int* status) {
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<batch*inputs;index+=stride) {
        const auto sample=index/inputs,i=index%inputs;double sum=0;
        for(std::size_t o=0;o<outputs;++o)sum+=upstream[sample*outputs+o]*derivatives[(o*inputs+i)*capacity+sample];
        input_gradient[index]=sum;report(sum,status);
    }
}
__global__ void rational_parameter_kernel(const double* input,const double* values,const double* denominator_values,const double* upstream,
                                          const double* parameters,double* gradients,std::size_t batch,std::size_t capacity,
                                          std::size_t inputs,std::size_t outputs,RationalConfig config,double lambda,int* status) {
    // Forward validated z and every parameter VJP of these cached samples with
    // bit-identical operations; backward runs only after a successful forward.
    const CheckedEarlier guard;
    const auto m=config.numerator_degree+1,n=config.denominator_degree,acount=inputs*outputs*m;
    const auto total=acount+outputs+inputs*outputs*n;
    // A full warp owns each parameter and scans contiguous edge-major
    // cache samples. Warp reduction preserves bounded launches and avoids
    // atomics or execution scratch allocations.
    const auto lane=threadIdx.x%32;
    const auto stride=static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for(auto index=(static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;index<total;index+=stride) {
        const bool numerator=index<acount,bias=index>=acount&&index<acount+outputs;
        const auto relative=numerator?index:bias?index-acount:index-acount-outputs;
        const auto edge=bias?0:relative/(numerator?m:n),k=bias?0:relative%(numerator?m:n)+(numerator?0:1);
        const auto o=bias?relative:edge/inputs,i=edge%inputs;double sum=0;
        for(std::size_t sample=lane;sample<batch;sample+=32) {
            double derivative=1;
            if(!bias) {
                const double z=detail::rational_argument(config,input[sample*inputs+i],guard);double power=1;
                for(std::size_t j=0;j<k;++j)power*=z;
                const auto cache=edge*capacity+sample;
                const double q=denominator_values[cache],divided=power/q;
                if(numerator)derivative=detail::rational_numerator_vjp(q,z,k,power,divided,guard);
                else {
                    const double p=values[cache];
                    derivative=detail::rational_denominator_vjp(p,q,p/q,z,k,power,divided,guard);
                }
            }
            const double term=upstream[sample*outputs+o]*derivative;report(term,status);sum+=term;
        }
        report(sum,status);
        for(unsigned offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffffU,sum,offset);
        if(lane==0) {
            if(numerator)sum+=lambda*parameters[index];gradients[index]=sum;report(sum,status);
        }
    }
}

struct Layout {
    std::size_t inputs, outputs, terms, coefficients, parameter_offset;
    std::size_t values, derivatives, centers, log_derivatives=0, scales=0, knots=0, nonlinear_partials=0;
    bool trainable=false;
    unsigned partial_tiles=1;
    bool rational=false;
    std::size_t denominator_count=0,denominator_values=0;
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
            const bool rational=layer.is_rational();
            const auto terms=rational?layer.rational_config().numerator_degree+1:layer.basis().size;
            const auto extra=rational?layer.denominators().size():layer.basis().trainable_rbf?product(terms,2):0;
            const auto count=layer.coefficients().size()+layer.outputs();
            if (count>std::vector<double>().max_size() || extra>std::vector<double>().max_size()-count || count+extra>std::vector<double>().max_size()-parameter_count)
                throw std::overflow_error("resident parameter size overflow");
            layers.push_back({layer.inputs(), layer.outputs(), terms, layer.coefficients().size(), parameter_count, 0, 0, 0});
            layers.back().rational=rational;
            layers.back().denominator_count=rational?extra:0;
            layers.back().trainable=!rational&&layer.basis().trainable_rbf;
            parameter_count += count+extra;
        }
        parameters = reserve(parameter_count); gradients = reserve(parameter_count); candidates = reserve(parameter_count);
        activation.push_back(reserve(product(capacity, layers.front().inputs)));
        upstream.push_back(reserve(product(capacity, layers.front().inputs)));
        for (std::size_t j = 0; j < layers.size(); ++j) {
            auto& layout = layers[j];
            activation.push_back(reserve(product(capacity, layout.outputs)));
            upstream.push_back(reserve(product(capacity, layout.outputs)));
            if(layout.rational) {
                const auto count=product(product(capacity,layout.inputs),layout.outputs);
                layout.values=reserve(count);layout.derivatives=reserve(count);layout.denominator_values=reserve(count);
                continue;
            }
            layout.values = reserve(product(product(capacity, layout.inputs), layout.terms));
            layout.derivatives = reserve(product(product(capacity, layout.inputs), layout.terms));
            const auto& basis=model.layers()[j].basis();
            if ((basis.kind == BasisKind::GaussianRbf && !layout.trainable) || basis.kind==BasisKind::MexicanHat)
                layout.centers = reserve(layout.terms);
            if(layout.trainable) {
                layout.log_derivatives=reserve(product(product(capacity,layout.inputs),layout.terms));
                const auto count=product(product(capacity,layout.inputs),layout.outputs);
                layout.partial_tiles=static_cast<unsigned>(std::min<std::size_t>(nonlinear_tiles,count?((count-1)/256+1):1));
                layout.nonlinear_partials=reserve(product(layout.terms,2*layout.partial_tiles));
            }
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
                if(layout.rational) {
                    upload(ptr(parameters+layout.parameter_offset+layout.coefficients+layout.outputs),layer.denominators());
                    continue;
                }
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
        if (value&1) throw std::overflow_error("nonfinite resident numerical result");
        if (value&2) throw std::domain_error("unsafe resident rational denominator");
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
        const auto& l = s.layers[j];
        if(l.rational) {
            rational_forward_kernel<<<blocks(s.batch*l.outputs),256,0,s.stream>>>(s.ptr(s.activation[j]),s.ptr(s.parameters+l.parameter_offset),
                s.ptr(s.parameters+l.parameter_offset+l.coefficients+l.outputs),s.ptr(s.parameters+l.parameter_offset+l.coefficients),
                s.ptr(l.values),s.ptr(l.denominator_values),s.ptr(l.derivatives),s.ptr(s.activation[j+1]),
                s.batch,s.capacity,l.inputs,l.outputs,s.model.layers()[j].rational_config(),s.status);
            check(cudaGetLastError(),"resident rational forward launch");continue;
        }
        const auto& b = s.model.layers()[j].basis();
        const auto nonlinear=s.parameters+l.parameter_offset+l.coefficients+l.outputs;
        const detail::BasisView basis{b.kind,b.size,b.alpha,b.beta,b.frequency,b.width,
                                      s.ptr(l.trainable?nonlinear:l.centers),s.ptr(nonlinear+l.terms),
                                      s.ptr(l.scales),s.ptr(l.knots),b.degree,l.trainable};
        detail::visit_basis_family(b.kind, [&](auto family) {
            basis_kernel<decltype(family)::value><<<blocks(s.batch*l.inputs), 256, 0, s.stream>>>(
                s.ptr(s.activation[j]), s.ptr(l.values), s.ptr(l.derivatives), s.ptr(l.log_derivatives), s.batch*l.inputs, basis, s.status);
        });
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
        if(l.rational) {
            if(s.batch) {
                rational_input_kernel<<<blocks(s.batch*l.inputs),256,0,s.stream>>>(s.ptr(l.derivatives),s.ptr(s.upstream[j+1]),s.ptr(s.upstream[j]),
                    s.batch,s.capacity,l.inputs,l.outputs,s.status);
                check(cudaGetLastError(),"resident rational input gradient launch");
            }
            rational_parameter_kernel<<<blocks(l.coefficients+l.outputs+l.denominator_count,8),256,0,s.stream>>>(s.ptr(s.activation[j]),s.ptr(l.values),s.ptr(l.denominator_values),
                s.ptr(s.upstream[j+1]),s.ptr(s.parameters+l.parameter_offset),s.ptr(s.gradients+l.parameter_offset),s.batch,s.capacity,l.inputs,l.outputs,
                s.model.layers()[j].rational_config(),coefficient_l2,s.status);
            check(cudaGetLastError(),"resident rational parameter gradient launch");continue;
        }
        if (s.batch) {
            input_kernel<<<blocks(s.batch*l.inputs), 256, 0, s.stream>>>(s.ptr(l.derivatives), s.ptr(s.parameters+l.parameter_offset), s.ptr(s.upstream[j+1]), s.ptr(s.upstream[j]), s.batch*l.inputs, l.inputs, l.outputs, l.terms, s.status);
            check(cudaGetLastError(), "resident input gradient launch");
        }
        parameter_kernel<<<blocks(l.coefficients+l.outputs), 256, 0, s.stream>>>(s.ptr(l.values), s.ptr(s.upstream[j+1]), s.ptr(s.parameters+l.parameter_offset), s.ptr(s.gradients+l.parameter_offset), s.batch, l.inputs, l.outputs, l.terms, coefficient_l2, s.status);
        check(cudaGetLastError(), "resident parameter gradient launch");
        if(l.trainable) {
            const auto count=product(product(s.batch,l.inputs),l.outputs);
            const auto tiles=static_cast<unsigned>(std::min<std::size_t>(l.partial_tiles,count?((count-1)/256+1):1));
            // Bound the launch dimension even for large valid basis term counts.
            if(l.terms>2147483647U/tiles)throw std::overflow_error("resident nonlinear launch size overflow");
            nonlinear_partial_kernel<<<static_cast<unsigned>(l.terms)*tiles,256,0,s.stream>>>(s.ptr(l.derivatives),s.ptr(l.log_derivatives),s.ptr(s.parameters+l.parameter_offset),
                s.ptr(s.upstream[j+1]),s.ptr(l.nonlinear_partials),count,l.inputs,l.outputs,l.terms,tiles,s.status);
            check(cudaGetLastError(),"resident nonlinear partial launch");
            nonlinear_finish_kernel<<<blocks(2*l.terms),256,0,s.stream>>>(s.ptr(l.nonlinear_partials),s.ptr(s.gradients+l.parameter_offset+l.coefficients+l.outputs),l.terms,tiles,s.status);
            check(cudaGetLastError(),"resident nonlinear reduction launch");
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
        if(l.rational) {
            g.denominators.resize(l.denominator_count);
            s.download(g.denominators,s.ptr(s.gradients+l.parameter_offset+l.coefficients+l.outputs));
        }
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
        if(l.rational) {
            std::vector<double> denominators(l.denominator_count);
            s.download(denominators,s.ptr(s.parameters+l.parameter_offset+l.coefficients+l.outputs));
            s.sync();layers[j].set_rational_parameters(coefficients,denominators,bias);continue;
        }
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

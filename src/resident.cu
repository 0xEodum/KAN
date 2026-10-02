#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include "detail/basis_view.hpp"
#include "detail/rational_formulas.hpp"
#include "detail/input_map_formulas.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

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
void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cublasGetStatusString(status));
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
template<detail::BasisKind Kind>
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
// Contraction engine epilogues (backlog C2). cuBLAS computes the products:
// forward Y = Phi*C^T, coefficient VJP dC = U^T*Phi (+ lambda*C), bias VJP
// db = U^T*1 and W = U*C; these kernels add the bias, reduce W against Phi'
// to the input VJP and report nonfinite results.
__global__ void bias_kernel(double* output, const double* bias, std::size_t count, std::size_t outputs, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        const double sum = output[index]+bias[index%outputs];
        output[index] = sum; report(sum, status);
    }
}
// Small forward contractions: one warp per output, Y[b,o] = bias[o] +
// dot(Phi[b,:], C[o,:]) over two contiguous rows, with the bias and the
// nonfinite check fused. cuBLAS runs a tiny DGEMM as one latency-bound CTA
// (42-64 us at 24x32x112 on the RTX 3090; this kernel: 7 us); see C2 evidence.
__global__ void forward_dot_kernel(const double* v, const double* c, const double* bias, double* output,
                                   std::size_t count, std::size_t outputs, std::size_t length, int* status) {
    const auto lane = threadIdx.x%32;
    const auto warps = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for (auto index = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32; index < count; index += warps) {
        const double* row = v+(index/outputs)*length;
        const double* column = c+(index%outputs)*length;
        double sum = 0;
        for (std::size_t k = lane; k < length; k += 32) sum += row[k]*column[k];
        for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (lane == 0) { sum += bias[index%outputs]; output[index] = sum; report(sum, status); }
    }
}
// Largest forward contraction (batch*outputs*inputs*terms multiply-adds) run by
// forward_dot_kernel; cuBLAS above. Both are FP64-ALU bound on GA102 and meet
// at a few million multiply-adds (C2 evidence, dot_vs_gemm).
constexpr std::size_t small_forward_contraction = std::size_t{1} << 23;
// Largest coefficients+outputs whose VJP runs parameter_partial_kernel, in at
// most parameter_tiles batch tiles of at least parameter_tile_rows samples.
constexpr std::size_t small_parameter_vjp = std::size_t{1} << 15, parameter_tile_rows = 64;
constexpr unsigned parameter_tiles = 64;
unsigned parameter_tile_count(std::size_t batch) {
    return static_cast<unsigned>(std::min<std::size_t>(parameter_tiles, batch ? (batch-1)/parameter_tile_rows+1 : 1));
}

// Small parameter VJPs (at most small_parameter_vjp coefficients+outputs):
// block row t reduces samples [t*chunk, (t+1)*chunk) into partial[t][c] for
// c = o*IK+ik (U[b,o]*Phi[b,ik]) and the bias c = IK*O+o (U[b,o]). cuBLAS
// runs these long, narrow products without split-K (one 32x32 tile over the
// whole batch: up to 331 us at 112x24x1024 on the RTX 3090; C2 evidence).
__global__ void parameter_partial_kernel(const double* v, const double* u, double* partial, std::size_t batch,
                                         std::size_t outputs, std::size_t length, std::size_t chunk) {
    const auto coefficients = outputs*length, count = coefficients+outputs;
    const auto begin = static_cast<std::size_t>(blockIdx.y)*chunk, end = begin+chunk < batch ? begin+chunk : batch;
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto c = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; c < count; c += stride) {
        double sum = 0;
        if (c < coefficients) {
            const auto o = c/length, ik = c%length;
            for (auto r = begin; r < end; ++r) sum += u[r*outputs+o]*v[r*length+ik];
        } else {
            for (auto r = begin; r < end; ++r) sum += u[r*outputs+c-coefficients];
        }
        partial[blockIdx.y*count+c] = sum;
    }
}
// One launch finishes a layer's backward: dx[r] = sum_k Phi'[r,k]*W[r,k] for
// the rows r = (sample, input), then the `checked` parameter VJPs [dC | db]:
// with partials, their fixed-order tile sum plus lambda*C; otherwise (written
// by cuBLAS earlier on the stream) only the nonfinite check.
__global__ void backward_finish_kernel(const double* derivatives, const double* w, double* dx,
                                       std::size_t rows, std::size_t terms, double* gradients, std::size_t checked,
                                       const double* partial, unsigned tiles, const double* c, std::size_t coefficients,
                                       double lambda, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < rows+checked; index += stride) {
        if (index >= rows) {
            const auto q = index-rows;
            if (partial) {
                double sum = 0;
                for (unsigned t = 0; t < tiles; ++t) sum += partial[t*checked+q];
                if (q < coefficients) sum += lambda*c[q];
                gradients[q] = sum;
            }
            report(gradients[q], status);
            continue;
        }
        double sum = 0;
        for (std::size_t k = 0; k < terms; ++k) sum += derivatives[index*terms+k]*w[index*terms+k];
        dx[index] = sum; report(sum, status);
    }
}
__global__ void fill_kernel(double* values, std::size_t count, double value) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride)
        values[index] = value;
}
constexpr unsigned nonlinear_tiles=64;
// Center/log-width VJPs: sum over rows r of W[r,k]*(-dPhi/dx) and W[r,k]*dPhi/dlog-width.
__global__ void nonlinear_partial_kernel(const double* dx, const double* dw, const double* w,
                                         double* partial, std::size_t rows, std::size_t terms, unsigned tiles, int* status) {
    __shared__ double centers[256], widths[256];
    const auto k=static_cast<std::size_t>(blockIdx.x)/tiles;
    const auto tile=blockIdx.x%tiles, lane=threadIdx.x;
    double center=0,width=0;
    const auto stride=static_cast<std::size_t>(tiles)*blockDim.x;
    for(auto r=static_cast<std::size_t>(tile)*blockDim.x+lane;r<rows;r+=stride) {
        const double factor=w[r*terms+k];
        center+=factor*(-dx[r*terms+k]);width+=factor*dw[r*terms+k];
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
// One instantiation per denominator policy; `gains` (g = dQ/dS) is cached
// only by the safe policies and is null for Guarded.
template<DenominatorPolicy Policy>
__global__ void rational_forward_kernel(const double* input,const double* a,const double* b,const double* bias,
                                        double* values,double* denominator_values,double* derivatives,double* gains,double* output,
                                        std::size_t batch,std::size_t capacity,std::size_t inputs,std::size_t outputs,
                                        RationalConfig config,int* status) {
    const StatusGuard guard{status};
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    const auto m=config.numerator_degree,n=config.denominator_degree;
    for(auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<batch*outputs;index+=stride) {
        const auto sample=index/outputs,o=index%outputs;double sum=bias[o];
        for(std::size_t i=0;i<inputs;++i) {
            const auto edge=o*inputs+i,cache=edge*capacity+sample;
            const auto h=detail::rational_horner<Policy>(config,input[sample*inputs+i],a+edge*(m+1),b+edge*n,guard);
            if(detail::rational_pole<Policy>(config,h)) {
                atomicOr(status,2);values[cache]=denominator_values[cache]=derivatives[cache]=0;continue;
            }
            const auto e=detail::rational_edge<Policy>(config,h,guard);
            // Derivative powers are part of the nonlinear contract, including
            // zero upstream. Detect unusable parameter VJPs during forward.
            double power=1;
            for(std::size_t k=0;k<=(m>n?m:n);++k) {
                if(k)power=guard(power*h.z);
                const double divided=guard(power/h.q);
                if(k<=m)detail::rational_numerator_vjp(h.q,h.z,k,power,divided,guard);
                if(k&&k<=n)detail::rational_denominator_vjp<Policy>(h.p,h.q,e.value,h.gain,h.z,k,power,divided,guard);
            }
            if constexpr(Policy!=DenominatorPolicy::Guarded)gains[cache]=h.gain;
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
template<DenominatorPolicy Policy>
__global__ void rational_parameter_kernel(const double* input,const double* values,const double* denominator_values,const double* gains,const double* upstream,
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
                    const double gain=Policy==DenominatorPolicy::Guarded?1.0:gains[cache];
                    derivative=detail::rational_denominator_vjp<Policy>(p,q,p/q,gain,z,k,power,divided,guard);
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

// Input maps (backlog M1). Affine and tanh are elementwise. LayerNorm uses one
// warp per row for the row moments and the input VJP, and a tiled column
// reduction with a fixed-order finish for the gain/bias VJPs: no
// floating-point atomics, deterministic results.
__global__ void affine_forward_kernel(const double* x, const double* scale, const double* shift, double* y,
                                      std::size_t count, std::size_t features, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        const auto i = index%features;
        y[index] = detail::affine_value(scale[i], shift[i], x[index]); report(y[index], status);
    }
}
__global__ void affine_input_kernel(const double* upstream, const double* scale, double* dx,
                                    std::size_t count, std::size_t features, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        dx[index] = scale[index%features]*upstream[index]; report(dx[index], status);
    }
}
__global__ void tanh_forward_kernel(const double* x, double* y, std::size_t count, double scale) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride)
        y[index] = detail::tanh_value(scale, x[index]); // bounded by 1 for finite input
}
// Derivative from the saved output y (activation[j+1]): three FP64 operations
// instead of cosh and a division per element.
__global__ void tanh_input_kernel(const double* y, const double* upstream, double* dx, std::size_t count,
                                  double scale, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        dx[index] = upstream[index]*detail::tanh_derivative(scale, y[index]); report(dx[index], status);
    }
}
// Butterfly sum over a group of Lanes consecutive lanes: every lane of the
// group ends with the same (commutative) result.
template<unsigned Lanes> __device__ double group_sum(double value) {
    for (unsigned offset = Lanes/2; offset; offset /= 2) value += __shfl_xor_sync(0xffffffffU, value, offset, Lanes);
    return value;
}
// LayerNorm rows are processed by groups of Lanes lanes (32/Lanes rows per
// warp), Lanes chosen per map so that each lane holds about eight features:
// the per-row scalar work (reductions, 1/sqrt) is FP64 and is executed by
// every lane, so narrower groups cut the FP64 instruction count. The row
// loop is warp-uniform; groups past the last row recompute a valid row so
// that every lane takes part in the shuffles, and store nothing.
template<unsigned Lanes>
__global__ void layer_norm_forward_kernel(const double* x, const double* gain, const double* bias, double* y,
                                          double* stats, std::size_t rows, std::size_t features,
                                          double inverse_count, double epsilon, int* status) {
    constexpr unsigned rows_per_warp = 32/Lanes;
    const auto lane = threadIdx.x%Lanes, group = (threadIdx.x%32)/Lanes;
    const auto warp = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32)*rows_per_warp;
    for (auto first = warp*rows_per_warp; first < rows; first += stride) {
        const auto row = first+group;
        const bool active = row < rows;
        const double* in = x+(active ? row : first)*features;
        double sum = 0;
        for (std::size_t f = lane; f < features; f += Lanes) sum += in[f];
        const double mean = detail::layer_norm_mean(group_sum<Lanes>(sum), inverse_count);
        double squares = 0;
        for (std::size_t f = lane; f < features; f += Lanes) { const double d = in[f]-mean; squares += d*d; }
        const double variance = detail::layer_norm_mean(group_sum<Lanes>(squares), inverse_count);
        const double rstd = detail::layer_norm_rstd(variance, epsilon);
        if (!active) continue;
        if (lane == 0) { report(mean, status); report(variance, status); stats[2*row] = mean; stats[2*row+1] = rstd; }
        for (std::size_t f = lane; f < features; f += Lanes) {
            const double xhat = detail::layer_norm_normalized(in[f], mean, rstd);
            const double value = gain ? gain[f]*xhat+bias[f] : xhat;
            y[row*features+f] = value; report(value, status);
        }
    }
}
template<unsigned Lanes>
__global__ void layer_norm_input_kernel(const double* x, const double* upstream, const double* gain, const double* stats,
                                        double* dx, std::size_t rows, std::size_t features, double inverse_count, int* status) {
    constexpr unsigned rows_per_warp = 32/Lanes;
    const auto lane = threadIdx.x%Lanes, group = (threadIdx.x%32)/Lanes;
    const auto warp = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32)*rows_per_warp;
    for (auto first = warp*rows_per_warp; first < rows; first += stride) {
        const auto row = first+group;
        const bool active = row < rows;
        const auto base = (active ? row : first)*features;
        const double mean = stats[2*(active ? row : first)], rstd = stats[2*(active ? row : first)+1];
        double sum_w = 0, sum_wx = 0;
        for (std::size_t f = lane; f < features; f += Lanes) {
            const double w = gain ? upstream[base+f]*gain[f] : upstream[base+f];
            sum_w += w; sum_wx += w*detail::layer_norm_normalized(x[base+f], mean, rstd);
        }
        const double mean_w = detail::layer_norm_mean(group_sum<Lanes>(sum_w), inverse_count);
        const double mean_wx = detail::layer_norm_mean(group_sum<Lanes>(sum_wx), inverse_count);
        if (!active) continue;
        for (std::size_t f = lane; f < features; f += Lanes) {
            const double w = gain ? upstream[base+f]*gain[f] : upstream[base+f];
            const double xhat = detail::layer_norm_normalized(x[base+f], mean, rstd);
            dx[base+f] = detail::layer_norm_input_vjp(rstd, w, mean_w, xhat, mean_wx); report(dx[base+f], status);
        }
    }
}
// Lanes per LayerNorm row: the power of two holding about eight features per lane.
unsigned norm_lanes(std::size_t features) {
    unsigned lanes = 1;
    while (lanes < 32 && lanes*8 < features) lanes *= 2;
    return lanes;
}
template<class F> void with_norm_lanes(unsigned lanes, F&& f) {
    switch (lanes) {
    case 1: f(std::integral_constant<unsigned, 1>{}); break;
    case 2: f(std::integral_constant<unsigned, 2>{}); break;
    case 4: f(std::integral_constant<unsigned, 4>{}); break;
    case 8: f(std::integral_constant<unsigned, 8>{}); break;
    case 16: f(std::integral_constant<unsigned, 16>{}); break;
    default: f(std::integral_constant<unsigned, 32>{}); break;
    }
}
// Small blocks (two warps) for `rows` rows at `lanes` lanes per row: the row
// kernels are latency bound, so spreading few warps over every SM matters
// more than block size (1024 rows x 64 features: 128 blocks, not 32).
constexpr unsigned norm_block = 64;
unsigned norm_blocks(std::size_t rows, unsigned lanes) {
    return blocks(rows, static_cast<std::size_t>(norm_block/lanes));
}
// Gain/bias VJPs: block (32 features x 8 row lanes) per feature chunk and row
// tile; coalesced along features. Partials are [tile][gain | bias]. Small
// tiles give enough blocks to fill the GPU; the finish sums tiles in order.
constexpr unsigned norm_row_lanes = 8, norm_rows_per_tile = 16, norm_max_tiles = 64;
unsigned norm_tiles(std::size_t rows) {
    return static_cast<unsigned>(std::clamp<std::size_t>((rows+norm_rows_per_tile-1)/norm_rows_per_tile, 1, norm_max_tiles));
}
__global__ void layer_norm_parameter_partial_kernel(const double* x, const double* upstream, const double* stats,
                                                    double* partial, std::size_t rows, std::size_t features, unsigned tiles) {
    __shared__ double gains[norm_row_lanes][32], biases[norm_row_lanes][32];
    const auto f = static_cast<std::size_t>(blockIdx.x)*32+threadIdx.x;
    double g = 0, b = 0;
    if (f < features)
        for (auto row = static_cast<std::size_t>(blockIdx.y)*norm_row_lanes+threadIdx.y; row < rows; row += static_cast<std::size_t>(tiles)*norm_row_lanes) {
            const double u = upstream[row*features+f];
            g += u*detail::layer_norm_normalized(x[row*features+f], stats[2*row], stats[2*row+1]); b += u;
        }
    gains[threadIdx.y][threadIdx.x] = g; biases[threadIdx.y][threadIdx.x] = b;
    __syncthreads();
    if (threadIdx.y == 0 && f < features) {
        double sg = 0, sb = 0;
        for (unsigned lane = 0; lane < norm_row_lanes; ++lane) { sg += gains[lane][threadIdx.x]; sb += biases[lane][threadIdx.x]; }
        partial[blockIdx.y*2*features+f] = sg; partial[blockIdx.y*2*features+features+f] = sb;
    }
}
// One warp per gain/bias parameter: lanes sum strided tiles, then a fixed
// butterfly reduction (independent loads instead of a serial tile chain).
__global__ void layer_norm_parameter_finish_kernel(const double* partial, double* gradient, std::size_t features,
                                                   unsigned tiles, int* status) {
    const auto lane = threadIdx.x%32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for (auto p = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32; p < 2*features; p += stride) {
        double sum = 0;
        for (unsigned tile = lane; tile < tiles; tile += 32) sum += partial[tile*2*features+p];
        sum = group_sum<32>(sum);
        if (lane == 0) { gradient[p] = sum; report(sum, status); }
    }
}

} // namespace

// Device state shared by every layer plan: the arena, stream and status word,
// the double-buffered parameter regions and the per-layer activations. Named
// (not anonymous) because ResidentNetwork::Impl derives from it.
struct ResidentContext {
    std::size_t capacity, batch = 0;
    std::size_t parameters = 0, gradients = 0, candidates = 0;
    // Contraction engine (C2): W = U*C scratch shared by all expansion layers
    // (largest capacity*inputs*terms), a ones vector for the bias VJP and the
    // cuBLAS workspace, so cuBLAS never allocates during execution.
    std::size_t scratch = 0, ones = 0, blas_workspace = 0;
    std::size_t partials = 0; // small parameter VJP tiles, shared like the scratch
    std::vector<std::size_t> activation, upstream;
    double* arena = nullptr;
    int* status = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    double* ptr(std::size_t offset) const { return arena+offset; }
    void sync() { check(cudaStreamSynchronize(stream), "resident synchronize"); }
    void upload(double* destination, std::span<const double> data) {
        if (!data.empty()) check(cudaMemcpyAsync(destination, data.data(), data.size_bytes(), cudaMemcpyHostToDevice, stream), "resident upload");
    }
    void download(std::span<double> destination, const double* data) {
        if (!destination.empty()) check(cudaMemcpyAsync(destination.data(), data, destination.size_bytes(), cudaMemcpyDeviceToHost, stream), "resident download");
    }
};

namespace {
using Context = ResidentContext;

// Bump allocator over the arena, checked before the single device allocation.
// Every region starts on a 256-byte boundary (cudaMalloc alignment), which
// cuBLAS needs for its workspace and prefers for vectorized operand loads.
struct Reservation {
    static constexpr std::size_t alignment = 256/sizeof(double);
    std::size_t total = 0;
    std::size_t operator()(std::size_t count) {
        const auto limit = std::vector<double>().max_size();
        const auto offset = (total+alignment-1)/alignment*alignment;
        if (offset > limit || count > limit - offset) throw std::overflow_error("resident workspace size overflow");
        total = offset+count; return offset;
    }
};
constexpr std::size_t blas_workspace_bytes = std::size_t{4} << 20;

// A layer's parameters occupy [coefficients | bias | nonlinear] at `offset`
// inside each of the parameter, gradient and candidate regions.
struct ParameterBlock {
    std::size_t inputs, outputs, terms, coefficients, nonlinear_count, offset;
    std::size_t bias() const noexcept { return offset+coefficients; }
    std::size_t nonlinear() const noexcept { return offset+coefficients+outputs; }
    std::size_t size() const noexcept { return coefficients+outputs+nonlinear_count; }
};

ParameterBlock parameter_block(const Layer& layer, std::size_t nonlinear_count, std::size_t offset) {
    const auto count = layer.coefficients().size()+layer.outputs(), maximum = std::vector<double>().max_size();
    if (count>maximum || nonlinear_count>maximum-count || count+nonlinear_count>maximum-offset)
        throw std::overflow_error("resident parameter size overflow");
    return {layer.inputs(), layer.outputs(), layer.terms(), layer.coefficients().size(), nonlinear_count, offset};
}

// Execution plans, one per carrier type (the alternatives of kan::Carrier).
// Each plan owns its workspace offsets and launches its own kernels; the
// executor dispatches on the plan once per layer and operation.

// Expansion + contraction engine: basis_kernel writes the rows of Phi and Phi',
// forward_kernel contracts Y = Phi*C^T + b, input_kernel and parameter_kernel
// are its VJPs. Used by BasisEdges and by the coefficients of TrainableRbfEdges.
struct ExpansionPlan {
    ParameterBlock block;
    detail::BasisView view; // host scalars; vector pointers are set per launch
    std::size_t values = 0, derivatives = 0, centers = 0, scales = 0, knots = 0;
};

struct BasisPlan {
    using edges_type = BasisEdges;
    ExpansionPlan expansion;
};

// Adds trainable shared centers/log widths (nonlinear block: centers, then
// log widths) and their tiled reductions.
struct TrainableRbfPlan {
    using edges_type = TrainableRbfEdges;
    ExpansionPlan expansion;
    std::size_t log_derivatives = 0, partials = 0;
    unsigned partial_tiles = 1;
};

// Rational edges with edge-major caches of P, Q and dr/dx, plus g = dQ/dS for
// the safe denominator policies (nonlinear block: denominators).
struct RationalPlan {
    using edges_type = RationalEdges;
    ParameterBlock block;
    RationalConfig config;
    std::size_t values = 0, derivatives = 0, denominator_values = 0, gains = 0;
    bool safe() const { return config.denominator_policy != DenominatorPolicy::Guarded; }
};

// Input-map plans (backlog M1), one per map kind. Trainable LayerNorm gain and
// bias occupy [gain | bias] at `offset` in the parameter, gradient and
// candidate regions, so the shared candidate kernel updates them; fixed map
// parameters live in the workspace.
struct MapBlock {
    std::size_t features, offset, parameters;
    std::size_t half() const noexcept { return parameters/2; }
};
struct AffinePlan {
    using map_type = AffineMap;
    MapBlock block;
    std::size_t scale = 0, shift = 0;
};
struct TanhPlan {
    using map_type = TanhMap;
    MapBlock block;
    double scale;
};
struct LayerNormPlan {
    using map_type = LayerNormMap;
    MapBlock block;
    double epsilon, inverse_count; // inverse_count = 1/features, as on the CPU
    unsigned lanes;                // lanes per row (norm_lanes)
    std::size_t stats = 0, partials = 0; // (capacity, 2) row moments; (tiles, 2*features) partial VJPs
    bool affine() const noexcept { return block.parameters != 0; }
};

using Plan = std::variant<BasisPlan, TrainableRbfPlan, RationalPlan, AffinePlan, TanhPlan, LayerNormPlan>;

const ParameterBlock& block_of(const BasisPlan& p) { return p.expansion.block; }
const ParameterBlock& block_of(const TrainableRbfPlan& p) { return p.expansion.block; }
const ParameterBlock& block_of(const RationalPlan& p) { return p.block; }

// Dimensions and parameter count of any plan.
struct Extent {
    std::size_t inputs, outputs, size;
};
template<class P> Extent extent_of(const P& p) {
    if constexpr (requires { typename P::edges_type; }) {
        const auto& b = block_of(p);
        return {b.inputs, b.outputs, b.size()};
    } else {
        return {p.block.features, p.block.features, p.block.parameters};
    }
}
Extent extent(const Plan& plan) {
    return std::visit([](const auto& p) { return extent_of(p); }, plan);
}

Plan make_plan(const Layer& layer, const BasisEdges& edges, std::size_t offset) {
    return BasisPlan{{parameter_block(layer, 0, offset), detail::basis_view(edges.basis)}};
}
Plan make_plan(const Layer& layer, const TrainableRbfEdges& edges, std::size_t offset) {
    return TrainableRbfPlan{{parameter_block(layer, product(layer.terms(), 2), offset), detail::basis_view(edges.basis)}};
}
Plan make_plan(const Layer& layer, const RationalEdges& edges, std::size_t offset) {
    return RationalPlan{parameter_block(layer, edges.denominators.size(), offset), edges.config};
}
MapBlock map_block(const InputMap& map, std::size_t parameters, std::size_t offset) {
    if (parameters > std::vector<double>().max_size()-offset) throw std::overflow_error("resident parameter size overflow");
    return {map.features(), offset, parameters};
}
Plan make_plan(const InputMap& map, const AffineMap&, std::size_t offset) { return AffinePlan{map_block(map, 0, offset)}; }
Plan make_plan(const InputMap& map, const TanhMap& tanh, std::size_t offset) {
    return TanhPlan{map_block(map, 0, offset), tanh.scale};
}
Plan make_plan(const InputMap& map, const LayerNormMap& norm, std::size_t offset) {
    return LayerNormPlan{map_block(map, product(norm.gain.size(), 2), offset), norm.epsilon,
                         1.0/static_cast<double>(map.features()), norm_lanes(map.features())};
}
Plan make_plan(const Layer& layer, std::size_t offset) {
    finite(layer.coefficients()); finite(layer.bias());
    return std::visit([&](const auto& edges) { return make_plan(layer, edges, offset); }, layer.carrier());
}
Plan make_plan(const InputMap& map, std::size_t offset) {
    return std::visit([&](const auto& kind) { return make_plan(map, kind, offset); }, map.map());
}

// Workspaces, reserved in the order of the arena layout.
void reserve_rows(ExpansionPlan& p, std::size_t capacity, Reservation& reserve) {
    const auto rows = product(product(capacity, p.block.inputs), p.block.terms);
    p.values = reserve(rows); p.derivatives = reserve(rows);
}
void reserve_workspace(BasisPlan& plan, std::size_t capacity, Reservation& reserve) {
    auto& p = plan.expansion;
    reserve_rows(p, capacity, reserve);
    if (p.view.kind == detail::BasisKind::GaussianRbf || p.view.kind == detail::BasisKind::MexicanHat)
        p.centers = reserve(p.block.terms);
    if (p.view.kind == detail::BasisKind::MexicanHat) p.scales = reserve(p.block.terms);
    if (p.view.kind == detail::BasisKind::BSpline) p.knots = reserve(p.block.terms+p.view.degree+1);
}
void reserve_workspace(TrainableRbfPlan& plan, std::size_t capacity, Reservation& reserve) {
    auto& p = plan.expansion;
    reserve_rows(p, capacity, reserve);
    plan.log_derivatives = reserve(product(product(capacity, p.block.inputs), p.block.terms));
    const auto count = product(capacity, p.block.inputs);
    plan.partial_tiles = static_cast<unsigned>(std::min<std::size_t>(nonlinear_tiles, count?((count-1)/256+1):1));
    plan.partials = reserve(product(p.block.terms, 2*plan.partial_tiles));
}
void reserve_workspace(RationalPlan& plan, std::size_t capacity, Reservation& reserve) {
    const auto count = product(product(capacity, plan.block.inputs), plan.block.outputs);
    plan.values = reserve(count); plan.derivatives = reserve(count); plan.denominator_values = reserve(count);
    if (plan.safe()) plan.gains = reserve(count);
}
void reserve_workspace(AffinePlan& plan, std::size_t, Reservation& reserve) {
    plan.scale = reserve(plan.block.features); plan.shift = reserve(plan.block.features);
}
void reserve_workspace(TanhPlan&, std::size_t, Reservation&) {}
void reserve_workspace(LayerNormPlan& plan, std::size_t capacity, Reservation& reserve) {
    plan.stats = reserve(product(capacity, 2));
    if (plan.affine()) plan.partials = reserve(product(plan.block.features, 2*static_cast<std::size_t>(norm_tiles(capacity))));
}

// Elements of the shared W = U*C scratch an expansion layer needs (zero otherwise).
std::size_t scratch_extent(const Plan& plan, std::size_t capacity) {
    return std::visit([&](const auto& p) -> std::size_t {
        using P = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<P, BasisPlan> || std::is_same_v<P, TrainableRbfPlan>)
            return product(product(capacity, p.expansion.block.inputs), p.expansion.block.terms);
        else return 0;
    }, plan);
}

// Elements of the shared small-parameter-VJP partials a layer needs (zero otherwise).
std::size_t partial_extent(const Plan& plan, std::size_t capacity) {
    return std::visit([&](const auto& p) -> std::size_t {
        using P = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<P, BasisPlan> || std::is_same_v<P, TrainableRbfPlan>) {
            const auto checked = p.expansion.block.coefficients+p.expansion.block.outputs;
            return checked <= small_parameter_vjp ? checked*parameter_tile_count(capacity) : 0;
        } else return 0;
    }, plan);
}

// Parameter upload at construction (coefficients and bias are uploaded by the executor).
void upload_carrier(Context& s, const BasisPlan& plan, const BasisEdges& edges) {
    const auto& p = plan.expansion;
    const auto terms = p.block.terms;
    std::visit([&](const auto& c) {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, GaussianRbfConfig> || std::is_same_v<T, MexicanHatConfig>) s.upload(s.ptr(p.centers), {c.centers.data(), terms});
        if constexpr (std::is_same_v<T, MexicanHatConfig>) s.upload(s.ptr(p.scales), {c.scales.data(), terms});
        if constexpr (std::is_same_v<T, BSplineConfig>) s.upload(s.ptr(p.knots), {c.knots.data(), terms+c.degree+1});
    }, edges.basis);
}
void upload_carrier(Context& s, const TrainableRbfPlan& plan, const TrainableRbfEdges& edges) {
    const auto& b = plan.expansion.block;
    s.upload(s.ptr(s.parameters+b.nonlinear()), edges.basis.centers);
    s.upload(s.ptr(s.parameters+b.nonlinear()+b.terms), edges.basis.log_widths);
}
void upload_carrier(Context& s, const RationalPlan& plan, const RationalEdges& edges) {
    s.upload(s.ptr(s.parameters+plan.block.nonlinear()), edges.denominators);
}

// Construction upload of one network layer: a KAN layer's coefficients, bias
// and carrier state, or an input map's fixed and trainable parameters.
template<class P, class Edges>
void upload_stage(Context& s, const P& plan, const Layer& layer, const Edges& edges) {
    const auto& b = block_of(plan);
    s.upload(s.ptr(s.parameters+b.offset), layer.coefficients());
    s.upload(s.ptr(s.parameters+b.bias()), layer.bias());
    upload_carrier(s, plan, edges);
}
void upload_stage(Context& s, const AffinePlan& plan, const InputMap&, const AffineMap& map) {
    s.upload(s.ptr(plan.scale), map.scale); s.upload(s.ptr(plan.shift), map.shift);
}
void upload_stage(Context&, const TanhPlan&, const InputMap&, const TanhMap&) {}
void upload_stage(Context& s, const LayerNormPlan& plan, const InputMap&, const LayerNormMap& map) {
    s.upload(s.ptr(s.parameters+plan.block.offset), map.gain);
    s.upload(s.ptr(s.parameters+plan.block.offset+plan.block.half()), map.bias);
}

// Forward of layer j: activation[j] -> activation[j+1].
void expansion_forward(Context& s, const ExpansionPlan& p, std::size_t j, double* log_derivatives,
                       const double* centers, const double* log_widths) {
    const auto& b = p.block;
    // Same scalars as the host view; vectors point into device storage.
    auto basis = p.view;
    basis.centers = centers; basis.log_widths = log_widths;
    basis.scales = s.ptr(p.scales); basis.knots = s.ptr(p.knots);
    detail::visit_basis_family(basis.kind, [&](auto family) {
        basis_kernel<decltype(family)::value><<<blocks(s.batch*b.inputs), 256, 0, s.stream>>>(
            s.ptr(s.activation[j]), s.ptr(p.values), s.ptr(p.derivatives), log_derivatives, s.batch*b.inputs, basis, s.status);
    });
    check(cudaGetLastError(), "resident basis launch");
    const auto count = s.batch*b.outputs, length = b.inputs*b.terms;
    if (count <= small_forward_contraction/length) {
        forward_dot_kernel<<<blocks(count, 256/32), 256, 0, s.stream>>>(s.ptr(p.values), s.ptr(s.parameters+b.offset),
            s.ptr(s.parameters+b.bias()), s.ptr(s.activation[j+1]), count, b.outputs, length, s.status);
        check(cudaGetLastError(), "resident forward contraction launch");
        return;
    }
    // Row-major Y (batch x O) = Phi (batch x IK) * C^T, i.e. column-major
    // Y^T = C^T(op T) * Phi^T with C stored as column-major IK x O.
    const auto ik = static_cast<std::int64_t>(b.inputs*b.terms), o = static_cast<std::int64_t>(b.outputs);
    const double one = 1, zero = 0;
    check(cublasDgemm_64(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, o, static_cast<std::int64_t>(s.batch), ik, &one,
                         s.ptr(s.parameters+b.offset), ik, s.ptr(p.values), ik, &zero, s.ptr(s.activation[j+1]), o),
          "resident forward contraction");
    bias_kernel<<<blocks(s.batch*b.outputs), 256, 0, s.stream>>>(s.ptr(s.activation[j+1]), s.ptr(s.parameters+b.bias()),
        s.batch*b.outputs, b.outputs, s.status);
    check(cudaGetLastError(), "resident forward bias launch");
}
void run_forward(Context& s, const BasisPlan& plan, std::size_t j) {
    const auto& p = plan.expansion;
    expansion_forward(s, p, j, nullptr, s.ptr(p.centers), nullptr);
}
void run_forward(Context& s, const TrainableRbfPlan& plan, std::size_t j) {
    const auto nonlinear = s.parameters+plan.expansion.block.nonlinear();
    expansion_forward(s, plan.expansion, j, s.ptr(plan.log_derivatives), s.ptr(nonlinear), s.ptr(nonlinear+plan.expansion.block.terms));
}
void run_forward(Context& s, const RationalPlan& plan, std::size_t j) {
    const auto& b = plan.block;
    double* gains = plan.safe() ? s.ptr(plan.gains) : nullptr;
    detail::visit_denominator_policy(plan.config.denominator_policy, [&](auto policy) {
        rational_forward_kernel<decltype(policy)::value><<<blocks(s.batch*b.outputs),256,0,s.stream>>>(
            s.ptr(s.activation[j]),s.ptr(s.parameters+b.offset),
            s.ptr(s.parameters+b.nonlinear()),s.ptr(s.parameters+b.bias()),
            s.ptr(plan.values),s.ptr(plan.denominator_values),s.ptr(plan.derivatives),gains,s.ptr(s.activation[j+1]),
            s.batch,s.capacity,b.inputs,b.outputs,plan.config,s.status);
    });
    check(cudaGetLastError(),"resident rational forward launch");
}
void run_forward(Context& s, const AffinePlan& plan, std::size_t j) {
    const auto count = s.batch*plan.block.features;
    affine_forward_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.activation[j]), s.ptr(plan.scale), s.ptr(plan.shift),
        s.ptr(s.activation[j+1]), count, plan.block.features, s.status);
    check(cudaGetLastError(), "resident affine map launch");
}
void run_forward(Context& s, const TanhPlan& plan, std::size_t j) {
    const auto count = s.batch*plan.block.features;
    tanh_forward_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.activation[j]), s.ptr(s.activation[j+1]), count, plan.scale);
    check(cudaGetLastError(), "resident tanh map launch");
}
// Trainable gain and bias pointers (null without them).
const double* norm_gain(const Context& s, const LayerNormPlan& plan, std::size_t region) {
    return plan.affine() ? s.ptr(region+plan.block.offset) : nullptr;
}
void run_forward(Context& s, const LayerNormPlan& plan, std::size_t j) {
    const auto gain = norm_gain(s, plan, s.parameters);
    with_norm_lanes(plan.lanes, [&](auto lanes) {
        layer_norm_forward_kernel<decltype(lanes)::value><<<norm_blocks(s.batch, plan.lanes), norm_block, 0, s.stream>>>(
            s.ptr(s.activation[j]), gain, gain ? gain+plan.block.half() : nullptr, s.ptr(s.activation[j+1]),
            s.ptr(plan.stats), s.batch, plan.block.features, plan.inverse_count, plan.epsilon, s.status);
    });
    check(cudaGetLastError(), "resident layer norm launch");
}

// Backward of layer j: upstream[j+1] -> upstream[j] and the parameter gradients.
void expansion_backward(Context& s, const ExpansionPlan& p, std::size_t j, double lambda) {
    const auto& b = p.block;
    const auto ik = static_cast<std::int64_t>(b.inputs*b.terms), o = static_cast<std::int64_t>(b.outputs);
    const auto n = static_cast<std::int64_t>(s.batch);
    const double* c = s.ptr(s.parameters+b.offset);
    const double* u = s.ptr(s.upstream[j+1]);
    double* dc = s.ptr(s.gradients+b.offset);
    double* db = s.ptr(s.gradients+b.bias());
    const double one = 1, zero = 0;
    const auto checked = b.coefficients+b.outputs;
    const bool small = s.batch && checked <= small_parameter_vjp;
    const auto tiles = parameter_tile_count(s.batch);
    if (small) {
        // Partial sums; backward_finish_kernel adds them with lambda*C.
        const auto chunk = (s.batch-1)/tiles+1;
        parameter_partial_kernel<<<dim3(blocks(checked), tiles), 256, 0, s.stream>>>(s.ptr(p.values), u, s.ptr(s.partials),
            s.batch, b.outputs, b.inputs*b.terms, chunk);
        check(cudaGetLastError(), "resident parameter partial launch");
    } else if (s.batch) {
        // Coefficient VJP + L2: column-major dC^T (IK x O) = Phi^T * U + lambda*C^T.
        // beta = 0 never reads the output, so lambda = 0 needs no copy.
        if (lambda != 0) check(cudaMemcpyAsync(dc, c, b.coefficients*sizeof(double), cudaMemcpyDeviceToDevice, s.stream), "resident L2 copy");
        check(cublasDgemm_64(s.blas, CUBLAS_OP_N, CUBLAS_OP_T, ik, o, n, &one, s.ptr(p.values), ik, u, o,
                             &lambda, dc, ik), "resident coefficient VJP");
        // Bias VJP: column sums of U, as U^T * 1 (U^T is column-major O x batch).
        check(cublasDgemv_64(s.blas, CUBLAS_OP_N, o, n, &one, u, o, s.ptr(s.ones), 1, &zero, db, 1), "resident bias VJP");
    } else {
        if (lambda != 0) {
            check(cudaMemcpyAsync(dc, c, b.coefficients*sizeof(double), cudaMemcpyDeviceToDevice, s.stream), "resident L2 copy");
            check(cublasDscal_64(s.blas, static_cast<std::int64_t>(b.coefficients), &lambda, dc, 1), "resident L2 scale");
        } else {
            check(cudaMemsetAsync(dc, 0, b.coefficients*sizeof(double), s.stream), "resident coefficient VJP reset");
        }
        check(cudaMemsetAsync(db, 0, b.outputs*sizeof(double), s.stream), "resident bias VJP reset");
    }
    if (s.batch) {
        // Input VJP: column-major W^T (IK x batch) = C^T * U^T, then dx = sum_k Phi' * W.
        check(cublasDgemm_64(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, ik, n, o, &one, c, ik, u, o, &zero, s.ptr(s.scratch), ik),
              "resident input VJP contraction");
    }
    const auto rows = s.batch*b.inputs;
    backward_finish_kernel<<<blocks(rows+checked), 256, 0, s.stream>>>(s.ptr(p.derivatives), s.ptr(s.scratch),
        s.ptr(s.upstream[j]), rows, b.terms, dc, checked, small ? s.ptr(s.partials) : nullptr, tiles, c,
        b.coefficients, lambda, s.status);
    check(cudaGetLastError(), "resident backward finish launch");
}

void run_backward(Context& s, const BasisPlan& plan, std::size_t j, double lambda) { expansion_backward(s, plan.expansion, j, lambda); }
void run_backward(Context& s, const TrainableRbfPlan& plan, std::size_t j, double lambda) {
    const auto& p = plan.expansion;
    const auto& b = p.block;
    expansion_backward(s, p, j, lambda);
    // Reuses W = U*C left in the scratch by expansion_backward (zero rows for an empty batch).
    const auto rows=product(s.batch,b.inputs);
    const auto tiles=static_cast<unsigned>(std::min<std::size_t>(plan.partial_tiles,rows?((rows-1)/256+1):1));
    // Bound the launch dimension even for large valid basis term counts.
    if(b.terms>2147483647U/tiles)throw std::overflow_error("resident nonlinear launch size overflow");
    nonlinear_partial_kernel<<<static_cast<unsigned>(b.terms)*tiles,256,0,s.stream>>>(s.ptr(p.derivatives),s.ptr(plan.log_derivatives),
        s.ptr(s.scratch),s.ptr(plan.partials),rows,b.terms,tiles,s.status);
    check(cudaGetLastError(),"resident nonlinear partial launch");
    nonlinear_finish_kernel<<<blocks(2*b.terms),256,0,s.stream>>>(s.ptr(plan.partials),s.ptr(s.gradients+b.nonlinear()),b.terms,tiles,s.status);
    check(cudaGetLastError(),"resident nonlinear reduction launch");
}
void run_backward(Context& s, const RationalPlan& plan, std::size_t j, double lambda) {
    const auto& b = plan.block;
    if(s.batch) {
        rational_input_kernel<<<blocks(s.batch*b.inputs),256,0,s.stream>>>(s.ptr(plan.derivatives),s.ptr(s.upstream[j+1]),s.ptr(s.upstream[j]),
            s.batch,s.capacity,b.inputs,b.outputs,s.status);
        check(cudaGetLastError(),"resident rational input gradient launch");
    }
    const double* gains = plan.safe() ? s.ptr(plan.gains) : nullptr;
    detail::visit_denominator_policy(plan.config.denominator_policy, [&](auto policy) {
        rational_parameter_kernel<decltype(policy)::value><<<blocks(b.size(),8),256,0,s.stream>>>(
            s.ptr(s.activation[j]),s.ptr(plan.values),s.ptr(plan.denominator_values),gains,
            s.ptr(s.upstream[j+1]),s.ptr(s.parameters+b.offset),s.ptr(s.gradients+b.offset),s.batch,s.capacity,b.inputs,b.outputs,
            plan.config,lambda,s.status);
    });
    check(cudaGetLastError(),"resident rational parameter gradient launch");
}
// Input maps are not penalized by the coefficient L2 (lambda unused).
void run_backward(Context& s, const AffinePlan& plan, std::size_t j, double) {
    const auto count = s.batch*plan.block.features;
    if (!count) return;
    affine_input_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.upstream[j+1]), s.ptr(plan.scale), s.ptr(s.upstream[j]),
        count, plan.block.features, s.status);
    check(cudaGetLastError(), "resident affine map gradient launch");
}
void run_backward(Context& s, const TanhPlan& plan, std::size_t j, double) {
    const auto count = s.batch*plan.block.features;
    if (!count) return;
    tanh_input_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.activation[j+1]), s.ptr(s.upstream[j+1]), s.ptr(s.upstream[j]),
        count, plan.scale, s.status);
    check(cudaGetLastError(), "resident tanh map gradient launch");
}
void run_backward(Context& s, const LayerNormPlan& plan, std::size_t j, double) {
    const auto features = plan.block.features;
    if (s.batch) {
        with_norm_lanes(plan.lanes, [&](auto lanes) {
            layer_norm_input_kernel<decltype(lanes)::value><<<norm_blocks(s.batch, plan.lanes), norm_block, 0, s.stream>>>(
                s.ptr(s.activation[j]), s.ptr(s.upstream[j+1]), norm_gain(s, plan, s.parameters), s.ptr(plan.stats),
                s.ptr(s.upstream[j]), s.batch, features, plan.inverse_count, s.status);
        });
        check(cudaGetLastError(), "resident layer norm gradient launch");
    }
    if (!plan.affine()) return;
    const auto tiles = norm_tiles(s.batch);
    const auto chunks = (features+31)/32;
    if (chunks > 2147483647U) throw std::overflow_error("resident layer norm launch size overflow");
    layer_norm_parameter_partial_kernel<<<dim3(static_cast<unsigned>(chunks), tiles), dim3(32, norm_row_lanes), 0, s.stream>>>(
        s.ptr(s.activation[j]), s.ptr(s.upstream[j+1]), s.ptr(plan.stats), s.ptr(plan.partials), s.batch, features, tiles);
    check(cudaGetLastError(), "resident layer norm parameter launch");
    layer_norm_parameter_finish_kernel<<<blocks(2*features, 256/32), 256, 0, s.stream>>>(s.ptr(plan.partials),
        s.ptr(s.gradients+plan.block.offset), features, tiles, s.status);
    check(cudaGetLastError(), "resident layer norm parameter reduction launch");
}

// Candidate validation beyond finiteness, before the SGD commit.
void validate_candidates(Context&, const BasisPlan&) {}
void validate_candidates(Context& s, const TrainableRbfPlan& plan) {
    const auto& b = plan.expansion.block;
    validate_width_kernel<<<blocks(b.terms),256,0,s.stream>>>(s.ptr(s.candidates+b.nonlinear()+b.terms),b.terms,s.status);
    check(cudaGetLastError(),"resident width validation launch");
}
void validate_candidates(Context&, const RationalPlan&) {}
void validate_candidates(Context&, const AffinePlan&) {}
void validate_candidates(Context&, const TanhPlan&) {}
void validate_candidates(Context&, const LayerNormPlan&) {}

// Nonlinear gradients, downloaded asynchronously (the caller synchronizes).
// The destination vectors are moved, never copied, into the result, so the
// heap buffers the pending copies target stay the same.
NonlinearGradients download_nonlinear(Context&, const BasisPlan&) { return std::monostate{}; }
NonlinearGradients download_nonlinear(Context& s, const TrainableRbfPlan& plan) {
    const auto& b = plan.expansion.block;
    TrainableRbfGradients g{std::vector<double>(b.terms), std::vector<double>(b.terms)};
    s.download(g.centers,s.ptr(s.gradients+b.nonlinear()));
    s.download(g.log_widths,s.ptr(s.gradients+b.nonlinear()+b.terms));
    return g;
}
NonlinearGradients download_nonlinear(Context& s, const RationalPlan& plan) {
    RationalGradients g{std::vector<double>(plan.block.nonlinear_count)};
    s.download(g.denominators,s.ptr(s.gradients+plan.block.nonlinear()));
    return g;
}

// The trained carrier: the snapshot's configuration with current parameters.
Carrier download_carrier(Context&, const BasisPlan&, const BasisEdges& snapshot, std::vector<double> coefficients) {
    return BasisEdges{snapshot.basis, std::move(coefficients)};
}
Carrier download_carrier(Context& s, const TrainableRbfPlan& plan, const TrainableRbfEdges&, std::vector<double> coefficients) {
    const auto& b = plan.expansion.block;
    TrainableRbfConfig basis{std::vector<double>(b.terms), std::vector<double>(b.terms)};
    s.download(basis.centers,s.ptr(s.parameters+b.nonlinear()));
    s.download(basis.log_widths,s.ptr(s.parameters+b.nonlinear()+b.terms));
    return TrainableRbfEdges{std::move(basis), std::move(coefficients)};
}
Carrier download_carrier(Context& s, const RationalPlan& plan, const RationalEdges& snapshot, std::vector<double> coefficients) {
    std::vector<double> denominators(plan.block.nonlinear_count);
    s.download(denominators,s.ptr(s.parameters+plan.block.nonlinear()));
    return RationalEdges{snapshot.config, std::move(coefficients), std::move(denominators)};
}

// Gradients of one network layer, downloaded asynchronously (the caller
// synchronizes); destination vectors are moved, never copied, into the result.
template<class P> requires requires { typename P::edges_type; }
NetworkLayerGradients download_stage_gradients(Context& s, const P& plan, std::size_t j) {
    const auto& b = block_of(plan);
    LayerGradients g;
    g.input.resize(product(s.batch, b.inputs)); g.coefficients.resize(b.coefficients); g.bias.resize(b.outputs);
    s.download(g.input, s.ptr(s.upstream[j])); s.download(g.coefficients, s.ptr(s.gradients+b.offset));
    s.download(g.bias, s.ptr(s.gradients+b.bias()));
    g.nonlinear = download_nonlinear(s, plan);
    return NetworkLayerGradients(std::move(g));
}
NetworkLayerGradients map_gradients(Context& s, const MapBlock& b, std::size_t j) {
    InputMapGradients g{std::vector<double>(product(s.batch, b.features)), std::vector<double>(b.half()), std::vector<double>(b.half())};
    s.download(g.input, s.ptr(s.upstream[j]));
    s.download(g.gain, s.ptr(s.gradients+b.offset));
    s.download(g.bias, s.ptr(s.gradients+b.offset+b.half()));
    return NetworkLayerGradients(std::move(g));
}
NetworkLayerGradients download_stage_gradients(Context& s, const AffinePlan& plan, std::size_t j) { return map_gradients(s, plan.block, j); }
NetworkLayerGradients download_stage_gradients(Context& s, const TanhPlan& plan, std::size_t j) { return map_gradients(s, plan.block, j); }
NetworkLayerGradients download_stage_gradients(Context& s, const LayerNormPlan& plan, std::size_t j) { return map_gradients(s, plan.block, j); }

// The trained network layer: the snapshot with the current device parameters.
template<class P, class Edges>
NetworkLayer download_stage(Context& s, const P& plan, const Layer& snapshot, const Edges& edges) {
    const auto& b = block_of(plan);
    std::vector<double> coefficients(b.coefficients), bias(b.outputs);
    s.download(coefficients, s.ptr(s.parameters+b.offset)); s.download(bias, s.ptr(s.parameters+b.bias()));
    auto carrier = download_carrier(s, plan, edges, std::move(coefficients));
    s.sync();
    Layer layer = snapshot;
    layer.set_carrier(std::move(carrier), bias);
    return layer;
}
NetworkLayer download_stage(Context&, const AffinePlan&, const InputMap& snapshot, const AffineMap&) { return snapshot; }
NetworkLayer download_stage(Context&, const TanhPlan&, const InputMap& snapshot, const TanhMap&) { return snapshot; }
NetworkLayer download_stage(Context& s, const LayerNormPlan& plan, const InputMap& snapshot, const LayerNormMap& norm) {
    if (!plan.affine()) return snapshot;
    LayerNormMap trained{norm.epsilon, std::vector<double>(plan.block.half()), std::vector<double>(plan.block.half())};
    s.download(trained.gain, s.ptr(s.parameters+plan.block.offset));
    s.download(trained.bias, s.ptr(s.parameters+plan.block.offset+plan.block.half()));
    s.sync();
    InputMap map = snapshot;
    map.set_map(std::move(trained));
    return map;
}

// Applies f(plan, stage, kind) to a plan, its network layer and the matching
// carrier (KAN layer) or map (input map) alternative.
template<class F> decltype(auto) with_stage(const Plan& plan, const NetworkLayer& stage, F&& f) {
    return std::visit([&](const auto& p) -> decltype(auto) {
        using P = std::decay_t<decltype(p)>;
        if constexpr (requires { typename P::edges_type; }) {
            const auto& layer = std::get<Layer>(stage);
            return f(p, layer, std::get<typename P::edges_type>(layer.carrier()));
        } else {
            const auto& map = std::get<InputMap>(stage);
            return f(p, map, std::get<typename P::map_type>(map.map()));
        }
    }, plan);
}
}

struct ResidentNetwork::Impl : ResidentContext {
    Network model;
    std::size_t allocations = 0, parameter_count = 0;
    std::vector<Plan> plans;
    bool has_input = false, has_upstream = false, has_forward = false, has_backward = false;
    explicit Impl(const Network& source, std::size_t maximum) : ResidentContext{maximum}, model(source) {
        if (model.layers().empty()) throw std::invalid_argument("resident network is empty or moved from");
        // Validate the copied CPU state and all shape arithmetic before CUDA allocation.
        model.forward({}, 0);
        for (const auto& stage : model.layers()) {
            plans.push_back(std::visit([&](const auto& s) { return make_plan(s, parameter_count); }, stage));
            parameter_count += extent(plans.back()).size;
        }
        Reservation reserve;
        parameters = reserve(parameter_count); gradients = reserve(parameter_count); candidates = reserve(parameter_count);
        activation.push_back(reserve(product(capacity, extent(plans.front()).inputs)));
        upstream.push_back(reserve(product(capacity, extent(plans.front()).inputs)));
        for (auto& plan : plans) {
            const auto outputs = extent(plan).outputs;
            activation.push_back(reserve(product(capacity, outputs)));
            upstream.push_back(reserve(product(capacity, outputs)));
            std::visit([&](auto& p) { reserve_workspace(p, capacity, reserve); }, plan);
        }
        std::size_t scratch_size = 0;
        for (const auto& plan : plans) scratch_size = std::max(scratch_size, scratch_extent(plan, capacity));
        std::size_t partial_size = 0;
        for (const auto& plan : plans) partial_size = std::max(partial_size, partial_extent(plan, capacity));
        scratch = reserve(scratch_size); partials = reserve(partial_size); ones = reserve(capacity);
        blas_workspace = reserve(blas_workspace_bytes/sizeof(double));
        const auto bytes = product(reserve.total, sizeof(double));
        try {
            if (!available()) throw std::runtime_error("no CUDA device available");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "resident stream create");
            check(cudaMalloc(&arena, bytes), "resident arena allocation"); ++allocations;
            check(cudaMalloc(&status, sizeof(int)), "resident status allocation"); ++allocations;
            // The handle runs on the network stream with a workspace inside the
            // arena (set after the stream: cublasSetStream resets it).
            check(cublasCreate(&blas), "resident cuBLAS handle create");
            check(cublasSetStream(blas, stream), "resident cuBLAS stream");
            check(cublasSetWorkspace(blas, ptr(blas_workspace), blas_workspace_bytes), "resident cuBLAS workspace");
            if (capacity) {
                fill_kernel<<<blocks(capacity), 256, 0, stream>>>(ptr(ones), capacity, 1.0);
                check(cudaGetLastError(), "resident ones launch");
            }
            for (std::size_t j = 0; j < plans.size(); ++j)
                with_stage(plans[j], model.layers()[j], [&](const auto& p, const auto& stage, const auto& kind) {
                    upload_stage(*this, p, stage, kind);
                });
            sync();
        } catch (...) { cleanup(); throw; }
    }
    ~Impl() { cleanup(); }
    void cleanup() noexcept {
        if (stream) cudaStreamSynchronize(stream);
        if (blas) cublasDestroy(blas);
        if (arena) cudaFree(arena);
        if (status) cudaFree(status);
        if (stream) cudaStreamDestroy(stream);
        arena = nullptr; status = nullptr; stream = nullptr; blas = nullptr;
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
    auto& s = state(); const auto count = product(batch, extent(s.plans.front()).inputs);
    if (batch > s.capacity || input.size() != count) throw std::invalid_argument("resident input shape or capacity mismatch");
    finite(input);
    s.upload(s.ptr(s.activation.front()), input); s.sync();
    s.batch = batch; s.has_input = true; s.has_upstream = s.has_forward = s.has_backward = false;
}
void ResidentNetwork::upload_output_gradient(std::span<const double> gradient) {
    auto& s = state();
    if (!s.has_input) throw std::logic_error("resident input must be uploaded first");
    if (gradient.size() != product(s.batch, extent(s.plans.back()).outputs)) throw std::invalid_argument("resident upstream shape mismatch");
    finite(gradient); s.upload(s.ptr(s.upstream.back()), gradient); s.sync();
    s.has_upstream = true; s.has_backward = false;
}
void ResidentNetwork::forward() {
    auto& s = state();
    if (!s.has_input) throw std::logic_error("resident input has not been uploaded");
    s.has_forward = s.has_backward = false; s.reset_status();
    for (std::size_t j = 0; j < s.plans.size() && s.batch; ++j)
        std::visit([&](const auto& plan) { run_forward(s, plan, j); }, s.plans[j]);
    s.result(); s.has_forward = true;
}
void ResidentNetwork::backward(double coefficient_l2) {
    auto& s = state();
    if(!std::isfinite(coefficient_l2)||coefficient_l2<0)throw std::invalid_argument("coefficient L2 must be finite and nonnegative");
    if (!s.has_forward || !s.has_upstream) throw std::logic_error("resident backward requires current forward and upstream");
    s.has_backward = false; s.reset_status();
    for (std::size_t j = s.plans.size(); j-- > 0;)
        std::visit([&](const auto& plan) { run_backward(s, plan, j, coefficient_l2); }, s.plans[j]);
    s.result(); s.has_backward = true;
}
void ResidentNetwork::sgd(double learning_rate) {
    auto& s = state();
    if (!std::isfinite(learning_rate) || learning_rate <= 0) throw std::invalid_argument("learning rate must be finite and positive");
    if (!s.has_backward) throw std::logic_error("resident SGD requires current gradients");
    s.reset_status();
    if (s.parameter_count) { // zero only for networks of fixed input maps
        candidate_kernel<<<blocks(s.parameter_count), 256, 0, s.stream>>>(s.ptr(s.parameters), s.ptr(s.gradients), s.ptr(s.candidates), s.parameter_count, learning_rate, s.status);
        check(cudaGetLastError(),"resident candidate launch");
    }
    for (const auto& plan : s.plans) std::visit([&](const auto& p) { validate_candidates(s, p); }, plan);
    s.result(); // All layers validated before any parameter mutation.
    // Both regions are permanently reserved and candidate execution is complete.
    // Changing the active region commits the whole network without a tensor copy.
    std::swap(s.parameters, s.candidates);
    s.has_forward = s.has_backward = false;
}
std::vector<double> ResidentNetwork::download_output() {
    auto& s = state();
    if (!s.has_forward) throw std::logic_error("resident output requires current forward");
    std::vector<double> result(product(s.batch, extent(s.plans.back()).outputs));
    s.download(result, s.ptr(s.activation.back())); s.sync(); return result;
}
NetworkGradients ResidentNetwork::download_gradients() {
    auto& s = state();
    if (!s.has_backward) throw std::logic_error("resident gradients require current backward");
    NetworkGradients result; result.layers.reserve(s.plans.size());
    for (std::size_t j = 0; j < s.plans.size(); ++j)
        result.layers.push_back(std::visit([&](const auto& plan) { return download_stage_gradients(s, plan, j); }, s.plans[j]));
    s.sync();
    result.input = std::visit([](const auto& g) { return g.input; }, result.layers.front());
    return result;
}
Network ResidentNetwork::download_parameters() {
    auto& s = state(); std::vector<NetworkLayer> layers;
    layers.reserve(s.plans.size());
    for (std::size_t j = 0; j < s.plans.size(); ++j)
        layers.push_back(with_stage(s.plans[j], s.model.layers()[j], [&](const auto& plan, const auto& stage, const auto& kind) {
            return download_stage(s, plan, stage, kind);
        }));
    return Network(std::move(layers));
}
void ResidentNetwork::synchronize() { state().sync(); }
std::size_t ResidentNetwork::capacity() const { return state().capacity; }
std::size_t ResidentNetwork::batch() const { return state().batch; }
std::size_t ResidentNetwork::workspace_allocations() const { return state().allocations; }
} // namespace kan::cuda

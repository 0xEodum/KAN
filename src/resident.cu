#include "kan/resident.hpp"
#include "detail/basis_view.hpp"
#include "detail/rational_formulas.hpp"
#include "detail/input_map_formulas.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

// Precision policy (backlog C1): every kernel, plan and the execution engine
// are templates over the device scalar T. Engine<double> is the FP64 executor
// (the parity reference; its operations are those of C2), Engine<float> the
// opt-in FP32 executor, optionally with TF32 tensor-op cuBLAS GEMMs. Host
// data and the public API stay double.

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

// Conversion of a finite host double to the executor scalar. FP32 rejects
// magnitudes beyond FLT_MAX (they would become infinite) with
// std::invalid_argument; values below the FP32 range round (to subnormals or
// zero) like any other value.
template<class T> T narrow(double value) {
    if constexpr (std::is_same_v<T, double>) {
        return value;
    } else {
        if (!(std::abs(value) <= static_cast<double>(FLT_MAX)))
            throw std::invalid_argument("resident data is not representable in float32");
        return static_cast<float>(value);
    }
}
template<class T> T narrow_positive(double value) {
    const T result = narrow<T>(value);
    if (!(result > 0)) throw std::invalid_argument("resident float32 configuration rounds to a nonpositive value");
    return result;
}
template<class T> void representable(std::span<const double> values) {
    if constexpr (!std::is_same_v<T, double>) for (double v : values) narrow<T>(v);
}

template<class T> __device__ void report(T value, int* status) { if (!isfinite(value)) atomicOr(status, 1); }
// Device guard for the shared formulas: record a nonfinite status bit and
// continue; the host raises after the launch sequence completes.
struct StatusGuard {
    int* status;
    template<class T> __device__ T operator()(T value) const { report(value, status); return value; }
};
// Identity guard for recomputing values that an earlier kernel of the same
// step already checked with StatusGuard (keeps hot reduction loops lean).
struct CheckedEarlier {
    template<class T> __device__ T operator()(T value) const { return value; }
};
// Row tiles staged in shared memory (backlog C1 profiling). A row of Phi,
// Phi' or W holds `terms` contiguous values; one thread per row reading or
// writing it directly makes every warp access 32 addresses `terms` elements
// apart (uncoalesced; about 7x the DRAM traffic of the 256-wide FP32 step).
// Kernels instead move tiles of `tile_rows` consecutive rows, i.e. contiguous
// tile_rows*terms elements, between global and shared memory with coalesced
// accesses, and each thread works on its row in shared memory. The arithmetic
// and summation order are unchanged. tile_rows = 0 (rows too long for the
// shared budget) keeps the direct per-thread access.
constexpr unsigned stage_threads = 256;
constexpr std::size_t stage_bytes = std::size_t{32} << 10;
// FP64 keeps the direct access (tile_rows = 0): its basis kernels are FP64-ALU
// bound on GA102, and the 32 KiB stage per block costs them occupancy (m2
// Gaussian RBF resident 1.9x slower); FP32 kernels are memory bound.
template<class T> unsigned stage_rows(std::size_t terms, std::size_t planes) {
    if constexpr (std::is_same_v<T, double>) {
        return 0;
    } else {
        const auto rows = std::min<std::size_t>(stage_threads, stage_bytes/(terms*planes*sizeof(T)))/32*32;
        return static_cast<unsigned>(rows);
    }
}
template<class T> __device__ T* stage_memory() {
    extern __shared__ __align__(16) unsigned char stage_raw[];
    return reinterpret_cast<T*>(stage_raw);
}

// derivatives = null (staged path only, backlog C3): Phi' is evaluated into the
// stage but not written; the backward pass recomputes it.
template<detail::BasisKind Kind, class T>
__global__ void basis_kernel(const T* input, T* values, T* derivatives, T* log_derivatives,
                             std::size_t count, detail::BasisViewOf<T> basis, unsigned tile_rows, int* status) {
    const StatusGuard guard{status};
    const auto terms = basis.terms;
    if (tile_rows == 0) {
        const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
            const auto row = index * terms;
            const detail::BasisRowOf<T> out{values + row, derivatives + row, nullptr,
                                            basis.trainable ? log_derivatives + row : nullptr};
            detail::basis_terms_for<Kind>(basis, input[index], out, guard);
        }
        return;
    }
    T* stage = stage_memory<T>();
    const auto plane = static_cast<std::size_t>(tile_rows) * terms;
    for (auto base = static_cast<std::size_t>(blockIdx.x) * tile_rows; base < count;
         base += static_cast<std::size_t>(gridDim.x) * tile_rows) {
        const auto rows = count - base < tile_rows ? count - base : static_cast<std::size_t>(tile_rows);
        if (threadIdx.x < rows) {
            const auto row = threadIdx.x * terms;
            const detail::BasisRowOf<T> out{stage + row, stage + plane + row, nullptr,
                                            basis.trainable ? stage + 2 * plane + row : nullptr};
            detail::basis_terms_for<Kind>(basis, input[base + threadIdx.x], out, guard);
        }
        __syncthreads();
        const auto length = rows * terms, offset = base * terms;
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < length; i += blockDim.x) {
            values[offset + i] = stage[i];
            if (derivatives) derivatives[offset + i] = stage[plane + i];
            if (basis.trainable) log_derivatives[offset + i] = stage[2 * plane + i];
        }
        __syncthreads();
    }
}
// Contraction engine epilogues (backlog C2). cuBLAS computes the products:
// forward Y = Phi*C^T, coefficient VJP dC = U^T*Phi (+ lambda*C), bias VJP
// db = U^T*1 and W = U*C; these kernels add the bias, reduce W against Phi'
// to the input VJP and report nonfinite results.
template<class T>
__global__ void bias_kernel(T* output, const T* bias, std::size_t count, std::size_t outputs, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        const T sum = output[index]+bias[index%outputs];
        output[index] = sum; report(sum, status);
    }
}
// Small forward contractions: one warp per output, Y[b,o] = bias[o] +
// dot(Phi[b,:], C[o,:]) over two contiguous rows, with the bias and the
// nonfinite check fused. cuBLAS runs a tiny GEMM as one latency-bound CTA
// (42-64 us at 24x32x112 on the RTX 3090; this kernel: 7 us); see C2 evidence.
template<class T>
__global__ void forward_dot_kernel(const T* v, const T* c, const T* bias, T* output,
                                   std::size_t count, std::size_t outputs, std::size_t length, int* status) {
    const auto lane = threadIdx.x%32;
    const auto warps = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for (auto index = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32; index < count; index += warps) {
        const T* row = v+(index/outputs)*length;
        const T* column = c+(index%outputs)*length;
        T sum = 0;
        for (std::size_t k = lane; k < length; k += 32) sum += row[k]*column[k];
        for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (lane == 0) { sum += bias[index%outputs]; output[index] = sum; report(sum, status); }
    }
}
// Largest forward contraction (batch*outputs*inputs*terms multiply-adds) run by
// forward_dot_kernel; cuBLAS above. FP64: both are FP64-ALU bound on GA102 and
// meet at a few million multiply-adds (C2 evidence, dot_vs_gemm). FP32: the
// warp kernel wins up to about 14M multiply-adds (80x400x448: 22 vs 29 us),
// SGEMM above (64x1024x448: 20 vs 39 us; C1 evidence, small_shapes_f32).
template<class T> constexpr std::size_t small_forward_contraction = std::size_t{1} << 23;
template<> constexpr std::size_t small_forward_contraction<float> = std::size_t{1} << 24;
// Largest coefficients+outputs whose VJP runs parameter_partial_kernel, in at
// most parameter_tiles batch tiles of at least parameter_tile_rows samples.
// FP32 also bounds the tiled reduction's work (batch*(coefficients+outputs)):
// above 2^22 SGEMM+SGEMV is faster (32x1024x448: 22 vs 39 us; 16x1024x224,
// 3.7M: tiled 18 vs 23 us; C1 evidence).
template<class T> constexpr std::size_t small_parameter_vjp = std::size_t{1} << 15;
template<class T> constexpr std::size_t small_parameter_work = std::numeric_limits<std::size_t>::max();
template<> constexpr std::size_t small_parameter_work<float> = std::size_t{1} << 22;
constexpr std::size_t parameter_tile_rows = 64;
constexpr unsigned parameter_tiles = 64;
unsigned parameter_tile_count(std::size_t batch) {
    return static_cast<unsigned>(std::min<std::size_t>(parameter_tiles, batch ? (batch-1)/parameter_tile_rows+1 : 1));
}

// Small parameter VJPs (at most small_parameter_vjp coefficients+outputs):
// block row t reduces samples [t*chunk, (t+1)*chunk) into partial[t][c] for
// c = o*IK+ik (U[b,o]*Phi[b,ik]) and the bias c = IK*O+o (U[b,o]). cuBLAS
// runs these long, narrow products without split-K (one 32x32 tile over the
// whole batch: up to 331 us at 112x24x1024 on the RTX 3090; C2 evidence).
template<class T>
__global__ void parameter_partial_kernel(const T* v, const T* u, T* partial, std::size_t batch,
                                         std::size_t outputs, std::size_t length, std::size_t chunk) {
    const auto coefficients = outputs*length, count = coefficients+outputs;
    const auto begin = static_cast<std::size_t>(blockIdx.y)*chunk, end = begin+chunk < batch ? begin+chunk : batch;
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto c = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; c < count; c += stride) {
        T sum = 0;
        if (c < coefficients) {
            const auto o = c/length, ik = c%length;
            for (auto r = begin; r < end; ++r) sum += u[r*outputs+o]*v[r*length+ik];
        } else {
            for (auto r = begin; r < end; ++r) sum += u[r*outputs+c-coefficients];
        }
        partial[blockIdx.y*count+c] = sum;
    }
}
// Source of the Phi' rows reduced by backward_finish_kernel: rows stored by
// the forward basis_kernel, or (backlog C3) recomputed from the layer input in
// the staged tile, which needs a third plane for the values the formulas
// produce alongside. Recomputation evaluates the forward's formula on the same
// input, so the rows are the forward's; the forward already checked them.
template<class T> struct StoredDerivatives {
    const T* rows;
};
template<detail::BasisKind Kind, class T> struct RecomputedDerivatives {
    const T* input;
    detail::BasisViewOf<T> basis;
};
template<class Source> constexpr bool stored_rows = true;
template<detail::BasisKind Kind, class T> constexpr bool stored_rows<RecomputedDerivatives<Kind, T>> = false;
template<class Source> constexpr detail::BasisKind recomputed_kind = detail::BasisKind::Chebyshev;
template<detail::BasisKind Kind, class T> constexpr detail::BasisKind recomputed_kind<RecomputedDerivatives<Kind, T>> = Kind;
// Stage planes: Phi' and W, plus the recomputed values.
template<class Source> constexpr unsigned stage_planes = stored_rows<Source> ? 2 : 3;

// One launch finishes a layer's backward: dx[r] = sum_k Phi'[r,k]*W[r,k] for
// the rows r = (sample, input), then the `checked` parameter VJPs [dC | db]:
// with partials, their fixed-order tile sum plus lambda*C; otherwise (written
// by cuBLAS earlier on the stream) only the nonfinite check.
// Rows are reduced from staged tiles of Phi' and W (tile_rows > 0, see
// stage_rows) or directly (stored rows only); the parameter part is a
// grid-stride loop.
template<class Source, class T>
__global__ void backward_finish_kernel(Source source, const T* w, T* dx,
                                       std::size_t rows, std::size_t terms, T* gradients, std::size_t checked,
                                       const T* partial, unsigned tiles, const T* c, std::size_t coefficients,
                                       T lambda, unsigned tile_rows, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    if (tile_rows == 0) {
        if constexpr (stored_rows<Source>) {
            for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < rows; index += stride) {
                T sum = 0;
                for (std::size_t k = 0; k < terms; ++k) sum += source.rows[index*terms+k]*w[index*terms+k];
                dx[index] = sum; report(sum, status);
            }
        }
    } else {
        T* stage = stage_memory<T>();
        const auto plane = static_cast<std::size_t>(tile_rows)*terms;
        for (auto base = static_cast<std::size_t>(blockIdx.x)*tile_rows; base < rows;
             base += static_cast<std::size_t>(gridDim.x)*tile_rows) {
            const auto count = rows-base < tile_rows ? rows-base : static_cast<std::size_t>(tile_rows);
            const auto length = count*terms, offset = base*terms;
            for (auto i = static_cast<std::size_t>(threadIdx.x); i < length; i += blockDim.x) {
                if constexpr (stored_rows<Source>) stage[i] = source.rows[offset+i];
                stage[plane+i] = w[offset+i];
            }
            if constexpr (!stored_rows<Source>) {
                if (threadIdx.x < count) {
                    const auto row = threadIdx.x*terms;
                    detail::basis_terms_for<recomputed_kind<Source>>(source.basis, source.input[base+threadIdx.x],
                        detail::BasisRowOf<T>{stage+2*plane+row, stage+row, nullptr, nullptr}, CheckedEarlier{});
                }
            }
            __syncthreads();
            if (threadIdx.x < count) {
                T sum = 0;
                const T* d = stage+threadIdx.x*terms;
                for (std::size_t k = 0; k < terms; ++k) sum += d[k]*d[plane+k];
                dx[base+threadIdx.x] = sum; report(sum, status);
            }
            __syncthreads();
        }
    }
    for (auto q = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; q < checked; q += stride) {
        if (partial) {
            T sum = 0;
            for (unsigned t = 0; t < tiles; ++t) sum += partial[t*checked+q];
            if (q < coefficients) sum += lambda*c[q];
            gradients[q] = sum;
        }
        report(gradients[q], status);
    }
}
template<class T>
__global__ void fill_kernel(T* values, std::size_t count, T value) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride)
        values[index] = value;
}
constexpr unsigned nonlinear_tiles=64;
// Center/log-width VJPs: sum over rows r of W[r,k]*(-dPhi/dx) and W[r,k]*dPhi/dlog-width.
template<class T>
__global__ void nonlinear_partial_kernel(const T* dx, const T* dw, const T* w,
                                         T* partial, std::size_t rows, std::size_t terms, unsigned tiles, int* status) {
    __shared__ T centers[256], widths[256];
    const auto k=static_cast<std::size_t>(blockIdx.x)/tiles;
    const auto tile=blockIdx.x%tiles, lane=threadIdx.x;
    T center=0,width=0;
    const auto stride=static_cast<std::size_t>(tiles)*blockDim.x;
    for(auto r=static_cast<std::size_t>(tile)*blockDim.x+lane;r<rows;r+=stride) {
        const T factor=w[r*terms+k];
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
// Staged variant (C1 profiling): the kernel above reads column k of three
// row-major (rows, terms) tensors, i.e. addresses `terms` apart per lane.
// Here block `tile` moves chunks of tile_rows consecutive rows of dx, dw and
// W into shared memory with coalesced loads; thread t accumulates term
// k = t % terms over the chunk rows t / terms, t / terms + per, ... (per =
// blockDim / terms), and a fixed-order sum over those row lanes gives the
// same partial layout [k*tiles + tile] that nonlinear_finish_kernel reduces.
template<class T>
__global__ void nonlinear_staged_kernel(const T* dx, const T* dw, const T* w, T* partial, std::size_t rows,
                                        std::size_t terms, unsigned tiles, unsigned tile_rows, int* status) {
    __shared__ T centers[stage_threads], widths[stage_threads];
    T* stage = stage_memory<T>();
    const auto plane = static_cast<std::size_t>(tile_rows)*terms;
    const auto per = blockDim.x/terms, k = threadIdx.x%terms, lane_row = threadIdx.x/terms;
    const bool active = lane_row < per;
    T center = 0, width = 0;
    for (auto base = static_cast<std::size_t>(blockIdx.x)*tile_rows; base < rows; base += static_cast<std::size_t>(tiles)*tile_rows) {
        const auto count = rows-base < tile_rows ? rows-base : static_cast<std::size_t>(tile_rows);
        const auto length = count*terms, offset = base*terms;
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < length; i += blockDim.x) {
            stage[i] = dx[offset+i]; stage[plane+i] = dw[offset+i]; stage[2*plane+i] = w[offset+i];
        }
        __syncthreads();
        if (active)
            for (std::size_t r = lane_row; r < count; r += per) {
                const auto i = r*terms+k;
                const T factor = stage[2*plane+i];
                center += factor*(-stage[i]); width += factor*stage[plane+i];
            }
        __syncthreads();
    }
    report(center, status); report(width, status);
    centers[threadIdx.x] = center; widths[threadIdx.x] = width;
    __syncthreads();
    if (threadIdx.x < terms) {
        T c = 0, v = 0;
        for (unsigned r = 0; r < per; ++r) { c += centers[r*terms+threadIdx.x]; v += widths[r*terms+threadIdx.x]; }
        partial[threadIdx.x*tiles+blockIdx.x] = c; partial[(terms+threadIdx.x)*tiles+blockIdx.x] = v;
        report(c, status); report(v, status);
    }
}
template<class T>
__global__ void nonlinear_finish_kernel(const T* partial, T* gradient, std::size_t terms, unsigned tiles, int* status) {
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto k=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;k<2*terms;k+=stride) {
        T sum=0;for(unsigned tile=0;tile<tiles;++tile)sum+=partial[k*tiles+tile];
        gradient[k]=sum;report(sum,status);
    }
}
template<class T>
__global__ void validate_width_kernel(const T* next, std::size_t count, int* status) {
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto k=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=stride) {
        // atomicOr (not Exch): the status bits accumulate (C9 deferred status).
        const T width=detail::math::exp(next[k]);if(!isfinite(width)||width<=0)atomicOr(status,1);
    }
}
template<class T>
__global__ void candidate_kernel(const T* parameters, const T* gradients, T* next,
                                 std::size_t count, T rate, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += stride) {
        next[i] = parameters[i] - rate*gradients[i]; report(next[i], status);
    }
}

// Device status of the executor (one allocation). bits[0] is the status word
// of every eager call; a captured training step (backlog C9) reports its
// forward + loss, backward and SGD phases into bits[0], bits[1] and bits[2],
// which stay set (sticky) until the host's deferred check, so that the step
// that failed first and its first failing phase can be told apart.
struct DeviceControl {
    int bits[3];
    int first_bits;               // phase bits of the first failing step since the last check
    unsigned ticket;              // last-block ticket of mse_kernel (reset by its last block)
    unsigned reserved;
    unsigned long long committed; // training steps whose update was committed
};
// Commit of a captured training step. The host swaps the parameter and
// candidate regions after every step; this kernel makes the swap a commit or
// a rollback on the device: with no status bit set the step counts as
// committed, otherwise the first failing step records its phase bits and the
// current parameters are copied over the candidates, so the region that
// becomes active holds the parameters of the last good step. Bits stay set,
// so every later step of the interval rolls back as well.
template<class T>
__global__ void commit_kernel(const T* parameters, T* candidates, std::size_t count, DeviceControl* control) {
    const volatile int* bits = control->bits;
    const int forward = bits[0], backward = bits[1], update = bits[2];
    if (!(forward | backward | update)) {
        if (blockIdx.x == 0 && threadIdx.x == 0) ++control->committed;
        return;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && control->first_bits == 0)
        control->first_bits = forward ? forward : backward ? backward : update;
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += stride)
        candidates[i] = parameters[i];
}
constexpr unsigned commit_max_blocks = 160;
// Host threads converting a staged batch: at most 8, at least 2^18 values each.
constexpr std::size_t stage_threads_max = 8, stage_chunk = std::size_t{1} << 18;

// Mean squared error against the resident target (backlog C9): the upstream
// u = (y - t)*scale (scale = 2/count in T) and L = sum (y - t)^2 / count.
// Each block writes a fixed-order tree sum of its grid-stride rows; the last
// block (ticket) adds the block sums in index order: deterministic for a
// given grid, no floating-point atomics.
constexpr unsigned mse_threads = 256, mse_max_blocks = 256;
template<class T>
__global__ void mse_kernel(const T* y, const T* t, T* upstream, std::size_t count, T scale, T divisor,
                           T* partial, T* loss, unsigned* ticket, int* status) {
    __shared__ T sums[mse_threads];
    __shared__ bool last;
    T sum = 0;
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += stride) {
        const T d = y[i]-t[i];
        const T u = d*scale;
        upstream[i] = u; report(u, status);
        sum += d*d;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned step = blockDim.x/2; step; step /= 2) {
        if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x+step];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sums[0];
        __threadfence();
        last = atomicAdd(ticket, 1U) == gridDim.x-1;
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        const volatile T* blocks = partial;
        T total = 0;
        for (unsigned b = 0; b < gridDim.x; ++b) total += blocks[b];
        const T value = total/divisor;
        *loss = value; report(value, status);
        *ticket = 0;
    }
}
// Distinct rational execution: caches are edge-major to expose contiguous
// samples to nonlinear parameter reductions. All caches live in the arena.
// One instantiation per denominator policy; `gains` (g = dQ/dS) is cached
// only by the safe policies and is null for Guarded.
template<DenominatorPolicy Policy, class T>
__global__ void rational_forward_kernel(const T* input,const T* a,const T* b,const T* bias,
                                        T* values,T* denominator_values,T* derivatives,T* gains,T* output,
                                        std::size_t batch,std::size_t capacity,std::size_t inputs,std::size_t outputs,
                                        detail::RationalScalars<T> config,int* status) {
    const StatusGuard guard{status};
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    const auto m=config.numerator_degree,n=config.denominator_degree;
    for(auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<batch*outputs;index+=stride) {
        const auto sample=index/outputs,o=index%outputs;T sum=bias[o];
        for(std::size_t i=0;i<inputs;++i) {
            const auto edge=o*inputs+i,cache=edge*capacity+sample;
            const auto h=detail::rational_horner<Policy>(config,input[sample*inputs+i],a+edge*(m+1),b+edge*n,guard);
            if(detail::rational_pole<Policy>(config,h)) {
                atomicOr(status,2);values[cache]=denominator_values[cache]=derivatives[cache]=0;continue;
            }
            const auto e=detail::rational_edge<Policy>(config,h,guard);
            // Derivative powers are part of the nonlinear contract, including
            // zero upstream. Detect unusable parameter VJPs during forward.
            T power=1;
            for(std::size_t k=0;k<=(m>n?m:n);++k) {
                if(k)power=guard(power*h.z);
                const T divided=guard(power/h.q);
                if(k<=m)detail::rational_numerator_vjp(h.q,h.z,k,power,divided,guard);
                if(k&&k<=n)detail::rational_denominator_vjp<Policy>(h.p,h.q,e.value,h.gain,h.z,k,power,divided,guard);
            }
            if constexpr(Policy!=DenominatorPolicy::Guarded)gains[cache]=h.gain;
            values[cache]=h.p;denominator_values[cache]=h.q;derivatives[cache]=e.input_derivative;sum+=e.value;report(sum,status);
        }
        output[index]=sum;report(sum,status);
    }
}
template<class T>
__global__ void rational_input_kernel(const T* derivatives,const T* upstream,T* input_gradient,
                                      std::size_t batch,std::size_t capacity,std::size_t inputs,std::size_t outputs,int* status) {
    const auto stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for(auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<batch*inputs;index+=stride) {
        const auto sample=index/inputs,i=index%inputs;T sum=0;
        for(std::size_t o=0;o<outputs;++o)sum+=upstream[sample*outputs+o]*derivatives[(o*inputs+i)*capacity+sample];
        input_gradient[index]=sum;report(sum,status);
    }
}
template<DenominatorPolicy Policy, class T>
__global__ void rational_parameter_kernel(const T* input,const T* values,const T* denominator_values,const T* gains,const T* upstream,
                                          const T* parameters,T* gradients,std::size_t batch,std::size_t capacity,
                                          std::size_t inputs,std::size_t outputs,detail::RationalScalars<T> config,T lambda,int* status) {
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
        const auto o=bias?relative:edge/inputs,i=edge%inputs;T sum=0;
        for(std::size_t sample=lane;sample<batch;sample+=32) {
            T derivative=1;
            if(!bias) {
                const T z=detail::rational_argument(config,input[sample*inputs+i],guard);T power=1;
                for(std::size_t j=0;j<k;++j)power*=z;
                const auto cache=edge*capacity+sample;
                const T q=denominator_values[cache],divided=power/q;
                if(numerator)derivative=detail::rational_numerator_vjp(q,z,k,power,divided,guard);
                else {
                    const T p=values[cache];
                    const T gain=Policy==DenominatorPolicy::Guarded?T(1):gains[cache];
                    derivative=detail::rational_denominator_vjp<Policy>(p,q,p/q,gain,z,k,power,divided,guard);
                }
            }
            const T term=upstream[sample*outputs+o]*derivative;report(term,status);sum+=term;
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
template<class T>
__global__ void affine_forward_kernel(const T* x, const T* scale, const T* shift, T* y,
                                      std::size_t count, std::size_t features, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        const auto i = index%features;
        y[index] = detail::affine_value(scale[i], shift[i], x[index]); report(y[index], status);
    }
}
template<class T>
__global__ void affine_input_kernel(const T* upstream, const T* scale, T* dx,
                                    std::size_t count, std::size_t features, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        dx[index] = scale[index%features]*upstream[index]; report(dx[index], status);
    }
}
template<class T>
__global__ void tanh_forward_kernel(const T* x, T* y, std::size_t count, T scale) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride)
        y[index] = detail::tanh_value(scale, x[index]); // bounded by 1 for finite input
}
// Derivative from the saved output y (activation[j+1]): three FP64 operations
// instead of cosh and a division per element.
template<class T>
__global__ void tanh_input_kernel(const T* y, const T* upstream, T* dx, std::size_t count,
                                  T scale, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride) {
        dx[index] = upstream[index]*detail::tanh_derivative(scale, y[index]); report(dx[index], status);
    }
}
// Butterfly sum over a group of Lanes consecutive lanes: every lane of the
// group ends with the same (commutative) result.
template<unsigned Lanes, class T> __device__ T group_sum(T value) {
    for (unsigned offset = Lanes/2; offset; offset /= 2) value += __shfl_xor_sync(0xffffffffU, value, offset, Lanes);
    return value;
}
// LayerNorm rows are processed by groups of Lanes lanes (32/Lanes rows per
// warp), Lanes chosen per map so that each lane holds about eight features:
// the per-row scalar work (reductions, 1/sqrt) is FP64 and is executed by
// every lane, so narrower groups cut the FP64 instruction count. The row
// loop is warp-uniform; groups past the last row recompute a valid row so
// that every lane takes part in the shuffles, and store nothing.
template<unsigned Lanes, class T>
__global__ void layer_norm_forward_kernel(const T* x, const T* gain, const T* bias, T* y,
                                          T* stats, std::size_t rows, std::size_t features,
                                          T inverse_count, T epsilon, int* status) {
    constexpr unsigned rows_per_warp = 32/Lanes;
    const auto lane = threadIdx.x%Lanes, group = (threadIdx.x%32)/Lanes;
    const auto warp = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32)*rows_per_warp;
    for (auto first = warp*rows_per_warp; first < rows; first += stride) {
        const auto row = first+group;
        const bool active = row < rows;
        const T* in = x+(active ? row : first)*features;
        T sum = 0;
        for (std::size_t f = lane; f < features; f += Lanes) sum += in[f];
        const T mean = detail::layer_norm_mean(group_sum<Lanes>(sum), inverse_count);
        T squares = 0;
        for (std::size_t f = lane; f < features; f += Lanes) { const T d = in[f]-mean; squares += d*d; }
        const T variance = detail::layer_norm_mean(group_sum<Lanes>(squares), inverse_count);
        const T rstd = detail::layer_norm_rstd(variance, epsilon);
        if (!active) continue;
        if (lane == 0) { report(mean, status); report(variance, status); stats[2*row] = mean; stats[2*row+1] = rstd; }
        for (std::size_t f = lane; f < features; f += Lanes) {
            const T xhat = detail::layer_norm_normalized(in[f], mean, rstd);
            const T value = gain ? gain[f]*xhat+bias[f] : xhat;
            y[row*features+f] = value; report(value, status);
        }
    }
}
template<unsigned Lanes, class T>
__global__ void layer_norm_input_kernel(const T* x, const T* upstream, const T* gain, const T* stats,
                                        T* dx, std::size_t rows, std::size_t features, T inverse_count, int* status) {
    constexpr unsigned rows_per_warp = 32/Lanes;
    const auto lane = threadIdx.x%Lanes, group = (threadIdx.x%32)/Lanes;
    const auto warp = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32)*rows_per_warp;
    for (auto first = warp*rows_per_warp; first < rows; first += stride) {
        const auto row = first+group;
        const bool active = row < rows;
        const auto base = (active ? row : first)*features;
        const T mean = stats[2*(active ? row : first)], rstd = stats[2*(active ? row : first)+1];
        T sum_w = 0, sum_wx = 0;
        for (std::size_t f = lane; f < features; f += Lanes) {
            const T w = gain ? upstream[base+f]*gain[f] : upstream[base+f];
            sum_w += w; sum_wx += w*detail::layer_norm_normalized(x[base+f], mean, rstd);
        }
        const T mean_w = detail::layer_norm_mean(group_sum<Lanes>(sum_w), inverse_count);
        const T mean_wx = detail::layer_norm_mean(group_sum<Lanes>(sum_wx), inverse_count);
        if (!active) continue;
        for (std::size_t f = lane; f < features; f += Lanes) {
            const T w = gain ? upstream[base+f]*gain[f] : upstream[base+f];
            const T xhat = detail::layer_norm_normalized(x[base+f], mean, rstd);
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
template<class T>
__global__ void layer_norm_parameter_partial_kernel(const T* x, const T* upstream, const T* stats,
                                                    T* partial, std::size_t rows, std::size_t features, unsigned tiles) {
    __shared__ T gains[norm_row_lanes][32], biases[norm_row_lanes][32];
    const auto f = static_cast<std::size_t>(blockIdx.x)*32+threadIdx.x;
    T g = 0, b = 0;
    if (f < features)
        for (auto row = static_cast<std::size_t>(blockIdx.y)*norm_row_lanes+threadIdx.y; row < rows; row += static_cast<std::size_t>(tiles)*norm_row_lanes) {
            const T u = upstream[row*features+f];
            g += u*detail::layer_norm_normalized(x[row*features+f], stats[2*row], stats[2*row+1]); b += u;
        }
    gains[threadIdx.y][threadIdx.x] = g; biases[threadIdx.y][threadIdx.x] = b;
    __syncthreads();
    if (threadIdx.y == 0 && f < features) {
        T sg = 0, sb = 0;
        for (unsigned lane = 0; lane < norm_row_lanes; ++lane) { sg += gains[lane][threadIdx.x]; sb += biases[lane][threadIdx.x]; }
        partial[blockIdx.y*2*features+f] = sg; partial[blockIdx.y*2*features+features+f] = sb;
    }
}
// One warp per gain/bias parameter: lanes sum strided tiles, then a fixed
// butterfly reduction (independent loads instead of a serial tile chain).
template<class T>
__global__ void layer_norm_parameter_finish_kernel(const T* partial, T* gradient, std::size_t features,
                                                   unsigned tiles, int* status) {
    const auto lane = threadIdx.x%32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for (auto p = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32; p < 2*features; p += stride) {
        T sum = 0;
        for (unsigned tile = lane; tile < tiles; tile += 32) sum += partial[tile*2*features+p];
        sum = group_sum<32>(sum);
        if (lane == 0) { gradient[p] = sum; report(sum, status); }
    }
}

// cuBLAS entry points per scalar type (64-bit API, CUDA 12+).
cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, std::int64_t m, std::int64_t n,
                    std::int64_t k, const double* alpha, const double* x, std::int64_t ldx, const double* y,
                    std::int64_t ldy, const double* beta, double* z, std::int64_t ldz) {
    return cublasDgemm_64(h, a, b, m, n, k, alpha, x, ldx, y, ldy, beta, z, ldz);
}
cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, std::int64_t m, std::int64_t n,
                    std::int64_t k, const float* alpha, const float* x, std::int64_t ldx, const float* y,
                    std::int64_t ldy, const float* beta, float* z, std::int64_t ldz) {
    return cublasSgemm_64(h, a, b, m, n, k, alpha, x, ldx, y, ldy, beta, z, ldz);
}
cublasStatus_t gemv(cublasHandle_t h, std::int64_t m, std::int64_t n, const double* alpha, const double* a,
                    std::int64_t lda, const double* x, const double* beta, double* y) {
    return cublasDgemv_64(h, CUBLAS_OP_N, m, n, alpha, a, lda, x, 1, beta, y, 1);
}
cublasStatus_t gemv(cublasHandle_t h, std::int64_t m, std::int64_t n, const float* alpha, const float* a,
                    std::int64_t lda, const float* x, const float* beta, float* y) {
    return cublasSgemv_64(h, CUBLAS_OP_N, m, n, alpha, a, lda, x, 1, beta, y, 1);
}
cublasStatus_t scal(cublasHandle_t h, std::int64_t n, const double* alpha, double* x) { return cublasDscal_64(h, n, alpha, x, 1); }
cublasStatus_t scal(cublasHandle_t h, std::int64_t n, const float* alpha, float* x) { return cublasSscal_64(h, n, alpha, x, 1); }

// Device state shared by every layer plan: the arena, stream and status word,
// the double-buffered parameter regions and the per-layer activations.
template<class T>
struct Context {
    std::size_t capacity, batch = 0;
    // Training steps (C9) never expose the gradient with respect to the
    // network input (gradients are stale after a step), so they skip it, as
    // PyTorch does for an input that does not require grad.
    bool skip_network_input_gradient = false;
    bool input_gradient(std::size_t j) const noexcept { return j != 0 || !skip_network_input_gradient; }
    std::size_t parameters = 0, gradients = 0, candidates = 0;
    // Contraction engine (C2): W = U*C scratch shared by all expansion layers
    // (largest capacity*inputs*terms), a ones vector for the bias VJP and the
    // cuBLAS workspace, so cuBLAS never allocates during execution.
    std::size_t scratch = 0, ones = 0, blas_workspace = 0;
    std::size_t partials = 0; // small parameter VJP tiles, shared like the scratch
    std::vector<std::size_t> activation, upstream;
    T* arena = nullptr;
    int* status = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    // FP32 host staging: converted upload sources stay alive, and downloaded
    // device values are widened into their double destinations, at sync().
    std::vector<std::vector<T>> staged;
    std::vector<std::pair<std::span<double>, std::vector<T>>> pending;
    // FP32: page-locked host buffer for the per-step input/upstream uploads
    // (capacity * widest end of the network), converted in place: pageable
    // copies ran at about 2.4 GB/s (32 MiB upstream of the 1024-wide step:
    // 13.5 ms). Host memory, not counted by workspace_allocations().
    T* pinned = nullptr;
    std::size_t pinned_size = 0;
    // Validates (finite, then representable: std::invalid_argument) before any
    // device change, in the same single pass that converts into the buffer.
    void upload_pinned(T* destination, std::span<const double> data) {
        if (!pinned || data.size() > pinned_size) { finite(data); upload(destination, data); return; }
        bool nonfinite = false, outside = false;
        for (std::size_t i = 0; i < data.size(); ++i) {
            const double v = data[i];
            nonfinite |= !std::isfinite(v);
            outside |= !(std::abs(v) <= static_cast<double>(FLT_MAX));
            pinned[i] = static_cast<T>(outside ? 0.0 : v);
        }
        if (nonfinite) throw std::invalid_argument("resident data must be finite");
        if (outside) throw std::invalid_argument("resident data is not representable in float32");
        if (!data.empty())
            check(cudaMemcpyAsync(destination, pinned, data.size()*sizeof(T), cudaMemcpyHostToDevice, stream), "resident upload");
    }
    T* ptr(std::size_t offset) const { return arena+offset; }
    void sync() {
        check(cudaStreamSynchronize(stream), "resident synchronize");
        staged.clear();
        for (auto& [destination, values] : pending) std::copy(values.begin(), values.end(), destination.begin());
        pending.clear();
    }
    void upload(T* destination, std::span<const double> data) {
        if (data.empty()) return;
        if constexpr (std::is_same_v<T, double>) {
            check(cudaMemcpyAsync(destination, data.data(), data.size_bytes(), cudaMemcpyHostToDevice, stream), "resident upload");
        } else {
            std::vector<T> converted(data.size());
            std::transform(data.begin(), data.end(), converted.begin(), [](double v) { return narrow<T>(v); });
            check(cudaMemcpyAsync(destination, converted.data(), converted.size()*sizeof(T), cudaMemcpyHostToDevice, stream), "resident upload");
            staged.push_back(std::move(converted));
        }
    }
    void download(std::span<double> destination, const T* data) {
        if (destination.empty()) return;
        if constexpr (std::is_same_v<T, double>) {
            check(cudaMemcpyAsync(destination.data(), data, destination.size_bytes(), cudaMemcpyDeviceToHost, stream), "resident download");
        } else {
            pending.emplace_back(destination, std::vector<T>(destination.size()));
            check(cudaMemcpyAsync(pending.back().second.data(), data, destination.size()*sizeof(T), cudaMemcpyDeviceToHost, stream), "resident download");
        }
    }
};

// Bump allocator over the arena, checked before the single device allocation.
// Every region starts on a 256-byte boundary (cudaMalloc alignment), which
// cuBLAS needs for its workspace and prefers for vectorized operand loads.
template<class T>
struct Reservation {
    static constexpr std::size_t alignment = 256/sizeof(T);
    std::size_t total = 0;
    std::size_t operator()(std::size_t count) {
        const auto limit = std::vector<T>().max_size();
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

// Execution plans, one per carrier type (the alternatives of kan::Carrier),
// over the device scalar T. Each plan owns its workspace offsets and the
// configuration scalars converted to T; the executor dispatches on the plan
// once per layer and operation.

// Expansion + contraction engine: basis_kernel writes the rows of Phi and Phi',
// the contraction computes Y = Phi*C^T + b and its VJPs. Used by BasisEdges and
// by the coefficients of TrainableRbfEdges.
template<class T>
struct ExpansionPlan {
    ParameterBlock block;
    detail::BasisViewOf<T> view; // host scalars; vector pointers are set per launch
    std::size_t values = 0, derivatives = 0, centers = 0, scales = 0, knots = 0;
    // Phi' rows kept from forward to backward; otherwise (backlog C3) the
    // backward finish recomputes them from the layer input (no derivatives region).
    bool stored_derivatives = true;
};

template<class T>
struct BasisPlan {
    using edges_type = BasisEdges;
    ExpansionPlan<T> expansion;
};

// Adds trainable shared centers/log widths (nonlinear block: centers, then
// log widths) and their tiled reductions.
template<class T>
struct TrainableRbfPlan {
    using edges_type = TrainableRbfEdges;
    ExpansionPlan<T> expansion;
    std::size_t log_derivatives = 0, partials = 0;
    unsigned partial_tiles = 1;
};

// Rational edges with edge-major caches of P, Q and dr/dx, plus g = dQ/dS for
// the safe denominator policies (nonlinear block: denominators).
template<class T>
struct RationalPlan {
    using edges_type = RationalEdges;
    ParameterBlock block;
    RationalConfig config;
    detail::RationalScalars<T> scalars; // center, scale and the pole threshold in T
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
template<class T>
struct AffinePlan {
    using map_type = AffineMap;
    MapBlock block;
    std::size_t scale = 0, shift = 0;
};
template<class T>
struct TanhPlan {
    using map_type = TanhMap;
    MapBlock block;
    T scale;
};
template<class T>
struct LayerNormPlan {
    using map_type = LayerNormMap;
    MapBlock block;
    T epsilon, inverse_count; // inverse_count = 1/features, as on the CPU
    unsigned lanes;           // lanes per row (norm_lanes)
    std::size_t stats = 0, partials = 0; // (capacity, 2) row moments; (tiles, 2*features) partial VJPs
    bool affine() const noexcept { return block.parameters != 0; }
};

template<class T>
using Plan = std::variant<BasisPlan<T>, TrainableRbfPlan<T>, RationalPlan<T>, AffinePlan<T>, TanhPlan<T>, LayerNormPlan<T>>;

template<class T> const ParameterBlock& block_of(const BasisPlan<T>& p) { return p.expansion.block; }
template<class T> const ParameterBlock& block_of(const TrainableRbfPlan<T>& p) { return p.expansion.block; }
template<class T> const ParameterBlock& block_of(const RationalPlan<T>& p) { return p.block; }

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
template<class T> Extent extent(const Plan<T>& plan) {
    return std::visit([](const auto& p) { return extent_of(p); }, plan);
}

// FP32 representability of a basis configuration: every scalar the kernels
// read, and the structure the CPU validated (positive widths and scales,
// alpha/beta > -1, distinct knots stay distinct). std::invalid_argument.
template<class T, class C> void check_basis(const C& c) {
    if constexpr (std::is_same_v<T, double>) {
        return;
    } else if constexpr (std::is_same_v<C, BasisConfig>) {
        std::visit([](const auto& alternative) { check_basis<T>(alternative); }, c);
    } else {
        const auto invalid = [] { throw std::invalid_argument("resident float32 basis configuration is not representable"); };
        if constexpr (std::is_same_v<C, JacobiConfig>) {
            if (!(narrow<T>(c.alpha) > -1) || !(narrow<T>(c.beta) > -1)) invalid();
        } else if constexpr (std::is_same_v<C, FourierConfig>) {
            narrow_positive<T>(c.frequency);
        } else if constexpr (std::is_same_v<C, GaussianRbfConfig>) {
            representable<T>(c.centers); narrow_positive<T>(c.width);
        } else if constexpr (std::is_same_v<C, TrainableRbfConfig>) {
            representable<T>(c.centers);
            for (double w : c.log_widths) {
                const T width = std::exp(narrow<T>(w));
                if (!std::isfinite(width) || !(width > 0)) invalid();
            }
        } else if constexpr (std::is_same_v<C, BSplineConfig>) {
            representable<T>(c.knots);
            for (std::size_t k = 0; k + 1 < c.knots.size(); ++k)
                if (c.knots[k] < c.knots[k+1] && !(static_cast<T>(c.knots[k]) < static_cast<T>(c.knots[k+1]))) invalid();
        } else if constexpr (std::is_same_v<C, MexicanHatConfig>) {
            representable<T>(c.centers);
            for (double scale : c.scales) narrow_positive<T>(scale);
        }
    }
}
// Host view of a basis in T: the configuration scalars converted; vector
// pointers are set per launch.
template<class T, class C> detail::BasisViewOf<T> device_view(const C& config) {
    check_basis<T>(config);
    const auto view = detail::basis_view(config);
    return {view.kind, view.terms, narrow<T>(view.alpha), narrow<T>(view.beta), narrow<T>(view.frequency),
            narrow<T>(view.width), nullptr, nullptr, nullptr, nullptr, view.degree, view.trainable};
}
template<class T> detail::RationalScalars<T> device_scalars(const RationalConfig& c) {
    T epsilon = narrow<T>(c.epsilon);
    if constexpr (!std::is_same_v<T, double>) {
        // FP32 cannot resolve |Q| below the rounding of its Horner evaluation
        // (about n*u*bound, u = 2^-24): the relative guard is at least n*2^-23.
        epsilon = std::max(epsilon, static_cast<T>(c.denominator_degree) * static_cast<T>(0x1p-23));
    }
    return {c.numerator_degree, c.denominator_degree, narrow<T>(c.center), narrow_positive<T>(c.scale), epsilon};
}

template<class T> Plan<T> make_plan(const Layer& layer, const BasisEdges& edges, std::size_t offset) {
    return BasisPlan<T>{{parameter_block(layer, 0, offset), device_view<T>(edges.basis)}};
}
template<class T> Plan<T> make_plan(const Layer& layer, const TrainableRbfEdges& edges, std::size_t offset) {
    return TrainableRbfPlan<T>{{parameter_block(layer, product(layer.terms(), 2), offset), device_view<T>(edges.basis)}};
}
template<class T> Plan<T> make_plan(const Layer& layer, const RationalEdges& edges, std::size_t offset) {
    representable<T>(edges.denominators);
    return RationalPlan<T>{parameter_block(layer, edges.denominators.size(), offset), edges.config, device_scalars<T>(edges.config)};
}
MapBlock map_block(const InputMap& map, std::size_t parameters, std::size_t offset) {
    if (parameters > std::vector<double>().max_size()-offset) throw std::overflow_error("resident parameter size overflow");
    return {map.features(), offset, parameters};
}
template<class T> Plan<T> make_plan(const InputMap& map, const AffineMap& affine, std::size_t offset) {
    representable<T>(affine.shift);
    for (double s : affine.scale)
        if (narrow<T>(s) == 0) throw std::invalid_argument("resident float32 affine scale rounds to zero");
    return AffinePlan<T>{map_block(map, 0, offset)};
}
template<class T> Plan<T> make_plan(const InputMap& map, const TanhMap& tanh, std::size_t offset) {
    return TanhPlan<T>{map_block(map, 0, offset), narrow_positive<T>(tanh.scale)};
}
template<class T> Plan<T> make_plan(const InputMap& map, const LayerNormMap& norm, std::size_t offset) {
    representable<T>(norm.gain); representable<T>(norm.bias);
    return LayerNormPlan<T>{map_block(map, product(norm.gain.size(), 2), offset), narrow_positive<T>(norm.epsilon),
                            static_cast<T>(1.0/static_cast<double>(map.features())), norm_lanes(map.features())};
}
template<class T> Plan<T> make_plan(const Layer& layer, std::size_t offset) {
    finite(layer.coefficients()); finite(layer.bias());
    representable<T>(layer.coefficients()); representable<T>(layer.bias());
    return std::visit([&](const auto& edges) { return make_plan<T>(layer, edges, offset); }, layer.carrier());
}
template<class T> Plan<T> make_plan(const InputMap& map, std::size_t offset) {
    return std::visit([&](const auto& kind) { return make_plan<T>(map, kind, offset); }, map.map());
}

// Workspaces, reserved in the order of the arena layout.
template<class T> void reserve_rows(ExpansionPlan<T>& p, std::size_t capacity, Reservation<T>& reserve) {
    const auto rows = product(product(capacity, p.block.inputs), p.block.terms);
    p.values = reserve(rows);
    if (p.stored_derivatives) p.derivatives = reserve(rows);
}
// Fixed bases recompute Phi' in the backward finish (backlog C3) wherever its
// three-plane stage fits: the FP32 executor. FP64 basis kernels are FP64-ALU
// bound on GA102 and unstaged (stage_rows), so FP64 keeps the stored rows.
template<class T> bool recompute_derivatives(const ExpansionPlan<T>& p) {
    return !p.view.trainable && stage_rows<T>(p.block.terms, stage_planes<RecomputedDerivatives<detail::BasisKind::Chebyshev, T>>) > 0;
}
template<class T> void reserve_workspace(BasisPlan<T>& plan, std::size_t capacity, Reservation<T>& reserve) {
    auto& p = plan.expansion;
    p.stored_derivatives = !recompute_derivatives(p);
    reserve_rows(p, capacity, reserve);
    if (p.view.kind == detail::BasisKind::GaussianRbf || p.view.kind == detail::BasisKind::MexicanHat)
        p.centers = reserve(p.block.terms);
    if (p.view.kind == detail::BasisKind::MexicanHat) p.scales = reserve(p.block.terms);
    if (p.view.kind == detail::BasisKind::BSpline) p.knots = reserve(p.block.terms+p.view.degree+1);
}
template<class T> void reserve_workspace(TrainableRbfPlan<T>& plan, std::size_t capacity, Reservation<T>& reserve) {
    auto& p = plan.expansion;
    reserve_rows(p, capacity, reserve);
    plan.log_derivatives = reserve(product(product(capacity, p.block.inputs), p.block.terms));
    const auto count = product(capacity, p.block.inputs);
    plan.partial_tiles = static_cast<unsigned>(std::min<std::size_t>(nonlinear_tiles, count?((count-1)/256+1):1));
    plan.partials = reserve(product(p.block.terms, 2*plan.partial_tiles));
}
template<class T> void reserve_workspace(RationalPlan<T>& plan, std::size_t capacity, Reservation<T>& reserve) {
    const auto count = product(product(capacity, plan.block.inputs), plan.block.outputs);
    plan.values = reserve(count); plan.derivatives = reserve(count); plan.denominator_values = reserve(count);
    if (plan.safe()) plan.gains = reserve(count);
}
template<class T> void reserve_workspace(AffinePlan<T>& plan, std::size_t, Reservation<T>& reserve) {
    plan.scale = reserve(plan.block.features); plan.shift = reserve(plan.block.features);
}
template<class T> void reserve_workspace(TanhPlan<T>&, std::size_t, Reservation<T>&) {}
template<class T> void reserve_workspace(LayerNormPlan<T>& plan, std::size_t capacity, Reservation<T>& reserve) {
    plan.stats = reserve(product(capacity, 2));
    if (plan.affine()) plan.partials = reserve(product(plan.block.features, 2*static_cast<std::size_t>(norm_tiles(capacity))));
}

template<class P> constexpr bool is_expansion = requires(const P& p) { p.expansion; };

// Elements of the shared W = U*C scratch an expansion layer needs (zero otherwise).
template<class T> std::size_t scratch_extent(const Plan<T>& plan, std::size_t capacity) {
    return std::visit([&](const auto& p) -> std::size_t {
        if constexpr (is_expansion<std::decay_t<decltype(p)>>)
            return product(product(capacity, p.expansion.block.inputs), p.expansion.block.terms);
        else return 0;
    }, plan);
}

// Elements of the shared small-parameter-VJP partials a layer needs (zero otherwise).
template<class T> std::size_t partial_extent(const Plan<T>& plan, std::size_t capacity) {
    return std::visit([&](const auto& p) -> std::size_t {
        if constexpr (is_expansion<std::decay_t<decltype(p)>>) {
            const auto checked = p.expansion.block.coefficients+p.expansion.block.outputs;
            return checked <= small_parameter_vjp<T> ? checked*parameter_tile_count(capacity) : 0;
        } else return 0;
    }, plan);
}

// Parameter upload at construction (coefficients and bias are uploaded by the executor).
template<class T> void upload_carrier(Context<T>& s, const BasisPlan<T>& plan, const BasisEdges& edges) {
    const auto& p = plan.expansion;
    const auto terms = p.block.terms;
    std::visit([&](const auto& c) {
        using C = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<C, GaussianRbfConfig> || std::is_same_v<C, MexicanHatConfig>) s.upload(s.ptr(p.centers), {c.centers.data(), terms});
        if constexpr (std::is_same_v<C, MexicanHatConfig>) s.upload(s.ptr(p.scales), {c.scales.data(), terms});
        if constexpr (std::is_same_v<C, BSplineConfig>) s.upload(s.ptr(p.knots), {c.knots.data(), terms+c.degree+1});
    }, edges.basis);
}
template<class T> void upload_carrier(Context<T>& s, const TrainableRbfPlan<T>& plan, const TrainableRbfEdges& edges) {
    const auto& b = plan.expansion.block;
    s.upload(s.ptr(s.parameters+b.nonlinear()), edges.basis.centers);
    s.upload(s.ptr(s.parameters+b.nonlinear()+b.terms), edges.basis.log_widths);
}
template<class T> void upload_carrier(Context<T>& s, const RationalPlan<T>& plan, const RationalEdges& edges) {
    s.upload(s.ptr(s.parameters+plan.block.nonlinear()), edges.denominators);
}

// Construction upload of one network layer: a KAN layer's coefficients, bias
// and carrier state, or an input map's fixed and trainable parameters.
template<class T, class P, class Edges>
void upload_stage(Context<T>& s, const P& plan, const Layer& layer, const Edges& edges) {
    const auto& b = block_of(plan);
    s.upload(s.ptr(s.parameters+b.offset), layer.coefficients());
    s.upload(s.ptr(s.parameters+b.bias()), layer.bias());
    upload_carrier(s, plan, edges);
}
template<class T> void upload_stage(Context<T>& s, const AffinePlan<T>& plan, const InputMap&, const AffineMap& map) {
    s.upload(s.ptr(plan.scale), map.scale); s.upload(s.ptr(plan.shift), map.shift);
}
template<class T> void upload_stage(Context<T>&, const TanhPlan<T>&, const InputMap&, const TanhMap&) {}
template<class T> void upload_stage(Context<T>& s, const LayerNormPlan<T>& plan, const InputMap&, const LayerNormMap& map) {
    s.upload(s.ptr(s.parameters+plan.block.offset), map.gain);
    s.upload(s.ptr(s.parameters+plan.block.offset+plan.block.half()), map.bias);
}

// Same scalars as the host view; vectors point into device storage.
template<class T>
detail::BasisViewOf<T> device_basis(const Context<T>& s, const ExpansionPlan<T>& p, std::type_identity_t<const T*> centers,
                                    std::type_identity_t<const T*> log_widths) {
    auto basis = p.view;
    basis.centers = centers; basis.log_widths = log_widths;
    basis.scales = s.ptr(p.scales); basis.knots = s.ptr(p.knots);
    return basis;
}

// Forward of layer j: activation[j] -> activation[j+1].
template<class T>
void expansion_forward(Context<T>& s, const ExpansionPlan<T>& p, std::size_t j, T* log_derivatives,
                       const T* centers, const T* log_widths) {
    const auto& b = p.block;
    const auto basis = device_basis(s, p, centers, log_widths);
    const auto count = s.batch*b.inputs;
    const auto tile_rows = stage_rows<T>(b.terms, basis.trainable ? 3 : 2);
    const auto shared = static_cast<std::size_t>(tile_rows)*b.terms*(basis.trainable ? 3 : 2)*sizeof(T);
    detail::visit_basis_family(basis.kind, [&](auto family) {
        basis_kernel<decltype(family)::value><<<blocks(count, tile_rows ? tile_rows : stage_threads), stage_threads, shared, s.stream>>>(
            s.ptr(s.activation[j]), s.ptr(p.values), p.stored_derivatives ? s.ptr(p.derivatives) : nullptr, log_derivatives,
            count, basis, tile_rows, s.status);
    });
    check(cudaGetLastError(), "resident basis launch");
    const auto outputs = s.batch*b.outputs, length = b.inputs*b.terms;
    if (outputs <= small_forward_contraction<T>/length) {
        forward_dot_kernel<<<blocks(outputs, 256/32), 256, 0, s.stream>>>(s.ptr(p.values), s.ptr(s.parameters+b.offset),
            s.ptr(s.parameters+b.bias()), s.ptr(s.activation[j+1]), outputs, b.outputs, length, s.status);
        check(cudaGetLastError(), "resident forward contraction launch");
        return;
    }
    // Row-major Y (batch x O) = Phi (batch x IK) * C^T, i.e. column-major
    // Y^T = C^T(op T) * Phi^T with C stored as column-major IK x O.
    const auto ik = static_cast<std::int64_t>(b.inputs*b.terms), o = static_cast<std::int64_t>(b.outputs);
    const T one = 1, zero = 0;
    check(gemm(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, o, static_cast<std::int64_t>(s.batch), ik, &one,
               s.ptr(s.parameters+b.offset), ik, s.ptr(p.values), ik, &zero, s.ptr(s.activation[j+1]), o),
          "resident forward contraction");
    bias_kernel<<<blocks(s.batch*b.outputs), 256, 0, s.stream>>>(s.ptr(s.activation[j+1]), s.ptr(s.parameters+b.bias()),
        s.batch*b.outputs, b.outputs, s.status);
    check(cudaGetLastError(), "resident forward bias launch");
}
template<class T> void run_forward(Context<T>& s, const BasisPlan<T>& plan, std::size_t j) {
    const auto& p = plan.expansion;
    expansion_forward<T>(s, p, j, nullptr, s.ptr(p.centers), nullptr);
}
template<class T> void run_forward(Context<T>& s, const TrainableRbfPlan<T>& plan, std::size_t j) {
    const auto nonlinear = s.parameters+plan.expansion.block.nonlinear();
    expansion_forward<T>(s, plan.expansion, j, s.ptr(plan.log_derivatives), s.ptr(nonlinear), s.ptr(nonlinear+plan.expansion.block.terms));
}
template<class T> void run_forward(Context<T>& s, const RationalPlan<T>& plan, std::size_t j) {
    const auto& b = plan.block;
    T* gains = plan.safe() ? s.ptr(plan.gains) : nullptr;
    detail::visit_denominator_policy(plan.config.denominator_policy, [&](auto policy) {
        rational_forward_kernel<decltype(policy)::value><<<blocks(s.batch*b.outputs),256,0,s.stream>>>(
            s.ptr(s.activation[j]),s.ptr(s.parameters+b.offset),
            s.ptr(s.parameters+b.nonlinear()),s.ptr(s.parameters+b.bias()),
            s.ptr(plan.values),s.ptr(plan.denominator_values),s.ptr(plan.derivatives),gains,s.ptr(s.activation[j+1]),
            s.batch,s.capacity,b.inputs,b.outputs,plan.scalars,s.status);
    });
    check(cudaGetLastError(),"resident rational forward launch");
}
template<class T> void run_forward(Context<T>& s, const AffinePlan<T>& plan, std::size_t j) {
    const auto count = s.batch*plan.block.features;
    affine_forward_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.activation[j]), s.ptr(plan.scale), s.ptr(plan.shift),
        s.ptr(s.activation[j+1]), count, plan.block.features, s.status);
    check(cudaGetLastError(), "resident affine map launch");
}
template<class T> void run_forward(Context<T>& s, const TanhPlan<T>& plan, std::size_t j) {
    const auto count = s.batch*plan.block.features;
    tanh_forward_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.activation[j]), s.ptr(s.activation[j+1]), count, plan.scale);
    check(cudaGetLastError(), "resident tanh map launch");
}
// Trainable gain and bias pointers (null without them).
template<class T> const T* norm_gain(const Context<T>& s, const LayerNormPlan<T>& plan, std::size_t region) {
    return plan.affine() ? s.ptr(region+plan.block.offset) : nullptr;
}
template<class T> void run_forward(Context<T>& s, const LayerNormPlan<T>& plan, std::size_t j) {
    const auto gain = norm_gain(s, plan, s.parameters);
    with_norm_lanes(plan.lanes, [&](auto lanes) {
        layer_norm_forward_kernel<decltype(lanes)::value><<<norm_blocks(s.batch, plan.lanes), norm_block, 0, s.stream>>>(
            s.ptr(s.activation[j]), gain, gain ? gain+plan.block.half() : nullptr, s.ptr(s.activation[j+1]),
            s.ptr(plan.stats), s.batch, plan.block.features, plan.inverse_count, plan.epsilon, s.status);
    });
    check(cudaGetLastError(), "resident layer norm launch");
}

// Backward of layer j: upstream[j+1] -> upstream[j] and the parameter gradients.
template<class T>
// with_w: compute W = U*C (input VJP, and the trainable-RBF reductions);
// with_dx: reduce W against Phi' to the input VJP.
void expansion_backward(Context<T>& s, const ExpansionPlan<T>& p, std::size_t j, T lambda, bool with_w, bool with_dx) {
    const auto& b = p.block;
    const auto ik = static_cast<std::int64_t>(b.inputs*b.terms), o = static_cast<std::int64_t>(b.outputs);
    const auto n = static_cast<std::int64_t>(s.batch);
    const T* c = s.ptr(s.parameters+b.offset);
    const T* u = s.ptr(s.upstream[j+1]);
    T* dc = s.ptr(s.gradients+b.offset);
    T* db = s.ptr(s.gradients+b.bias());
    const T one = 1, zero = 0;
    const auto checked = b.coefficients+b.outputs;
    const bool small = s.batch && checked <= small_parameter_vjp<T> && s.batch <= small_parameter_work<T>/checked;
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
        if (lambda != 0) check(cudaMemcpyAsync(dc, c, b.coefficients*sizeof(T), cudaMemcpyDeviceToDevice, s.stream), "resident L2 copy");
        check(gemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_T, ik, o, n, &one, s.ptr(p.values), ik, u, o,
                   &lambda, dc, ik), "resident coefficient VJP");
        // Bias VJP: column sums of U, as U^T * 1 (U^T is column-major O x batch).
        check(gemv(s.blas, o, n, &one, u, o, s.ptr(s.ones), &zero, db), "resident bias VJP");
    } else {
        if (lambda != 0) {
            check(cudaMemcpyAsync(dc, c, b.coefficients*sizeof(T), cudaMemcpyDeviceToDevice, s.stream), "resident L2 copy");
            check(scal(s.blas, static_cast<std::int64_t>(b.coefficients), &lambda, dc), "resident L2 scale");
        } else {
            check(cudaMemsetAsync(dc, 0, b.coefficients*sizeof(T), s.stream), "resident coefficient VJP reset");
        }
        check(cudaMemsetAsync(db, 0, b.outputs*sizeof(T), s.stream), "resident bias VJP reset");
    }
    if (s.batch && with_w) {
        // Input VJP: column-major W^T (IK x batch) = C^T * U^T, then dx = sum_k Phi' * W.
        check(gemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, ik, n, o, &one, c, ik, u, o, &zero, s.ptr(s.scratch), ik),
              "resident input VJP contraction");
    }
    const auto rows = with_dx ? s.batch*b.inputs : 0;
    const auto finish = [&](auto source) {
        constexpr auto planes = stage_planes<decltype(source)>;
        const auto tile_rows = stage_rows<T>(b.terms, planes);
        const auto shared = static_cast<std::size_t>(tile_rows)*b.terms*planes*sizeof(T);
        const auto grid = std::max(rows ? blocks(rows, tile_rows ? tile_rows : stage_threads) : 1u, blocks(checked, stage_threads));
        backward_finish_kernel<<<grid, stage_threads, shared, s.stream>>>(source, s.ptr(s.scratch),
            s.ptr(s.upstream[j]), rows, b.terms, dc, checked, small ? s.ptr(s.partials) : nullptr, tiles, c,
            b.coefficients, lambda, tile_rows, s.status);
    };
    if (p.stored_derivatives) {
        finish(StoredDerivatives<T>{s.ptr(p.derivatives)});
    } else if constexpr (!std::is_same_v<T, double>) {
        const auto basis = device_basis(s, p, s.ptr(p.centers), nullptr);
        detail::visit_basis_family(basis.kind, [&](auto family) {
            finish(RecomputedDerivatives<decltype(family)::value, T>{s.ptr(s.activation[j]), basis});
        });
    }
    check(cudaGetLastError(), "resident backward finish launch");
}

template<class T> void run_backward(Context<T>& s, const BasisPlan<T>& plan, std::size_t j, T lambda) {
    const bool dx = s.input_gradient(j);
    expansion_backward(s, plan.expansion, j, lambda, dx, dx);
}
template<class T> void run_backward(Context<T>& s, const TrainableRbfPlan<T>& plan, std::size_t j, T lambda) {
    const auto& p = plan.expansion;
    const auto& b = p.block;
    expansion_backward(s, p, j, lambda, true, s.input_gradient(j));
    // Reuses W = U*C left in the scratch by expansion_backward (zero rows for an empty batch).
    const auto rows=product(s.batch,b.inputs);
    const auto tiles=static_cast<unsigned>(std::min<std::size_t>(plan.partial_tiles,rows?((rows-1)/256+1):1));
    // Bound the launch dimension even for large valid basis term counts.
    if(b.terms>2147483647U/tiles)throw std::overflow_error("resident nonlinear launch size overflow");
    const auto tile_rows = b.terms <= stage_threads ? stage_rows<T>(b.terms, 3) : 0;
    if (rows && tile_rows) {
        nonlinear_staged_kernel<<<tiles, stage_threads, static_cast<std::size_t>(tile_rows)*b.terms*3*sizeof(T), s.stream>>>(
            s.ptr(p.derivatives), s.ptr(plan.log_derivatives), s.ptr(s.scratch), s.ptr(plan.partials), rows, b.terms, tiles,
            tile_rows, s.status);
    } else {
        nonlinear_partial_kernel<<<static_cast<unsigned>(b.terms)*tiles,256,0,s.stream>>>(s.ptr(p.derivatives),s.ptr(plan.log_derivatives),
            s.ptr(s.scratch),s.ptr(plan.partials),rows,b.terms,tiles,s.status);
    }
    check(cudaGetLastError(),"resident nonlinear partial launch");
    nonlinear_finish_kernel<<<blocks(2*b.terms),256,0,s.stream>>>(s.ptr(plan.partials),s.ptr(s.gradients+b.nonlinear()),b.terms,tiles,s.status);
    check(cudaGetLastError(),"resident nonlinear reduction launch");
}
template<class T> void run_backward(Context<T>& s, const RationalPlan<T>& plan, std::size_t j, T lambda) {
    const auto& b = plan.block;
    if(s.batch && s.input_gradient(j)) {
        rational_input_kernel<<<blocks(s.batch*b.inputs),256,0,s.stream>>>(s.ptr(plan.derivatives),s.ptr(s.upstream[j+1]),s.ptr(s.upstream[j]),
            s.batch,s.capacity,b.inputs,b.outputs,s.status);
        check(cudaGetLastError(),"resident rational input gradient launch");
    }
    const T* gains = plan.safe() ? s.ptr(plan.gains) : nullptr;
    detail::visit_denominator_policy(plan.config.denominator_policy, [&](auto policy) {
        rational_parameter_kernel<decltype(policy)::value><<<blocks(b.size(),8),256,0,s.stream>>>(
            s.ptr(s.activation[j]),s.ptr(plan.values),s.ptr(plan.denominator_values),gains,
            s.ptr(s.upstream[j+1]),s.ptr(s.parameters+b.offset),s.ptr(s.gradients+b.offset),s.batch,s.capacity,b.inputs,b.outputs,
            plan.scalars,lambda,s.status);
    });
    check(cudaGetLastError(),"resident rational parameter gradient launch");
}
// Input maps are not penalized by the coefficient L2 (lambda unused).
template<class T> void run_backward(Context<T>& s, const AffinePlan<T>& plan, std::size_t j, T) {
    const auto count = s.batch*plan.block.features;
    if (!count || !s.input_gradient(j)) return;
    affine_input_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.upstream[j+1]), s.ptr(plan.scale), s.ptr(s.upstream[j]),
        count, plan.block.features, s.status);
    check(cudaGetLastError(), "resident affine map gradient launch");
}
template<class T> void run_backward(Context<T>& s, const TanhPlan<T>& plan, std::size_t j, T) {
    const auto count = s.batch*plan.block.features;
    if (!count || !s.input_gradient(j)) return;
    tanh_input_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.activation[j+1]), s.ptr(s.upstream[j+1]), s.ptr(s.upstream[j]),
        count, plan.scale, s.status);
    check(cudaGetLastError(), "resident tanh map gradient launch");
}
template<class T> void run_backward(Context<T>& s, const LayerNormPlan<T>& plan, std::size_t j, T) {
    const auto features = plan.block.features;
    if (s.batch && s.input_gradient(j)) {
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
template<class T> void validate_candidates(Context<T>&, const BasisPlan<T>&) {}
template<class T> void validate_candidates(Context<T>& s, const TrainableRbfPlan<T>& plan) {
    const auto& b = plan.expansion.block;
    validate_width_kernel<<<blocks(b.terms),256,0,s.stream>>>(s.ptr(s.candidates+b.nonlinear()+b.terms),b.terms,s.status);
    check(cudaGetLastError(),"resident width validation launch");
}
template<class T> void validate_candidates(Context<T>&, const RationalPlan<T>&) {}
template<class T> void validate_candidates(Context<T>&, const AffinePlan<T>&) {}
template<class T> void validate_candidates(Context<T>&, const TanhPlan<T>&) {}
template<class T> void validate_candidates(Context<T>&, const LayerNormPlan<T>&) {}

// Nonlinear gradients, downloaded asynchronously (the caller synchronizes).
// The destination vectors are moved, never copied, into the result, so the
// heap buffers the pending copies target stay the same.
template<class T> NonlinearGradients download_nonlinear(Context<T>&, const BasisPlan<T>&) { return std::monostate{}; }
template<class T> NonlinearGradients download_nonlinear(Context<T>& s, const TrainableRbfPlan<T>& plan) {
    const auto& b = plan.expansion.block;
    TrainableRbfGradients g{std::vector<double>(b.terms), std::vector<double>(b.terms)};
    s.download(g.centers,s.ptr(s.gradients+b.nonlinear()));
    s.download(g.log_widths,s.ptr(s.gradients+b.nonlinear()+b.terms));
    return g;
}
template<class T> NonlinearGradients download_nonlinear(Context<T>& s, const RationalPlan<T>& plan) {
    RationalGradients g{std::vector<double>(plan.block.nonlinear_count)};
    s.download(g.denominators,s.ptr(s.gradients+plan.block.nonlinear()));
    return g;
}

// The trained carrier: the snapshot's configuration with current parameters.
template<class T>
Carrier download_carrier(Context<T>&, const BasisPlan<T>&, const BasisEdges& snapshot, std::vector<double> coefficients) {
    return BasisEdges{snapshot.basis, std::move(coefficients)};
}
template<class T>
Carrier download_carrier(Context<T>& s, const TrainableRbfPlan<T>& plan, const TrainableRbfEdges&, std::vector<double> coefficients) {
    const auto& b = plan.expansion.block;
    TrainableRbfConfig basis{std::vector<double>(b.terms), std::vector<double>(b.terms)};
    s.download(basis.centers,s.ptr(s.parameters+b.nonlinear()));
    s.download(basis.log_widths,s.ptr(s.parameters+b.nonlinear()+b.terms));
    return TrainableRbfEdges{std::move(basis), std::move(coefficients)};
}
template<class T>
Carrier download_carrier(Context<T>& s, const RationalPlan<T>& plan, const RationalEdges& snapshot, std::vector<double> coefficients) {
    std::vector<double> denominators(plan.block.nonlinear_count);
    s.download(denominators,s.ptr(s.parameters+plan.block.nonlinear()));
    return RationalEdges{snapshot.config, std::move(coefficients), std::move(denominators)};
}

// Gradients of one network layer, downloaded asynchronously (the caller
// synchronizes); destination vectors are moved, never copied, into the result.
template<class T, class P> requires requires { typename P::edges_type; }
NetworkLayerGradients download_stage_gradients(Context<T>& s, const P& plan, std::size_t j) {
    const auto& b = block_of(plan);
    LayerGradients g;
    g.input.resize(product(s.batch, b.inputs)); g.coefficients.resize(b.coefficients); g.bias.resize(b.outputs);
    s.download(g.input, s.ptr(s.upstream[j])); s.download(g.coefficients, s.ptr(s.gradients+b.offset));
    s.download(g.bias, s.ptr(s.gradients+b.bias()));
    g.nonlinear = download_nonlinear(s, plan);
    return NetworkLayerGradients(std::move(g));
}
template<class T> NetworkLayerGradients map_gradients(Context<T>& s, const MapBlock& b, std::size_t j) {
    InputMapGradients g{std::vector<double>(product(s.batch, b.features)), std::vector<double>(b.half()), std::vector<double>(b.half())};
    s.download(g.input, s.ptr(s.upstream[j]));
    s.download(g.gain, s.ptr(s.gradients+b.offset));
    s.download(g.bias, s.ptr(s.gradients+b.offset+b.half()));
    return NetworkLayerGradients(std::move(g));
}
template<class T> NetworkLayerGradients download_stage_gradients(Context<T>& s, const AffinePlan<T>& plan, std::size_t j) { return map_gradients(s, plan.block, j); }
template<class T> NetworkLayerGradients download_stage_gradients(Context<T>& s, const TanhPlan<T>& plan, std::size_t j) { return map_gradients(s, plan.block, j); }
template<class T> NetworkLayerGradients download_stage_gradients(Context<T>& s, const LayerNormPlan<T>& plan, std::size_t j) { return map_gradients(s, plan.block, j); }

// The trained network layer: the snapshot with the current device parameters.
template<class T, class P, class Edges>
NetworkLayer download_stage(Context<T>& s, const P& plan, const Layer& snapshot, const Edges& edges) {
    const auto& b = block_of(plan);
    std::vector<double> coefficients(b.coefficients), bias(b.outputs);
    s.download(coefficients, s.ptr(s.parameters+b.offset)); s.download(bias, s.ptr(s.parameters+b.bias()));
    auto carrier = download_carrier(s, plan, edges, std::move(coefficients));
    s.sync();
    Layer layer = snapshot;
    layer.set_carrier(std::move(carrier), bias);
    return layer;
}
template<class T> NetworkLayer download_stage(Context<T>&, const AffinePlan<T>&, const InputMap& snapshot, const AffineMap&) { return snapshot; }
template<class T> NetworkLayer download_stage(Context<T>&, const TanhPlan<T>&, const InputMap& snapshot, const TanhMap&) { return snapshot; }
template<class T> NetworkLayer download_stage(Context<T>& s, const LayerNormPlan<T>& plan, const InputMap& snapshot, const LayerNormMap& norm) {
    if (!plan.affine()) return snapshot;
    LayerNormMap trained{norm.epsilon, std::vector<double>(plan.block.half()), std::vector<double>(plan.block.half())};
    s.download(trained.gain, s.ptr(s.parameters+plan.block.offset));
    s.download(trained.bias, s.ptr(s.parameters+plan.block.offset+plan.block.half()));
    s.sync();
    InputMap map = snapshot;
    map.set_map(std::move(trained));
    return map;
}

// Parameter upload into an existing executor (backlog R9). The active
// parameter region is one dense vector of parameter_count values (each plan's
// block at its offset). Every source tensor is validated on the host before
// the commit copies anything.

// Structure: what the executor was built for and does not upload. Trainable
// RBF centers/log widths and LayerNorm gain/bias are state (only their counts
// are structure); every other configuration value is compared exactly.
bool same_structure(const BasisEdges& a, const BasisEdges& b) { return a.basis == b.basis; }
bool same_structure(const TrainableRbfEdges& a, const TrainableRbfEdges& b) {
    return a.basis.centers.size() == b.basis.centers.size() && a.basis.log_widths.size() == b.basis.log_widths.size();
}
bool same_structure(const RationalEdges& a, const RationalEdges& b) { return a.config == b.config; }
bool same_structure(const AffineMap& a, const AffineMap& b) { return a == b; }
bool same_structure(const TanhMap& a, const TanhMap& b) { return a == b; }
bool same_structure(const LayerNormMap& a, const LayerNormMap& b) {
    return a.epsilon == b.epsilon && a.gain.size() == b.gain.size() && a.bias.size() == b.bias.size();
}
bool same_dimensions(const Layer& a, const Layer& b) { return a.inputs() == b.inputs() && a.outputs() == b.outputs(); }
bool same_dimensions(const InputMap& a, const InputMap& b) { return a.features() == b.features(); }
const Carrier& kind_of(const Layer& layer) { return layer.carrier(); }
const InputMapKind& kind_of(const InputMap& map) { return map.map(); }
[[noreturn]] void structure_mismatch(std::size_t j, const char* what) {
    throw std::invalid_argument("resident parameter upload: layer " + std::to_string(j) +
                                " differs from the executor's network in its " + what);
}

// Host image of the parameter region in T. put() checks the tensor length,
// finiteness and (FP32) representability with the construction messages.
// Packed: the tensors are converted into one host buffer covering the region
// (the blocks tile it, so every element is written), committed with a single
// copy; always for FP32, which must convert. Direct (FP64 regions above
// direct_upload_bytes): nothing to convert, so the validated source tensors
// are copied directly, one copy per nonempty tensor (at most four per KAN
// layer, two per LayerNorm map). R9 profiling: packing a 117 MB FP64 region
// was 40% of its upload (62 -> 45 ms direct, the copy alone 36 ms), while for
// a 0.4 MB region the six direct copies cost more than packing (0.16 -> 0.24 ms).
constexpr std::size_t direct_upload_bytes = std::size_t{1} << 20;
template<class T>
struct ParameterImage {
    static constexpr bool may_copy_directly = std::is_same_v<T, double>; // no conversion needed
    bool direct;
    std::unique_ptr<T[]> values;                                          // packed
    std::vector<std::pair<std::size_t, std::span<const double>>> sources; // direct
    explicit ParameterImage(std::size_t count)
        : direct(may_copy_directly && count > direct_upload_bytes/sizeof(T)) {
        if (!direct) values = std::make_unique_for_overwrite<T[]>(count);
    }
    void put(std::size_t offset, std::span<const double> data, std::size_t expected) {
        if (data.size() != expected) throw std::invalid_argument("resident parameter upload: parameter shape mismatch");
        if constexpr (may_copy_directly) {
            if (direct) {
                finite(data);
                if (!data.empty()) sources.emplace_back(offset, data);
                return;
            }
        }
        {
            bool nonfinite = false, outside = false;
            T* out = values.get()+offset;
            for (std::size_t i = 0; i < data.size(); ++i) {
                const double v = data[i];
                nonfinite |= !std::isfinite(v);
                if constexpr (!std::is_same_v<T, double>) outside |= !(std::abs(v) <= static_cast<double>(FLT_MAX));
                out[i] = static_cast<T>(outside ? 0.0 : v);
            }
            if (nonfinite) throw std::invalid_argument("resident data must be finite");
            if (outside) throw std::invalid_argument("resident data is not representable in float32");
        }
    }
    // Asynchronous on the context's stream; the caller synchronizes before
    // the sources (the caller's network) or the buffer go away.
    void commit(Context<T>& s, std::size_t region, std::size_t count) const {
        if constexpr (may_copy_directly) {
            if (direct) {
                for (const auto& [offset, data] : sources)
                    check(cudaMemcpyAsync(s.ptr(region+offset), data.data(), data.size_bytes(), cudaMemcpyHostToDevice, s.stream),
                          "resident parameter upload");
                return;
            }
        }
        if (count) {
            check(cudaMemcpyAsync(s.ptr(region), values.get(), count*sizeof(T), cudaMemcpyHostToDevice, s.stream),
                  "resident parameter upload");
        }
    }
};
template<class T> void image_carrier(ParameterImage<T>&, const BasisPlan<T>&, const BasisEdges&) {}
template<class T> void image_carrier(ParameterImage<T>& image, const TrainableRbfPlan<T>& plan, const TrainableRbfEdges& edges) {
    check_basis<T>(edges.basis); // FP32: centers representable, exp(log width) finite and positive
    const auto& b = plan.expansion.block;
    image.put(b.nonlinear(), edges.basis.centers, b.terms);
    image.put(b.nonlinear()+b.terms, edges.basis.log_widths, b.terms);
}
template<class T> void image_carrier(ParameterImage<T>& image, const RationalPlan<T>& plan, const RationalEdges& edges) {
    image.put(plan.block.nonlinear(), edges.denominators, plan.block.nonlinear_count);
}
template<class T, class P, class Edges>
void image_stage(ParameterImage<T>& image, const P& plan, const Layer& layer, const Edges& edges) {
    const auto& b = block_of(plan);
    image.put(b.offset, layer.coefficients(), b.coefficients);
    image.put(b.bias(), layer.bias(), b.outputs);
    image_carrier(image, plan, edges);
}
template<class T> void image_stage(ParameterImage<T>&, const AffinePlan<T>&, const InputMap&, const AffineMap&) {}
template<class T> void image_stage(ParameterImage<T>&, const TanhPlan<T>&, const InputMap&, const TanhMap&) {}
template<class T> void image_stage(ParameterImage<T>& image, const LayerNormPlan<T>& plan, const InputMap&, const LayerNormMap& map) {
    image.put(plan.block.offset, map.gain, plan.block.half());
    image.put(plan.block.offset+plan.block.half(), map.bias, plan.block.half());
}

// Applies f(plan, stage, kind) to a plan, its network layer and the matching
// carrier (KAN layer) or map (input map) alternative.
template<class T, class F> decltype(auto) with_stage(const Plan<T>& plan, const NetworkLayer& stage, F&& f) {
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
// Applies f(plan, stage, kind) to a plan and the matching alternatives of
// `source`, after checking that source's layer j has the structure of the
// executor's snapshot layer (std::invalid_argument otherwise).
template<class T, class F>
void with_matching_stage(const Plan<T>& plan, const NetworkLayer& snapshot, const NetworkLayer& source, std::size_t j, F&& f) {
    with_stage(plan, snapshot, [&](const auto& p, const auto& expected, const auto& expected_kind) {
        using Stage = std::decay_t<decltype(expected)>;
        using Kind = std::decay_t<decltype(expected_kind)>;
        const auto* actual = std::get_if<Stage>(&source);
        if (!actual) structure_mismatch(j, "kind (KAN layer or input map)");
        if (!same_dimensions(expected, *actual)) structure_mismatch(j, "dimensions");
        const auto* actual_kind = std::get_if<Kind>(&kind_of(*actual));
        if (!actual_kind) structure_mismatch(j, std::is_same_v<Stage, Layer> ? "carrier" : "map kind");
        if (!same_structure(expected_kind, *actual_kind)) structure_mismatch(j, "fixed configuration");
        f(p, *actual, *actual_kind);
    });
}
} // namespace

// Precision-independent interface of an execution engine (one per Precision).
struct ResidentExecutor {
    virtual ~ResidentExecutor() = default;
    virtual void upload_input(std::span<const double> input, std::size_t batch) = 0;
    virtual void upload_output_gradient(std::span<const double> gradient) = 0;
    virtual void forward() = 0;
    virtual void backward(double coefficient_l2) = 0;
    virtual void sgd(double learning_rate) = 0;
    virtual void upload_parameters(const Network& network) = 0;
    virtual void train_step(double learning_rate, double coefficient_l2, Loss loss) = 0;
    virtual void train_batch(std::span<const double> input, std::span<const double> target, std::size_t batch,
                             double learning_rate, double coefficient_l2) = 0;
    virtual void upload_target(std::span<const double> target) = 0;
    virtual double download_loss() = 0;
    virtual void set_status_interval(std::size_t steps) = 0;
    virtual std::size_t status_interval() const = 0;
    virtual void check_status() = 0;
    virtual std::size_t trained_steps() const = 0;
    virtual std::vector<double> download_output() = 0;
    virtual NetworkGradients download_gradients() = 0;
    virtual Network download_parameters() = 0;
    virtual void synchronize() = 0;
    virtual std::size_t max_batch() const = 0;
    virtual std::size_t current_batch() const = 0;
    virtual std::size_t allocation_count() const = 0;
};

namespace {
template<class T>
struct Engine final : ResidentExecutor, Context<T> {
    using Context<T>::capacity; using Context<T>::batch; using Context<T>::parameters; using Context<T>::gradients;
    using Context<T>::candidates; using Context<T>::scratch; using Context<T>::ones; using Context<T>::blas_workspace;
    using Context<T>::partials; using Context<T>::activation; using Context<T>::upstream; using Context<T>::arena;
    using Context<T>::status; using Context<T>::stream; using Context<T>::pinned_size; using Context<T>::blas; using Context<T>::ptr; using Context<T>::sync;
    Network model;
    std::size_t allocations = 0, parameter_count = 0;
    std::vector<Plan<T>> plans;
    bool has_input = false, has_upstream = false, has_forward = false, has_backward = false;
    // Training steps (backlog C9). The input and target each have two arena
    // regions: a staged batch is copied into the region the running steps do
    // not read, and activation.front() / target() name the current one.
    DeviceControl* control = nullptr; // the status allocation (status == control->bits)
    std::size_t input_regions[2] = {}, target_regions[2] = {}, loss_value = 0, loss_partials = 0;
    unsigned current = 0; // index of the current input/target region
    bool has_target = false, has_loss = false;
    // Deferred status: `pending` steps issued since the last check, which
    // happens when it reaches `interval`; `confirmed` committed steps as of
    // that check.
    std::size_t interval = 1, pending = 0, confirmed = 0;
    // Captured steps, keyed by everything a replay bakes in except the
    // learning rate, which is a kernel parameter updated in place.
    struct StepKey {
        std::size_t batch, parameters, input, target;
        Loss loss;
        T lambda;
        bool staged; // waits for the staged target inside the graph
        bool operator==(const StepKey&) const = default;
    };
    struct CandidateArgs { const T* parameters; const T* gradients; T* next; std::size_t count; T rate; int* status; };
    struct StepGraph {
        StepKey key;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaGraphNode_t candidate = nullptr; // the SGD candidate kernel (null without parameters)
        CandidateArgs args{};
        std::uint64_t used = 0;
    };
    static constexpr std::size_t max_graphs = 8;
    std::vector<StepGraph> graphs;
    std::uint64_t graph_clock = 0;
    // Staged batches: a copy stream, two page-locked host slots (input then
    // target, `stage_stride` elements each) and per-slot events on the copy
    // stream, `input_copied` / `target_copied` (the slot's input / target is
    // on the device), and on the main stream `released` (the last step reading
    // the slot's device regions has finished). The step waits for its input
    // before the graph launch and for its target inside the graph, before the
    // loss, so the target copy overlaps the forward pass.
    // Created on first use; host memory, not counted by workspace_allocations().
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t input_copied[2] = {}, target_copied[2] = {}, released[2] = {};
    T* stage_host = nullptr;
    std::size_t stage_stride = 0;
    bool staging = false;
    // tensor_ops: cuBLAS TF32 tensor-op math for the FP32 GEMMs (Precision::TensorFloat32).
    Engine(const Network& source, std::size_t maximum, bool tensor_ops = false) : Context<T>{maximum}, model(source) {
        if (model.layers().empty()) throw std::invalid_argument("resident network is empty or moved from");
        // Validate the copied CPU state and all shape arithmetic before CUDA allocation.
        model.forward({}, 0);
        for (const auto& stage : model.layers()) {
            plans.push_back(std::visit([&](const auto& s) { return make_plan<T>(s, parameter_count); }, stage));
            parameter_count += extent(plans.back()).size;
        }
        Reservation<T> reserve;
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
        blas_workspace = reserve(blas_workspace_bytes/sizeof(T));
        // C9 regions after every earlier one, so the pre-C9 layout (and the
        // operands cuBLAS sees) is unchanged.
        input_regions[0] = activation.front();
        input_regions[1] = reserve(product(capacity, extent(plans.front()).inputs));
        target_regions[0] = reserve(product(capacity, extent(plans.back()).outputs));
        target_regions[1] = reserve(product(capacity, extent(plans.back()).outputs));
        loss_value = reserve(1); loss_partials = reserve(mse_max_blocks);
        graphs.reserve(max_graphs);
        const auto bytes = product(reserve.total, sizeof(T));
        try {
            if (!available()) throw std::runtime_error("no CUDA device available");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "resident stream create");
            check(cudaMalloc(&arena, bytes), "resident arena allocation"); ++allocations;
            check(cudaMalloc(&control, sizeof(DeviceControl)), "resident status allocation"); ++allocations;
            status = control->bits;
            check(cudaMemsetAsync(control, 0, sizeof(DeviceControl), stream), "resident status reset");
            if constexpr (!std::is_same_v<T, double>) {
                pinned_size = product(capacity, std::max(extent(plans.front()).inputs, extent(plans.back()).outputs));
                if (pinned_size) check(cudaMallocHost(&this->pinned, pinned_size*sizeof(T)), "resident pinned staging allocation");
            }
            // The handle runs on the network stream with a workspace inside the
            // arena (set after the stream: cublasSetStream resets it).
            check(cublasCreate(&blas), "resident cuBLAS handle create");
            check(cublasSetStream(blas, stream), "resident cuBLAS stream");
            check(cublasSetWorkspace(blas, ptr(blas_workspace), blas_workspace_bytes), "resident cuBLAS workspace");
            if (tensor_ops) check(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH), "resident cuBLAS TF32 math");
            if (capacity) {
                fill_kernel<<<blocks(capacity), 256, 0, stream>>>(ptr(ones), capacity, T(1));
                check(cudaGetLastError(), "resident ones launch");
            }
            for (std::size_t j = 0; j < plans.size(); ++j)
                with_stage(plans[j], model.layers()[j], [&](const auto& p, const auto& stage, const auto& kind) {
                    upload_stage(*this, p, stage, kind);
                });
            sync();
        } catch (...) { cleanup(); throw; }
    }
    ~Engine() override { cleanup(); }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    void cleanup() noexcept {
        if (stream) cudaStreamSynchronize(stream);
        if (copy_stream) cudaStreamSynchronize(copy_stream);
        for (auto& g : graphs) destroy(g);
        graphs.clear();
        for (auto* e : {&input_copied[0], &input_copied[1], &target_copied[0], &target_copied[1], &released[0], &released[1]}) if (*e) { cudaEventDestroy(*e); *e = nullptr; }
        if (copy_stream) cudaStreamDestroy(copy_stream);
        if (stage_host) cudaFreeHost(stage_host);
        copy_stream = nullptr; stage_host = nullptr; staging = false; control = nullptr;
        if (blas) cublasDestroy(blas);
        if (arena) cudaFree(arena);
        if (status) cudaFree(status);
        if (stream) cudaStreamDestroy(stream);
        if (this->pinned) cudaFreeHost(this->pinned);
        arena = nullptr; status = nullptr; stream = nullptr; blas = nullptr; this->pinned = nullptr;
    }
    void reset_status() { check(cudaMemsetAsync(status, 0, sizeof(int), stream), "resident status reset"); }
    // Eager calls report through bits[0] and leave it clear, so that a
    // training interval always starts with every status word zero.
    void result() {
        check(cudaGetLastError(), "resident kernel launch");
        int value = 0;
        check(cudaMemcpyAsync(&value, status, sizeof(int), cudaMemcpyDeviceToHost, stream), "resident status download");
        sync();
        if (value) reset_status();
        if (value&1) throw std::overflow_error("nonfinite resident numerical result");
        if (value&2) throw std::domain_error("unsafe resident rational denominator");
    }

    // Launch sequences shared by the eager calls and the captured steps, so
    // that a replay runs exactly the eager kernels.
    void enqueue_forward() {
        for (std::size_t j = 0; j < plans.size() && batch; ++j)
            std::visit([&](const auto& plan) { run_forward(*this, plan, j); }, plans[j]);
    }
    void enqueue_backward(T lambda) {
        for (std::size_t j = plans.size(); j-- > 0;)
            std::visit([&](const auto& plan) { run_backward(*this, plan, j, lambda); }, plans[j]);
    }
    // `node` (during capture only): receives the candidate kernel's graph node.
    void enqueue_update(T rate, cudaGraphNode_t* node = nullptr) {
        if (parameter_count) { // zero only for networks of fixed input maps
            candidate_kernel<<<blocks(parameter_count), 256, 0, stream>>>(ptr(parameters), ptr(gradients), ptr(candidates), parameter_count, rate, status);
            check(cudaGetLastError(),"resident candidate launch");
            if (node) *node = last_captured_node();
        }
        for (const auto& plan : plans) std::visit([&](const auto& p) { validate_candidates(*this, p); }, plan);
    }
    void enqueue_loss() {
        const auto count = batch*extent(plans.back()).outputs;
        const auto grid = std::min(blocks(count, mse_threads), mse_max_blocks);
        mse_kernel<<<grid, mse_threads, 0, stream>>>(ptr(activation.back()), ptr(target_regions[current]), ptr(upstream.back()),
            count, narrow<T>(2.0/static_cast<double>(count)), static_cast<T>(count), ptr(loss_partials), ptr(loss_value),
            &control->ticket, status);
        check(cudaGetLastError(), "resident loss launch");
    }

    // Deferred status check (C9): reads the device control block once and
    // raises the first failure of the interval, attributed to its step.
    void flush() {
        if (!pending) return;
        const auto issued = pending;
        pending = 0;
        DeviceControl c{};
        check(cudaMemcpyAsync(&c, control, sizeof c, cudaMemcpyDeviceToHost, stream), "resident status download");
        sync();
        const auto before = confirmed;
        confirmed = static_cast<std::size_t>(c.committed);
        if (!c.first_bits) return;
        check(cudaMemsetAsync(control, 0, offsetof(DeviceControl, ticket), stream), "resident status reset");
        sync();
        has_forward = has_backward = has_loss = false;
        const auto skipped = before + issued - confirmed - 1;
        const auto where = " in training step " + std::to_string(confirmed) +
            " (deferred status check; that step and the " + std::to_string(skipped) +
            " later steps of the interval committed no update)";
        if (c.first_bits&1) throw std::overflow_error("nonfinite resident numerical result" + where);
        throw std::domain_error("unsafe resident rational denominator" + where);
    }
    static std::pair<T, T> step_scalars(double learning_rate, double coefficient_l2) {
        if (!std::isfinite(learning_rate) || learning_rate <= 0) throw std::invalid_argument("learning rate must be finite and positive");
        if (!std::isfinite(coefficient_l2) || coefficient_l2 < 0) throw std::invalid_argument("coefficient L2 must be finite and nonnegative");
        return {narrow_positive<T>(learning_rate), narrow<T>(coefficient_l2)};
    }

    static void destroy(StepGraph& g) noexcept {
        if (g.exec) cudaGraphExecDestroy(g.exec);
        if (g.graph) cudaGraphDestroy(g.graph);
        g.exec = nullptr; g.graph = nullptr;
    }
    // Captures one training step on the stream (nothing executes): forward
    // and loss report into bits[0], backward into bits[1], SGD into bits[2].
    // The node the next captured operation would depend on: right after a
    // kernel launch, that kernel's node (CUDA 13 added an edge-data argument).
    cudaGraphNode_t last_captured_node() {
        cudaStreamCaptureStatus capturing{};
        const cudaGraphNode_t* dependencies = nullptr;
        std::size_t count = 0;
#if CUDART_VERSION >= 13000
        check(cudaStreamGetCaptureInfo(stream, &capturing, nullptr, nullptr, &dependencies, nullptr, &count), "resident capture info");
#else
        check(cudaStreamGetCaptureInfo(stream, &capturing, nullptr, nullptr, &dependencies, &count), "resident capture info");
#endif
        if (capturing != cudaStreamCaptureStatusActive || count != 1) throw std::runtime_error("resident step capture: SGD node not found");
        return dependencies[0];
    }
    cudaGraph_t record_step(Loss loss, T rate, T lambda, bool staged, cudaGraphNode_t* candidate) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "resident step capture");
        try {
            status = control->bits;
            enqueue_forward();
            if (staged) check(cudaStreamWaitEvent(stream, target_copied[current], cudaEventWaitExternal), "resident staged target wait");
            if (loss == Loss::MeanSquaredError) enqueue_loss();
            status = control->bits+1;
            this->skip_network_input_gradient = true;
            enqueue_backward(lambda);
            this->skip_network_input_gradient = false;
            status = control->bits+2;
            enqueue_update(rate, candidate);
            commit_kernel<<<parameter_count ? std::min(blocks(parameter_count), commit_max_blocks) : 1U, 256, 0, stream>>>(
                ptr(parameters), ptr(candidates), parameter_count, control);
            check(cudaGetLastError(), "resident commit launch");
        } catch (...) {
            status = control->bits;
            this->skip_network_input_gradient = false;
            cudaGraph_t partial = nullptr;
            cudaStreamEndCapture(stream, &partial);
            if (partial) cudaGraphDestroy(partial);
            cudaGetLastError();
            throw;
        }
        status = control->bits;
        cudaGraph_t graph = nullptr;
        check(cudaStreamEndCapture(stream, &graph), "resident step capture");
        return graph;
    }
    StepGraph& step_graph(const StepKey& key, T rate) {
        for (auto& g : graphs) if (g.key == key) { g.used = ++graph_clock; return g; }
        StepGraph entry{key};
        entry.graph = record_step(key.loss, rate, key.lambda, key.staged, &entry.candidate);
        try {
            entry.args = {ptr(parameters), ptr(gradients), ptr(candidates), parameter_count, rate, control->bits+2};
            check(cudaGraphInstantiate(&entry.exec, entry.graph, 0), "resident step graph instantiate");
        } catch (...) { destroy(entry); throw; }
        entry.used = ++graph_clock;
        if (graphs.size() < max_graphs) { graphs.push_back(entry); return graphs.back(); }
        auto& lru = *std::min_element(graphs.begin(), graphs.end(), [](const auto& a, const auto& b) { return a.used < b.used; });
        destroy(lru);
        lru = entry;
        return lru;
    }
    // The learning rate is the candidate kernel's parameter: a schedule
    // updates the instantiated node instead of re-capturing (future launches only).
    void set_rate(StepGraph& g, T rate) {
        if (!g.candidate || g.args.rate == rate) return;
        cudaKernelNodeParams p{};
        check(cudaGraphKernelNodeGetParams(g.candidate, &p), "resident step rate update");
        auto args = g.args;
        args.rate = rate;
        void* values[] = {&args.parameters, &args.gradients, &args.next, &args.count, &args.rate, &args.status};
        p.kernelParams = values; p.extra = nullptr;
        check(cudaGraphExecKernelNodeSetParams(g.exec, g.candidate, &p), "resident step rate update");
        g.args.rate = rate;
    }
    void launch_step(Loss loss, T rate, T lambda, bool staged = false) {
        const StepKey key{batch, parameters, activation.front(), target_regions[current], loss, lambda, staged};
        auto& graph = step_graph(key, rate);
        set_rate(graph, rate);
        check(cudaGraphLaunch(graph.exec, stream), "resident step launch");
        if (staging) check(cudaEventRecord(released[current], stream), "resident staging event");
        // Commit or rollback happened on the device (commit_kernel).
        std::swap(parameters, candidates);
        has_forward = has_backward = false;
        has_loss = loss == Loss::MeanSquaredError;
        if (has_loss) has_upstream = false;
        if (++pending >= interval) flush();
    }
    void ensure_staging() {
        if (staging) return;
        stage_stride = product(capacity, extent(plans.front()).inputs) + product(capacity, extent(plans.back()).outputs);
        if (!copy_stream) check(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "resident copy stream create");
        for (auto* e : {&input_copied[0], &input_copied[1], &target_copied[0], &target_copied[1], &released[0], &released[1]})
            if (!*e) check(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "resident staging event create");
        if (!stage_host) check(cudaMallocHost(&stage_host, product(product(stage_stride, 2), sizeof(T))), "resident staging allocation");
        staging = true;
    }
    // Validating conversion into staging: finite and (FP32) at most FLT_MAX
    // in magnitude, with the upload messages. One branch-free pass (the clamp
    // keeps the conversion defined for rejected values); large batches are
    // split over host threads, because one core converts only about 1 GB/s
    // of doubles (C9 profiling: 9.5 ms for the 1024-wide step's 8M values,
    // 3-4 ms on 4-8 threads), which would otherwise serialize with the copy.
    static void stage_values(T* out, std::span<const double> data) {
        constexpr double limit = std::is_same_v<T, double> ? DBL_MAX : static_cast<double>(FLT_MAX);
        const auto convert = [&](std::size_t begin, std::size_t end) {
            int bad = 0;
            for (std::size_t i = begin; i < end; ++i) {
                const double v = data[i];
                bad |= !(std::abs(v) <= limit);
                out[i] = static_cast<T>(std::clamp(v, -limit, limit));
            }
            return bad != 0;
        };
        bool bad = false;
        const std::size_t hardware = std::max(1U, std::thread::hardware_concurrency());
        const auto threads = std::min({stage_threads_max, hardware, data.size()/stage_chunk});
        if (threads < 2) {
            bad = convert(0, data.size());
        } else {
            std::vector<char> flags(threads, 0);
            std::vector<std::thread> pool;
            pool.reserve(threads-1);
            const auto range = [&](std::size_t k) { return data.size()*k/threads; };
            for (std::size_t k = 1; k < threads; ++k)
                pool.emplace_back([&, k] { flags[k] = convert(range(k), range(k+1)); });
            flags[0] = convert(0, range(1));
            for (auto& thread : pool) thread.join();
            bad = std::find(flags.begin(), flags.end(), char{1}) != flags.end();
        }
        if (!bad) return;
        for (double v : data) if (!std::isfinite(v)) throw std::invalid_argument("resident data must be finite");
        throw std::invalid_argument("resident data is not representable in float32");
    }

    void upload_input(std::span<const double> input, std::size_t rows) override {
        flush();
        const auto count = product(rows, extent(plans.front()).inputs);
        if (rows > capacity || input.size() != count) throw std::invalid_argument("resident input shape or capacity mismatch");
        if constexpr (std::is_same_v<T, double>) { finite(input); this->upload(ptr(activation.front()), input); }
        else this->upload_pinned(ptr(activation.front()), input);
        sync();
        batch = rows; has_input = true; has_upstream = has_forward = has_backward = has_target = false;
    }
    void upload_output_gradient(std::span<const double> gradient) override {
        flush();
        if (!has_input) throw std::logic_error("resident input must be uploaded first");
        if (gradient.size() != product(batch, extent(plans.back()).outputs)) throw std::invalid_argument("resident upstream shape mismatch");
        if constexpr (std::is_same_v<T, double>) { finite(gradient); this->upload(ptr(upstream.back()), gradient); }
        else this->upload_pinned(ptr(upstream.back()), gradient);
        sync();
        has_upstream = true; has_backward = false;
    }
    void upload_target(std::span<const double> target) override {
        flush();
        if (!has_input) throw std::logic_error("resident input must be uploaded first");
        if (target.size() != product(batch, extent(plans.back()).outputs)) throw std::invalid_argument("resident target shape mismatch");
        if constexpr (std::is_same_v<T, double>) { finite(target); this->upload(ptr(target_regions[current]), target); }
        else this->upload_pinned(ptr(target_regions[current]), target);
        sync();
        has_target = true;
    }
    void forward() override {
        flush();
        if (!has_input) throw std::logic_error("resident input has not been uploaded");
        has_forward = has_backward = false; reset_status();
        enqueue_forward();
        result(); has_forward = true;
    }
    void backward(double coefficient_l2) override {
        flush();
        if(!std::isfinite(coefficient_l2)||coefficient_l2<0)throw std::invalid_argument("coefficient L2 must be finite and nonnegative");
        const T lambda = narrow<T>(coefficient_l2);
        if (!has_forward || !has_upstream) throw std::logic_error("resident backward requires current forward and upstream");
        has_backward = false; reset_status();
        enqueue_backward(lambda);
        result(); has_backward = true;
    }
    void sgd(double learning_rate) override {
        flush();
        if (!std::isfinite(learning_rate) || learning_rate <= 0) throw std::invalid_argument("learning rate must be finite and positive");
        const T rate = narrow_positive<T>(learning_rate);
        if (!has_backward) throw std::logic_error("resident SGD requires current gradients");
        reset_status();
        enqueue_update(rate);
        result(); // All layers validated before any parameter mutation.
        // Both regions are permanently reserved and candidate execution is complete.
        // Changing the active region commits the whole network without a tensor copy.
        std::swap(parameters, candidates);
        has_forward = has_backward = false;
    }
    void train_step(double learning_rate, double coefficient_l2, Loss loss) override {
        const auto [rate, lambda] = step_scalars(learning_rate, coefficient_l2);
        if (loss != Loss::OutputGradient && loss != Loss::MeanSquaredError) throw std::invalid_argument("invalid resident loss");
        if (!has_input) throw std::logic_error("resident input has not been uploaded");
        if (loss == Loss::OutputGradient && !has_upstream)
            throw std::logic_error("resident output-gradient training step requires an uploaded upstream");
        if (loss == Loss::MeanSquaredError) {
            if (!has_target) throw std::logic_error("resident mean squared error requires an uploaded target");
            if (!batch) throw std::invalid_argument("resident mean squared error requires a nonempty batch");
        }
        launch_step(loss, rate, lambda);
    }
    void train_batch(std::span<const double> input, std::span<const double> target, std::size_t rows,
                     double learning_rate, double coefficient_l2) override {
        const auto [rate, lambda] = step_scalars(learning_rate, coefficient_l2);
        if (rows > capacity) throw std::invalid_argument("resident input shape or capacity mismatch");
        const auto inputs = product(rows, extent(plans.front()).inputs), outputs = product(rows, extent(plans.back()).outputs);
        if (input.size() != inputs) throw std::invalid_argument("resident input shape or capacity mismatch");
        if (target.size() != outputs) throw std::invalid_argument("resident target shape mismatch");
        if (!rows) throw std::invalid_argument("resident mean squared error requires a nonempty batch");
        ensure_staging();
        // Slot `next` was last read by the step two staged steps ago: wait for
        // it (normally long done) before reusing its host and device buffers.
        const unsigned next = 1-current;
        check(cudaEventSynchronize(released[next]), "resident staging wait");
        T* host = stage_host+next*stage_stride;
        // The input copy starts while the target converts. Slot `next` is not
        // the executor's input until the switch below, so a rejected target
        // leaves nothing observable (once the copy has drained).
        stage_values(host, input);
        check(cudaMemcpyAsync(ptr(input_regions[next]), host, inputs*sizeof(T), cudaMemcpyHostToDevice, copy_stream), "resident staged upload");
        check(cudaEventRecord(input_copied[next], copy_stream), "resident staging event");
        try { stage_values(host+inputs, target); }
        catch (...) { cudaStreamSynchronize(copy_stream); throw; }
        check(cudaMemcpyAsync(ptr(target_regions[next]), host+inputs, outputs*sizeof(T), cudaMemcpyHostToDevice, copy_stream), "resident staged upload");
        check(cudaEventRecord(target_copied[next], copy_stream), "resident staging event");
        check(cudaStreamWaitEvent(stream, input_copied[next], 0), "resident staging wait");
        current = next; activation.front() = input_regions[next]; batch = rows;
        has_input = has_target = true; has_upstream = false;
        launch_step(Loss::MeanSquaredError, rate, lambda, true);
    }
    double download_loss() override {
        flush();
        if (!has_loss) throw std::logic_error("resident loss requires a current mean squared error training step");
        T value{};
        check(cudaMemcpyAsync(&value, ptr(loss_value), sizeof(T), cudaMemcpyDeviceToHost, stream), "resident loss download");
        sync();
        return static_cast<double>(value);
    }
    void set_status_interval(std::size_t steps) override {
        if (!steps) throw std::invalid_argument("resident status interval must be positive");
        flush();
        interval = steps;
    }
    std::size_t status_interval() const override { return interval; }
    void check_status() override { flush(); }
    std::size_t trained_steps() const override { return confirmed; }
    void upload_parameters(const Network& source) override {
        flush();
        const auto stages = source.layers();
        if (stages.size() != plans.size())
            throw std::invalid_argument("resident parameter upload: the layer count differs from the executor's network");
        // Host only until every layer is checked and staged: structure first,
        // then values with the construction rules. Nothing on the device or
        // in the lifecycle state changes before the commit below.
        for (std::size_t j = 0; j < plans.size(); ++j)
            with_matching_stage(plans[j], model.layers()[j], stages[j], j, [](const auto&, const auto&, const auto&) {});
        ParameterImage<T> image(parameter_count);
        for (std::size_t j = 0; j < plans.size(); ++j)
            with_matching_stage(plans[j], model.layers()[j], stages[j], j, [&](const auto& plan, const auto& stage, const auto& kind) {
                image_stage(image, plan, stage, kind);
            });
        // Commit into the active region. The stream is idle (every call
        // completes before returning); outputs and gradients of the old
        // parameters become stale, the input and upstream stay valid.
        // Captured training steps read the region at replay and see the values.
        has_forward = has_backward = false;
        image.commit(*this, parameters, parameter_count);
        sync();
    }
    std::vector<double> download_output() override {
        flush();
        if (!has_forward) throw std::logic_error("resident output requires current forward");
        std::vector<double> output(product(batch, extent(plans.back()).outputs));
        this->download(output, ptr(activation.back())); sync(); return output;
    }
    NetworkGradients download_gradients() override {
        flush();
        if (!has_backward) throw std::logic_error("resident gradients require current backward");
        NetworkGradients gradient; gradient.layers.reserve(plans.size());
        for (std::size_t j = 0; j < plans.size(); ++j)
            gradient.layers.push_back(std::visit([&](const auto& plan) { return download_stage_gradients(*this, plan, j); }, plans[j]));
        sync();
        gradient.input = std::visit([](const auto& g) { return g.input; }, gradient.layers.front());
        return gradient;
    }
    Network download_parameters() override {
        flush();
        std::vector<NetworkLayer> layers;
        layers.reserve(plans.size());
        for (std::size_t j = 0; j < plans.size(); ++j)
            layers.push_back(with_stage(plans[j], model.layers()[j], [&](const auto& plan, const auto& stage, const auto& kind) {
                return download_stage(*this, plan, stage, kind);
            }));
        return Network(std::move(layers));
    }
    void synchronize() override { flush(); sync(); }
    std::size_t max_batch() const override { return capacity; }
    std::size_t current_batch() const override { return batch; }
    std::size_t allocation_count() const override { return allocations; }
};
} // namespace

struct ResidentNetwork::Impl {
    Precision precision;
    std::unique_ptr<ResidentExecutor> engine;
};

ResidentNetwork::ResidentNetwork(const Network& network, std::size_t capacity, Precision precision) {
    switch (precision) {
    case Precision::Float64: impl_ = std::make_unique<Impl>(Impl{precision, std::make_unique<Engine<double>>(network, capacity)}); break;
    case Precision::Float32: impl_ = std::make_unique<Impl>(Impl{precision, std::make_unique<Engine<float>>(network, capacity)}); break;
    case Precision::TensorFloat32:
        impl_ = std::make_unique<Impl>(Impl{precision, std::make_unique<Engine<float>>(network, capacity, true)}); break;
    default: throw std::invalid_argument("invalid resident precision");
    }
}
ResidentNetwork::~ResidentNetwork() = default;
ResidentNetwork::ResidentNetwork(ResidentNetwork&&) noexcept = default;
ResidentNetwork& ResidentNetwork::operator=(ResidentNetwork&&) noexcept = default;
ResidentNetwork::Impl& ResidentNetwork::state() const {
    if (!impl_) throw std::logic_error("resident network is moved from");
    return *impl_;
}
void ResidentNetwork::upload_input(std::span<const double> input, std::size_t batch) { state().engine->upload_input(input, batch); }
void ResidentNetwork::upload_output_gradient(std::span<const double> gradient) { state().engine->upload_output_gradient(gradient); }
void ResidentNetwork::forward() { state().engine->forward(); }
void ResidentNetwork::backward(double coefficient_l2) { state().engine->backward(coefficient_l2); }
void ResidentNetwork::sgd(double learning_rate) { state().engine->sgd(learning_rate); }
void ResidentNetwork::upload_parameters(const Network& network) { state().engine->upload_parameters(network); }
void ResidentNetwork::train_step(double learning_rate, double coefficient_l2, Loss loss) {
    state().engine->train_step(learning_rate, coefficient_l2, loss);
}
void ResidentNetwork::train_step(std::span<const double> input, std::span<const double> target, std::size_t batch,
                                 double learning_rate, double coefficient_l2) {
    state().engine->train_batch(input, target, batch, learning_rate, coefficient_l2);
}
void ResidentNetwork::upload_target(std::span<const double> target) { state().engine->upload_target(target); }
double ResidentNetwork::download_loss() { return state().engine->download_loss(); }
void ResidentNetwork::set_status_interval(std::size_t steps) { state().engine->set_status_interval(steps); }
std::size_t ResidentNetwork::status_interval() const { return state().engine->status_interval(); }
void ResidentNetwork::check_status() { state().engine->check_status(); }
std::size_t ResidentNetwork::trained_steps() const { return state().engine->trained_steps(); }
std::vector<double> ResidentNetwork::download_output() { return state().engine->download_output(); }
NetworkGradients ResidentNetwork::download_gradients() { return state().engine->download_gradients(); }
Network ResidentNetwork::download_parameters() { return state().engine->download_parameters(); }
void ResidentNetwork::synchronize() { state().engine->synchronize(); }
std::size_t ResidentNetwork::capacity() const { return state().engine->max_batch(); }
std::size_t ResidentNetwork::batch() const { return state().engine->current_batch(); }
std::size_t ResidentNetwork::workspace_allocations() const { return state().engine->allocation_count(); }
Precision ResidentNetwork::precision() const { return state().precision; }
} // namespace kan::cuda

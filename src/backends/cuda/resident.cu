#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include "carriers/basis_view.hpp"
#include "carriers/local/grid_refit.hpp"
#include "carriers/rational/rational_formulas.hpp"
#include "core/input_map_formulas.hpp"
#include "core/residual_formulas.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <numeric>
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
namespace testing {
namespace {
// Active ResidentTestScope of this thread (backlog C5 test hooks).
struct TestScopeState {
    SplinePath path = SplinePath::Automatic;
    ResidentLayout* layout = nullptr;
    NonlinearReduction reduction = NonlinearReduction::Automatic;
    WaveletLaunch wavelet{};
};
thread_local TestScopeState test_scope;
} // namespace
ResidentTestScope::ResidentTestScope(SplinePath path, ResidentLayout* layout) { test_scope = {path, layout}; }
ResidentTestScope::ResidentTestScope(NonlinearReduction reduction, ResidentLayout* layout) { test_scope = {SplinePath::Automatic, layout, reduction}; }
ResidentTestScope::ResidentTestScope(WaveletLaunch launch, ResidentLayout* layout) {
    test_scope = {SplinePath::Automatic, layout, NonlinearReduction::Automatic, launch};
}
ResidentTestScope::~ResidentTestScope() { test_scope = {}; }
} // namespace testing
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

// Finiteness test of the device status checks. FP64: an integer test of the
// exponent field (all ones for inf and NaN), the exact equivalent of
// isfinite; isfinite compiles to DSETP on the FP64 pipe, which GA102 runs at
// 1/64 of the FP32 rate, where these checks were a large share of the FP64
// instructions of the rational and basis kernels (backlog C7 profiling).
__device__ __forceinline__ bool finite_value(double value) {
    return (__double2hiint(value) & 0x7ff00000) != 0x7ff00000;
}
__device__ __forceinline__ bool finite_value(float value) { return isfinite(value); }
template<class T> __device__ void report(T value, int* status) { if (!finite_value(value)) atomicOr(status, 1); }
// Status bits: 1 a nonfinite result, 2 an unsafe rational denominator (domain
// error), status_parameter_range a finite SGD candidate outside its parameter's
// valid range (wavelet log scales, backlog M7; RBF log widths still report bit 1).
constexpr int status_parameter_range = 4;
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

// S = silu(X) and silu'(X), (batch, inputs) row-major, of a layer with the
// SiLU residual branch (backlog M3).
template<class T> struct SiluRows {
    T* values;
    T* derivatives;
};
// Both from one sigmoid (one exp), checked like every formula result.
template<class T> __device__ void write_silu(SiluRows<T> silu, std::size_t index, T x, const StatusGuard& guard) {
    const auto f = detail::silu(x, guard);
    silu.values[index] = f.value; silu.derivatives[index] = f.derivative;
}

// derivatives = null (staged path only, backlog C3): Phi' is evaluated into the
// stage but not written; the backward pass recomputes it.
// Residual (backlog M3): the pass that reads X also writes the layer's S and
// silu' rows; the instantiation without it is the kernel as before.
// Per-input spline grids (backlog M6): row r = (sample, input) of a layer
// reads the knot vector at knots + (r % inputs) * stride. The input index of a
// grid-stride loop advances incrementally (one 64-bit remainder per thread).
struct KnotRows {
    std::size_t inputs, stride;
};
template<bool Grids> struct RowInput {
    __device__ RowInput(std::size_t, std::size_t, const KnotRows&) {}
    __device__ void advance(const KnotRows&) {}
    template<class T> __device__ const T* knots(const T* knots, const KnotRows&) const { return knots; }
};
template<> struct RowInput<true> {
    std::size_t input, step; // the row's input and the loop step's remainder, both < inputs
    __device__ RowInput(std::size_t row, std::size_t loop_step, const KnotRows& rows)
        : input(row % rows.inputs), step(loop_step % rows.inputs) {}
    __device__ void advance(const KnotRows& rows) {
        input += step;
        if (input >= rows.inputs) input -= rows.inputs;
    }
    template<class T> __device__ const T* knots(const T* knots, const KnotRows& rows) const { return knots + input * rows.stride; }
};
template<detail::BasisKind Kind, bool Residual, class T>
__global__ void basis_kernel(const T* input, T* values, T* derivatives, T* log_derivatives,
                             std::size_t count, detail::BasisViewOf<T> basis, unsigned tile_rows, SiluRows<T> silu,
                             int* status) {
    const StatusGuard guard{status};
    const auto terms = basis.terms;
    if (tile_rows == 0) {
        const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += stride) {
            const auto row = index * terms;
            if constexpr (Kind == detail::BasisKind::BSpline && !std::is_same_v<T, double>) {
                if (!derivatives) {
                    const T x = input[index];
                    const detail::BasisRowOf<T> out{values + row, nullptr, nullptr, nullptr};
                    // FP32/TF32 B-spline rows beyond the stage (backlog C13): the input
                    // VJP reads no Phi', so only the values are written, from the
                    // sparse form of the same evaluation (identical values; the
                    // derivatives are still evaluated and checked).
                    T v[detail::max_spline_degree+1], d[detail::max_spline_degree+1];
                    for (std::size_t k = 0; k < terms; ++k) out.values[k] = 0;
                    const auto g = detail::spline_span_terms(basis.knots, terms, basis.degree, x, v, d, guard);
                    if (g != terms) for (std::size_t r = 0; r <= basis.degree; ++r) out.values[g+r] = v[r];
                    if constexpr (Residual) write_silu(silu, index, x, guard);
                    continue;
                }
            }
            const detail::BasisRowOf<T> out{values + row, derivatives + row, nullptr,
                                            basis.trainable ? log_derivatives + row : nullptr};
            const T x = input[index];
            detail::basis_terms_for<Kind>(basis, x, out, guard);
            if constexpr (Residual) write_silu(silu, index, x, guard);
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
            const T x = input[base + threadIdx.x];
            detail::basis_terms_for<Kind>(basis, x, out, guard);
            if constexpr (Residual) write_silu(silu, base + threadIdx.x, x, guard);
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
// B-spline layers with per-input grids (backlog M6): basis_kernel's B-spline
// rows with each row's own knot vector (a separate kernel, so the
// basis_kernel instantiations are unchanged). Consecutive rows belong to
// consecutive inputs, so a warp's knot reads diverge over up to 32 knot rows
// (nsys: 1.17x the shared-grid kernel at 256 inputs). With `staged` != 0 the
// block first copies the knot rows its rows use into shared memory after the
// stage (row stride `pad`, odd: conflict-free), which needs every block to
// see the same inputs in every iteration of its grid-stride loop: the host
// launches a grid whose loop step is a multiple of the inputs, or a single
// iteration. staged rows: the block's rows per iteration if they are at most
// the inputs (row l of the block reads staged row l), otherwise all inputs
// (staged row = input).
// Degree: the layer's degree at compile time for degrees 0..3 (the
// Cox-de Boor scratch then lives in registers; with a runtime degree it is
// local memory, whose L1 traffic made the kernel depend on the shared/L1
// split: 1.2x slower at 4 resident blocks per SM than at 2 on GB10), -1 for
// any degree.
template<bool Residual, int Degree, class T>
__global__ void basis_grid_kernel(const T* input, T* values, T* derivatives, std::size_t count, detail::BasisViewOf<T> basis,
                                  unsigned tile_rows, SiluRows<T> silu, int* status, KnotRows knot_rows, unsigned staged, unsigned pad) {
    const StatusGuard guard{status};
    const auto terms = basis.terms;
    if constexpr (Degree >= 0) basis.degree = Degree;
    const auto per_block = static_cast<std::size_t>(tile_rows ? tile_rows : blockDim.x);
    const auto knot_count = terms + basis.degree + 1;
    T* stage = stage_memory<T>();
    T* knot_stage = stage + 2 * static_cast<std::size_t>(tile_rows) * terms;
    const auto first = static_cast<std::size_t>(blockIdx.x) * per_block % knot_rows.inputs;
    const bool all_inputs = per_block > knot_rows.inputs;
    if (staged) {
        for (auto e = static_cast<std::size_t>(threadIdx.x); e < staged * knot_count; e += blockDim.x) {
            const auto row = e / knot_count, k = e % knot_count;
            const auto i = all_inputs ? row : (first + row) % knot_rows.inputs;
            knot_stage[row * pad + k] = basis.knots[i * knot_rows.stride + k];
        }
        __syncthreads();
    }
    // The knots of a thread's rows (the same input in every iteration when staged).
    const auto staged_row = all_inputs ? (first + threadIdx.x) % knot_rows.inputs : threadIdx.x;
    const auto row_knots = [&](const RowInput<true>& at) {
        return staged ? knot_stage + staged_row * pad : at.knots(basis.knots, knot_rows);
    };
    if (tile_rows == 0) {
        const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        for (RowInput<true> at(index, stride, knot_rows); index < count; index += stride, at.advance(knot_rows)) {
            const auto row = index * terms;
            const T x = input[index];
            const T* knots = row_knots(at);
            if (std::is_same_v<T, double> || derivatives) {
                auto view = basis;
                view.knots = knots;
                detail::basis_terms_for<detail::BasisKind::BSpline>(view, x, detail::BasisRowOf<T>{values + row, derivatives + row, nullptr, nullptr}, guard);
            } else {
                // FP32/TF32 rows beyond the stage (backlog C13): values only, as basis_kernel.
                T v[detail::max_spline_degree+1], d[detail::max_spline_degree+1];
                for (std::size_t k = 0; k < terms; ++k) values[row + k] = 0;
                const auto g = detail::spline_span_terms(knots, terms, basis.degree, x, v, d, guard);
                if (g != terms) for (std::size_t r = 0; r <= basis.degree; ++r) values[row + g + r] = v[r];
            }
            if constexpr (Residual) write_silu(silu, index, x, guard);
        }
        return;
    }
    const auto plane = static_cast<std::size_t>(tile_rows) * terms;
    const auto step = static_cast<std::size_t>(gridDim.x) * tile_rows;
    auto base = static_cast<std::size_t>(blockIdx.x) * tile_rows;
    for (RowInput<true> at(base + threadIdx.x, step, knot_rows); base < count; base += step, at.advance(knot_rows)) {
        const auto rows = count - base < tile_rows ? count - base : static_cast<std::size_t>(tile_rows);
        if (threadIdx.x < rows) {
            const auto row = threadIdx.x * terms;
            const T x = input[base + threadIdx.x];
            auto view = basis;
            view.knots = row_knots(at);
            detail::basis_terms_for<detail::BasisKind::BSpline>(view, x, detail::BasisRowOf<T>{stage + row, stage + plane + row, nullptr, nullptr}, guard);
            if constexpr (Residual) write_silu(silu, base + threadIdx.x, x, guard);
        }
        __syncthreads();
        const auto length = rows * terms, offset = base * terms;
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < length; i += blockDim.x) {
            values[offset + i] = stage[i];
            if (derivatives) derivatives[offset + i] = stage[plane + i];
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
// Residual (backlog M3): each lane also accumulates its share of
// r = dot(S[b,:], W[o,:]) into the same partial sum, so one shuffle tree
// reduces carrier and branch together (a separate tree for r cost 25-60% of
// the kernel; the device order of the sums is tolerance-based), then the
// bias is added; the instantiation without it is the kernel as before.
template<bool Residual, class T>
__global__ void forward_dot_kernel(const T* v, const T* c, const T* bias, T* output,
                                   std::size_t count, std::size_t outputs, std::size_t length,
                                   const T* s, const T* w, std::size_t inputs, int* status) {
    const auto lane = threadIdx.x%32;
    const auto warps = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for (auto index = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32; index < count; index += warps) {
        const T* row = v+(index/outputs)*length;
        const T* column = c+(index%outputs)*length;
        T sum = 0;
        for (std::size_t k = lane; k < length; k += 32) sum += row[k]*column[k];
        if constexpr (Residual) {
            const T* silu = s+(index/outputs)*inputs;
            const T* weights = w+(index%outputs)*inputs;
            for (std::size_t i = lane; i < inputs; i += 32) sum += silu[i]*weights[i];
        }
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
// Input VJP of FP32/TF32 B-spline layers (backlog C13): W holds W* = U*C*
// over the derivative coefficients C* (spline_derivative_coefficient, rows of
// terms-1), and dx[r] = sum_j B_{g+1+j,p-1}(x) W*[r, g+j] over the degree
// nonzero degree-(p-1) values of x = input[r] (spline_lower_terms; g the
// first term of x). The weights are nonnegative and sum to one, so the
// rounding of W* (TF32 operands) is not amplified as it is by the signed
// derivatives B'_{k,p} of order degree/knot spacing, which sum to zero. FP64
// keeps W = U*C and Phi' (results unchanged).
// Degree: the layer's degree as a compile-time constant for degrees 1..3 (the
// spline evaluation then stays in registers; C13 profiling: the runtime-degree
// form was 1.4x slower than the C3 finish at 12 terms), 0 for any degree.
template<class T> constexpr bool spline_difference_vjp = !std::is_same_v<T, double>;
template<class T, unsigned Degree> struct SplineDifferences {
    const T* input;
    const T* knots;
    std::size_t terms, degree;
};
// Per-input grids (backlog M6): row r reads the knots of input r % inputs.
template<class T, unsigned Degree> struct GridSplineDifferences {
    const T* input;
    const T* knots;
    std::size_t terms, degree;
    KnotRows rows;
};
template<class Source> constexpr bool difference_rows = false;
template<class T, unsigned Degree> constexpr bool difference_rows<SplineDifferences<T, Degree>> = true;
template<class T, unsigned Degree> constexpr bool difference_rows<GridSplineDifferences<T, Degree>> = true;
template<class Source> constexpr unsigned difference_degree = 0;
template<class T, unsigned Degree> constexpr unsigned difference_degree<SplineDifferences<T, Degree>> = Degree;
template<class T, unsigned Degree> constexpr unsigned difference_degree<GridSplineDifferences<T, Degree>> = Degree;
template<class Source> constexpr bool grid_rows = false;
template<class T, unsigned Degree> constexpr bool grid_rows<GridSplineDifferences<T, Degree>> = true;
template<class Source> __device__ KnotRows knot_rows_of(const Source& source) {
    if constexpr (grid_rows<Source>) return source.rows; else return KnotRows{};
}

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
        if constexpr (difference_rows<Source>) {
            // W* rows of terms-1 values; a row outside the domain (or of degree zero) is zero.
            constexpr unsigned D = difference_degree<Source>;
            constexpr std::size_t width = D ? D : detail::max_spline_degree;
            const auto span = source.terms-1;
            const std::size_t degree = D ? std::size_t{D} : source.degree;
            const auto knot_rows = knot_rows_of(source);
            auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
            RowInput<grid_rows<Source>> at(index, stride, knot_rows);
            for (; index < rows; index += stride, at.advance(knot_rows)) {
                T sum = 0;
                if (degree != 0) {
                    T values[width+1], lower[width];
                    const auto g = detail::spline_lower_terms(at.knots(source.knots, knot_rows), source.terms, degree, source.input[index],
                                                              values, lower, CheckedEarlier{});
                    if (g != source.terms) {
                        const T* row = w+index*span+g;
                        for (std::size_t j = 0; j < degree; ++j) sum += lower[j]*row[j];
                    }
                }
                dx[index] = sum; report(sum, status);
            }
        } else if constexpr (stored_rows<Source>) {
            for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < rows; index += stride) {
                T sum = 0;
                for (std::size_t k = 0; k < terms; ++k) sum += source.rows[index*terms+k]*w[index*terms+k];
                dx[index] = sum; report(sum, status);
            }
        }
    } else if constexpr (!difference_rows<Source>) {
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
// Derivative coefficients of a B-spline layer (backlog C13): for every edge
// q = (o, i) and j = 1..terms-1, C*[q*span + j-1] = (c_j - c_{j-1}) s_j
// (s_j = p/(t_{j+p} - t_j), the layer's knot scales, uploaded at
// construction) in rows of span = terms-1 (dense: the W* GEMM operand) or
// span = terms with a zero last term (sparse: the C layout the input-VJP
// kernel stages; staging the differences there instead cost 10% of the
// 1024-wide step, C13 evidence). Computed in every backward pass from the current C, so
// graphs, uploads and rollback need nothing; not checked here: a nonfinite
// C* reaches every input VJP whose degree-(p-1) terms include it.
// Index: 32-bit when every coefficient index fits (the division per element
// is the kernel's main instruction cost), 64-bit otherwise.
template<class Index, class T>
__global__ void spline_difference_kernel(const T* c, const T* scales, T* differences, Index count, Index span, Index terms) {
    const auto stride = static_cast<Index>(gridDim.x)*blockDim.x;
    for (auto e = static_cast<Index>(blockIdx.x)*blockDim.x+threadIdx.x; e < count; e += stride) {
        const Index q = e/span, j = e-q*span, k = q*terms+j+1;
        differences[e] = j+1 < terms ? detail::spline_derivative_coefficient(c[k-1], c[k], scales[j]) : T(0);
    }
}
// Per-input grids (backlog M6): the knot scales of input i at
// scales[i*(terms-1) + j-1], i = q % inputs for edge q = (o, i).
template<class Index, class T>
__global__ void spline_grid_difference_kernel(const T* c, const T* scales, T* differences, Index count, Index span, Index terms,
                                              Index inputs) {
    const auto stride = static_cast<Index>(gridDim.x)*blockDim.x;
    for (auto e = static_cast<Index>(blockIdx.x)*blockDim.x+threadIdx.x; e < count; e += stride) {
        const Index q = e/span, j = e-q*span, k = q*terms+j+1;
        differences[e] = j+1 < terms ? detail::spline_derivative_coefficient(c[k-1], c[k], scales[(q%inputs)*(terms-1)+j]) : T(0);
    }
}
// Sparse B-spline path (backlog C5). At any input only degree+1 = P of a
// spline's terms can be nonzero, so a sparse layer keeps per input and sample
// the index g of the first of them (`first`, unsigned [inputs][capacity]) and
// their P values and input derivatives ([inputs][P][capacity]; samples
// fastest; g = 0 and zero values outside the domain) instead of the dense
// rows, and contracts only those terms:
//   y[b,o]    = bias[o] + sum_i sum_j phi[b,i,j]*C[o,i,g+j]
//   dC[o,i,k] = sum over the samples with g <= k < g+P of U[b,o]*phi[b,i,k-g], + lambda*C
//   dx[b,i]   = sum_j phi'[b,i,j] * sum_o U[b,o]*C[o,i,g+j]
// The forward and input-VJP kernels map lanes to samples (each thread owns its
// outputs: fixed order, no atomics) and read C tiles staged in shared memory
// with outputs fastest as 16-byte vectors: GB10 issues 32-bit shared loads at
// half a warp instruction per clock (1.8 T loads/s), 16-byte loads deliver
// 6.5 T values/s (C5 evidence). The coefficient VJP walks every input's
// samples ordered by g (a stable segmented radix sort, CUB): one thread per
// (output, input) keeps the P coefficients g..g+P-1 in a register window and
// writes a coefficient when the window passes it, so every sum has a fixed
// order. P (1..4) is a template parameter: phi and the accumulators live in
// registers.
template<class T> struct Vector16;
template<> struct Vector16<float> { using type = float4; static constexpr unsigned size = 4; };
template<> struct Vector16<double> { using type = double2; static constexpr unsigned size = 2; };
template<class T> constexpr unsigned vector16 = Vector16<T>::size;
template<class T> __device__ __forceinline__ void load16(const T* p, T (&out)[vector16<T>]) {
    const auto v = *reinterpret_cast<const typename Vector16<T>::type*>(p);
    if constexpr (vector16<T> == 4) { out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w; }
    else { out[0] = v.x; out[1] = v.y; }
}
constexpr unsigned max_sparse_spline_width = 4; // degree + 1 of the instantiated sparse kernels
template<class T> struct SparseRows {
    unsigned* first;
    T* values;
    T* derivatives;
    std::size_t capacity;
};

// Sparse rows of `batch` samples from X (batch, inputs). A block takes 32
// inputs x spline_tile_samples samples: it reads X inputs fastest
// (coalesced), evaluates the spline into shared memory and writes the rows
// samples fastest (coalesced). With the residual branch the same pass writes
// the layer's S and silu' rows (backlog M3).
template<class T> constexpr unsigned spline_tile_samples = std::is_same_v<T, double> ? 16 : 32;
// FP32/TF32 (backlog C13): the derivative rows hold the degree-(p-1) values
// of the input VJP (spline_lower_terms; P-1 rows, none for degree zero)
// instead of the P input derivatives.
template<class T, unsigned P> constexpr unsigned sparse_derivative_rows = spline_difference_vjp<T> ? P-1 : P;
template<class T> std::size_t sparse_derivative_count(unsigned width) { return spline_difference_vjp<T> ? width-1 : width; }
template<unsigned P, bool Residual, bool Grids, class T>
__device__ __forceinline__ void spline_rows(const T* input, SparseRows<T> rows, std::size_t batch, std::size_t inputs,
                                            const T* knots, std::size_t terms, SiluRows<T> silu, int* status, std::size_t knot_stride,
                                            unsigned pad) {
    constexpr unsigned TS = spline_tile_samples<T>, D = sparse_derivative_rows<T, P>;
    __shared__ unsigned first[32][TS+1];
    __shared__ T values[P][32][TS+1], derivatives[D ? D : 1][32][TS+1];
    const StatusGuard guard{status};
    const auto i0 = static_cast<std::size_t>(blockIdx.x)*32;
    if constexpr (Grids) {
        // Per-input grids (backlog M6): with pad != 0 the block's 32 knot rows
        // are staged once (row stride pad, odd) and read from shared memory.
        if (pad) {
            T* staged = stage_memory<T>();
            for (auto e = static_cast<std::size_t>(threadIdx.x); e < 32*knot_stride; e += blockDim.x) {
                const auto ii = e/knot_stride, k = e%knot_stride;
                if (i0+ii < inputs) staged[ii*pad+k] = knots[(i0+ii)*knot_stride+k];
            }
            __syncthreads();
            knots = staged;
        }
    }
    for (auto b0 = static_cast<std::size_t>(blockIdx.y)*TS; b0 < batch; b0 += static_cast<std::size_t>(gridDim.y)*TS) {
        for (unsigned e = threadIdx.x; e < 32*TS; e += blockDim.x) {
            const unsigned ii = e%32, bb = e/32;
            const auto i = i0+ii, b = b0+bb;
            if (i >= inputs || b >= batch) continue;
            const T x = input[b*inputs+i];
            T v[P], d[P];
            std::size_t g;
            const T* t = knots;
            if constexpr (Grids) t += pad ? ii*pad : i*knot_stride;
            if constexpr (spline_difference_vjp<T>) g = detail::spline_lower_terms(t, terms, std::size_t{P-1}, x, v, d, guard);
            else g = detail::spline_span_terms(t, terms, std::size_t{P-1}, x, v, d, guard);
            if (g == terms) {
                g = 0;
                for (unsigned j = 0; j < P; ++j) { v[j] = 0; d[j] = 0; }
            }
            first[ii][bb] = static_cast<unsigned>(g);
            if constexpr (spline_difference_vjp<T>) {
                for (unsigned j = 0; j < P; ++j) values[j][ii][bb] = v[j];
                if constexpr (D != 0) for (unsigned j = 0; j < D; ++j) derivatives[j][ii][bb] = d[j];
            } else {
                for (unsigned j = 0; j < P; ++j) { values[j][ii][bb] = v[j]; derivatives[j][ii][bb] = d[j]; }
            }
            if constexpr (Residual) write_silu(silu, b*inputs+i, x, guard);
        }
        __syncthreads();
        for (unsigned e = threadIdx.x; e < 32*TS; e += blockDim.x) {
            const unsigned bb = e%TS, ii = e/TS;
            const auto i = i0+ii, b = b0+bb;
            if (i >= inputs || b >= batch) continue;
            rows.first[i*rows.capacity+b] = first[ii][bb];
            if constexpr (spline_difference_vjp<T>) {
                for (unsigned j = 0; j < P; ++j) rows.values[(i*P+j)*rows.capacity+b] = values[j][ii][bb];
                if constexpr (D != 0) for (unsigned j = 0; j < D; ++j) rows.derivatives[(i*D+j)*rows.capacity+b] = derivatives[j][ii][bb];
            } else {
                for (unsigned j = 0; j < P; ++j) {
                    rows.values[(i*P+j)*rows.capacity+b] = values[j][ii][bb];
                    rows.derivatives[(i*P+j)*rows.capacity+b] = derivatives[j][ii][bb];
                }
            }
        }
        __syncthreads();
    }
}

template<unsigned P, bool Residual, class T>
__global__ void spline_basis_kernel(const T* input, SparseRows<T> rows, std::size_t batch, std::size_t inputs,
                                    const T* knots, std::size_t terms, SiluRows<T> silu, int* status) {
    spline_rows<P, Residual, false>(input, rows, batch, inputs, knots, terms, silu, status, 0, 0);
}
// Per-input grids (backlog M6): input i reads knots + i*knot_stride.
// pad != 0: the block's knot rows staged in dynamic shared memory (32*pad values).
template<unsigned P, bool Residual, class T>
__global__ void spline_grid_basis_kernel(const T* input, SparseRows<T> rows, std::size_t batch, std::size_t inputs,
                                         const T* knots, std::size_t terms, SiluRows<T> silu, int* status, std::size_t knot_stride,
                                         unsigned pad) {
    spline_rows<P, Residual, true>(input, rows, batch, inputs, knots, terms, silu, status, knot_stride, pad);
}

// Forward: a block of stage_threads samples (lane = sample) and
// spline_outputs outputs; C[o0..o0+15, i, :] for `chunk` inputs at a time is
// staged as rows (i, k) of the 16 outputs (padded by one vector), so a sample
// reads its P rows g..g+P-1 as 16-byte vectors. Each thread sums its outputs
// over i, then j in ascending order and adds the bias (when given; the
// residual branch adds S*W^T first and bias_kernel then adds the bias).
constexpr unsigned spline_outputs = 16;
template<class T> constexpr unsigned spline_forward_stride = spline_outputs+vector16<T>;
template<unsigned P, class T>
__global__ void __launch_bounds__(stage_threads) spline_forward_kernel(const unsigned* first, const T* values, std::size_t capacity,
        const T* c, const T* bias, T* y, std::size_t batch, std::size_t inputs, std::size_t outputs, std::size_t terms,
        unsigned chunk, int* status) {
    constexpr unsigned TO = spline_outputs, V = vector16<T>, RS = spline_forward_stride<T>;
    T* cs = stage_memory<T>();
    const auto b = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto o0 = static_cast<std::size_t>(blockIdx.y)*TO;
    const auto sample = b < batch ? b : batch-1;
    const unsigned warp = threadIdx.x/32, lane = threadIdx.x%32, warps = blockDim.x/32;
    T acc[TO];
#pragma unroll
    for (unsigned t = 0; t < TO; ++t) acc[t] = 0;
    for (std::size_t i0 = 0; i0 < inputs; i0 += chunk) {
        const auto count = inputs-i0 < chunk ? inputs-i0 : static_cast<std::size_t>(chunk);
        const auto row = count*terms;
        __syncthreads();
        for (unsigned oo = warp; oo < TO; oo += warps) {
            const auto o = o0+oo;
            const T* source = c+(o*inputs+i0)*terms;
            for (auto r = static_cast<std::size_t>(lane); r < row; r += 32) cs[r*RS+oo] = o < outputs ? source[r] : T(0);
        }
        __syncthreads();
        // The next input's rows are loaded while the current one is reduced.
        unsigned g = first[i0*capacity+sample];
        T phi[P];
#pragma unroll
        for (unsigned j = 0; j < P; ++j) phi[j] = values[(i0*P+j)*capacity+sample];
        for (std::size_t ii = 0; ii < count; ++ii) {
            unsigned next_g = 0;
            T next_phi[P];
            if (ii+1 < count) {
                const auto i = i0+ii+1;
                next_g = first[i*capacity+sample];
#pragma unroll
                for (unsigned j = 0; j < P; ++j) next_phi[j] = values[(i*P+j)*capacity+sample];
            }
            const T* base = cs+(ii*terms+g)*RS;
#pragma unroll
            for (unsigned j = 0; j < P; ++j)
#pragma unroll
                for (unsigned q = 0; q < TO; q += V) {
                    T cv[V];
                    load16(base+j*RS+q, cv);
#pragma unroll
                    for (unsigned e = 0; e < V; ++e) acc[q+e] += phi[j]*cv[e];
                }
            g = next_g;
#pragma unroll
            for (unsigned j = 0; j < P; ++j) phi[j] = next_phi[j];
        }
    }
    if (b >= batch) return;
#pragma unroll
    for (unsigned t = 0; t < TO; ++t) {
        const auto o = o0+t;
        if (o >= outputs) break;
        T sum = acc[t];
        if (bias) { sum += bias[o]; report(sum, status); }
        y[b*outputs+o] = sum;
    }
}

// Input VJP: a block of stage_threads samples (lane = sample) and `group`
// (<= spline_dx_inputs, fewer for long rows that would not fit the shared
// memory) inputs; output chunks of two vectors at a time: U[b, chunk]
// rows (padded by one vector) and C[chunk, i, :] staged as rows (i, k) of the
// chunk's outputs. Each thread accumulates w_j = sum_o U[b,o]*C[o,i,g+j] in
// ascending o, then dx = sum_j phi'_j*w_j. FP32/TF32 layers (backlog C13)
// launch it with P-1 terms per sample over the derivative coefficients C*
// (rows of terms-1) and the degree-(p-1) rows: w*_j = sum_o U[b,o]*C*[o,i,g+j],
// dx = sum_j B_{g+1+j,p-1}*w*_j (nonnegative weights summing to one).
constexpr unsigned spline_dx_inputs = 8;
template<class T> constexpr unsigned spline_dx_outputs = 2*vector16<T>;
template<class T> constexpr unsigned spline_dx_stride = spline_dx_outputs<T>+vector16<T>;
template<unsigned P, class T>
__global__ void __launch_bounds__(stage_threads) spline_input_kernel(const unsigned* first, const T* derivatives, std::size_t capacity,
        const T* c, const T* u, T* dx, std::size_t batch, std::size_t inputs, std::size_t outputs, std::size_t terms, unsigned group,
        int* status) {
    constexpr unsigned TI = spline_dx_inputs, V = vector16<T>, OC = spline_dx_outputs<T>, RS = spline_dx_stride<T>;
    T* us = stage_memory<T>();
    T* cs = us+blockDim.x*RS;
    const auto b0 = static_cast<std::size_t>(blockIdx.x)*blockDim.x, b = b0+threadIdx.x;
    const auto i0 = static_cast<std::size_t>(blockIdx.y)*group;
    const auto sample = b < batch ? b : batch-1;
    const unsigned count = inputs-i0 < group ? static_cast<unsigned>(inputs-i0) : group;
    const unsigned warp = threadIdx.x/32, lane = threadIdx.x%32, warps = blockDim.x/32;
    unsigned g[TI];
    T acc[TI][P];
#pragma unroll
    for (unsigned t = 0; t < TI; ++t) {
        g[t] = t < count ? first[(i0+t)*capacity+sample] : 0;
#pragma unroll
        for (unsigned j = 0; j < P; ++j) acc[t][j] = 0;
    }
    const auto row = static_cast<std::size_t>(count)*terms;
    for (std::size_t oc0 = 0; oc0 < outputs; oc0 += OC) {
        const auto width = outputs-oc0 < OC ? outputs-oc0 : static_cast<std::size_t>(OC);
        __syncthreads();
        for (auto e = static_cast<std::size_t>(threadIdx.x); e < static_cast<std::size_t>(blockDim.x)*OC; e += blockDim.x) {
            const auto r = e/OC, o = e%OC;
            us[r*RS+o] = b0+r < batch && o < width ? u[(b0+r)*outputs+oc0+o] : T(0);
        }
        for (unsigned o = warp; o < OC; o += warps) {
            const T* source = c+((oc0+o)*inputs+i0)*terms;
            for (auto r = static_cast<std::size_t>(lane); r < group*terms; r += 32) cs[r*RS+o] = o < width && r < row ? source[r] : T(0);
        }
        __syncthreads();
        const T* own = us+threadIdx.x*RS;
#pragma unroll
        for (unsigned o = 0; o < OC; o += V) {
            T uv[V];
            load16(own+o, uv);
#pragma unroll
            for (unsigned t = 0; t < TI; ++t) {
                if (t >= group) break;
                const T* base = cs+(t*terms+g[t])*RS+o;
#pragma unroll
                for (unsigned j = 0; j < P; ++j) {
                    T cv[V];
                    load16(base+j*RS, cv);
#pragma unroll
                    for (unsigned e = 0; e < V; ++e) acc[t][j] += uv[e]*cv[e];
                }
            }
        }
    }
    if (b >= batch) return;
#pragma unroll
    for (unsigned t = 0; t < TI; ++t) {
        if (t >= count) break;
        T sum = 0;
#pragma unroll
        for (unsigned j = 0; j < P; ++j) sum += derivatives[((i0+t)*P+j)*capacity+b]*acc[t][j];
        dx[b*inputs+i0+t] = sum; report(sum, status);
    }
}

// Coefficient and bias VJPs. Block (blockIdx.x, i): spline_parameter_warps
// warps of 32 lanes, lane = output o (and o + 32, ...), over input i's
// samples in sorted order (g ascending, then sample index), staged
// spline_parameter_chunk at a time. Lane accumulators acc[j] belong to
// coefficient `current` + j; when the samples move past `current`, its sum
// (+ lambda*C) is written and the window shifts. The input-0 blocks also sum
// U over all samples (in the same order) into the bias VJP. Blocks along z
// split the coefficients of an input into `ranges` contiguous ranges [k0, k1)
// (layers with few outputs: enough blocks for the GPU); a block walks only the
// sorted samples with k0-P < g < k1 (found by binary search) and writes only
// its coefficients, whose sums keep the same order for any split.
constexpr unsigned spline_parameter_outputs = 2, spline_parameter_warps = 4, spline_parameter_chunk = 256, spline_parameter_lookahead = 8;
template<unsigned P, class T>
__global__ void spline_parameter_kernel(const unsigned* sorted_first, const unsigned* order, const T* values, std::size_t capacity,
        const T* u, const T* c, T* dc, T* db, T lambda, std::size_t batch, std::size_t inputs, std::size_t outputs,
        std::size_t terms, unsigned ranges, int* status) {
    constexpr unsigned TO = spline_parameter_outputs, R = spline_parameter_chunk;
    __shared__ unsigned sample_of[R], first_of[R];
    __shared__ T phi_of[R][P];
    const unsigned lane = threadIdx.x%32, warp = threadIdx.x/32;
    const auto obase = (static_cast<std::size_t>(blockIdx.x)*(blockDim.x/32)+warp)*32*TO+lane;
    const auto k0 = terms*blockIdx.z/ranges, k1 = terms*(blockIdx.z+1)/ranges;
    const bool bias_range = blockIdx.z == 0;
    for (auto i = static_cast<std::size_t>(blockIdx.y); i < inputs; i += gridDim.y) {
        T acc[TO][P], bias_sum[TO];
#pragma unroll
        for (unsigned t = 0; t < TO; ++t) {
            bias_sum[t] = 0;
#pragma unroll
            for (unsigned j = 0; j < P; ++j) acc[t][j] = 0;
        }
        // Sorted positions [begin, end) of the samples with first+P > k0 and first < k1
        // (all samples for the bias range of input 0).
        const unsigned* keys = sorted_first+i*capacity;
        const auto lower = [&](std::size_t key) {
            std::size_t lo = 0, hi = batch;
            while (lo < hi) { const auto mid = (lo+hi)/2; if (keys[mid] < key) lo = mid+1; else hi = mid; }
            return lo;
        };
        const bool all = i == 0 && bias_range;
        const std::size_t begin = all || k0 < P ? 0 : lower(k0-P+1), end = all || k1 >= terms ? batch : lower(k1);
        std::size_t current = k0 < P ? 0 : k0-P+1;
        const auto emit = [&] {
#pragma unroll
            for (unsigned t = 0; t < TO; ++t) {
                const auto o = obase+32*t;
                if (o < outputs && current >= k0 && current < k1) {
                    const auto q = (o*inputs+i)*terms+current;
                    // Without L2 the coefficients are not read (acc + 0*c == acc:
                    // acc starts at +0 and is never -0). Streaming stores: dC
                    // passes once, the U rows are reread from L2.
                    const T sum = lambda != 0 ? acc[t][0]+lambda*c[q] : acc[t][0];
                    __stcs(dc+q, sum); report(sum, status);
                }
#pragma unroll
                for (unsigned j = 0; j+1 < P; ++j) acc[t][j] = acc[t][j+1];
                acc[t][P-1] = 0;
            }
            ++current;
        };
        for (std::size_t r0 = begin; r0 < end; r0 += R) {
            const auto n = end-r0 < R ? static_cast<unsigned>(end-r0) : R;
            __syncthreads();
            for (unsigned r = threadIdx.x; r < n; r += blockDim.x) {
                const auto b = order[i*capacity+r0+r];
                sample_of[r] = b; first_of[r] = sorted_first[i*capacity+r0+r];
#pragma unroll
                for (unsigned j = 0; j < P; ++j) phi_of[r][j] = values[(i*P+j)*capacity+b];
            }
            __syncthreads();
            // U rows of D samples are loaded before they are accumulated: the
            // loop is bound by the latency of these L2 loads otherwise (C5
            // profiling: 1.3-1.9x for D = 8).
            constexpr unsigned D = spline_parameter_lookahead;
            for (unsigned r = 0; r < n; r += D) {
                T uv[D][TO];
#pragma unroll
                for (unsigned q = 0; q < D; ++q) {
                    const bool valid = r+q < n;
                    const auto row = valid ? static_cast<std::size_t>(sample_of[r+q])*outputs : 0;
#pragma unroll
                    for (unsigned t = 0; t < TO; ++t) {
                        const auto o = obase+32*t;
                        uv[q][t] = valid && o < outputs ? u[row+o] : T(0);
                    }
                }
#pragma unroll
                for (unsigned q = 0; q < D; ++q) {
                    if (r+q >= n) break;
                    const unsigned g = first_of[r+q];
                    while (current < g && current < k1) emit();
#pragma unroll
                    for (unsigned t = 0; t < TO; ++t) {
#pragma unroll
                        for (unsigned j = 0; j < P; ++j) acc[t][j] += uv[q][t]*phi_of[r+q][j];
                        if (all) bias_sum[t] += uv[q][t];
                    }
                }
            }
        }
        while (current < k1) emit();
        if (all) {
#pragma unroll
            for (unsigned t = 0; t < TO; ++t) {
                const auto o = obase+32*t;
                if (o < outputs) { db[o] = bias_sum[t]; report(bias_sum[t], status); }
            }
        }
        __syncthreads();
    }
}
// Segment bounds of the per-input sort: [i*capacity, i*capacity + batch).
__global__ void spline_segments_kernel(int* offsets, std::size_t inputs, std::size_t capacity, std::size_t batch) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; i < inputs; i += stride) {
        offsets[i] = static_cast<int>(i*capacity);
        offsets[inputs+i] = static_cast<int>(i*capacity+batch);
    }
}
// Sample indices b at [i*capacity + b], the values of the per-input sort.
__global__ void spline_indices_kernel(unsigned* indices, std::size_t inputs, std::size_t capacity) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto e = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; e < inputs*capacity; e += stride)
        indices[e] = static_cast<unsigned>(e%capacity);
}
// Device grid refit (backlog M6, ResidentNetwork::refit_grids): the
// arithmetic of kan::refit_grid (carriers/local/grid_refit.hpp, in double in
// every precision) over the layer input of the executor's forward pass.
// Per layer: the input transposed to (inputs, batch) and sorted per input (CUB
// segmented radix sort; a shared grid also sorts all values for its knots),
// refit_knots_kernel (one thread per grid), the runs of the sorted samples
// with one first term on each grid (CUB run-length encoding of the run key),
// refit_run_kernel (one warp per run for degrees 0..3, one thread otherwise:
// the run's products),
// refit_factor_kernel (one thread per input: the band of the normal matrix
// from its runs in order, ridge, Cholesky) and refit_solve_kernel (one thread
// per (input, output): the right-hand side from its runs in order and the
// banded solve; new coefficients rounded to T). Every sum has a fixed order:
// deterministic, and in FP64 the CPU refit's operations on the same samples.
// Status bits of a refit (refit_status reads them).
constexpr int status_refit_interval = 8, status_refit_knots = 16, status_refit_singular = 32, status_refit_internal = 64,
              status_refit_nonfinite = 128;
// New and current knots in double (the run keys) and in T (the products).
template<class T> struct RefitGrids {
    const double* fresh;   // new knots, `fresh_grids` rows of `knots` values
    const double* current; // current knots, `current_grids` rows
    const T* fresh_values;
    const T* current_values;
    std::size_t knots, terms, degree, fresh_grids, current_grids;
    __host__ __device__ std::size_t fresh_offset(std::size_t i) const { return (fresh_grids == 1 ? 0 : i)*knots; }
    __host__ __device__ std::size_t current_offset(std::size_t i) const { return (current_grids == 1 ? 0 : i)*knots; }
};
// Run key of sorted sample e (input e / batch): (input, first new term,
// first current term or `terms` outside its domain) in one integer.
template<class T> struct RefitKey {
    const T* sorted;
    RefitGrids<T> grids;
    std::size_t batch;
    __host__ __device__ unsigned long long operator()(long long e) const {
        const auto index = static_cast<std::size_t>(e), i = index/batch;
        const double x = sorted[index];
        const auto g = detail::spline_first_term(grids.fresh+grids.fresh_offset(i), grids.terms, grids.degree, x);
        const auto gc = detail::spline_first_term(grids.current+grids.current_offset(i), grids.terms, grids.degree, x);
        return (static_cast<unsigned long long>(i)*(grids.terms+1)+g)*(grids.terms+1)+gc;
    }
};
template<class T> struct RefitSample {
    const T* sorted;
    __device__ T operator()(std::size_t b) const { return sorted[b]; }
};
// One thread per grid g: pykan's knots from its n sorted samples
// (sorted + g*n), rounded to T; T-strictly increasing interior knots, else a
// status bit. Writes the knots (T and double), the FP32/TF32 knot scales
// (degree >= 1) and the current knots in double.
template<class T>
__global__ void refit_knots_kernel(const T* sorted, std::size_t n, std::size_t grids, std::size_t intervals, std::size_t degree,
                                   double eps, double margin, T* knots, T* scales, double* fresh, const T* current, double* current_double,
                                   std::size_t current_grids, int* status) {
    const auto count = intervals+2*degree+1, terms = intervals+degree;
    for (auto g = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; g < grids || g < current_grids;
         g += static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        if (g < current_grids)
            for (std::size_t k = 0; k < count; ++k) current_double[g*count+k] = current[g*count+k];
        if (g >= grids) continue;
        const RefitSample<T> load{sorted+g*n};
        // The radix sort puts nonfinite values (NaN, +-inf) at the ends (review H1).
        if (!finite_value(load(0)) || !finite_value(load(n-1))) { atomicOr(status, status_refit_nonfinite); continue; }
        if (!(load(0) < load(n-1))) atomicOr(status, status_refit_interval);
        T* t = knots+g*count;
        for (std::size_t k = 0; k < count; ++k) {
            const auto j = k < degree ? 0 : k-degree > intervals ? intervals : k-degree;
            t[k] = static_cast<T>(detail::refit_knot(load, n, intervals, j, eps, margin));
            fresh[g*count+k] = t[k];
        }
        bool increasing = finite_value(t[degree]); // FP32: the margin can round the endpoint beyond the range (review L3)
        for (std::size_t k = degree; k < degree+intervals; ++k) increasing = increasing && t[k] < t[k+1] && finite_value(t[k+1]);
        if (!increasing) atomicOr(status, status_refit_knots);
        if (scales && degree != 0)
            for (std::size_t j = 1; j < terms; ++j) scales[g*(terms-1)+j-1] = detail::spline_derivative_scale(degree, t[j+degree], t[j]);
    }
}
// Runs r < *runs: their products (refit_sample, P x P each, at sums +
// 2*P*P*r) and the first run of every input (first_run[i], and
// first_run[inputs] = *runs). Width 1..4 (degrees 0..3): one warp per run,
// lane l adds samples begin+l, begin+l+32, ... in order, then a fixed
// shuffle tree (nsys: a thread per run took 92% of the 256-wide refit, latency
// bound at two warps per SM); Width 0 (any degree): one thread per run, the
// samples in order.
template<unsigned Width, class T>
__global__ void refit_run_kernel(const T* sorted, RefitGrids<T> grids, std::size_t inputs,
                                 const unsigned long long* keys, const unsigned* counts, const unsigned* offsets, const unsigned* runs,
                                 unsigned* first_run, double* sums, int* status) {
    constexpr std::size_t lanes = Width ? 32 : 1;
    const auto P = Width ? std::size_t{Width} : grids.degree+1, total = static_cast<std::size_t>(*runs);
    const auto lane = Width ? threadIdx.x%32 : 0u;
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x/lanes;
    for (auto r = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/lanes; r <= total; r += stride) {
        if (r == total) { if (lane == 0) first_run[inputs] = static_cast<unsigned>(total); continue; }
        const auto key = keys[r], span = static_cast<unsigned long long>(grids.terms+1);
        const auto gc = static_cast<std::size_t>(key%span), g = static_cast<std::size_t>(key/span%span);
        const auto i = static_cast<std::size_t>(key/span/span);
        if (lane == 0 && (r == 0 || keys[r-1]/span/span != i)) first_run[i] = static_cast<unsigned>(r);
        const auto begin = static_cast<std::size_t>(offsets[r]), end = begin+counts[r];
        const T* fresh = grids.fresh_values+grids.fresh_offset(i);
        const T* current = grids.current_values+grids.current_offset(i);
        double* out = sums+2*P*P*r;
        bool consistent = true;
        if constexpr (Width != 0) {
            double nn[Width*Width] = {}, no[Width*Width] = {};
            for (auto b = begin+lane; b < end; b += 32)
                consistent = detail::refit_sample<Width>(fresh, grids.terms, current, grids.terms, grids.degree, sorted[b], g, gc, nn, no) && consistent;
            for (unsigned k = 0; k < Width*Width; ++k)
                for (unsigned offset = 16; offset; offset /= 2) {
                    nn[k] += __shfl_down_sync(0xffffffffU, nn[k], offset);
                    no[k] += __shfl_down_sync(0xffffffffU, no[k], offset);
                }
            if (lane == 0) for (unsigned k = 0; k < Width*Width; ++k) { out[k] = nn[k]; out[Width*Width+k] = no[k]; }
        } else {
            consistent = detail::refit_run_sums<0>(fresh, grids.terms, current, grids.terms, grids.degree, RefitSample<T>{sorted}, begin, end,
                                                   g, gc, out, out+P*P);
        }
        if (!consistent) atomicOr(status, status_refit_internal);
    }
}
// One thread per input: the band of its normal matrix (runs in order), ridge, Cholesky.
__global__ void refit_factor_kernel(const unsigned long long* keys, const unsigned* first_run, const double* sums,
                                    std::size_t inputs, std::size_t terms, std::size_t degree, double ridge, double* bands, int* status) {
    const auto P = degree+1;
    const auto span = static_cast<unsigned long long>(terms+1);
    for (auto i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; i < inputs; i += static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        double* band = bands+i*terms*P;
        for (std::size_t k = 0; k < terms*P; ++k) band[k] = 0;
        for (auto r = static_cast<std::size_t>(first_run[i]); r < first_run[i+1]; ++r) {
            const auto g = static_cast<std::size_t>(keys[r]/span%span);
            if (g >= terms) continue; // not for finite samples (all inside the new domain)
            const double* nn = sums+2*P*P*r;
            for (std::size_t a = 0; a < P; ++a)
                for (std::size_t q = a; q < P; ++q) band[(g+q)*P+(q-a)] += nn[a*P+q];
        }
        if (!detail::band_cholesky(band, terms, degree, ridge)) atomicOr(status, status_refit_singular);
    }
}
// One thread per (input i, output o) of the output chunk [o0, o0 + chunk),
// t = i*chunk + (o - o0): the right-hand side sum_runs sum M c[o,i,:] (runs in
// order; rhs[(i*terms + k)*chunk + o - o0], outputs fastest), the banded
// solve, and the coefficients rounded to T. Chunks bound the double
// right-hand-side workspace (refit_outputs outputs at a time).
template<class T>
__global__ void refit_solve_kernel(const unsigned long long* keys, const unsigned* first_run, const double* sums,
                                   const double* bands, double* rhs, const T* current, T* fresh, std::size_t inputs,
                                   std::size_t outputs, std::size_t o0, std::size_t chunk, std::size_t terms, std::size_t degree,
                                   int* status) {
    const auto P = degree+1;
    const auto span = static_cast<unsigned long long>(terms+1);
    for (auto t = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; t < inputs*chunk; t += static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        const auto i = t/chunk, o = o0+t%chunk;
        if (o >= outputs) continue;
        double* x = rhs+i*terms*chunk+t%chunk;
        for (std::size_t k = 0; k < terms; ++k) x[k*chunk] = 0;
        const T* c = current+(o*inputs+i)*terms;
        for (auto r = static_cast<std::size_t>(first_run[i]); r < first_run[i+1]; ++r) {
            const auto key = keys[r];
            const auto gc = static_cast<std::size_t>(key%span), g = static_cast<std::size_t>(key/span%span);
            if (gc == terms || g >= terms) continue;
            const double* no = sums+2*P*P*r+P*P;
            for (std::size_t a = 0; a < P; ++a) {
                double sum = x[(g+a)*chunk];
                for (std::size_t q = 0; q < P; ++q) sum += no[a*P+q]*static_cast<double>(c[gc+q]);
                x[(g+a)*chunk] = sum;
            }
        }
        detail::band_solve(bands+i*terms*P, terms, degree, x, chunk);
        T* next = fresh+(o*inputs+i)*terms;
        for (std::size_t k = 0; k < terms; ++k) {
            next[k] = static_cast<T>(x[k*chunk]);
            report(next[k], status);
        }
    }
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
// Tiled variant with the summation order of nonlinear_partial_kernel (backlog
// C12). There, lane l of block (k, tile) accumulates rows tile*256 + l +
// n*tiles*256 for n = 0, 1, ... and a shared tree sums the 256 lanes. When
// the column blocks of one tile are not resident together, each warp load
// fetches 32 sectors for 32 useful values and the rows come from DRAM once per
// term (FP64 K=8: 44 GB/s effective on GB10). Here block (tile, group) owns
// the chains of all 256 lanes for `group` consecutive terms k0.. as shared
// accumulators at item lane*g + j: per chunk of 256 rows its threads read the
// g values of every row (contiguous when g = terms) and update the items in
// the same order, then run the same tree per term, so the partials are
// bitwise those of the column kernel in either FMA mode.
template<class T>
__global__ void nonlinear_tile_kernel(const T* dx, const T* dw, const T* w, T* partial, std::size_t rows,
                                      std::size_t terms, unsigned tiles, unsigned group, int* status) {
    T* centers = stage_memory<T>();
    T* widths = centers+std::size_t{256}*group;
    const auto tile = blockIdx.x%tiles;
    const auto k0 = static_cast<std::size_t>(blockIdx.x/tiles)*group;
    const auto g = terms-k0 < group ? terms-k0 : static_cast<std::size_t>(group);
    const auto items = 256*g;
    for (auto i = static_cast<std::size_t>(threadIdx.x); i < items; i += blockDim.x) { centers[i] = 0; widths[i] = 0; }
    const auto stride = static_cast<std::size_t>(tiles)*256;
    for (auto base = static_cast<std::size_t>(tile)*256; base < rows; base += stride) {
        // Each item is updated by one thread (the same one for every chunk).
        const auto count = (rows-base < 256 ? rows-base : std::size_t{256})*g;
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < count; i += blockDim.x) {
            const auto r = i/g, o = (base+r)*terms+k0+(i-r*g);
            const T factor = w[o];
            centers[i] += factor*(-dx[o]); widths[i] += factor*dw[o];
        }
    }
    __syncthreads();
    for (auto i = static_cast<std::size_t>(threadIdx.x); i < items; i += blockDim.x) { report(centers[i], status); report(widths[i], status); }
    for (std::size_t step = 128; step; step /= 2) {
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < step*g; i += blockDim.x) {
            centers[i] += centers[i+step*g]; widths[i] += widths[i+step*g];
        }
        __syncthreads();
    }
    for (auto j = static_cast<std::size_t>(threadIdx.x); j < g; j += blockDim.x) {
        partial[(k0+j)*tiles+tile] = centers[j]; partial[(terms+k0+j)*tiles+tile] = widths[j];
        report(centers[j], status); report(widths[j], status);
    }
}
// Shared accumulators of one nonlinear_tile_kernel block.
template<class T> constexpr std::size_t nonlinear_tile_bytes(std::size_t group) { return std::size_t{2}*256*group*sizeof(T); }
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

// SiLU residual branch (backlog M3): y += S*W^T with S = silu(X), dW =
// U^T*S + lambda*W and dX += silu'(X) (.) (U*W), after the carrier's own
// VJPs. W is (outputs, inputs) row-major; S and silu'(X) are (batch, inputs)
// rows kept from forward to backward, written by the expansion carriers'
// basis_kernel (which reads X anyway) or by silu_kernel. The expansion
// carriers fuse the forward sum into forward_dot_kernel (small shapes) or run
// the cuBLAS GEMM before bias_kernel, whose check then covers it; other
// carriers use the kernels below after their forward. Larger shapes use
// cuBLAS (residual_contract, residual_backward).
template<class T>
__global__ void silu_kernel(const T* x, SiluRows<T> silu, std::size_t count, int* status) {
    const StatusGuard guard{status};
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride)
        write_silu(silu, index, x[index], guard);
}
// Small Y += S*W^T: a block takes 32 samples (lane = sample) and stages their
// S rows transposed (padded, conflict-free; one warp per row, coalesced) in
// shared memory; each warp then copies one W row into its own shared slice
// and sums that output over the inputs from shared memory only, adds it to Y
// and checks. Blocks along y split the outputs, one per warp. Profiling
// (M3 evidence, phase 2): reading W[o,i] from global memory inside the
// multiply-add chain made the kernel long-scoreboard bound, and fewer, wider
// output groups (register tiles of 4 outputs) starved the GPU of blocks at
// batch 1024. Needs the S tile and the warps' W rows in the stage
// (residual_tile_inputs).
constexpr unsigned residual_tile = 32;
template<class T> constexpr std::size_t residual_tile_inputs =
    stage_bytes/((residual_tile+1+stage_threads/32)*sizeof(T));
template<class T>
__global__ void residual_dot_kernel(const T* s, const T* w, T* y, std::size_t batch, std::size_t inputs,
                                    std::size_t outputs, int* status) {
    constexpr auto pitch = residual_tile+1;
    const auto lane = threadIdx.x%32, warp = threadIdx.x/32, warps = blockDim.x/32;
    T* tile = stage_memory<T>();                            // [input][pitch]
    T* row = tile+static_cast<std::size_t>(inputs)*pitch+warp*inputs; // this warp's W row
    for (auto first = static_cast<std::size_t>(blockIdx.x)*residual_tile; first < batch;
         first += static_cast<std::size_t>(gridDim.x)*residual_tile) {
        const auto rows = batch-first < residual_tile ? batch-first : static_cast<std::size_t>(residual_tile);
        for (auto r = static_cast<std::size_t>(warp); r < rows; r += warps)
            for (auto i = static_cast<std::size_t>(lane); i < inputs; i += 32) tile[i*pitch+r] = s[(first+r)*inputs+i];
        __syncthreads();
        for (auto o = static_cast<std::size_t>(blockIdx.y)*warps+warp; o < outputs; o += static_cast<std::size_t>(gridDim.y)*warps) {
            for (auto i = static_cast<std::size_t>(lane); i < inputs; i += 32) row[i] = w[o*inputs+i];
            __syncwarp();
            if (lane < rows) {
                T sum = 0;
#pragma unroll 8
                for (std::size_t i = 0; i < inputs; ++i) sum += row[i]*tile[i*pitch+lane];
                const auto at = (first+lane)*outputs+o;
                const T value = y[at]+sum;
                y[at] = value; report(value, status);
            }
            __syncwarp(); // the row is overwritten for the warp's next output
        }
        __syncthreads(); // the tile is overwritten for the block's next samples
    }
}
// Nonfinite check of a result cuBLAS wrote (Y after Y += S*W^T).
template<class T>
__global__ void check_kernel(const T* values, std::size_t count, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < count; index += stride)
        report(values[index], status);
}
// Backward finish. Weights q < count: with `reduce`, dW[q] = the fixed-order
// sum of `tiles` partials (parameter_partial_kernel over S, rows `checked`
// apart; no tiles for an empty batch) + lambda*W[q]; otherwise cuBLAS wrote
// dW and only the check remains. Rows r = (sample, input) < rows:
// dx[r] += silu'(x[r]) * t (silu' from the forward's rows) with t = tw[r]
// (cuBLAS U*W) or, without tw, sum_o U[b,o]*W[o,i] in ascending o.
template<class T>
__global__ void residual_finish_kernel(const T* partial, unsigned tiles, std::size_t checked, bool reduce,
                                       const T* w, T* dw, std::size_t count, T lambda,
                                       const T* derivative, const T* u, const T* tw, T* dx, std::size_t rows,
                                       std::size_t inputs, std::size_t outputs, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    const auto first = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    for (auto q = first; q < count; q += stride) {
        if (reduce) {
            T sum = 0;
            for (unsigned t = 0; t < tiles; ++t) sum += partial[t*checked+q];
            sum += lambda*w[q];
            dw[q] = sum;
        }
        report(dw[q], status);
    }
    for (auto r = first; r < rows; r += stride) {
        T t = 0;
        if (tw) {
            t = tw[r];
        } else {
            const auto b = r/inputs, i = r%inputs;
            for (std::size_t o = 0; o < outputs; ++o) t += u[b*outputs+o]*w[o*inputs+i];
        }
        const T value = dx[r]+derivative[r]*t;
        dx[r] = value; report(value, status);
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
// Host message of status_parameter_range (the CPU's wording).
constexpr const char* parameter_range_message = "resident wavelet log-scale candidate leaves the valid range";
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
// Rational edges. The caches of P, Q, dr/dx and, for the safe policies only,
// g = dQ/dS (`gains`, null for Guarded) are edge-major, [edge*capacity +
// sample], edge = o*inputs + i, so that the samples of one edge are
// contiguous. Every rational kernel maps consecutive lanes to consecutive
// samples (backlog C6), which makes the cache accesses coalesced. One
// instantiation per denominator policy; all caches live in the arena.
//
// Arguments: z = (x - center)/scale depends on the input and sample only, so
// it is computed (and checked) once into `arguments`, [i*capacity + sample],
// instead of once per output in the forward pass and once per edge in the
// parameter VJP (a division each, and the layer input read with stride
// `inputs`); both kernels then read it coalesced, the same value bitwise.
template<class T>
__global__ void rational_argument_kernel(const T* input, T* arguments, std::size_t batch, std::size_t capacity,
                                         std::size_t inputs, detail::RationalScalars<T> config, int* status) {
    const StatusGuard guard{status};
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < batch*inputs; index += stride) {
        const auto i = index/batch, sample = index%batch;
        arguments[i*capacity+sample] = detail::rational_argument(config, input[sample*inputs+i], guard);
    }
}
// Forward: one thread per (output, sample), sample fastest; it sums its
// output over the inputs in order. The parameter-VJP intermediates of every
// executed sample are checked here (the backward pass recomputes them with
// CheckedEarlier), with the full check only where the cheap bound of
// rational_parameter_vjps_bounded does not hold (backlog C7).
template<DenominatorPolicy Policy, class T>
__global__ void rational_forward_kernel(const T* arguments, const T* a, const T* b, const T* bias,
                                        T* values, T* denominator_values, T* derivatives, T* gains, T* output,
                                        std::size_t batch, std::size_t capacity, std::size_t inputs, std::size_t outputs,
                                        detail::RationalScalars<T> config, int* status) {
    const StatusGuard guard{status};
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    const auto m = config.numerator_degree, n = config.denominator_degree;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < batch*outputs; index += stride) {
        const auto o = index/batch, sample = index%batch;
        T sum = bias[o];
        for (std::size_t i = 0; i < inputs; ++i) {
            const auto edge = o*inputs+i, cache = edge*capacity+sample;
            const auto h = detail::rational_horner_at<Policy>(config, arguments[i*capacity+sample], a+edge*(m+1), b+edge*n, guard);
            if (detail::rational_pole<Policy>(config, h)) {
                atomicOr(status, 2);
                values[cache] = denominator_values[cache] = derivatives[cache] = 0;
                continue;
            }
            const auto e = detail::rational_edge<Policy>(config, h, guard);
            // The parameter VJPs are part of the contract for every executed
            // sample, including a zero upstream.
            if (!detail::rational_parameter_vjps_bounded<Policy>(config, h, e.value))
                detail::rational_parameter_vjps_check<Policy>(config, h, e.value, guard);
            if constexpr (Policy != DenominatorPolicy::Guarded) gains[cache] = h.gain;
            values[cache] = h.p; denominator_values[cache] = h.q; derivatives[cache] = e.input_derivative;
            sum += e.value; report(sum, status);
        }
        output[sample*outputs+o] = sum; report(sum, status);
    }
}
// Input VJP: one thread per (input, sample), sample fastest, summing over the
// outputs in order.
template<class T>
__global__ void rational_input_kernel(const T* derivatives, const T* upstream, T* input_gradient,
                                      std::size_t batch, std::size_t capacity, std::size_t inputs, std::size_t outputs, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; index < batch*inputs; index += stride) {
        const auto i = index/batch, sample = index%batch;
        T sum = 0;
#pragma unroll 8
        for (std::size_t o = 0; o < outputs; ++o) sum += upstream[sample*outputs+o]*derivatives[(o*inputs+i)*capacity+sample];
        input_gradient[sample*inputs+i] = sum; report(sum, status);
    }
}
// Largest numerator or denominator degree (RationalConfig: 0..16).
constexpr std::size_t rational_max_degree = 16;
// Parameter VJPs (backlog C8): one warp per edge accumulates all m+1
// numerator and n denominator sums, and the warp of edge (o, 0) also the
// bias of output o, in one pass over the samples: z, z^k, P/Q and the cached
// P, Q, g are read or formed once per sample, not once per parameter. Lane l
// sums the samples l, l+32, ... in order and the warp reduces each sum with
// the same shuffle tree, so every gradient is bitwise that of the former
// warp-per-parameter kernel. The forward pass validated z and every parameter
// VJP of these cached samples with the same operations (CheckedEarlier);
// backward runs only after a successful forward. The upstream products are
// backward results and are reported here.
template<DenominatorPolicy Policy, class T>
__global__ void rational_parameter_kernel(const T* arguments, const T* values, const T* denominator_values, const T* gains,
                                          const T* upstream, const T* parameters, T* gradients,
                                          std::size_t batch, std::size_t capacity, std::size_t inputs, std::size_t outputs,
                                          detail::RationalScalars<T> config, T lambda, int* status) {
    const CheckedEarlier guard;
    const auto m = config.numerator_degree, n = config.denominator_degree, degree = m > n ? m : n;
    const auto edges = inputs*outputs;
    // Gradient layout [numerators (edge, m+1) | bias (outputs) | denominators (edge, n)].
    T* bias_gradients = gradients+edges*(m+1);
    T* denominator_gradients = bias_gradients+outputs;
    const auto lane = threadIdx.x%32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for (auto edge = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32; edge < edges; edge += stride) {
        const auto o = edge/inputs, i = edge%inputs;
        const bool owns_bias = i == 0;
        T numerator_sums[rational_max_degree+1], denominator_sums[rational_max_degree], bias_sum = 0;
#pragma unroll
        for (std::size_t k = 0; k <= rational_max_degree; ++k) numerator_sums[k] = 0;
#pragma unroll
        for (std::size_t k = 0; k < rational_max_degree; ++k) denominator_sums[k] = 0;
#pragma unroll 2
        for (auto sample = static_cast<std::size_t>(lane); sample < batch; sample += 32) {
            const auto cache = edge*capacity+sample;
            const T u = upstream[sample*outputs+o];
            const T z = arguments[i*capacity+sample];
            const T q = denominator_values[cache];
            const T p = n ? values[cache] : T(0);
            const T value = n ? p/q : T(0);
            const T gain = Policy == DenominatorPolicy::Guarded || !n ? T(1) : gains[cache];
            T power = 1;
#pragma unroll
            for (std::size_t k = 0; k <= rational_max_degree; ++k) {
                if (k > degree) break;
                if (k) power *= z;
                const T divided = power/q;
                if (k <= m) {
                    const T term = u*detail::rational_numerator_vjp(q, z, k, power, divided, guard);
                    report(term, status); numerator_sums[k] += term;
                }
                if (k && k <= n) {
                    const T term = u*detail::rational_denominator_vjp<Policy>(p, q, value, gain, z, k, power, divided, guard);
                    report(term, status); denominator_sums[k-1] += term;
                }
            }
            if (owns_bias) { report(u, status); bias_sum += u; }
        }
        const auto reduce = [&](T sum) {
            report(sum, status);
            for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffffU, sum, offset);
            return sum;
        };
#pragma unroll
        for (std::size_t k = 0; k <= rational_max_degree; ++k) {
            if (k > m) break;
            T sum = reduce(numerator_sums[k]);
            if (lane == 0) {
                const auto index = edge*(m+1)+k;
                sum += lambda*parameters[index]; gradients[index] = sum; report(sum, status);
            }
        }
#pragma unroll
        for (std::size_t k = 0; k < rational_max_degree; ++k) {
            if (k >= n) break;
            const T sum = reduce(denominator_sums[k]);
            if (lane == 0) { denominator_gradients[edge*n+k] = sum; report(sum, status); }
        }
        if (owns_bias) {
            const T sum = reduce(bias_sum);
            if (lane == 0) { bias_gradients[o] = sum; report(sum, status); }
        }
    }
}

// Per-edge trainable Mexican-hat wavelets (backlog M7). Every edge (o,i) has
// its own wavelets j = (o*inputs + i)*size + k, so no expansion is shared
// between edges and no contraction maps to a GEMM: y[b,o] = bias[o] +
// sum_i sum_k w_j psi_j(x[b,i]) is O(batch*outputs*inputs*size) evaluations,
// one exp each, and the backward pass recomputes them (caching any per-sample
// row would cost capacity*outputs*inputs*size values). Layouts: the layer
// input is transposed once per forward into xt [i*capacity + sample] and the
// upstream once per backward into ut [o*capacity + sample], so that lanes =
// consecutive samples read both coalesced; parameters are read at block-
// uniform addresses (one broadcast transaction per warp). The per-wavelet
// scales r = exp(-l), a = A sqrt(r) are formed once per forward
// (wavelet_scales_kernel) and reused by the backward pass. Intermediates are
// not checked one by one (CheckedEarlier): the CPU's guarded intermediates
// are all summed into a result (a nonfinite one makes the sum nonfinite), so
// reporting the finished sums gives the same status.
constexpr unsigned wavelet_threads = 256;
template<class T> struct WaveletArgs {
    const T* xt;                                     // (inputs, capacity)
    const T* w; const T* tau; const T* inverse; const T* amplitude; // (outputs, inputs, size)
    std::size_t batch, capacity, inputs, outputs, size;
};
// Branch-free direct path for the kernels: E and the direct terms of x, and
// in `rare` whether wavelet_terms would take its out-of-line branches for x
// (the terms are then zero here). The kernels evaluate a group of samples
// this way and, when any lane of the warp has a rare sample (warp vote),
// recompute the group with wavelet_terms before using it, so every lane gets
// exactly wavelet_terms' result while the common case has no branch per
// evaluation (a per-evaluation branch around the out-of-line call cost the
// backward kernel about a quarter of its time on GB10).
template<bool Derivatives, class T>
__device__ __forceinline__ detail::WaveletTerms<T> wavelet_fast_terms(T x, T tau, detail::WaveletScales<T> scales, bool& rare) {
    const CheckedEarlier unchecked;
    const T q = (x-tau)*scales.inverse, q2 = q*q, e = detail::wavelet_envelope(q2);
    const bool direct = e >= detail::math::constants<T>::min_normal;
    rare |= !direct;
    auto r = detail::wavelet_direct_terms<Derivatives>(q, q2, e, scales, unchecked);
    if (!direct) r = {T(0), T(0), T(0)};
    return r;
}
template<class T>
__global__ void wavelet_scales_kernel(const T* log_scales, T* inverse, T* amplitude, std::size_t count) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto j = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; j < count; j += stride) {
        const auto s = detail::wavelet_scales(log_scales[j]);
        inverse[j] = s.inverse; amplitude[j] = s.amplitude;
    }
}
// rows x columns row-major -> columns x pitch (out[c*pitch + r]), 32x32 tiles.
template<class T>
__global__ void transpose_kernel(const T* in, T* out, std::size_t rows, std::size_t columns, std::size_t pitch) {
    __shared__ T tile[32][33];
    const auto c0 = static_cast<std::size_t>(blockIdx.x)*32;
    // Row tiles stride over grid y (any batch fits the 65535 limit).
    for (auto r0 = static_cast<std::size_t>(blockIdx.y)*32; r0 < rows; r0 += static_cast<std::size_t>(gridDim.y)*32) {
        for (unsigned y = threadIdx.y; y < 32; y += blockDim.y) {
            const auto r = r0+y, c = c0+threadIdx.x;
            if (r < rows && c < columns) tile[y][threadIdx.x] = in[r*columns+c];
        }
        __syncthreads();
        for (unsigned y = threadIdx.y; y < 32; y += blockDim.y) {
            const auto c = c0+y, r = r0+threadIdx.x;
            if (r < rows && c < columns) out[c*pitch+r] = tile[threadIdx.x][y];
        }
        __syncthreads();
    }
}
// Forward: block (o, chunk slot, input split); thread t owns samples
// chunk*256*S + t + 256*s, s < S, and sums each over (i, k) in ascending
// order. Chunks beyond gridDim.y stride. Unsplit (gridDim.z = 1) the sum
// starts from bias[o] and is y (the CPU's order); split z sums the inputs
// [z*per, (z+1)*per) from zero into partial[z*batch*outputs + b*outputs + o],
// which wavelet_forward_finish_kernel adds to the bias in the order of z.
template<unsigned S, class T>
__global__ void __launch_bounds__(wavelet_threads)
wavelet_forward_kernel(WaveletArgs<T> a, const T* bias, T* y, int* status, std::size_t per, T* partial) {
    const CheckedEarlier unchecked;
    const auto o = static_cast<std::size_t>(blockIdx.x);
    const bool whole = gridDim.z == 1;
    const auto i_begin = static_cast<std::size_t>(blockIdx.z)*per, i_end = a.inputs-i_begin < per ? a.inputs : i_begin+per;
    T* const out = whole ? y : partial+static_cast<std::size_t>(blockIdx.z)*a.batch*a.outputs;
    constexpr auto span = static_cast<std::size_t>(wavelet_threads)*S;
    for (auto first = static_cast<std::size_t>(blockIdx.y)*span; first < a.batch; first += static_cast<std::size_t>(gridDim.y)*span) {
      // Full chunks compile without any test of the batch end. In the last,
      // partial chunk a warp whose lanes are all beyond the batch skips that
      // sample slot (warp-uniform); other lanes beyond it evaluate at x = tau
      // (q = 0, the direct path) and their sums are not written.
      const auto end = a.batch-first < span ? a.batch : first+span;
      const auto chunk = [&](auto partial) {
        constexpr bool Partial = decltype(partial)::value;
        T sum[S];
        bool valid[S], active[S];
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            sum[s] = whole ? bias[o] : T(0); valid[s] = !Partial || first+threadIdx.x+s*wavelet_threads < end;
            active[s] = !Partial || __any_sync(0xffffffffU, valid[s]);
        }
        for (std::size_t i = i_begin; i < i_end; ++i) {
            T x[S];
#pragma unroll
            for (unsigned s = 0; s < S; ++s) x[s] = valid[s] ? a.xt[i*a.capacity+first+threadIdx.x+s*wavelet_threads] : T(0);
            const auto base = (o*a.inputs+i)*a.size;
            for (std::size_t k = 0; k < a.size; ++k) {
                const auto j = base+k;
                const T w = a.w[j], tau = a.tau[j];
                const detail::WaveletScales<T> scales{a.inverse[j], a.amplitude[j]};
                T value[S];
                bool rare = false;
#pragma unroll
                for (unsigned s = 0; s < S; ++s)
                    if (active[s]) value[s] = wavelet_fast_terms<false>(valid[s] ? x[s] : tau, tau, scales, rare).value;
                if (__any_sync(0xffffffffU, rare)) {
#pragma unroll
                    for (unsigned s = 0; s < S; ++s)
                        if (active[s]) value[s] = detail::wavelet_terms<false>(valid[s] ? x[s] : tau, tau, scales, unchecked).value;
                }
#pragma unroll
                for (unsigned s = 0; s < S; ++s)
                    if (active[s]) sum[s] += w*value[s];
            }
        }
#pragma unroll
        for (unsigned s = 0; s < S; ++s)
            if (valid[s]) {
                out[(first+threadIdx.x+s*wavelet_threads)*a.outputs+o] = sum[s];
                if (whole) report(sum[s], status);
            }
      };
      if (end-first == span) chunk(std::false_type{}); else chunk(std::true_type{});
    }
}
// y = bias + the `splits` partial sums of a split forward, in order of z.
template<class T>
__global__ void wavelet_forward_finish_kernel(const T* partial, unsigned splits, const T* bias, T* y, std::size_t batch,
                                              std::size_t outputs, int* status) {
    const auto count = batch*outputs, stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto r = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; r < count; r += stride) {
        T sum = bias[r%outputs];
        for (unsigned z = 0; z < splits; ++z) sum += partial[z*count+r];
        y[r] = sum; report(sum, status);
    }
}
// Largest samples per thread S of the forward and backward kernels.
template<class T> constexpr unsigned wavelet_forward_samples = std::is_same_v<T, double> ? 1 : 8;
template<class T> constexpr unsigned wavelet_backward_samples = std::is_same_v<T, double> ? 4 : 8;
// Sample chunks of a batch with S samples per thread.
inline std::size_t wavelet_chunks(std::size_t batch, unsigned samples) {
    const auto span = static_cast<std::size_t>(wavelet_threads)*samples;
    return (batch+span-1)/span;
}

// Resident blocks per multiprocessor the backward kernel is compiled for
// (__launch_bounds__ minimum; register cap 65536/(256*blocks)). FP32 3: the
// warp-vote recompute path of every unrolled sample otherwise doubles the
// registers (64 -> 128, two blocks); on GB10 a cap of 85 registers ran the
// 256-wide step in 2.21 ms against 2.29 (64 registers) and 3.36 (51, with
// spills). FP64 1 (no cap): 2 and 3 were 2-4% slower. The cap is the
// same on sm_86 and sm_121 (64K registers, 1536 threads per multiprocessor);
// both compile every instantiation without spills (M7 review: per-round
// addresses in shared memory, u read again in the rare path).
template<class T> constexpr int wavelet_backward_blocks = 3;
template<> constexpr int wavelet_backward_blocks<double> = 1;
// Wavelets per reduction round of the backward kernel.
constexpr unsigned wavelet_round = 8;
// Backward: block (i, slot, split) runs the chunks slot, slot + gridDim.y, ...
// of its input's samples (thread t owns S samples of a chunk, as forward) over
// the outputs [split*per, ...) of the split: for every wavelet j of those
// edges and owned sample, with t = w dpsi/dx,
//   dx[b,i] += u t (ascending o, then k), and the sample sums
//   sum u psi, sum u t, sum u w dpsi/dl over the chunk,
// the latter reduced per round of wavelet_round wavelets by a fixed lane
// butterfly, then over the 8 warps in order, into this slot's partials
// (assigned by its first chunk, accumulated by the next ones, in order).
// dx goes to dx_out[split*dx_stride + b*inputs + i]; WithDx = false skips it.
template<unsigned S, bool WithDx, class T>
__global__ void __launch_bounds__(wavelet_threads, wavelet_backward_blocks<T>)
wavelet_backward_kernel(WaveletArgs<T> a, const T* ut, T* partial, std::size_t per, T* dx_out, std::size_t dx_stride, int* status) {
    const CheckedEarlier unchecked;
    constexpr unsigned warps = wavelet_threads/32, values = 3*wavelet_round;
    __shared__ T sums[2][warps][values];
    // Partial of each pair of the round (thread 0 writes slot_partial + j);
    // in shared memory, as the plane stride, so that the reduction keeps no
    // 64-bit addresses live in registers across the evaluation loop.
    __shared__ T* partial_of[2][wavelet_round];
    __shared__ std::size_t plane_stride, dx_offset;
    const auto i = static_cast<std::size_t>(blockIdx.x), slot = static_cast<std::size_t>(blockIdx.y);
    const auto o_begin = static_cast<std::size_t>(blockIdx.z)*per;
    const auto o_end = o_begin+per < a.outputs ? o_begin+per : a.outputs;
    const auto pairs = o_end > o_begin ? (o_end-o_begin)*a.size : 0;
    const auto count = a.outputs*a.inputs*a.size;          // wavelets: partial planes are `count` apart
    T* slot_partial = partial+slot*3*count;
    if (threadIdx.x == 0) { plane_stride = count; dx_offset = static_cast<std::size_t>(blockIdx.z)*dx_stride+i; }
    __syncthreads();
    const unsigned lane = threadIdx.x%32, warp = threadIdx.x/32;
    constexpr auto span = static_cast<std::size_t>(wavelet_threads)*S;
    unsigned round = 0;
    for (auto chunk = slot; chunk*span < a.batch; chunk += gridDim.y) {
      // Full chunks compile without any test of the batch end (see the forward).
      const auto begin = chunk*span, end = a.batch-begin < span ? a.batch : begin+span;
      const auto run = [&](auto partial) {
        constexpr bool Partial = decltype(partial)::value;
        const auto first = begin+threadIdx.x;
        const bool assign = chunk == slot;
        // Samples beyond the batch: a warp whose lanes are all beyond it skips
        // that sample slot (warp-uniform); otherwise they evaluate at x = tau
        // (q = 0: the direct path, finite terms) with u = 0, contributing
        // exact zeros.
        T x[S], dx[S];
        bool valid[S], active[S];
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            valid[s] = !Partial || first+s*wavelet_threads < end;
            active[s] = !Partial || __any_sync(0xffffffffU, valid[s]);
            x[s] = valid[s] ? a.xt[i*a.capacity+first+s*wavelet_threads] : T(0);
            dx[s] = 0;
        }
        // Pair p = (o - o_begin)*size + k of wavelet j = (o*inputs + i)*size + k,
        // carried from pair to pair (no 64-bit division per wavelet).
        std::size_t o = o_begin, k = 0, j = (o_begin*a.inputs+i)*a.size;
        const auto next_output = (a.inputs-1)*a.size;
        for (std::size_t p0 = 0; p0 < pairs; p0 += wavelet_round, ++round) {
            T* buffer = sums[round&1][warp];
            // Each wavelet's three sample sums are reduced over the warp as soon
            // as they are complete, so only three of them are live at a time.
            // Not unrolled: the body is already S evaluations; unrolled 8 times it
            // multiplied the kernel's code (FP32 S = 8: about 192 KB of SASS
            // against 38 KB) for no measured gain (instruction-cache footprint).
#pragma unroll 1
            for (unsigned v = 0; v < wavelet_round; ++v) {
                T psum = 0, tsum = 0, lsum = 0;
                if (p0+v < pairs) {
                    if (threadIdx.x == 0) partial_of[round&1][v] = slot_partial+j;
                    const T w = a.w[j], tau = a.tau[j];
                    const detail::WaveletScales<T> scales{a.inverse[j], a.amplitude[j]};
                    T step[S];
                    bool rare = false;
#pragma unroll
                    for (unsigned s = 0; s < S; ++s) {
                        const T u = valid[s] ? ut[o*a.capacity+first+s*wavelet_threads] : T(0);
                        step[s] = 0;
                        if (!active[s]) continue;
                        const auto r = wavelet_fast_terms<true>(valid[s] ? x[s] : tau, tau, scales, rare);
                        const T t = w*r.input_derivative;
                        step[s] = u*t;
                        psum += u*r.value; tsum += u*t; lsum += u*(w*r.log_scale_derivative);
                    }
                    if (__any_sync(0xffffffffU, rare)) {
                        psum = tsum = lsum = 0;
#pragma unroll
                        for (unsigned s = 0; s < S; ++s) {
                            if (!active[s]) continue;
                            // u read again (not kept live across the vote: register pressure).
                            const T us = valid[s] ? ut[o*a.capacity+first+s*wavelet_threads] : T(0);
                            const auto r = detail::wavelet_terms<true>(valid[s] ? x[s] : tau, tau, scales, unchecked);
                            const T t = w*r.input_derivative;
                            step[s] = us*t;
                            psum += us*r.value; tsum += us*t; lsum += us*(w*r.log_scale_derivative);
                        }
                    }
                    if constexpr (WithDx) {
#pragma unroll
                        for (unsigned s = 0; s < S; ++s) dx[s] += step[s];
                    }
                    ++j;
                    if (++k == a.size) { k = 0; ++o; j += next_output; }
                }
#pragma unroll
                for (unsigned offset = 16; offset; offset /= 2) {
                    psum += __shfl_xor_sync(0xffffffffU, psum, offset);
                    tsum += __shfl_xor_sync(0xffffffffU, tsum, offset);
                    lsum += __shfl_xor_sync(0xffffffffU, lsum, offset);
                }
                if (lane == 0) { buffer[v] = psum; buffer[wavelet_round+v] = tsum; buffer[2*wavelet_round+v] = lsum; }
            }
            __syncthreads();
            if (threadIdx.x < values) {
                const auto v = threadIdx.x%wavelet_round, plane = threadIdx.x/wavelet_round;
                if (p0+v < pairs) {
                    T total = 0;
                    for (unsigned w = 0; w < warps; ++w) total += sums[round&1][w][threadIdx.x];
                    T* at = partial_of[round&1][v]+plane*plane_stride;
                    *at = assign ? total : *at+total;
                }
            }
            // The other buffer is written next round; no second barrier needed.
        }
        if constexpr (WithDx) {
#pragma unroll
            for (unsigned s = 0; s < S; ++s)
                if (valid[s]) {
                    const T value = dx[s];
                    dx_out[dx_offset+(first+s*wavelet_threads)*a.inputs] = value;
                    if (gridDim.z == 1) report(value, status);
                }
        }
      };
      if (end-begin == span) run(std::false_type{}); else run(std::true_type{});
    }
}
// Parameter VJPs from `slots` partial planes (fixed order): dw = sum + lambda*w,
// dtau = -sum, dl = sum; zero (plus lambda*w) for an empty batch (slots = 0).
template<class T>
__global__ void wavelet_finish_kernel(const T* partial, unsigned slots, const T* w, T lambda, T* dw, T* dtau, T* dl,
                                      std::size_t count, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto j = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; j < count; j += stride) {
        T p = 0, t = 0, l = 0;
        for (unsigned slot = 0; slot < slots; ++slot) {
            const T* at = partial+static_cast<std::size_t>(slot)*3*count+j;
            p += at[0]; t += at[count]; l += at[2*count];
        }
        p += lambda*w[j];
        dw[j] = p; dtau[j] = -t; dl[j] = l;
        report(p, status); report(t, status); report(l, status);
    }
}
// dx of a split backward: the `splits` partial rows summed in order.
template<class T>
__global__ void wavelet_input_finish_kernel(const T* partial, unsigned splits, std::size_t stride_split, T* dx, std::size_t count, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto r = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; r < count; r += stride) {
        T sum = 0;
        for (unsigned p = 0; p < splits; ++p) sum += partial[p*stride_split+r];
        dx[r] = sum; report(sum, status);
    }
}
// Bias VJP: one warp per output sums its transposed upstream row (lane l
// samples l, l+32, ..., then the shuffle tree).
template<class T>
__global__ void wavelet_bias_kernel(const T* ut, T* gradient, std::size_t batch, std::size_t capacity, std::size_t outputs, int* status) {
    const auto lane = threadIdx.x%32;
    const auto stride = static_cast<std::size_t>(gridDim.x)*(blockDim.x/32);
    for (auto o = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32; o < outputs; o += stride) {
        T sum = 0;
        for (auto b = static_cast<std::size_t>(lane); b < batch; b += 32) sum += ut[o*capacity+b];
        for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (lane == 0) { gradient[o] = sum; report(sum, status); }
    }
}
// SGD candidates of the log scales: exp(l) and exp(-l) finite and nonzero.
// A nonfinite candidate is reported by the candidate kernel (status bit 1);
// a finite one outside the range sets status_parameter_range, so that the
// host names the cause as the CPU does.
template<class T>
__global__ void validate_log_scale_kernel(const T* next, std::size_t count, int* status) {
    const auto stride = static_cast<std::size_t>(gridDim.x)*blockDim.x;
    for (auto j = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; j < count; j += stride) {
        const T l = next[j], scale = detail::math::exp(l), inverse = detail::math::exp(-l);
        if (!isfinite(scale) || !(scale > 0) || !isfinite(inverse) || !(inverse > 0))
            atomicOr(status, isfinite(l) ? status_parameter_range : 1);
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

// cuBLAS workspace of every contraction, and the larger one the residual
// branch's GEMMs (backlog M3) switch to: cuBLAS splits long-k products such
// as dW = U^T*S (256 x 256 x 8192) over k only with enough workspace (M3
// evidence, phase 2: 105 us at 4 MiB, 54 us at 32 MiB, FP32 RTX 3090). The
// carriers keep 4 MiB, so their kernel selection and results are unchanged.
constexpr std::size_t blas_workspace_bytes = std::size_t{4} << 20;
constexpr std::size_t residual_workspace_bytes = std::size_t{32} << 20;

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
    // Derivative coefficients C* of the FP32/TF32 B-spline layers (backlog C13),
    // shared like the scratch (largest outputs*inputs*(terms-1), or *terms for
    // a sparse layer); none otherwise.
    std::size_t differences = 0;
    // Residual branch GEMM workspace (residual_workspace_bytes), reserved only
    // when some branch takes a cuBLAS path at capacity.
    std::size_t residual_workspace = 0;
    bool has_residual_workspace = false;
    // Runs the residual branch's cuBLAS calls `f` with its workspace, then
    // restores the shared one (host-side handle state; graph capture records
    // the pointers the kernels use).
    template<class F> void with_residual_workspace(F&& f) {
        if (!has_residual_workspace) { f(); return; }
        check(cublasSetWorkspace(blas, ptr(residual_workspace), residual_workspace_bytes), "resident cuBLAS workspace");
        try { f(); } catch (...) { cublasSetWorkspace(blas, ptr(blas_workspace), blas_workspace_bytes); throw; }
        check(cublasSetWorkspace(blas, ptr(blas_workspace), blas_workspace_bytes), "resident cuBLAS workspace");
    }
    // Runs the cuBLAS calls `f` with FP32 (non-tensor-op) math, then restores the
    // handle's mode (TF32 under TensorFloat32; a no-op otherwise). Backlog C13:
    // input-VJP GEMMs of the families whose TF32 rounding the derivatives amplify.
    template<class F> void with_fp32_math(F&& f) {
        cublasMath_t mode{};
        check(cublasGetMathMode(blas, &mode), "resident cuBLAS math mode");
        if (mode == CUBLAS_DEFAULT_MATH) { f(); return; }
        check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "resident cuBLAS FP32 math");
        try { f(); } catch (...) { cublasSetMathMode(blas, mode); throw; }
        check(cublasSetMathMode(blas, mode), "resident cuBLAS math mode");
    }
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

// A layer's parameters occupy [coefficients | bias | nonlinear | residual] at
// `offset` inside each of the parameter, gradient and candidate regions. The
// residual part holds the SiLU branch weights W (outputs, inputs) of backlog
// M3 and is empty without the branch, so the blocks (and the arena) of layers
// without it are unchanged; the shared candidate and commit kernels and
// (through dW) the L2 penalty cover W like every other parameter.
struct ParameterBlock {
    std::size_t inputs, outputs, terms, coefficients, nonlinear_count, offset;
    std::size_t residual_count = 0; // outputs*inputs with the branch, else 0
    std::size_t silu = 0, silu_derivative = 0; // workspace: S, silu'(X), (capacity, inputs) each; branch only
    std::size_t bias() const noexcept { return offset+coefficients; }
    std::size_t nonlinear() const noexcept { return offset+coefficients+outputs; }
    std::size_t residual() const noexcept { return offset+coefficients+outputs+nonlinear_count; }
    std::size_t size() const noexcept { return coefficients+outputs+nonlinear_count+residual_count; }
};

ParameterBlock parameter_block(const Layer& layer, std::size_t nonlinear_count, std::size_t offset) {
    const auto count = layer.coefficients().size()+layer.outputs(), maximum = std::vector<double>().max_size();
    const auto residual = layer.residual() ? product(layer.inputs(), layer.outputs()) : 0;
    if (count>maximum || nonlinear_count>maximum-count || residual>maximum-count-nonlinear_count ||
        count+nonlinear_count+residual>maximum-offset)
        throw std::overflow_error("resident parameter size overflow");
    return {layer.inputs(), layer.outputs(), layer.terms(), layer.coefficients().size(), nonlinear_count, offset, residual};
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
    // FP32/TF32 B-spline layers of degree >= 1 (backlog C13): the knot scales
    // s_j = p/(t_{j+p} - t_j), j = 1..terms-1, of the derivative coefficients.
    std::size_t knot_scales = 0;
    // Per-input spline grids (backlog M6): `grids` knot vectors of knot_stride
    // knots each (grids = 1: the shared grid, knot_stride 0 in the kernels).
    std::size_t grids = 1, knot_stride = 0;
    // Dense per-input layers: knot rows staged per block of basis_grid_kernel
    // (0: read from global memory), their padded stride, and the grid bound
    // (multiprocessors x resident blocks), from select_knot_stage.
    unsigned knot_stage_rows = 0, knot_pad = 0, grid_blocks = 0;
    // Phi' rows kept from forward to backward; otherwise (backlog C3) the
    // backward finish recomputes them from the layer input (no derivatives region).
    bool stored_derivatives = true;
};

// Sparse B-spline path (backlog C5, kernels above): arena offsets in elements
// of T (the index regions are unsigned/int, rounded up to whole elements).
// Inactive: the layer keeps the dense rows of ExpansionPlan.
struct SparseSplinePlan {
    bool active = false;
    unsigned width = 0;   // degree + 1
    unsigned chunk = 1;   // inputs per staged C tile of spline_forward_kernel
    int key_bits = 1;     // significant bits of the largest first index
    unsigned ranges = 1;  // coefficient ranges per input of spline_parameter_kernel (grid z)
    unsigned threads = 0; // samples per block of spline_forward_kernel
    unsigned group = 0;   // inputs per block of spline_input_kernel
    std::size_t first = 0, values = 0, derivatives = 0;              // sparse rows
    std::size_t sorted_first = 0, order = 0, indices = 0, offsets = 0; // per-input sort
    std::size_t sort_storage = 0, sort_bytes = 0;                      // CUB temporary storage
};

template<class T>
struct BasisPlan {
    using edges_type = BasisEdges;
    ExpansionPlan<T> expansion;
    SparseSplinePlan sparse;
    // After a device grid refit (backlog M6): the layer's configuration with
    // the refit knots, which replaces the snapshot's in downloads and in the
    // upload structure check (the snapshot keeps its construction state).
    std::optional<BSplineConfig> refit;
};

// Adds trainable shared centers/log widths (nonlinear block: centers, then
// log widths) and their tiled reductions.
template<class T>
struct TrainableRbfPlan {
    using edges_type = TrainableRbfEdges;
    ExpansionPlan<T> expansion;
    std::size_t log_derivatives = 0, partials = 0;
    unsigned partial_tiles = 1;
    // Center/log-width reduction (backlog C12, select_reduction): terms per
    // nonlinear_tile_kernel block (0: never tiled), the least tile-kernel grid
    // that the automatic choice accepts, and the path a test scope forces.
    unsigned tile_group = 0, tile_min_blocks = 1;
    testing::NonlinearReduction reduction = testing::NonlinearReduction::Automatic;
};

// Rational edges with the arguments z per (input, sample), edge-major caches
// of P, Q and dr/dx, plus g = dQ/dS for the safe denominator policies
// (nonlinear block: denominators).
template<class T>
struct RationalPlan {
    using edges_type = RationalEdges;
    ParameterBlock block;
    RationalConfig config;
    detail::RationalScalars<T> scalars; // center, scale and the pole threshold in T
    std::size_t arguments = 0, values = 0, derivatives = 0, denominator_values = 0, gains = 0;
    bool safe() const { return config.denominator_policy != DenominatorPolicy::Guarded; }
};

// Per-edge wavelets (backlog M7; nonlinear block: translations, then log
// scales, `count` = outputs*inputs*size each). Workspace: the transposed layer
// input xt (inputs, capacity) and upstream ut (outputs, capacity), the
// per-wavelet scales r and a, the backward kernel's partial sums (`slots`
// planes of 3*count) and, with output splits, the dx partials (`splits` rows of
// capacity*inputs) and, with forward input splits, the forward partial sums.
// Launch geometry (select_wavelet_launch): slots and both splits are maxima
// fixed at construction; every launch chooses within them (cost model).
template<class T>
struct WaveletPlan {
    using edges_type = WaveletEdges;
    ParameterBlock block;
    std::size_t count = 0;
    std::size_t xt = 0, ut = 0, inverse = 0, amplitude = 0, partials = 0, dx_partials = 0;
    unsigned slots = 1, splits = 1;
    // Forward input splits: partial sums of at most y_partial_values values.
    std::size_t y_partials = 0, y_partial_values = 0;
    // Device: multiprocessors and resident blocks per multiprocessor of every
    // instantiation, index log2(S) (occupancy API; backward with dx).
    unsigned multiprocessors = 1;
    unsigned forward_resident[4] = {1, 1, 1, 1}, backward_resident[4] = {1, 1, 1, 1};
    testing::WaveletLaunch forced{}; // test hook: nonzero values replace the choice
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
using Plan = std::variant<BasisPlan<T>, TrainableRbfPlan<T>, RationalPlan<T>, AffinePlan<T>, TanhPlan<T>, LayerNormPlan<T>,
                          WaveletPlan<T>>;

template<class T> const ParameterBlock& block_of(const BasisPlan<T>& p) { return p.expansion.block; }
template<class T> const ParameterBlock& block_of(const TrainableRbfPlan<T>& p) { return p.expansion.block; }
template<class T> const ParameterBlock& block_of(const RationalPlan<T>& p) { return p.block; }
template<class T> ParameterBlock& block_of(BasisPlan<T>& p) { return p.expansion.block; }
template<class T> ParameterBlock& block_of(TrainableRbfPlan<T>& p) { return p.expansion.block; }
template<class T> ParameterBlock& block_of(RationalPlan<T>& p) { return p.block; }
template<class T> const ParameterBlock& block_of(const WaveletPlan<T>& p) { return p.block; }
template<class T> ParameterBlock& block_of(WaveletPlan<T>& p) { return p.block; }

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
            // Within each grid (backlog M6: one per input, or one shared).
            const auto n = c.knots.size()/c.grids;
            for (std::size_t k = 0; k + 1 < c.knots.size(); ++k)
                if ((k+1)%n != 0 && c.knots[k] < c.knots[k+1] && !(static_cast<T>(c.knots[k]) < static_cast<T>(c.knots[k+1]))) invalid();
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
    BasisPlan<T> plan{{parameter_block(layer, 0, offset), device_view<T>(edges.basis)}};
    if (const auto* spline = std::get_if<BSplineConfig>(&edges.basis); spline && spline->grids != 1) {
        plan.expansion.grids = spline->grids;
        plan.expansion.knot_stride = spline->knots.size()/spline->grids;
    }
    return plan;
}
template<class T> Plan<T> make_plan(const Layer& layer, const TrainableRbfEdges& edges, std::size_t offset) {
    return TrainableRbfPlan<T>{{parameter_block(layer, product(layer.terms(), 2), offset), device_view<T>(edges.basis)}};
}
template<class T> Plan<T> make_plan(const Layer& layer, const RationalEdges& edges, std::size_t offset) {
    representable<T>(edges.denominators);
    return RationalPlan<T>{parameter_block(layer, edges.denominators.size(), offset), edges.config, device_scalars<T>(edges.config)};
}
// FP32: translations representable, exp(+-l) finite and nonzero in float
// (FP64: the CPU validation). std::invalid_argument.
template<class T> void check_wavelet(const WaveletEdges& edges) {
    if constexpr (!std::is_same_v<T, double>) {
        representable<T>(edges.translations);
        for (double l : edges.log_scales) {
            const T value = narrow<T>(l), scale = std::exp(value), inverse = std::exp(-value);
            if (!std::isfinite(scale) || !(scale > 0) || !std::isfinite(inverse) || !(inverse > 0))
                throw std::invalid_argument("resident float32 wavelet log scale is not representable");
        }
    }
}
template<class T> Plan<T> make_plan(const Layer& layer, const WaveletEdges& edges, std::size_t offset) {
    check_wavelet<T>(edges);
    WaveletPlan<T> plan{parameter_block(layer, product(edges.translations.size(), 2), offset)};
    plan.count = edges.translations.size();
    return plan;
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
    if (layer.residual()) { finite(layer.residual()->weights); representable<T>(layer.residual()->weights); }
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
// Elements of T holding `count` 32-bit indices.
template<class T> std::size_t index_elements(std::size_t count) { return (product(count, sizeof(unsigned))+sizeof(T)-1)/sizeof(T); }
template<class T> void reserve_sparse(SparseSplinePlan& p, const ParameterBlock& b, std::size_t capacity, Reservation<T>& reserve) {
    const auto samples = product(capacity, b.inputs), rows = product(samples, p.width);
    p.first = reserve(index_elements<T>(samples));
    p.values = reserve(rows); p.derivatives = reserve(product(samples, sparse_derivative_count<T>(p.width)));
    p.sorted_first = reserve(index_elements<T>(samples)); p.order = reserve(index_elements<T>(samples));
    p.indices = reserve(index_elements<T>(samples)); p.offsets = reserve(index_elements<T>(product(b.inputs, 2)));
    p.sort_storage = reserve((p.sort_bytes+sizeof(T)-1)/sizeof(T));
}
template<class T> void reserve_knot_scales(ExpansionPlan<T>& p, Reservation<T>& reserve) {
    if constexpr (spline_difference_vjp<T>) if (p.view.degree != 0) p.knot_scales = reserve(product(p.grids, p.block.terms-1));
}
// Knots of every grid (backlog M6).
template<class T> std::size_t knot_count(const ExpansionPlan<T>& p) { return product(p.grids, p.block.terms+p.view.degree+1); }
template<class T> void reserve_workspace(BasisPlan<T>& plan, std::size_t capacity, Reservation<T>& reserve) {
    auto& p = plan.expansion;
    if (plan.sparse.active) {
        reserve_sparse(plan.sparse, p.block, capacity, reserve);
        p.knots = reserve(knot_count(p));
        reserve_knot_scales(p, reserve);
        return;
    }
    p.stored_derivatives = !recompute_derivatives(p);
    // FP32/TF32 B-spline layers never read Phi' (backlog C13), also beyond the stage.
    if (spline_difference_vjp<T> && p.view.kind == detail::BasisKind::BSpline) p.stored_derivatives = false;
    reserve_rows(p, capacity, reserve);
    if (p.view.kind == detail::BasisKind::GaussianRbf || p.view.kind == detail::BasisKind::MexicanHat)
        p.centers = reserve(p.block.terms);
    if (p.view.kind == detail::BasisKind::MexicanHat) p.scales = reserve(p.block.terms);
    if (p.view.kind == detail::BasisKind::BSpline) { p.knots = reserve(knot_count(p)); reserve_knot_scales(p, reserve); }
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
    plan.arguments = reserve(product(capacity, plan.block.inputs));
    plan.values = reserve(count); plan.derivatives = reserve(count); plan.denominator_values = reserve(count);
    if (plan.safe()) plan.gains = reserve(count);
}
template<class T> void reserve_workspace(WaveletPlan<T>& plan, std::size_t capacity, Reservation<T>& reserve) {
    const auto& b = plan.block;
    plan.xt = reserve(product(capacity, b.inputs)); plan.ut = reserve(product(capacity, b.outputs));
    plan.inverse = reserve(plan.count); plan.amplitude = reserve(plan.count);
    plan.partials = reserve(product(product(plan.count, 3), plan.slots));
    if (plan.splits > 1) plan.dx_partials = reserve(product(product(capacity, b.inputs), plan.splits));
    if (plan.y_partial_values) plan.y_partials = reserve(plan.y_partial_values);
}
template<class T> void reserve_workspace(AffinePlan<T>& plan, std::size_t, Reservation<T>& reserve) {
    plan.scale = reserve(plan.block.features); plan.shift = reserve(plan.block.features);
}
template<class T> void reserve_workspace(TanhPlan<T>&, std::size_t, Reservation<T>&) {}
template<class T> void reserve_workspace(LayerNormPlan<T>& plan, std::size_t capacity, Reservation<T>& reserve) {
    plan.stats = reserve(product(capacity, 2));
    if (plan.affine()) plan.partials = reserve(product(plan.block.features, 2*static_cast<std::size_t>(norm_tiles(capacity))));
}

// The S and silu'(X) regions of a layer with the residual branch, after its
// carrier's workspace (nothing is reserved without the branch).
template<class T, class P> void reserve_residual(P& plan, std::size_t capacity, Reservation<T>& reserve) {
    if constexpr (requires { typename P::edges_type; }) {
        auto& b = block_of(plan);
        if (b.residual_count) { b.silu = reserve(product(capacity, b.inputs)); b.silu_derivative = reserve(product(capacity, b.inputs)); }
    }
}

template<class P> constexpr bool is_expansion = requires(const P& p) { p.expansion; };
// A B-spline layer on the sparse path (no dense rows, W scratch or partial tiles).
template<class P> bool sparse_layer(const P& p) {
    if constexpr (requires { p.sparse; }) return p.sparse.active; else return false;
}
template<class P> constexpr bool is_kan_layer = requires { typename P::edges_type; };

// Elements of the shared W = U*C scratch an expansion layer needs, and the
// U*W (capacity, inputs) of a residual branch (zero otherwise).
template<class T> std::size_t scratch_extent(const Plan<T>& plan, std::size_t capacity) {
    return std::visit([&](const auto& p) -> std::size_t {
        using P = std::decay_t<decltype(p)>;
        std::size_t extent = 0;
        if constexpr (is_expansion<P>)
            if (!sparse_layer(p)) extent = product(product(capacity, p.expansion.block.inputs), p.expansion.block.terms);
        if constexpr (is_kan_layer<P>)
            if (block_of(p).residual_count) extent = std::max(extent, product(capacity, block_of(p).inputs));
        return extent;
    }, plan);
}

// A residual dW of at most small_parameter_vjp weights+outputs (and, FP32,
// small_parameter_work batch*(weights+outputs)) is reduced by
// parameter_partial_kernel over S, the expansion carriers' rule.
template<class T> bool small_residual_vjp(const ParameterBlock& b, std::size_t batch) {
    const auto checked = b.residual_count+b.outputs;
    return checked <= small_parameter_vjp<T> && batch <= small_parameter_work<T>/checked;
}
// Elements of the shared small-parameter-VJP partials a layer needs (zero otherwise).
template<class T> std::size_t partial_extent(const Plan<T>& plan, std::size_t capacity) {
    return std::visit([&](const auto& p) -> std::size_t {
        using P = std::decay_t<decltype(p)>;
        std::size_t extent = 0;
        if constexpr (is_expansion<P>) {
            const auto checked = p.expansion.block.coefficients+p.expansion.block.outputs;
            extent = checked <= small_parameter_vjp<T> && !sparse_layer(p) ? checked*parameter_tile_count(capacity) : 0;
        }
        if constexpr (is_kan_layer<P>) {
            const auto& b = block_of(p);
            const auto checked = b.residual_count+b.outputs;
            if (b.residual_count && checked <= small_parameter_vjp<T>)
                extent = std::max(extent, checked*parameter_tile_count(capacity));
        }
        return extent;
    }, plan);
}

// Device properties queried at construction by the sparse B-spline selection
// (backlog C5) and the trainable-RBF reduction (C12).
struct ExecutorDevice {
    int fp64_ratio = 0;           // cudaDevAttrSingleToDoublePrecisionPerfRatio
    unsigned multiprocessors = 1; // cudaDevAttrMultiProcessorCount
    std::size_t l2_bytes = 0;     // cudaDevAttrL2CacheSize
    std::size_t shared_optin = 0; // cudaDevAttrMaxSharedMemoryPerBlockOptin
    unsigned threads_per_multiprocessor = 0; // cudaDevAttrMaxThreadsPerMultiProcessor
};
ExecutorDevice executor_device() {
    ExecutorDevice d;
    int device = 0, value = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return d;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) == cudaSuccess) d.fp64_ratio = value;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device) == cudaSuccess && value > 0)
        d.multiprocessors = static_cast<unsigned>(value);
    if (cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, device) == cudaSuccess && value > 0)
        d.l2_bytes = static_cast<std::size_t>(value);
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess)
        d.shared_optin = static_cast<std::size_t>(value);
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerMultiProcessor, device) == cudaSuccess && value > 0)
        d.threads_per_multiprocessor = static_cast<unsigned>(value);
    return d;
}
// Knot staging of a dense per-input B-spline layer (backlog M6): the rows a
// block of basis_grid_kernel reads (its rows per iteration, at most the
// inputs) with an odd stride, when stage and knots fit the default 48 KiB of
// shared memory; the grid bound is the device's multiprocessors times the
// blocks of the instantiation resident at that shared size (occupancy API).
// Instantiation of basis_grid_kernel for a layer's degree (degrees 0..3 at compile time).
template<class F> void with_grid_degree(std::size_t degree, F&& f) {
    switch (degree) {
    case 0: f(std::integral_constant<int, 0>{}); return;
    case 1: f(std::integral_constant<int, 1>{}); return;
    case 2: f(std::integral_constant<int, 2>{}); return;
    case 3: f(std::integral_constant<int, 3>{}); return;
    default: f(std::integral_constant<int, -1>{});
    }
}
template<class T> void select_knot_stage(ExpansionPlan<T>& p, const ExecutorDevice& device, bool residual) {
    if (p.view.kind != detail::BasisKind::BSpline) return;
    const auto tile_rows = stage_rows<T>(p.block.terms, 2);
    const auto per_block = tile_rows ? tile_rows : stage_threads;
    const auto grid_inputs = p.knot_stride ? p.block.inputs : 1;
    const auto rows = static_cast<unsigned>(std::min<std::size_t>(grid_inputs, per_block));
    const auto pad = static_cast<unsigned>((p.block.terms+p.view.degree+1) | 1);
    const auto bytes = static_cast<std::size_t>(tile_rows)*p.block.terms*2*sizeof(T) + std::size_t{rows}*pad*sizeof(T);
    if (bytes > (std::size_t{48} << 10)) return;
    int resident = 0;
    cudaError_t query = cudaSuccess;
    with_grid_degree(p.view.degree, [&](auto degree) {
        constexpr int D = decltype(degree)::value;
        query = residual ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, basis_grid_kernel<true, D, T>, stage_threads, bytes)
                         : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, basis_grid_kernel<false, D, T>, stage_threads, bytes);
    });
    if (query != cudaSuccess || resident <= 0) { cudaGetLastError(); return; }
    p.knot_stage_rows = rows; p.knot_pad = pad;
    p.grid_blocks = static_cast<unsigned>(std::min<std::size_t>(std::size_t{device.multiprocessors}*static_cast<unsigned>(resident), 65535));
}
// Sparse per-input layers (backlog M6): spline_grid_basis_kernel stages its
// 32 knot rows when they fit the device's per-block opt-in shared memory with
// the kernel's static shared memory (cudaFuncGetAttributes; above the default
// 48 KiB the kernel's dynamic limit is raised); otherwise it reads global
// memory.
template<class T> void select_sparse_knot_stage(BasisPlan<T>& plan, const ExecutorDevice& device) {
    auto& p = plan.expansion;
    if (!p.knot_stride) return;
    const auto pad = static_cast<unsigned>(p.knot_stride | 1);
    const bool residual = p.block.residual_count != 0;
    with_spline_width(plan.sparse.width, [&](auto width) {
        constexpr unsigned P = decltype(width)::value;
        cudaFuncAttributes attributes{};
        const auto query = residual ? cudaFuncGetAttributes(&attributes, spline_grid_basis_kernel<P, true, T>)
                                    : cudaFuncGetAttributes(&attributes, spline_grid_basis_kernel<P, false, T>);
        if (query != cudaSuccess) { cudaGetLastError(); return; }
        const auto dynamic = std::size_t{32}*pad*sizeof(T);
        const auto limit = std::max(std::size_t{48} << 10, device.shared_optin);
        if (attributes.sharedSizeBytes + dynamic > limit) return;
        if (attributes.sharedSizeBytes + dynamic > (std::size_t{48} << 10)) {
            // The largest dynamic size any executor may need (a per-function attribute).
            const auto most = static_cast<int>(limit-attributes.sharedSizeBytes);
            const auto set = residual ? cudaFuncSetAttribute(spline_grid_basis_kernel<P, true, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, most)
                                      : cudaFuncSetAttribute(spline_grid_basis_kernel<P, false, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, most);
            if (set != cudaSuccess) { cudaGetLastError(); return; }
        }
        p.knot_pad = pad;
    });
}
// Selection of the sparse B-spline path (backlog C5), per layer at
// construction, so that the arena holds either the dense rows or the sparse
// ones and graphs, uploads and rollback see a fixed layout.
// Automatic selection (C5 evidence: GB10 sweep of complete training steps,
// forced dense against forced sparse, 64-, 256- and 1024-wide networks, K/(p+1)
// 1.5 to 24). A layer takes the sparse path when
// - terms/(degree+1) reaches the precision's ratio: FP64 on a device with a
//   reduced FP64 rate (fp64_ratio >= 8) 2, where both paths are FP64-ALU bound
//   at about the same rate per multiply-add; FP32 4 against SGEMM plus the
//   dense rows' memory traffic; TF32 5 against tensor-op GEMMs (backlog C13
//   re-measured both after its cheaper dense input VJP: C5's FP32 2 and TF32 4
//   now gave 0.81 and 0.94 of the dense step at 256 wide);
// - and, when the layer's upstream U (capacity*outputs values) exceeds half the
//   L2 cache, a higher ratio: FP32 6, TF32 20 (C13; C5's 16 gave 0.93), FP64 8
//   (FP64 not measured: extrapolated as four times its in-L2 ratio, like the
//   first FP32 rule): the coefficient VJP rereads U once per input, from L2 while it
//   fits and from DRAM otherwise (1024-wide steps: 4096*1024 FP32 = 16 MiB);
// - and its dense forward contraction has at least 2^22 (FP32), 2^23 (TF32) or
//   2^21 (FP64) multiply-adds per SM at capacity (2.0e8 / 4.0e8 / 1.0e8 on GB10's
//   48 SMs): below, the sparse step's extra launches and serial per-warp sample
//   walks cost more than the saved multiply-adds (64x64x32x16 at batch 1024).
// The ratios and floors are the smallest values at which no configuration of
// the C5 sweep ran slower with the automatic choice than with the dense rows;
// C13's re-measured values leave two boundary configurations within 3% of the
// dense step (FP32 64-wide K=48 0.97, TF32 degree 1 at ratio 5 0.98, where
// degree 3 gains 1.11; one ratio for all degrees, C13 evidence).
// Without a reduced FP64 rate FP64 uses C5's FP32 values (2, and 6 out of L2;
// C13 changed only the FP32/TF32 kernels).
double sparse_spline_ratio(bool fp64, bool tensor_ops, bool upstream_in_l2, const ExecutorDevice& d) {
    const bool slow_fp64 = fp64 && d.fp64_ratio >= 8;
    if (fp64) return slow_fp64 ? (upstream_in_l2 ? 2.0 : 8.0) : (upstream_in_l2 ? 2.0 : 6.0);
    if (upstream_in_l2) return tensor_ops ? 5.0 : 4.0;
    return tensor_ops ? 20.0 : 6.0;
}
std::size_t sparse_spline_work(bool fp64, bool tensor_ops, const ExecutorDevice& d) {
    const unsigned bits = fp64 && d.fp64_ratio >= 8 ? 21 : tensor_ops && !fp64 ? 23 : 22;
    return (std::size_t{1} << bits)*d.multiprocessors;
}
// Bytes of shared memory the sparse kernels need for a layer (forward with one
// input per tile; input VJP).
template<class T> std::size_t spline_forward_row_bytes(std::size_t terms) {
    return product(product(spline_forward_stride<T>, terms), sizeof(T));
}
template<class T> std::size_t spline_input_bytes(std::size_t terms, unsigned group = spline_dx_inputs) {
    return product(product(stage_threads+product(group, terms), spline_dx_stride<T>), sizeof(T));
}
// Shared-memory budget of a forward tile (C5 sweep: 24 KiB keeps four blocks
// per SM on GB10's 100 KiB; more inputs per tile did not help).
constexpr std::size_t spline_stage_bytes = std::size_t{24} << 10;
template<class T>
void select_sparse(BasisPlan<T>& plan, std::size_t capacity, bool tensor_ops, const ExecutorDevice& device, testing::SplinePath path) {
    const auto& b = plan.expansion.block;
    const auto degree = plan.expansion.view.degree;
    if (plan.expansion.view.kind != detail::BasisKind::BSpline || path == testing::SplinePath::Dense) return;
    // Kernel limits: instantiated widths, shared memory, 32-bit sort indices, grid dimensions.
    const auto width = degree+1;
    const auto row_bytes = spline_forward_row_bytes<T>(b.terms);
    // Inputs per input-VJP block: spline_dx_inputs, halved until the staged rows fit.
    unsigned group = spline_dx_inputs;
    while (group > 1 && spline_input_bytes<T>(b.terms, group) > device.shared_optin) group /= 2;
    if (width > max_sparse_spline_width || row_bytes > device.shared_optin || spline_input_bytes<T>(b.terms, group) > device.shared_optin ||
        product(capacity, b.inputs) > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        (b.outputs-1)/spline_outputs+1 > 65535 || (b.inputs-1)/group+1 > 65535 || (b.inputs-1)/32+1 > 65535)
        return;
    if (path == testing::SplinePath::Automatic) {
        // Only layers whose dense forward contraction runs on cuBLAS at
        // capacity (the latency-bound small kernels are kept) and whose work
        // and term ratio pass the rule above.
        constexpr bool fp64 = std::is_same_v<T, double>;
        const auto per_sample = product(product(b.inputs, b.terms), b.outputs);
        const bool large = capacity > small_forward_contraction<T>/per_sample &&
                           capacity >= (sparse_spline_work(fp64, tensor_ops, device)+per_sample-1)/per_sample;
        const bool upstream_in_l2 = product(product(capacity, b.outputs), sizeof(T)) <= device.l2_bytes/2;
        const auto ratio = sparse_spline_ratio(fp64, tensor_ops, upstream_in_l2, device);
        if (!large || static_cast<double>(b.terms) < ratio*static_cast<double>(width)) return;
    }
    auto& p = plan.sparse;
    p.active = true;
    p.width = static_cast<unsigned>(width);
    p.group = group;
    p.chunk = static_cast<unsigned>(std::clamp<std::size_t>(spline_stage_bytes/row_bytes, 1, b.inputs));
    // Layers with few outputs give few blocks: split their coefficient ranges
    // (and shrink the forward blocks) until about four blocks per SM, keeping
    // at least 2*width coefficients per range (the ranges overlap by width-1).
    // The coefficient VJP counts warps that own outputs (a warp walks its
    // samples serially, latency bound): about 16 per SM.
    const auto target = std::size_t{4}*device.multiprocessors;
    constexpr auto per_warp = std::size_t{32}*spline_parameter_outputs;
    const auto parameter_warps = ((b.outputs-1)/per_warp+1)*std::min<std::size_t>(b.inputs, 65535);
    const auto max_ranges = std::max<std::size_t>(1, b.terms/(2*width));
    p.ranges = static_cast<unsigned>(std::clamp<std::size_t>((4*target+parameter_warps-1)/parameter_warps, 1, std::min<std::size_t>(max_ranges, 64)));
    const auto output_blocks = (b.outputs-1)/spline_outputs+1, sample_blocks = capacity ? (capacity-1)/stage_threads+1 : 1;
    p.threads = stage_threads;
    while (p.threads > 64 && sample_blocks*(stage_threads/p.threads)*output_blocks < target) p.threads /= 2;
    p.key_bits = 1;
    while (p.key_bits < 32 && (std::size_t{1} << p.key_bits) <= b.terms-width) ++p.key_bits;
    const auto items = static_cast<int>(capacity*b.inputs), segments = static_cast<int>(b.inputs);
    check(cub::DeviceSegmentedRadixSort::SortPairs(nullptr, p.sort_bytes, static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr),
              static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr), items, segments,
              static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), 0, p.key_bits), "resident sparse sort storage");
}

// Trainable-RBF center/log-width reduction (backlog C12). A tile-kernel block
// holds 256 lanes of accumulators for `group` terms in shared memory: all
// terms when they fit the per-block opt-in limit, otherwise groups of a
// multiple of 128 bytes of terms (whole cache lines of a row when terms is a
// multiple of it). Per launch, FP32/TF32 layers whose rows fit the C1 stage
// keep nonlinear_staged_kernel (as fast or faster, its own summation order);
// the other layers take the tile kernel when its grid (term groups x tiles)
// has at least one block per SM, or reaches the most tiles on devices with
// more SMs than nonlinear_tiles (the C1 64-block staged kernel already won on
// GA102's 82 SMs), otherwise the column kernel: below that the
// few tile blocks serialize K times more work each (GB10: 2-5 us slower at
// 512-4096 rows), while the column blocks of a tile are then few enough to be
// resident together and to share the rows in L2.
template<class T>
void select_reduction(TrainableRbfPlan<T>& plan, const ExecutorDevice& device, testing::NonlinearReduction reduction) {
    const auto terms = plan.expansion.block.terms;
    plan.reduction = reduction;
    plan.tile_min_blocks = std::min(device.multiprocessors, nonlinear_tiles);
    const auto most = device.shared_optin/nonlinear_tile_bytes<T>(1);
    constexpr auto line = std::size_t{128}/sizeof(T);
    const auto group = terms <= most ? terms : most >= line ? most/line*line : most;
    if (group == 0 || terms == 0 || (terms-1)/group+1 > 2147483647U/nonlinear_tiles) return;
    plan.tile_group = static_cast<unsigned>(group);
}
template<class T>
bool tiled_reduction(const TrainableRbfPlan<T>& plan, unsigned tiles, unsigned staged_rows) {
    using testing::NonlinearReduction;
    if (plan.tile_group == 0 || plan.reduction == NonlinearReduction::Column) return false;
    if (plan.reduction == NonlinearReduction::Tile) return true;
    const auto groups = (plan.expansion.block.terms-1)/plan.tile_group+1;
    return staged_rows == 0 && groups*tiles >= plan.tile_min_blocks;
}

// Calls f(std::integral_constant<unsigned, S>) for every instantiated S <= Most
// (largest first).
template<unsigned Most, class F> void for_wavelet_samples(F&& f) {
    if constexpr (Most >= 8) f(std::integral_constant<unsigned, 8>{});
    if constexpr (Most >= 4) f(std::integral_constant<unsigned, 4>{});
    if constexpr (Most >= 2) f(std::integral_constant<unsigned, 2>{});
    f(std::integral_constant<unsigned, 1>{});
}
constexpr unsigned log2_samples(unsigned samples) { return samples >= 8 ? 3 : samples >= 4 ? 2 : samples >= 2 ? 1 : 0; }

// Cost model of a wavelet launch (backlog M7 review). It chooses the samples
// per thread S and the splits of every launch from the device's
// multiprocessors and each instantiation's resident blocks (CUDA runtime), so
// the choice adapts to the device; the constants below were fitted on GB10 to
// per-kernel times of 3080 forced geometries (M7 evidence, launch model).
// `groups` block groups (forward: outputs x input splits; backward: inputs x
// output splits) each run the batch's chunks of 256 S samples over `slots`
// blocks (forward: one per chunk), `iterations` wavelets per chunk. With
// n = min(blocks per multiprocessor, resident) blocks at a time, one
// iteration of a block costs max(n (S + overhead), latency): issue-bound when
// enough warps are resident, latency-bound otherwise (a partial chunk costs
// as much as a full one: its idle sample slots stay in the unrolled code).
// The busiest multiprocessor runs about max(1, per/n) + tail such blocks,
// each with a fixed cost per resident block. Units: one thread evaluation.
inline constexpr double wavelet_iteration_overhead = 3, wavelet_iteration_latency = 20, wavelet_wave_tail = 0.5,
                        wavelet_block_cost = 50, wavelet_split_launch = 100; // the extra reduction kernel of a split
inline double wavelet_launch_cost(std::size_t batch, unsigned samples, std::size_t groups, std::size_t slots, double iterations,
                                  unsigned multiprocessors, unsigned resident) {
    if (!batch || !groups) return 0;
    const auto chunks = wavelet_chunks(batch, samples);
    slots = std::max<std::size_t>(1, std::min(slots, chunks));
    const double blocks = static_cast<double>(groups)*static_cast<double>(slots), per = blocks/multiprocessors;
    const double n = std::min(std::ceil(per), static_cast<double>(std::max(resident, 1u)));
    const double block = static_cast<double>(chunks)/static_cast<double>(slots)*iterations*
                         std::max(n*(samples+wavelet_iteration_overhead), wavelet_iteration_latency) + wavelet_block_cost*n;
    return (std::max(1.0, per/n)+wavelet_wave_tail)*block;
}

// Forward launch for a batch: S, input splits and inputs per split. The
// choice minimizes the cost model over S and splits z; a split forward is
// considered only while outputs x chunks x z stays within one wave of its
// instantiation (resident blocks x multiprocessors), which bounds the partial
// sums (wavelet_forward_partials). Ties keep the larger S, then fewer splits.
struct WaveletForwardLaunch {
    unsigned samples = 1, splits = 1;
    std::size_t per = 1;
};
template<class T> std::size_t wavelet_most_input_splits(const WaveletPlan<T>& plan, std::size_t chunks, unsigned resident) {
    const auto& b = plan.block;
    const auto wave = static_cast<std::size_t>(plan.multiprocessors)*resident;
    if (b.inputs < 2 || !chunks || !b.outputs || b.outputs*chunks >= wave) return 1;
    return std::min<std::size_t>({b.inputs, wave/(b.outputs*chunks), 65535});
}
template<class T> WaveletForwardLaunch wavelet_forward_launch(const WaveletPlan<T>& plan, std::size_t batch) {
    const auto& b = plan.block;
    WaveletForwardLaunch best;
    best.per = std::max<std::size_t>(1, b.inputs);
    const auto& forced = plan.forced;
    double least = std::numeric_limits<double>::infinity();
    for_wavelet_samples<wavelet_forward_samples<T>>([&](auto S) {
        constexpr unsigned samples = decltype(S)::value;
        if (forced.forward_samples && forced.forward_samples != samples) return;
        const auto chunks = wavelet_chunks(batch, samples);
        const auto resident = plan.forward_resident[log2_samples(samples)];
        auto most = forced.input_splits ? forced.input_splits : wavelet_most_input_splits(plan, chunks, resident);
        if (batch*b.outputs) most = std::max<std::size_t>(1, std::min<std::size_t>(most, std::max<std::size_t>(1, plan.y_partial_values/(batch*b.outputs))));
        for (std::size_t z = forced.input_splits ? most : 1; z <= most; ++z) {
            const auto per = (b.inputs+z-1)/z, splits = b.inputs ? (b.inputs+per-1)/per : 1;
            if (splits != z && !forced.input_splits) continue; // the same partition as a smaller z
            const double cost = wavelet_launch_cost(batch, samples, b.outputs*splits, chunks, static_cast<double>(per*b.terms),
                                                    plan.multiprocessors, resident) +
                                (splits > 1 ? wavelet_split_launch : 0.0);
            if (cost < least) { least = cost; best = {samples, static_cast<unsigned>(splits), std::max<std::size_t>(1, per)}; }
        }
    });
    return best;
}
// Values of the forward partial sums any batch up to the capacity may need.
template<class T> std::size_t wavelet_forward_partials(const WaveletPlan<T>& plan, std::size_t capacity) {
    const auto& b = plan.block;
    if (plan.forced.input_splits) {
        const auto z = std::min<std::size_t>(plan.forced.input_splits, b.inputs);
        return z > 1 ? product(product(capacity, b.outputs), z) : 0;
    }
    std::size_t values = 0;
    for_wavelet_samples<wavelet_forward_samples<T>>([&](auto S) {
        constexpr unsigned samples = decltype(S)::value;
        const auto span = static_cast<std::size_t>(wavelet_threads)*samples;
        const auto resident = plan.forward_resident[log2_samples(samples)];
        for (std::size_t chunks = 1; chunks <= wavelet_chunks(capacity, samples); ++chunks) {
            const auto z = wavelet_most_input_splits(plan, chunks, resident);
            if (z < 2) break; // more chunks only lower the splits
            values = std::max(values, product(product(z, std::min(chunks*span, capacity)), b.outputs));
        }
    });
    return values;
}
// Backward launch for a batch: S, partial slots (at most plan.slots) and
// output splits (at most plan.splits) minimizing the cost model.
struct WaveletBackwardLaunch {
    unsigned samples = 1, slots = 1;
    std::size_t splits = 1, per = 1;
};
template<class T> WaveletBackwardLaunch wavelet_backward_launch(const WaveletPlan<T>& plan, std::size_t batch) {
    const auto& b = plan.block;
    const auto& forced = plan.forced;
    WaveletBackwardLaunch best;
    best.per = std::max<std::size_t>(1, b.outputs);
    double least = std::numeric_limits<double>::infinity();
    for_wavelet_samples<wavelet_backward_samples<T>>([&](auto S) {
        constexpr unsigned samples = decltype(S)::value;
        if (forced.backward_samples && forced.backward_samples != samples) return;
        const auto chunks = wavelet_chunks(batch, samples);
        const auto slots = static_cast<unsigned>(std::max<std::size_t>(1, std::min<std::size_t>(chunks, plan.slots)));
        const auto resident = plan.backward_resident[log2_samples(samples)];
        const std::size_t most = forced.splits ? forced.splits : std::max<std::size_t>(1, std::min<std::size_t>(plan.splits, b.outputs));
        for (std::size_t z = forced.splits ? most : 1; z <= most; ++z) {
            const auto per = std::max<std::size_t>(1, (b.outputs+z-1)/z), splits = b.outputs ? (b.outputs+per-1)/per : 1;
            if (splits != z && !forced.splits) continue;
            const double cost = wavelet_launch_cost(batch, samples, b.inputs*splits, slots, static_cast<double>(per*b.terms),
                                                    plan.multiprocessors, resident) +
                                (splits > 1 ? wavelet_split_launch : 0.0);
            if (cost < least) { least = cost; best = {samples, slots, splits, per}; }
        }
    });
    return best;
}

// Launch geometry of a wavelet layer (backlog M7), at construction: `wave`
// = multiprocessors x resident backward blocks per multiprocessor (occupancy
// API); `slots` = the chunks of a full batch at 4 samples per thread (FP64:
// 2), at least enough for inputs*slots >= wave, so a block runs about one
// chunk, but partial planes of at most the layer's xt + ut workspace; `splits`
// the most output splits the cost model may choose: dx partials of at most
// the same workspace, at most min(outputs, 16). Every instantiation's
// residency and the forward partials' bound are recorded for the launches.
template<class T> void select_wavelet_launch(WaveletPlan<T>& plan, std::size_t capacity, const ExecutorDevice& device,
                                             testing::WaveletLaunch forced) {
    const auto supported = [](unsigned v, unsigned most) { return v == 0 || ((v & (v-1)) == 0 && v <= most); };
    if (!supported(forced.forward_samples, wavelet_forward_samples<T>) || !supported(forced.backward_samples, wavelet_backward_samples<T>))
        throw std::invalid_argument("resident wavelet test launch: unsupported samples per thread");
    if (forced.slots > 65535 || forced.splits > 65535 || forced.input_splits > 65535)
        throw std::invalid_argument("resident wavelet test launch: slots or splits beyond the grid limit");
    // Occupancy of the instantiation the full-batch backward launches (largest S).
    int resident = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, wavelet_backward_kernel<wavelet_backward_samples<T>, true, T>,
                                                      wavelet_threads, 0) != cudaSuccess) {
        cudaGetLastError(); // clear: a stale error would be reported by a later launch check
        resident = 1;
    }
    const auto wave = std::max<std::size_t>(1, static_cast<std::size_t>(device.multiprocessors)*static_cast<std::size_t>(std::max(resident, 1)));
    const auto& b = plan.block;
    const auto most = std::max<std::size_t>(1, wavelet_chunks(capacity, 1));
    const auto budget = std::max<std::size_t>(1, product(capacity, b.inputs+b.outputs)/std::max<std::size_t>(1, product(plan.count, 3)));
    auto slots = std::max(wavelet_chunks(capacity, std::is_same_v<T, double> ? 2 : 4), (wave-1)/b.inputs+1);
    slots = std::max<std::size_t>(1, std::min({slots, most, budget, std::size_t{65535}}));
    const auto splits = std::max<std::size_t>(1, std::min<std::size_t>({(b.inputs+b.outputs)/std::max<std::size_t>(1, b.inputs), b.outputs, 16}));
    plan.slots = forced.slots ? forced.slots : static_cast<unsigned>(slots);
    plan.splits = forced.splits ? forced.splits : static_cast<unsigned>(splits);
    plan.forced = forced;
    plan.multiprocessors = std::max(1u, device.multiprocessors);
    const auto occupancy = [](auto kernel) {
        int blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, wavelet_threads, 0) != cudaSuccess) { cudaGetLastError(); blocks = 1; }
        return static_cast<unsigned>(std::max(blocks, 1));
    };
    for_wavelet_samples<wavelet_forward_samples<T>>([&](auto S) {
        plan.forward_resident[log2_samples(S)] = occupancy(wavelet_forward_kernel<decltype(S)::value, T>);
    });
    for_wavelet_samples<wavelet_backward_samples<T>>([&](auto S) {
        plan.backward_resident[log2_samples(S)] = occupancy(wavelet_backward_kernel<decltype(S)::value, true, T>);
    });
    plan.y_partial_values = wavelet_forward_partials(plan, capacity);
}

// Parameter upload at construction (coefficients and bias are uploaded by the executor).
template<class T> void upload_carrier(Context<T>& s, const BasisPlan<T>& plan, const BasisEdges& edges) {
    const auto& p = plan.expansion;
    const auto terms = p.block.terms;
    std::visit([&](const auto& c) {
        using C = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<C, GaussianRbfConfig> || std::is_same_v<C, MexicanHatConfig>) s.upload(s.ptr(p.centers), {c.centers.data(), terms});
        if constexpr (std::is_same_v<C, MexicanHatConfig>) s.upload(s.ptr(p.scales), {c.scales.data(), terms});
        if constexpr (std::is_same_v<C, BSplineConfig>) {
            s.upload(s.ptr(p.knots), {c.knots.data(), knot_count(p)});
            if constexpr (spline_difference_vjp<T>) {
                if (c.degree != 0) {
                    // Knot scales in T from the rounded knots, as the device would compute
                    // them; grid by grid (backlog M6), each grid's knots t[g*n ...].
                    const auto n = terms+c.degree+1;
                    std::vector<double> scales(product(p.grids, terms-1));
                    for (std::size_t g = 0; g < p.grids; ++g) {
                        const double* t = c.knots.data()+g*n;
                        for (std::size_t j = 1; j < terms; ++j)
                            scales[g*(terms-1)+j-1] = detail::spline_derivative_scale(c.degree, narrow<T>(t[j+c.degree]), narrow<T>(t[j]));
                    }
                    s.upload(s.ptr(p.knot_scales), scales);
                }
            }
        }
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
template<class T> void upload_carrier(Context<T>& s, const WaveletPlan<T>& plan, const WaveletEdges& edges) {
    s.upload(s.ptr(s.parameters+plan.block.nonlinear()), edges.translations);
    s.upload(s.ptr(s.parameters+plan.block.nonlinear()+plan.count), edges.log_scales);
}

// Construction upload of one network layer: a KAN layer's coefficients, bias
// and carrier state, or an input map's fixed and trainable parameters.
template<class T, class P, class Edges>
void upload_stage(Context<T>& s, const P& plan, const Layer& layer, const Edges& edges) {
    const auto& b = block_of(plan);
    s.upload(s.ptr(s.parameters+b.offset), layer.coefficients());
    s.upload(s.ptr(s.parameters+b.bias()), layer.bias());
    upload_carrier(s, plan, edges);
    if (b.residual_count) s.upload(s.ptr(s.parameters+b.residual()), layer.residual()->weights);
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

// SiLU residual branch (backlog M3): the layer's S and silu' rows, and
// Y += S*W^T without a check (the caller's bias_kernel or check_kernel
// reports): residual_dot_kernel where a 32-sample S tile fits the stage and
// batch*outputs*inputs is at most the carrier's small_forward_contraction,
// cuBLAS (beta = 1) above.
template<class T> SiluRows<T> silu_rows(Context<T>& s, const ParameterBlock& b) {
    return {s.ptr(b.silu), s.ptr(b.silu_derivative)};
}
template<class T> bool small_residual_work(const ParameterBlock& b, std::size_t batch) {
    return !batch || b.outputs*b.inputs <= small_forward_contraction<T>/batch;
}
template<class T> bool small_residual(const ParameterBlock& b, std::size_t batch) {
    return b.inputs <= residual_tile_inputs<T> && small_residual_work<T>(b, batch);
}
// Returns whether the contraction checked Y itself (residual_dot_kernel).
template<class T> bool residual_contract(Context<T>& s, const ParameterBlock& b, std::size_t j) {
    const T* w = s.ptr(s.parameters+b.residual());
    T* y = s.ptr(s.activation[j+1]);
    if (small_residual<T>(b, s.batch)) {
        constexpr unsigned warps = stage_threads/32;
        const auto groups = static_cast<unsigned>(std::min<std::size_t>((b.outputs-1)/warps+1, 65535));
        residual_dot_kernel<<<dim3(blocks(s.batch, residual_tile), groups), stage_threads,
                              b.inputs*(residual_tile+1+warps)*sizeof(T), s.stream>>>(s.ptr(b.silu), w, y, s.batch, b.inputs,
                                                                                      b.outputs, s.status);
        check(cudaGetLastError(), "resident residual forward launch");
        return true;
    }
    // Row-major Y (batch x O) += S (batch x I) * W^T: column-major
    // Y^T = W(op T, W stored as column-major I x O) * S^T.
    const auto i = static_cast<std::int64_t>(b.inputs), o = static_cast<std::int64_t>(b.outputs);
    const T one = 1;
    s.with_residual_workspace([&] {
        check(gemm(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, o, static_cast<std::int64_t>(s.batch), i, &one, w, i, s.ptr(b.silu), i,
                   &one, y, o), "resident residual forward contraction");
    });
    return false;
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
    // With the residual branch (backlog M3) the same pass writes S and silu'.
    const bool residual = b.residual_count != 0;
    const auto silu = residual ? silu_rows(s, b) : SiluRows<T>{nullptr, nullptr};
    if (p.knot_stride || p.knot_stage_rows) {
        // Per-input spline grids (backlog M6), and shared grids as one grid of every input.
        const KnotRows knot_rows{p.knot_stride ? b.inputs : 1, p.knot_stride};
        const auto launch = [&](auto with_residual) {
            const auto per_block = tile_rows ? tile_rows : stage_threads;
            auto grid = blocks(count, per_block);
            unsigned staged = 0;
            if (p.knot_stage_rows) {
                // One iteration per block, or a loop step that is a multiple of the inputs.
                const auto period = knot_rows.inputs/std::gcd(knot_rows.inputs, std::size_t{per_block});
                const auto needed = (count-1)/per_block+1;
                if (needed <= p.grid_blocks) { grid = static_cast<unsigned>(needed); staged = p.knot_stage_rows; }
                else if (period <= p.grid_blocks) { grid = static_cast<unsigned>(p.grid_blocks/period*period); staged = p.knot_stage_rows; }
            }
            const auto bytes = shared + (staged ? std::size_t{staged}*p.knot_pad*sizeof(T) : 0);
            with_grid_degree(p.view.degree, [&](auto degree) {
                basis_grid_kernel<decltype(with_residual)::value, decltype(degree)::value><<<grid, stage_threads, bytes, s.stream>>>(
                    s.ptr(s.activation[j]), s.ptr(p.values), p.stored_derivatives ? s.ptr(p.derivatives) : nullptr, count, basis,
                    tile_rows, silu, s.status, knot_rows, staged, p.knot_pad);
            });
        };
        if (residual) launch(std::true_type{}); else launch(std::false_type{});
    } else detail::visit_basis_family(basis.kind, [&](auto family) {
        const auto launch = [&](auto with_residual) {
            basis_kernel<decltype(family)::value, decltype(with_residual)::value>
                <<<blocks(count, tile_rows ? tile_rows : stage_threads), stage_threads, shared, s.stream>>>(
                s.ptr(s.activation[j]), s.ptr(p.values), p.stored_derivatives ? s.ptr(p.derivatives) : nullptr, log_derivatives,
                count, basis, tile_rows, silu, s.status);
        };
        if (residual) launch(std::true_type{}); else launch(std::false_type{});
    });
    check(cudaGetLastError(), "resident basis launch");
    const auto outputs = s.batch*b.outputs, length = b.inputs*b.terms;
    if (outputs <= small_forward_contraction<T>/length) {
        // The residual sum is reduced by the same warps (backlog M3).
        const auto launch = [&](auto with_residual) {
            forward_dot_kernel<decltype(with_residual)::value><<<blocks(outputs, 256/32), 256, 0, s.stream>>>(
                s.ptr(p.values), s.ptr(s.parameters+b.offset), s.ptr(s.parameters+b.bias()), s.ptr(s.activation[j+1]),
                outputs, b.outputs, length, silu.values, residual ? s.ptr(s.parameters+b.residual()) : nullptr, b.inputs,
                s.status);
        };
        if (residual) launch(std::true_type{}); else launch(std::false_type{});
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
    // Y += S*W^T before the bias, so that bias_kernel checks the sum (backlog M3).
    if (residual) residual_contract(s, b, j);
    bias_kernel<<<blocks(s.batch*b.outputs), 256, 0, s.stream>>>(s.ptr(s.activation[j+1]), s.ptr(s.parameters+b.bias()),
        s.batch*b.outputs, b.outputs, s.status);
    check(cudaGetLastError(), "resident forward bias launch");
}
// Derivative coefficients C* of a B-spline layer of degree >= 1 into the
// shared differences region (backlog C13), rows of `span` (terms-1 or terms).
template<class T> void spline_differences(Context<T>& s, const ExpansionPlan<T>& p, std::size_t span) {
    const auto& b = p.block;
    const auto edges = product(b.outputs, b.inputs), count = product(edges, span);
    const T* c = s.ptr(s.parameters+b.offset);
    // The kernel also forms coefficient indices up to edges*terms, and its
    // grid-stride index must not wrap past count.
    const bool narrow_index = product(edges, b.terms) <= std::numeric_limits<unsigned>::max()-std::size_t{blocks(count)}*256;
    if (p.knot_stride) { // per-input knot scales (backlog M6)
        if (narrow_index)
            spline_grid_difference_kernel<unsigned><<<blocks(count), 256, 0, s.stream>>>(c, s.ptr(p.knot_scales), s.ptr(s.differences),
                static_cast<unsigned>(count), static_cast<unsigned>(span), static_cast<unsigned>(b.terms), static_cast<unsigned>(b.inputs));
        else
            spline_grid_difference_kernel<std::size_t><<<blocks(count), 256, 0, s.stream>>>(c, s.ptr(p.knot_scales), s.ptr(s.differences),
                count, span, b.terms, b.inputs);
    } else if (narrow_index)
        spline_difference_kernel<unsigned><<<blocks(count), 256, 0, s.stream>>>(c, s.ptr(p.knot_scales), s.ptr(s.differences),
            static_cast<unsigned>(count), static_cast<unsigned>(span), static_cast<unsigned>(b.terms));
    else
        spline_difference_kernel<std::size_t><<<blocks(count), 256, 0, s.stream>>>(c, s.ptr(p.knot_scales), s.ptr(s.differences),
            count, span, b.terms);
    check(cudaGetLastError(), "resident spline difference launch");
}
// Sparse B-spline path (backlog C5).
template<class T> SparseRows<T> sparse_rows(Context<T>& s, const SparseSplinePlan& p) {
    return {reinterpret_cast<unsigned*>(s.ptr(p.first)), s.ptr(p.values), s.ptr(p.derivatives), s.capacity};
}
template<class F> void with_spline_width(unsigned width, F&& f) {
    switch (width) {
    case 1: f(std::integral_constant<unsigned, 1>{}); return;
    case 2: f(std::integral_constant<unsigned, 2>{}); return;
    case 3: f(std::integral_constant<unsigned, 3>{}); return;
    case 4: f(std::integral_constant<unsigned, 4>{}); return;
    }
    throw std::logic_error("unsupported sparse spline width");
}
// Sparse rows (with S and silu' for the residual branch), then the gather
// contraction; with the residual branch Y += S*W^T before bias_kernel adds
// the bias and checks, as on the dense cuBLAS path.
template<class T> void sparse_forward(Context<T>& s, const BasisPlan<T>& plan, std::size_t j) {
    if (!s.batch) return;
    const auto& p = plan.sparse;
    const auto& b = plan.expansion.block;
    const auto rows = sparse_rows(s, p);
    const bool residual = b.residual_count != 0;
    const auto silu = residual ? silu_rows(s, b) : SiluRows<T>{nullptr, nullptr};
    with_spline_width(p.width, [&](auto width) {
        constexpr unsigned P = decltype(width)::value;
        const dim3 tiles(static_cast<unsigned>((b.inputs-1)/32+1),
                         static_cast<unsigned>(std::min<std::size_t>((s.batch-1)/spline_tile_samples<T>+1, 65535)));
        const auto launch = [&](auto with_residual) {
            if (plan.expansion.knot_stride) // per-input grids (backlog M6)
                spline_grid_basis_kernel<P, decltype(with_residual)::value><<<tiles, stage_threads,
                                                                             std::size_t{32}*plan.expansion.knot_pad*sizeof(T), s.stream>>>(
                    s.ptr(s.activation[j]), rows, s.batch, b.inputs, s.ptr(plan.expansion.knots), b.terms, silu, s.status,
                    plan.expansion.knot_stride, plan.expansion.knot_pad);
            else
                spline_basis_kernel<P, decltype(with_residual)::value><<<tiles, stage_threads, 0, s.stream>>>(
                    s.ptr(s.activation[j]), rows, s.batch, b.inputs, s.ptr(plan.expansion.knots), b.terms, silu, s.status);
        };
        if (residual) launch(std::true_type{}); else launch(std::false_type{});
        check(cudaGetLastError(), "resident sparse basis launch");
        const auto shared = product(spline_forward_row_bytes<T>(b.terms), p.chunk);
        const dim3 grid(static_cast<unsigned>((s.batch-1)/p.threads+1), static_cast<unsigned>((b.outputs-1)/spline_outputs+1));
        spline_forward_kernel<P><<<grid, p.threads, shared, s.stream>>>(rows.first, rows.values, s.capacity,
            s.ptr(s.parameters+b.offset), residual ? nullptr : s.ptr(s.parameters+b.bias()), s.ptr(s.activation[j+1]),
            s.batch, b.inputs, b.outputs, b.terms, p.chunk, s.status);
        check(cudaGetLastError(), "resident sparse forward launch");
    });
    if (residual) {
        residual_contract(s, b, j);
        bias_kernel<<<blocks(s.batch*b.outputs), 256, 0, s.stream>>>(s.ptr(s.activation[j+1]), s.ptr(s.parameters+b.bias()),
            s.batch*b.outputs, b.outputs, s.status);
        check(cudaGetLastError(), "resident forward bias launch");
    }
}
// Per-input sort of the samples by first term, then the coefficient and bias
// VJPs (+ lambda*C) and, when needed, the input VJP.
template<class T> void sparse_backward(Context<T>& s, const BasisPlan<T>& plan, std::size_t j, T lambda, bool with_dx) {
    const auto& p = plan.sparse;
    const auto& b = plan.expansion.block;
    const auto rows = sparse_rows(s, p);
    auto* sorted_first = reinterpret_cast<unsigned*>(s.ptr(p.sorted_first));
    auto* order = reinterpret_cast<unsigned*>(s.ptr(p.order));
    auto* offsets = reinterpret_cast<int*>(s.ptr(p.offsets));
    const T* c = s.ptr(s.parameters+b.offset);
    const T* u = s.ptr(s.upstream[j+1]);
    if (s.batch) {
        spline_segments_kernel<<<blocks(b.inputs), 256, 0, s.stream>>>(offsets, b.inputs, s.capacity, s.batch);
        check(cudaGetLastError(), "resident sparse segment launch");
        auto bytes = p.sort_bytes;
        check(cub::DeviceSegmentedRadixSort::SortPairs(s.ptr(p.sort_storage), bytes, rows.first, sorted_first,
                  reinterpret_cast<const unsigned*>(s.ptr(p.indices)), order, static_cast<int>(b.inputs*s.capacity),
                  static_cast<int>(b.inputs), offsets, offsets+b.inputs, 0, p.key_bits, s.stream), "resident sparse sort");
    }
    with_spline_width(p.width, [&](auto width) {
        constexpr unsigned P = decltype(width)::value;
        // Only warps that own outputs (layers with few outputs).
        constexpr auto per_warp = std::size_t{32}*spline_parameter_outputs;
        const auto warps = static_cast<unsigned>(std::min<std::size_t>(spline_parameter_warps, (b.outputs-1)/per_warp+1));
        const dim3 grid(static_cast<unsigned>((b.outputs-1)/(per_warp*warps)+1), static_cast<unsigned>(std::min<std::size_t>(b.inputs, 65535)), p.ranges);
        spline_parameter_kernel<P><<<grid, 32*warps, 0, s.stream>>>(sorted_first, order, rows.values, s.capacity,
            u, c, s.ptr(s.gradients+b.offset), s.ptr(s.gradients+b.bias()), lambda, s.batch, b.inputs, b.outputs, b.terms, p.ranges, s.status);
        check(cudaGetLastError(), "resident sparse parameter launch");
        if (!with_dx || !s.batch) return;
        const dim3 inputs(static_cast<unsigned>((s.batch-1)/stage_threads+1), static_cast<unsigned>((b.inputs-1)/p.group+1));
        if constexpr (spline_difference_vjp<T>) {
            // Backlog C13: P-1 terms over C* (rows of terms, the last zero); degree zero has dx = 0.
            constexpr unsigned D = sparse_derivative_rows<T, P>;
            if constexpr (D == 0) {
                fill_kernel<<<blocks(s.batch*b.inputs), 256, 0, s.stream>>>(s.ptr(s.upstream[j]), s.batch*b.inputs, T(0));
            } else {
                spline_differences(s, plan.expansion, b.terms);
                spline_input_kernel<D><<<inputs, stage_threads, spline_input_bytes<T>(b.terms, p.group), s.stream>>>(rows.first,
                    rows.derivatives, s.capacity, s.ptr(s.differences), u, s.ptr(s.upstream[j]), s.batch, b.inputs, b.outputs, b.terms,
                    p.group, s.status);
            }
        } else {
            spline_input_kernel<P><<<inputs, stage_threads, spline_input_bytes<T>(b.terms, p.group), s.stream>>>(rows.first, rows.derivatives,
                s.capacity, c, u, s.ptr(s.upstream[j]), s.batch, b.inputs, b.outputs, b.terms, p.group, s.status);
        }
        check(cudaGetLastError(), "resident sparse input gradient launch");
    });
}
template<class T> void run_forward(Context<T>& s, const BasisPlan<T>& plan, std::size_t j) {
    if (plan.sparse.active) { sparse_forward(s, plan, j); return; }
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
    rational_argument_kernel<<<blocks(s.batch*b.inputs),256,0,s.stream>>>(s.ptr(s.activation[j]), s.ptr(plan.arguments),
        s.batch, s.capacity, b.inputs, plan.scalars, s.status);
    check(cudaGetLastError(), "resident rational argument launch");
    detail::visit_denominator_policy(plan.config.denominator_policy, [&](auto policy) {
        rational_forward_kernel<decltype(policy)::value><<<blocks(s.batch*b.outputs),256,0,s.stream>>>(
            s.ptr(plan.arguments),s.ptr(s.parameters+b.offset),
            s.ptr(s.parameters+b.nonlinear()),s.ptr(s.parameters+b.bias()),
            s.ptr(plan.values),s.ptr(plan.denominator_values),s.ptr(plan.derivatives),gains,s.ptr(s.activation[j+1]),
            s.batch,s.capacity,b.inputs,b.outputs,plan.scalars,s.status);
    });
    check(cudaGetLastError(),"resident rational forward launch");
}
// Per-edge wavelets (backlog M7, kernels above). Samples per thread S and
// the splits of every launch come from the cost model (wavelet_forward_launch,
// wavelet_backward_launch): FP64 instantiates forward S = 1 (FP64-ALU bound:
// occupancy beats amortization) and backward S <= 4 (8 doubles the registers).
// Instantiates only S <= Most (FP64: forward 1, backward 4).
template<unsigned Most, class F> void with_wavelet_samples(unsigned samples, F&& f) {
    if constexpr (Most >= 8) if (samples == 8) { f(std::integral_constant<unsigned, 8>{}); return; }
    if constexpr (Most >= 4) if (samples == 4) { f(std::integral_constant<unsigned, 4>{}); return; }
    if constexpr (Most >= 2) if (samples == 2) { f(std::integral_constant<unsigned, 2>{}); return; }
    f(std::integral_constant<unsigned, 1>{});
}
// rows x columns (row-major, at in) -> columns x pitch at out.
template<class T> void transpose(Context<T>& s, const T* in, T* out, std::size_t rows, std::size_t columns, std::size_t pitch) {
    if (!rows || !columns) return;
    const auto x = (columns+31)/32, y = std::min<std::size_t>((rows+31)/32, 65535);
    if (x > 2147483647U) throw std::overflow_error("resident transpose launch size overflow");
    transpose_kernel<<<dim3(static_cast<unsigned>(x), static_cast<unsigned>(y)), dim3(32, 8), 0, s.stream>>>(in, out, rows, columns, pitch);
    check(cudaGetLastError(), "resident transpose launch");
}
template<class T> WaveletArgs<T> wavelet_args(const Context<T>& s, const WaveletPlan<T>& plan) {
    const auto& b = plan.block;
    return {s.ptr(plan.xt), s.ptr(s.parameters+b.offset), s.ptr(s.parameters+b.nonlinear()), s.ptr(plan.inverse),
            s.ptr(plan.amplitude), s.batch, s.capacity, b.inputs, b.outputs, b.terms};
}
template<class T> void run_forward(Context<T>& s, const WaveletPlan<T>& plan, std::size_t j) {
    const auto& b = plan.block;
    wavelet_scales_kernel<<<blocks(plan.count), 256, 0, s.stream>>>(s.ptr(s.parameters+b.nonlinear()+plan.count), s.ptr(plan.inverse),
                                                                    s.ptr(plan.amplitude), plan.count);
    check(cudaGetLastError(), "resident wavelet scale launch");
    transpose(s, s.ptr(s.activation[j]), s.ptr(plan.xt), s.batch, b.inputs, s.capacity);
    const auto launch = wavelet_forward_launch(plan, s.batch);
    if (b.outputs > 2147483647U) throw std::overflow_error("resident wavelet launch size overflow");
    const auto chunks = static_cast<unsigned>(std::min<std::size_t>(wavelet_chunks(s.batch, launch.samples), 65535));
    T* y = s.ptr(s.activation[j+1]);
    T* partial = launch.splits > 1 ? s.ptr(plan.y_partials) : y;
    with_wavelet_samples<wavelet_forward_samples<T>>(launch.samples, [&](auto S) {
        wavelet_forward_kernel<decltype(S)::value><<<dim3(static_cast<unsigned>(b.outputs), chunks, launch.splits), wavelet_threads, 0, s.stream>>>(
            wavelet_args(s, plan), s.ptr(s.parameters+b.bias()), y, s.status, launch.per, partial);
    });
    check(cudaGetLastError(), "resident wavelet forward launch");
    if (launch.splits > 1) {
        wavelet_forward_finish_kernel<<<blocks(s.batch*b.outputs), 256, 0, s.stream>>>(partial, launch.splits, s.ptr(s.parameters+b.bias()), y,
                                                                                     s.batch, b.outputs, s.status);
        check(cudaGetLastError(), "resident wavelet forward reduction launch");
    }
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
    // Backlog C13: FP32/TF32 B-spline layers contract the derivative
    // coefficients instead (W* = U*C*, rows of terms-1; nothing for degree zero).
    bool differences = false;
    if constexpr (spline_difference_vjp<T>) differences = p.view.kind == detail::BasisKind::BSpline;
    if (s.batch && with_w) {
        if (!differences) {
            // Input VJP: column-major W^T (IK x batch) = C^T * U^T, then dx = sum_k Phi' * W.
            // Jacobi layers (backlog C13) run this GEMM in FP32 math under TF32: the
            // TF32 rounding of C, amplified by the derivatives' growth toward the
            // endpoints, exceeded the TF32 tolerance for fitted coefficients.
            const auto contract = [&] {
                check(gemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, ik, n, o, &one, c, ik, u, o, &zero, s.ptr(s.scratch), ik),
                      "resident input VJP contraction");
            };
            if (!std::is_same_v<T, double> && p.view.kind == detail::BasisKind::Jacobi) s.with_fp32_math(contract);
            else contract();
        } else if constexpr (spline_difference_vjp<T>) {
            if (p.view.degree != 0) {
                spline_differences(s, p, b.terms-1);
                const auto ik1 = static_cast<std::int64_t>(b.inputs*(b.terms-1));
                check(gemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, ik1, n, o, &one, s.ptr(s.differences), ik1, u, o, &zero,
                           s.ptr(s.scratch), ik1), "resident input VJP contraction");
            }
        }
    }
    const auto rows = with_dx ? s.batch*b.inputs : 0;
    const auto finish = [&](auto source) {
        constexpr auto planes = stage_planes<decltype(source)>;
        const auto tile_rows = difference_rows<decltype(source)> ? 0u : stage_rows<T>(b.terms, planes);
        const auto shared = static_cast<std::size_t>(tile_rows)*b.terms*planes*sizeof(T);
        const auto grid = std::max(rows ? blocks(rows, tile_rows ? tile_rows : stage_threads) : 1u, blocks(checked, stage_threads));
        backward_finish_kernel<<<grid, stage_threads, shared, s.stream>>>(source, s.ptr(s.scratch),
            s.ptr(s.upstream[j]), rows, b.terms, dc, checked, small ? s.ptr(s.partials) : nullptr, tiles, c,
            b.coefficients, lambda, tile_rows, s.status);
    };
    if (differences) {
        if constexpr (spline_difference_vjp<T>) {
            const auto launch = [&](auto degree) {
                if (p.knot_stride) // per-input grids (backlog M6)
                    finish(GridSplineDifferences<T, decltype(degree)::value>{s.ptr(s.activation[j]), s.ptr(p.knots), b.terms, p.view.degree,
                                                                              KnotRows{b.inputs, p.knot_stride}});
                else
                    finish(SplineDifferences<T, decltype(degree)::value>{s.ptr(s.activation[j]), s.ptr(p.knots), b.terms, p.view.degree});
            };
            switch (p.view.degree) {
            case 1: launch(std::integral_constant<unsigned, 1>{}); break;
            case 2: launch(std::integral_constant<unsigned, 2>{}); break;
            case 3: launch(std::integral_constant<unsigned, 3>{}); break;
            default: launch(std::integral_constant<unsigned, 0>{});
            }
        }
    } else if (p.stored_derivatives) {
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
    if (plan.sparse.active) { sparse_backward(s, plan, j, lambda, dx); return; }
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
    const auto tile_rows = b.terms <= stage_threads && plan.reduction == testing::NonlinearReduction::Automatic ? stage_rows<T>(b.terms, 3) : 0;
    if (rows && tiled_reduction(plan, tiles, tile_rows)) {
        const auto group = plan.tile_group;
        nonlinear_tile_kernel<<<static_cast<unsigned>((b.terms-1)/group+1)*tiles, stage_threads, nonlinear_tile_bytes<T>(group), s.stream>>>(
            s.ptr(p.derivatives), s.ptr(plan.log_derivatives), s.ptr(s.scratch), s.ptr(plan.partials), rows, b.terms, tiles, group, s.status);
    } else if (rows && tile_rows) {
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
        rational_parameter_kernel<decltype(policy)::value><<<blocks(b.inputs*b.outputs,8),256,0,s.stream>>>(
            s.ptr(plan.arguments),s.ptr(plan.values),s.ptr(plan.denominator_values),gains,
            s.ptr(s.upstream[j+1]),s.ptr(s.parameters+b.offset),s.ptr(s.gradients+b.offset),s.batch,s.capacity,b.inputs,b.outputs,
            plan.scalars,lambda,s.status);
    });
    check(cudaGetLastError(),"resident rational parameter gradient launch");
}
// Backward of per-edge wavelets: S, slots (at most plan.slots) and output
// splits over grid z (dx then reduced by the input finish) from the cost
// model (wavelet_backward_launch).
template<class T> void run_backward(Context<T>& s, const WaveletPlan<T>& plan, std::size_t j, T lambda) {
    const auto& b = plan.block;
    const bool dx = s.batch && s.input_gradient(j);
    transpose(s, s.ptr(s.upstream[j+1]), s.ptr(plan.ut), s.batch, b.outputs, s.capacity);
    wavelet_bias_kernel<<<blocks(b.outputs, 256/32), 256, 0, s.stream>>>(s.ptr(plan.ut), s.ptr(s.gradients+b.bias()), s.batch,
                                                                        s.capacity, b.outputs, s.status);
    check(cudaGetLastError(), "resident wavelet bias launch");
    unsigned slots = 0;
    if (s.batch) {
        const auto launch = wavelet_backward_launch(plan, s.batch);
        const unsigned samples = launch.samples;
        const std::size_t per = launch.per, splits = launch.splits;
        slots = launch.slots;
        if (b.inputs > 2147483647U) throw std::overflow_error("resident wavelet launch size overflow");
        T* dx_out = splits > 1 ? s.ptr(plan.dx_partials) : s.ptr(s.upstream[j]);
        const auto grid = dim3(static_cast<unsigned>(b.inputs), slots, static_cast<unsigned>(splits));
        with_wavelet_samples<wavelet_backward_samples<T>>(samples, [&](auto S) {
            constexpr unsigned samples_per_thread = decltype(S)::value;
            if (dx)
                wavelet_backward_kernel<samples_per_thread, true><<<grid, wavelet_threads, 0, s.stream>>>(
                    wavelet_args(s, plan), s.ptr(plan.ut), s.ptr(plan.partials), per, dx_out, s.capacity*b.inputs, s.status);
            else
                wavelet_backward_kernel<samples_per_thread, false><<<grid, wavelet_threads, 0, s.stream>>>(
                    wavelet_args(s, plan), s.ptr(plan.ut), s.ptr(plan.partials), per, dx_out, s.capacity*b.inputs, s.status);
        });
        check(cudaGetLastError(), "resident wavelet backward launch");
        if (dx && splits > 1) {
            const auto rows = s.batch*b.inputs;
            wavelet_input_finish_kernel<<<blocks(rows), 256, 0, s.stream>>>(s.ptr(plan.dx_partials), static_cast<unsigned>(splits),
                                                                            s.capacity*b.inputs, s.ptr(s.upstream[j]), rows, s.status);
            check(cudaGetLastError(), "resident wavelet input reduction launch");
        }
    }
    if (!plan.count) return;
    T* g = s.ptr(s.gradients+b.offset);
    wavelet_finish_kernel<<<blocks(plan.count), 256, 0, s.stream>>>(s.ptr(plan.partials), slots, s.ptr(s.parameters+b.offset), lambda,
        g, s.ptr(s.gradients+b.nonlinear()), s.ptr(s.gradients+b.nonlinear()+plan.count), plan.count, s.status);
    check(cudaGetLastError(), "resident wavelet parameter reduction launch");
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

// SiLU residual branch (backlog M3) of a KAN layer whose carrier does not
// fuse it (rational edges), enqueued after the carrier's run_forward: S and
// silu' rows, then Y += S*W^T, checked. Its backward always follows the
// carrier's run_backward.
template<class T> void residual_forward(Context<T>& s, const ParameterBlock& b, std::size_t j) {
    const auto count = s.batch*b.inputs;
    silu_kernel<<<blocks(count), 256, 0, s.stream>>>(s.ptr(s.activation[j]), silu_rows(s, b), count, s.status);
    check(cudaGetLastError(), "resident residual silu launch");
    if (residual_contract(s, b, j)) return;
    check_kernel<<<blocks(s.batch*b.outputs), 256, 0, s.stream>>>(s.ptr(s.activation[j+1]), s.batch*b.outputs, s.status);
    check(cudaGetLastError(), "resident residual check launch");
}
// dW = U^T*S + lambda*W (small: tiled partials, otherwise cuBLAS with beta =
// lambda), dX += silu'(X) (.) (U*W) when the layer's input gradient is
// needed (small: inline per row, otherwise T = U*W by cuBLAS into the
// scratch, whose earlier contents the carrier's backward has consumed).
template<class T> void residual_backward(Context<T>& s, const ParameterBlock& b, std::size_t j, T lambda) {
    const T* w = s.ptr(s.parameters+b.residual());
    T* dw = s.ptr(s.gradients+b.residual());
    const T* u = s.ptr(s.upstream[j+1]);
    const auto i = static_cast<std::int64_t>(b.inputs), o = static_cast<std::int64_t>(b.outputs);
    const auto n = static_cast<std::int64_t>(s.batch);
    const T one = 1, zero = 0;
    const auto checked = b.residual_count+b.outputs;
    unsigned tiles = 0;
    bool reduce = true;
    if (s.batch && small_residual_vjp<T>(b, s.batch)) {
        tiles = parameter_tile_count(s.batch);
        const auto chunk = (s.batch-1)/tiles+1;
        parameter_partial_kernel<<<dim3(blocks(checked), tiles), 256, 0, s.stream>>>(s.ptr(b.silu), u, s.ptr(s.partials),
            s.batch, b.outputs, b.inputs, chunk);
        check(cudaGetLastError(), "resident residual partial launch");
    } else if (s.batch) {
        // Column-major dW^T (I x O) = S^T * U + lambda*W^T; beta = 0 never reads dW.
        if (lambda != 0) check(cudaMemcpyAsync(dw, w, b.residual_count*sizeof(T), cudaMemcpyDeviceToDevice, s.stream), "resident L2 copy");
        s.with_residual_workspace([&] {
            check(gemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_T, i, o, n, &one, s.ptr(b.silu), i, u, o, &lambda, dw, i),
                  "resident residual weight VJP");
        });
        reduce = false;
    }
    const bool dx = s.batch && s.input_gradient(j);
    const T* tw = nullptr;
    if (dx && !small_residual_work<T>(b, s.batch)) {
        // Column-major T^T (I x batch) = W^T(W as column-major I x O) * U^T.
        s.with_residual_workspace([&] {
            check(gemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, i, n, o, &one, w, i, u, o, &zero, s.ptr(s.scratch), i),
                  "resident residual input VJP contraction");
        });
        tw = s.ptr(s.scratch);
    }
    const auto rows = dx ? s.batch*b.inputs : 0;
    residual_finish_kernel<<<std::max(blocks(b.residual_count), rows ? blocks(rows) : 1u), 256, 0, s.stream>>>(
        s.ptr(s.partials), tiles, checked, reduce, w, dw, b.residual_count, lambda, s.ptr(b.silu_derivative), u, tw,
        s.ptr(s.upstream[j]), rows, b.inputs, b.outputs, s.status);
    check(cudaGetLastError(), "resident residual finish launch");
}
// Expansion carriers fuse the residual forward into expansion_forward.
template<class T, class P> void run_residual_forward(Context<T>& s, const P& plan, std::size_t j) {
    if constexpr (is_kan_layer<P> && !is_expansion<P>) if (block_of(plan).residual_count) residual_forward(s, block_of(plan), j);
}
template<class T, class P> void run_residual_backward(Context<T>& s, const P& plan, std::size_t j, T lambda) {
    if constexpr (is_kan_layer<P>) if (block_of(plan).residual_count) residual_backward(s, block_of(plan), j, lambda);
}

// Candidate validation beyond finiteness, before the SGD commit.
template<class T> void validate_candidates(Context<T>&, const BasisPlan<T>&) {}
template<class T> void validate_candidates(Context<T>& s, const TrainableRbfPlan<T>& plan) {
    const auto& b = plan.expansion.block;
    validate_width_kernel<<<blocks(b.terms),256,0,s.stream>>>(s.ptr(s.candidates+b.nonlinear()+b.terms),b.terms,s.status);
    check(cudaGetLastError(),"resident width validation launch");
}
template<class T> void validate_candidates(Context<T>&, const RationalPlan<T>&) {}
template<class T> void validate_candidates(Context<T>& s, const WaveletPlan<T>& plan) {
    if (!plan.count) return;
    validate_log_scale_kernel<<<blocks(plan.count), 256, 0, s.stream>>>(s.ptr(s.candidates+plan.block.nonlinear()+plan.count),
                                                                        plan.count, s.status);
    check(cudaGetLastError(), "resident log-scale validation launch");
}
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
template<class T> NonlinearGradients download_nonlinear(Context<T>& s, const WaveletPlan<T>& plan) {
    WaveletGradients g{std::vector<double>(plan.count), std::vector<double>(plan.count)};
    s.download(g.translations, s.ptr(s.gradients+plan.block.nonlinear()));
    s.download(g.log_scales, s.ptr(s.gradients+plan.block.nonlinear()+plan.count));
    return g;
}
template<class T> NonlinearGradients download_nonlinear(Context<T>& s, const RationalPlan<T>& plan) {
    RationalGradients g{std::vector<double>(plan.block.nonlinear_count)};
    s.download(g.denominators,s.ptr(s.gradients+plan.block.nonlinear()));
    return g;
}

// The trained carrier: the snapshot's configuration with current parameters.
template<class T>
Carrier download_carrier(Context<T>&, const BasisPlan<T>& plan, const BasisEdges& snapshot, std::vector<double> coefficients) {
    if (plan.refit) return BasisEdges{*plan.refit, std::move(coefficients)};
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
Carrier download_carrier(Context<T>& s, const WaveletPlan<T>& plan, const WaveletEdges& snapshot, std::vector<double> coefficients) {
    std::vector<double> translations(plan.count), log_scales(plan.count);
    s.download(translations, s.ptr(s.parameters+plan.block.nonlinear()));
    s.download(log_scales, s.ptr(s.parameters+plan.block.nonlinear()+plan.count));
    return WaveletEdges{snapshot.config, std::move(coefficients), std::move(translations), std::move(log_scales)};
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
    g.residual.resize(b.residual_count);
    s.download(g.residual, s.ptr(s.gradients+b.residual()));
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
    std::vector<double> weights(b.residual_count);
    s.download(weights, s.ptr(s.parameters+b.residual()));
    s.sync();
    Layer layer = snapshot;
    layer.set_carrier(std::move(carrier), bias);
    if (b.residual_count) layer.set_residual(SiluResidual{std::move(weights)});
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
bool same_structure(const WaveletEdges& a, const WaveletEdges& b) { return a.config == b.config; }
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
// are copied directly, one copy per nonempty tensor (at most five per KAN
// layer with the residual branch, four without, two per LayerNorm map). R9 profiling: packing a 117 MB FP64 region
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
template<class T> void image_carrier(ParameterImage<T>& image, const WaveletPlan<T>& plan, const WaveletEdges& edges) {
    check_wavelet<T>(edges); // FP32: translations representable, exp(+-l) finite and nonzero
    image.put(plan.block.nonlinear(), edges.translations, plan.count);
    image.put(plan.block.nonlinear()+plan.count, edges.log_scales, plan.count);
}
template<class T, class P, class Edges>
void image_stage(ParameterImage<T>& image, const P& plan, const Layer& layer, const Edges& edges) {
    const auto& b = block_of(plan);
    image.put(b.offset, layer.coefficients(), b.coefficients);
    image.put(b.bias(), layer.bias(), b.outputs);
    image_carrier(image, plan, edges);
    if (b.residual_count) image.put(b.residual(), layer.residual()->weights, b.residual_count);
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
        if constexpr (std::is_same_v<Stage, Layer>)
            if (expected.residual().has_value() != actual->residual().has_value()) structure_mismatch(j, "residual branch presence");
        const auto* actual_kind = std::get_if<Kind>(&kind_of(*actual));
        if (!actual_kind) structure_mismatch(j, std::is_same_v<Stage, Layer> ? "carrier" : "map kind");
        bool same = true;
        if constexpr (requires { p.refit; } && std::is_same_v<Kind, BasisEdges>)
            same = p.refit ? actual_kind->basis == BasisConfig(*p.refit) : same_structure(expected_kind, *actual_kind);
        else
            same = same_structure(expected_kind, *actual_kind);
        if (!same) structure_mismatch(j, "fixed configuration");
        f(p, *actual, *actual_kind);
    });
}
} // namespace

// Precision-independent interface of an execution engine (one per Precision).
// Device grid refit workspace (backlog M6), reserved after every other region
// when the network has a B-spline layer, for its largest layer at capacity
// (one layer refits at a time): offsets in elements of T.
struct RefitSpace {
    bool available = false; // false: no B-spline layer, or inputs*capacity beyond 32-bit sort indices
    std::size_t runs = 0;   // run bound of the largest layer
    std::size_t xt = 0, sorted = 0, keys = 0, counts = 0, offsets = 0, first_run = 0, run_count = 0, segments = 0;
    std::size_t sums = 0, bands = 0, rhs = 0, fresh = 0, current = 0, knots = 0, scales = 0, backup = 0;
    std::size_t storage = 0, storage_bytes = 0;
};
// Runs of a layer's sorted samples: at most min(batch, 2*terms+1) per input
// (a monotone path through (first new term, first current term or outside)).
// Outputs per refit_solve_kernel launch (review L4): enough (input, output)
// threads to fill the device once (multiprocessors x resident threads), but
// right-hand sides (doubles, inputs x terms x outputs) of at most one sample
// buffer (inputs x capacity values of T); all outputs at once needed up to
// twice a layer's FP32 coefficients (arena +20% at 48 terms).
inline std::size_t refit_runs(std::size_t inputs, std::size_t capacity, std::size_t terms) {
    return product(inputs, std::min(capacity, product(terms, 2)+1));
}

struct ResidentExecutor {
    virtual ~ResidentExecutor() = default;
    virtual void upload_input(std::span<const double> input, std::size_t batch) = 0;
    virtual void upload_output_gradient(std::span<const double> gradient) = 0;
    virtual void forward() = 0;
    virtual void backward(double coefficient_l2) = 0;
    virtual void sgd(double learning_rate) = 0;
    virtual void upload_parameters(const Network& network) = 0;
    virtual void refit_grids(const GridRefit& options) = 0;
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
        // Backlog C5: path of every B-spline layer, which fixes the arena layout.
        // Backlog C12: reduction kernels of every trainable-RBF layer.
        ExecutorDevice device;
        const bool queried = std::any_of(plans.begin(), plans.end(), [](const Plan<T>& p) {
            const auto* basis = std::get_if<BasisPlan<T>>(&p);
            return std::holds_alternative<TrainableRbfPlan<T>>(p) || std::holds_alternative<WaveletPlan<T>>(p) ||
                   (basis && basis->expansion.view.kind == detail::BasisKind::BSpline);
        });
        if (queried && available()) {
            device = executor_device();
            for (auto& plan : plans) {
                if (auto* basis = std::get_if<BasisPlan<T>>(&plan)) {
                    select_sparse(*basis, capacity, tensor_ops, device, testing::test_scope.path);
                    if (!basis->sparse.active) select_knot_stage(basis->expansion, device, basis->expansion.block.residual_count != 0);
                    else select_sparse_knot_stage(*basis, device);
                }
                if (auto* rbf = std::get_if<TrainableRbfPlan<T>>(&plan)) select_reduction(*rbf, device, testing::test_scope.reduction);
                if (auto* wavelet = std::get_if<WaveletPlan<T>>(&plan)) select_wavelet_launch(*wavelet, capacity, device, testing::test_scope.wavelet);
            }
        }
        Reservation<T> reserve;
        parameters = reserve(parameter_count); gradients = reserve(parameter_count); candidates = reserve(parameter_count);
        activation.push_back(reserve(product(capacity, extent(plans.front()).inputs)));
        upstream.push_back(reserve(product(capacity, extent(plans.front()).inputs)));
        for (auto& plan : plans) {
            const auto outputs = extent(plan).outputs;
            activation.push_back(reserve(product(capacity, outputs)));
            upstream.push_back(reserve(product(capacity, outputs)));
            std::visit([&](auto& p) { reserve_workspace(p, capacity, reserve); reserve_residual(p, capacity, reserve); }, plan);
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
        // Backlog M3: after every earlier region, only for a branch that takes
        // a cuBLAS path at full capacity.
        for (const auto& plan : plans)
            std::visit([&](const auto& p) {
                if constexpr (is_kan_layer<std::decay_t<decltype(p)>>) {
                    const auto& b = block_of(p);
                    if (b.residual_count && (!small_residual<T>(b, this->capacity) || !small_residual_vjp<T>(b, this->capacity)))
                        this->has_residual_workspace = true;
                }
            }, plan);
        if (this->has_residual_workspace) this->residual_workspace = reserve(residual_workspace_bytes/sizeof(T));
        // Backlog C13: after every earlier region, only when some layer needs it.
        std::size_t difference_size = 0;
        if constexpr (spline_difference_vjp<T>)
            for (const auto& plan : plans)
                if (const auto* basis = std::get_if<BasisPlan<T>>(&plan)) {
                    const auto& e = basis->expansion;
                    if (e.view.kind == detail::BasisKind::BSpline && e.view.degree != 0)
                        difference_size = std::max(difference_size, product(product(e.block.outputs, e.block.inputs),
                                                                            basis->sparse.active ? e.block.terms : e.block.terms-1));
                }
        if (difference_size) this->differences = reserve(difference_size);
        // Backlog M6: the device refit workspace, after every earlier region.
        reserve_refit(reserve, device);
        graphs.reserve(max_graphs);
        const auto bytes = product(reserve.total, sizeof(T));
        if (auto* layout = testing::test_scope.layout) {
            layout->sparse.clear();
            for (const auto& plan : plans) layout->sparse.push_back(std::visit([](const auto& p) { return sparse_layer(p); }, plan));
            layout->arena_bytes = bytes;
            layout->reduction_groups.clear();
            layout->wavelet_forward.clear();
            layout->multiprocessors = device.multiprocessors;
            for (const auto& plan : plans) {
                const auto* rbf = std::get_if<TrainableRbfPlan<T>>(&plan);
                layout->reduction_groups.push_back(rbf ? rbf->tile_group : 0);
                testing::ResidentLayout::WaveletForward forward;
                if (const auto* wavelet = std::get_if<WaveletPlan<T>>(&plan)) {
                    const auto launch = wavelet_forward_launch(*wavelet, capacity);
                    forward = {launch.samples, launch.splits,
                               wavelet->block.outputs*wavelet_chunks(capacity, launch.samples)*launch.splits};
                }
                layout->wavelet_forward.push_back(forward);
            }
        }
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
            for (const auto& plan : plans) {
                const auto* basis = std::get_if<BasisPlan<T>>(&plan);
                if (!basis || !basis->sparse.active) continue;
                const auto& b = basis->expansion.block;
                if (capacity) {
                    spline_indices_kernel<<<blocks(capacity*b.inputs), 256, 0, stream>>>(
                        reinterpret_cast<unsigned*>(ptr(basis->sparse.indices)), b.inputs, capacity);
                    check(cudaGetLastError(), "resident sparse index launch");
                }
                // Dynamic shared memory beyond the default 48 KiB (large term counts).
                with_spline_width(basis->sparse.width, [&](auto width) {
                    constexpr unsigned P = decltype(width)::value;
                    const auto limit = static_cast<int>(device.shared_optin);
                    if (product(spline_forward_row_bytes<T>(b.terms), basis->sparse.chunk) > (std::size_t{48} << 10))
                        check(cudaFuncSetAttribute(spline_forward_kernel<P, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, limit), "resident sparse shared memory");
                    if constexpr (spline_difference_vjp<T>) {
                        constexpr unsigned D = sparse_derivative_rows<T, P>;
                        if constexpr (D != 0)
                            if (spline_input_bytes<T>(b.terms, basis->sparse.group) > (std::size_t{48} << 10))
                                check(cudaFuncSetAttribute(spline_input_kernel<D, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, limit), "resident sparse shared memory");
                    } else if (spline_input_bytes<T>(b.terms, basis->sparse.group) > (std::size_t{48} << 10)) {
                        check(cudaFuncSetAttribute(spline_input_kernel<P, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, limit), "resident sparse shared memory");
                    }
                });
            }
            for (const auto& plan : plans) {
                const auto* rbf = std::get_if<TrainableRbfPlan<T>>(&plan);
                if (rbf && nonlinear_tile_bytes<T>(rbf->tile_group) > (std::size_t{48} << 10))
                    check(cudaFuncSetAttribute(nonlinear_tile_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(device.shared_optin)), "resident nonlinear shared memory");
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
    // Backlog M6: device grid refit (refit kernels above).
    RefitSpace refit;
    // The expansion plan of B-spline layer j, or null.
    const ExpansionPlan<T>* spline_plan(std::size_t j) const {
        const auto* basis = std::get_if<BasisPlan<T>>(&plans[j]);
        return basis && basis->expansion.view.kind == detail::BasisKind::BSpline ? &basis->expansion : nullptr;
    }
    std::size_t knot_scale_count(const ExpansionPlan<T>& p) const {
        return spline_difference_vjp<T> && p.view.degree != 0 ? product(p.grids, p.block.terms-1) : 0;
    }
    using RefitIterator = thrust::transform_iterator<RefitKey<T>, thrust::counting_iterator<long long>>;
    static std::size_t double_elements(std::size_t count) { return (product(count, sizeof(double))+sizeof(T)-1)/sizeof(T); }
    std::size_t refit_threads = 0; // multiprocessors x threads per multiprocessor
    std::size_t refit_outputs(std::size_t inputs, std::size_t outputs, std::size_t terms) const {
        const auto fill = (refit_threads+inputs-1)/inputs;
        const auto memory = product(inputs, capacity)*sizeof(T)/(product(product(inputs, terms), sizeof(double)));
        return std::min(outputs, std::max<std::size_t>(1, std::min(fill, memory)));
    }
    std::string refit_unavailable = "the network has no B-spline layer";
    void reserve_refit(Reservation<T>& reserve, const ExecutorDevice& device) {
        refit_threads = std::max<std::size_t>(std::size_t{device.multiprocessors}*device.threads_per_multiprocessor, 2048);
        std::size_t samples = 0, runs = 0, sums = 0, bands = 0, rhs = 0, knots = 0, scales = 0, backup = 0, inputs = 0;
        bool any = false, pooled = false;
        for (std::size_t j = 0; j < plans.size(); ++j) {
            const auto* p = spline_plan(j);
            if (!p) continue;
            any = true;
            const auto& b = p->block;
            const auto P = p->view.degree+1;
            const auto r = refit_runs(b.inputs, capacity, b.terms);
            samples = std::max(samples, product(b.inputs, capacity));
            runs = std::max(runs, r);
            sums = std::max(sums, product(r, 2*P*P));
            bands = std::max(bands, product(product(b.inputs, b.terms), P));
            rhs = std::max(rhs, product(product(b.inputs, b.terms), refit_outputs(b.inputs, b.outputs, b.terms)));
            knots = std::max(knots, knot_count(*p));
            scales = std::max(scales, knot_scale_count(*p));
            backup += knot_count(*p)+knot_scale_count(*p);
            inputs = std::max(inputs, b.inputs);
            pooled = pooled || (p->grids == 1 && b.inputs > 1);
            // Run keys (input, g, g') in 64 bits.
            if (b.terms+1 > (std::size_t{1} << 21) || b.inputs > (std::size_t{1} << 21)) {
                refit_unavailable = "a layer's inputs or terms exceed the 21-bit run keys";
                return;
            }
        }
        if (!any) return;
        if (capacity == 0) { refit_unavailable = "capacity 0"; return; }
        if (samples > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            refit_unavailable = "inputs*capacity exceeds the 32-bit CUB item count";
            return;
        }
        const int items = static_cast<int>(samples);
        std::size_t bytes = 0, query = 0;
        check(cub::DeviceSegmentedRadixSort::SortKeys(nullptr, query, static_cast<const T*>(nullptr), static_cast<T*>(nullptr), items,
                  static_cast<int>(inputs), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr)), "resident refit sort size");
        bytes = std::max(bytes, query);
        if (pooled) {
            check(cub::DeviceRadixSort::SortKeys(nullptr, query, static_cast<const T*>(nullptr), static_cast<T*>(nullptr), items),
                  "resident refit sort size");
            bytes = std::max(bytes, query);
        }
        check(cub::DeviceRunLengthEncode::Encode(nullptr, query, RefitIterator(thrust::counting_iterator<long long>(0), RefitKey<T>{}),
                  static_cast<unsigned long long*>(nullptr), static_cast<unsigned*>(nullptr), static_cast<unsigned*>(nullptr), items),
              "resident refit run size");
        bytes = std::max(bytes, query);
        check(cub::DeviceScan::ExclusiveSum(nullptr, query, static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr),
                  static_cast<int>(runs)), "resident refit scan size");
        bytes = std::max(bytes, query);
        refit.available = true;
        refit.runs = runs;
        refit.xt = reserve(samples); refit.sorted = reserve(samples);
        refit.keys = reserve(double_elements(runs)); // 64-bit keys
        refit.counts = reserve(index_elements<T>(runs)); refit.offsets = reserve(index_elements<T>(runs));
        refit.first_run = reserve(index_elements<T>(inputs+1)); refit.run_count = reserve(index_elements<T>(1));
        refit.segments = reserve(index_elements<T>(product(inputs, 2)));
        refit.sums = reserve(double_elements(sums)); refit.bands = reserve(double_elements(bands)); refit.rhs = reserve(double_elements(rhs));
        refit.fresh = reserve(double_elements(knots)); refit.current = reserve(double_elements(knots));
        refit.knots = reserve(knots); refit.scales = reserve(scales); refit.backup = reserve(backup);
        refit.storage_bytes = bytes; refit.storage = reserve((bytes+sizeof(T)-1)/sizeof(T));
    }
    template<class U> U* refit_ptr(std::size_t offset) { return reinterpret_cast<U*>(ptr(offset)); }
    int read_status() {
        check(cudaGetLastError(), "resident kernel launch");
        int value = 0;
        check(cudaMemcpyAsync(&value, status, sizeof(int), cudaMemcpyDeviceToHost, stream), "resident status download");
        sync();
        if (value) reset_status();
        return value;
    }
    void copy(std::size_t to, std::size_t from, std::size_t count) {
        if (count) check(cudaMemcpyAsync(ptr(to), ptr(from), count*sizeof(T), cudaMemcpyDeviceToDevice, stream), "resident refit copy");
    }
    // Refits layer j from activation[j]: new coefficients into the active
    // region (a copy; the previous active region is `candidates`), new knots
    // and knot scales into the layer's regions.
    // Stage 1 (sort, knots) and stage 2 (runs, fit, solve, knot copy); the
    // caller checks the status between them, so nonfinite samples or invalid
    // knots never reach the run keys (review H1).
    void enqueue_refit(std::size_t j, const GridRefit& options, bool second) {
        const auto& p = *spline_plan(j);
        const auto& b = p.block;
        const auto degree = p.view.degree, terms = b.terms, count = knot_count(p)/p.grids, inputs = b.inputs;
        const auto all = product(inputs, batch);
        const int items = static_cast<int>(all);
        T* xt = ptr(refit.xt);
        T* sorted = ptr(refit.sorted);
        auto* segments = refit_ptr<int>(refit.segments);
        void* storage = ptr(refit.storage);
        auto* fresh = refit_ptr<double>(refit.fresh);
        auto* current = refit_ptr<double>(refit.current);
        const bool pooled = p.grids == 1 && inputs > 1;
        if (second) {
            enqueue_fit(p, options, pooled, items, xt, sorted, segments, storage, fresh, current);
            return;
        }
        transpose(*this, ptr(activation[j]), xt, batch, inputs, batch); // xt[i*batch + b]
        spline_segments_kernel<<<blocks(inputs), 256, 0, stream>>>(segments, inputs, batch, batch);
        check(cudaGetLastError(), "resident refit segment launch");
        const auto per_input = [&] {
            auto bytes = refit.storage_bytes;
            check(cub::DeviceSegmentedRadixSort::SortKeys(storage, bytes, static_cast<const T*>(xt), sorted, items, static_cast<int>(inputs),
                      static_cast<const int*>(segments), static_cast<const int*>(segments+inputs), 0, static_cast<int>(sizeof(T)*8), stream),
                  "resident refit sort");
        };
        if (pooled) {
            auto bytes = refit.storage_bytes;
            check(cub::DeviceRadixSort::SortKeys(storage, bytes, static_cast<const T*>(xt), sorted, items, 0, static_cast<int>(sizeof(T)*8), stream),
                  "resident refit sort");
        } else {
            per_input();
        }
        refit_knots_kernel<<<blocks(p.grids, 64), 64, 0, stream>>>(sorted, pooled ? all : batch, p.grids, terms-degree, degree,
            options.grid_eps, options.margin, ptr(refit.knots), knot_scale_count(p) ? ptr(refit.scales) : nullptr, fresh, ptr(p.knots), current,
            p.grids, status);
        check(cudaGetLastError(), "resident refit knot launch");
    }
    void enqueue_fit(const ExpansionPlan<T>& p, const GridRefit& options, bool pooled, int items, T* xt, T* sorted, int* segments,
                     void* storage, double* fresh, double* current) {
        const auto& b = p.block;
        const auto degree = p.view.degree, terms = b.terms, count = knot_count(p)/p.grids, inputs = b.inputs;
        if (pooled) {
            auto bytes = refit.storage_bytes;
            check(cub::DeviceSegmentedRadixSort::SortKeys(storage, bytes, static_cast<const T*>(xt), sorted, items, static_cast<int>(inputs),
                      static_cast<const int*>(segments), static_cast<const int*>(segments+inputs), 0, static_cast<int>(sizeof(T)*8), stream),
                  "resident refit sort");
        }
        // Runs of equal (input, first new term, first current term).
        const auto runs = refit_runs(inputs, batch, terms);
        auto* keys = refit_ptr<unsigned long long>(refit.keys);
        auto* counts = refit_ptr<unsigned>(refit.counts);
        auto* offsets = refit_ptr<unsigned>(refit.offsets);
        auto* run_count = refit_ptr<unsigned>(refit.run_count);
        auto* first_run = refit_ptr<unsigned>(refit.first_run);
        check(cudaMemsetAsync(counts, 0, runs*sizeof(unsigned), stream), "resident refit run reset");
        const RefitGrids<T> grids{fresh, current, ptr(refit.knots), ptr(p.knots), count, terms, degree, p.grids, p.grids};
        {
            auto bytes = refit.storage_bytes;
            check(cub::DeviceRunLengthEncode::Encode(storage, bytes, RefitIterator(thrust::counting_iterator<long long>(0), RefitKey<T>{sorted, grids, batch}),
                      keys, counts, run_count, items, stream), "resident refit runs");
            bytes = refit.storage_bytes;
            check(cub::DeviceScan::ExclusiveSum(storage, bytes, static_cast<const unsigned*>(counts), offsets, static_cast<int>(runs), stream),
                  "resident refit run offsets");
        }
        auto* sums = refit_ptr<double>(refit.sums);
        const auto launch_runs = [&](auto width) {
            constexpr unsigned lanes = decltype(width)::value ? 32 : 1;
            refit_run_kernel<decltype(width)::value><<<blocks((runs+1)*lanes, 128), 128, 0, stream>>>(sorted, grids, inputs, keys, counts,
                offsets, run_count, first_run, sums, status);
        };
        if (degree < max_sparse_spline_width) with_spline_width(static_cast<unsigned>(degree+1), launch_runs);
        else launch_runs(std::integral_constant<unsigned, 0>{});
        check(cudaGetLastError(), "resident refit run launch");
        auto* bands = refit_ptr<double>(refit.bands);
        refit_factor_kernel<<<blocks(inputs, 32), 32, 0, stream>>>(keys, first_run, sums, inputs, terms, degree, options.ridge, bands, status);
        check(cudaGetLastError(), "resident refit factor launch");
        const auto chunk = refit_outputs(b.inputs, b.outputs, b.terms);
        for (std::size_t o0 = 0; o0 < b.outputs; o0 += chunk) {
            refit_solve_kernel<<<blocks(inputs*chunk, 128), 128, 0, stream>>>(keys, first_run, sums, bands, refit_ptr<double>(refit.rhs),
                ptr(candidates+b.offset), ptr(parameters+b.offset), inputs, b.outputs, o0, chunk, terms, degree, status);
            check(cudaGetLastError(), "resident refit solve launch");
        }
        copy(p.knots, refit.knots, knot_count(p));
        copy(p.knot_scales, refit.scales, knot_scale_count(p));
    }
    void refit_grids(const GridRefit& options) override {
        flush();
        if (!(options.grid_eps >= 0 && options.grid_eps <= 1)) throw std::invalid_argument("grid_eps must be in [0, 1]");
        if (!(options.ridge >= 0) || !std::isfinite(options.ridge)) throw std::invalid_argument("refit ridge must be finite and nonnegative");
        if (!(options.margin >= 0) || !std::isfinite(options.margin)) throw std::invalid_argument("refit margin must be finite and nonnegative");
        if (!has_input) throw std::logic_error("resident input has not been uploaded");
        std::vector<char> refits(plans.size(), 0);
        bool any = false;
        for (std::size_t j = 0; j < plans.size(); ++j) {
            const auto* p = spline_plan(j);
            if (!p) continue;
            const auto& b = p->block;
            if (options.intervals && options.intervals != b.terms-p->view.degree)
                throw std::invalid_argument("resident grid refit keeps the term count: intervals must be 0 or the layer's");
            if (b.inputs > 1 && options.per_input != (p->grids != 1))
                throw std::invalid_argument("resident grid refit keeps each layer's grid form: per_input must match it");
            refits[j] = 1; any = true;
        }
        if (!any) throw std::invalid_argument("grid refit requires a network with B-spline layers");
        if (batch == 0) throw std::invalid_argument("grid refit needs samples");
        if (!refit.available) throw std::invalid_argument("resident grid refit unavailable: " + refit_unavailable);
        const auto last = static_cast<std::size_t>(std::find(refits.rbegin(), refits.rend(), 1) - refits.rbegin());
        const auto end = plans.size()-last;
        // Atomic: knots and knot scales are backed up, coefficients go to a
        // copy of the parameters that becomes the active region on success.
        reset_status();
        std::size_t backed = refit.backup;
        for (std::size_t j = 0; j < end; ++j)
            if (refits[j]) {
                const auto& p = *spline_plan(j);
                copy(backed, p.knots, knot_count(p)); backed += knot_count(p);
                copy(backed, p.knot_scales, knot_scale_count(p)); backed += knot_scale_count(p);
            }
        copy(candidates, parameters, parameter_count);
        std::swap(parameters, candidates);
        has_forward = has_backward = false;
        try {
            for (std::size_t j = 0; j < end; ++j) {
                if (refits[j]) {
                    const auto where = "resident grid refit of layer " + std::to_string(j) + ": ";
                    enqueue_refit(j, options, false);
                    int bits = read_status();
                    if (bits&1) throw std::overflow_error(where + "nonfinite activation before the layer");
                    if (bits&status_refit_nonfinite) throw std::overflow_error(where + "nonfinite samples");
                    if (!bits) {
                        enqueue_refit(j, options, true);
                        bits = read_status();
                    }
                    if (bits&status_refit_interval) throw std::invalid_argument(where + "the samples of an input span no interval");
                    if (bits&status_refit_knots) throw std::invalid_argument(where + "knots coincide or leave the precision's range (repeated samples: raise grid_eps)");
                    if (bits&status_refit_singular)
                        throw std::invalid_argument(where + "the least-squares system is singular (knot spans without samples; use ridge > 0)");
                    if (bits&status_refit_internal) throw std::logic_error(where + "inconsistent spline spans");
                    if (bits) throw std::overflow_error(where + "nonfinite result");
                }
                if (j+1 < end) std::visit([&](const auto& plan) { run_forward(*this, plan, j); run_residual_forward(*this, plan, j); }, plans[j]);
            }
        } catch (...) {
            std::swap(parameters, candidates);
            backed = refit.backup;
            for (std::size_t j = 0; j < end; ++j)
                if (refits[j]) {
                    const auto& p = *spline_plan(j);
                    copy(p.knots, backed, knot_count(p)); backed += knot_count(p);
                    copy(p.knot_scales, backed, knot_scale_count(p)); backed += knot_scale_count(p);
                }
            sync();
            reset_status();
            sync();
            throw;
        }
        // The layers' configurations with the refit knots (exact widenings of
        // the device's); the snapshot itself is not copied (its coefficients
        // are as large as the model: copying them cost 120 ms of a 140 ms
        // 1024-wide refit).
        std::vector<BSplineConfig> configs(plans.size());
        for (std::size_t j = 0; j < end; ++j)
            if (refits[j]) {
                const auto& p = *spline_plan(j);
                configs[j] = BSplineConfig{p.view.degree, std::vector<double>(knot_count(p)), p.grids};
                this->download(configs[j].knots, ptr(p.knots));
            }
        sync();
        for (std::size_t j = 0; j < end; ++j)
            if (refits[j]) std::get<BasisPlan<T>>(plans[j]).refit = std::move(configs[j]);
    }

    // Eager calls report through bits[0] and leave it clear, so that a
    // training interval always starts with every status word zero.
    void result() {
        check(cudaGetLastError(), "resident kernel launch");
        int value = 0;
        check(cudaMemcpyAsync(&value, status, sizeof(int), cudaMemcpyDeviceToHost, stream), "resident status download");
        sync();
        if (value) reset_status();
        if (value&1) throw std::overflow_error("nonfinite resident numerical result");
        if (value&status_parameter_range) throw std::overflow_error(parameter_range_message);
        if (value&2) throw std::domain_error("unsafe resident rational denominator");
    }

    // Launch sequences shared by the eager calls and the captured steps, so
    // that a replay runs exactly the eager kernels.
    void enqueue_forward() {
        for (std::size_t j = 0; j < plans.size() && batch; ++j)
            std::visit([&](const auto& plan) { run_forward(*this, plan, j); run_residual_forward(*this, plan, j); }, plans[j]);
    }
    void enqueue_backward(T lambda) {
        for (std::size_t j = plans.size(); j-- > 0;)
            std::visit([&](const auto& plan) {
                run_backward(*this, plan, j, lambda);
                run_residual_backward(*this, plan, j, lambda);
            }, plans[j]);
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
        if (c.first_bits&status_parameter_range) throw std::overflow_error(parameter_range_message + where);
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
void ResidentNetwork::refit_grids(const GridRefit& options) { state().engine->refit_grids(options); }
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

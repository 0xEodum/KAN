// C3 stage 2 prototype: input VJP dx[b,i] = sum_k Phi'_k(x[b,i]) * (U*C)[b, i*K+k]
// in one kernel (SGEMM over outputs with an epilogue that recomputes Phi').
// Reference: cuBLAS W = U*C then a staged recompute-finish kernel (as the library
// after C3). Main loop: 256x128 tile (16x8 per thread), 4-stage cp.async
// pipeline; column tiles hold whole inputs (128/K of them). Epilogue: the W tile
// in 64-row passes through shared memory, the pass's x values staged, one
// (row, input) derivative evaluation and dot product per task.
// -DNO_EPILOGUE times the main loop alone (results then invalid).
// Usage: fused_dx <batch> <inputs> <outputs> [terms]
#include "detail/basis_formulas.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace kan::detail;
#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1);} } while (0)
struct NoGuard { template<class T> __device__ T operator()(T v) const { return v; } };

constexpr int BM = 256, BN = 128, BK = 8, PITCH = 132, APITCH = BM + 4, THREADS = 256, TM = BM / 16;
constexpr int EPI_ROWS = 64; // rows of the W tile staged per epilogue pass
constexpr int STAGES = 4; // rows of the W tile staged per epilogue pass

// Shared (floats): main loop As[2][BK][PITCH] + Bs[2][BK][PITCH] = 4224;
// epilogue: W[EPI_ROWS][PITCH] = 4224 (aliases the main loop) + scratch[THREADS][2K].
template<BasisKind Kind>
__global__ void __launch_bounds__(THREADS, 1)
fused_dx(const float* __restrict__ u, const float* __restrict__ c, const float* __restrict__ x, float* __restrict__ dx,
         int batch, int inputs, int outputs, BasisViewOf<float> basis, int per_tile) {
    extern __shared__ __align__(16) float smem[];
    float* As = smem;
    float* Bs = smem + STAGES * BK * APITCH;
    const int K = static_cast<int>(basis.terms), L = inputs * K;
    float* scr = smem + STAGES * BK * (APITCH + PITCH) + threadIdx.x * 2 * K;
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    const int b0 = blockIdx.y * BM, in0 = blockIdx.x * per_tile;
    const int in_count = min(per_tile, inputs - in0), cols = in_count * K, l0 = in0 * K;
    const int ktiles = (outputs + BK - 1) / BK;
    float acc[TM][8] = {};
    // A = U tile (128 rows x BK outputs), transposed into As[kk][m]: 4 scalars per thread.
    // Asynchronous 4-byte copies into the transposed A layout and the natural B layout;
    // out-of-range elements are zero-filled.
    // Per-thread copy pattern (constant over tiles): A element r is row t/8 + 32r,
    // output column t%8; B element r is output row t/128 + 2r, column t%128.
    const int am = t / BK, ak = t % BK, bn = t % BN, bk = t / BN;
    const float* a_src = u + static_cast<size_t>(b0 + am) * outputs + ak;
    const float* b_src = c + static_cast<size_t>(bk) * L + l0 + bn;
    unsigned a_rows = 0; // bit r: row valid
    #pragma unroll
    for (int r = 0; r < BM * BK / THREADS; ++r) a_rows |= (b0 + am + 32 * r < batch) << r;
    const bool b_col = bn < cols;
    auto issue = [&](int kt) {
        const int k0 = kt * BK, st = kt % STAGES;
        float* A = As + st * BK * APITCH + ak * APITCH + am;
        float* B = Bs + st * BK * PITCH + bk * PITCH + bn;
        const bool a_col = k0 + ak < outputs;
        #pragma unroll
        for (int r = 0; r < BM * BK / THREADS; ++r) {
            const bool in = a_col && (a_rows >> r & 1);
            __pipeline_memcpy_async(A + 32 * r, in ? a_src + k0 + static_cast<size_t>(32 * r) * outputs : u, 4, in ? 0 : 4);
        }
        #pragma unroll
        for (int r = 0; r < BN * BK / THREADS; ++r) {
            const bool in = b_col && k0 + bk + 2 * r < outputs;
            __pipeline_memcpy_async(B + 2 * r * PITCH, in ? b_src + static_cast<size_t>(k0 + 2 * r) * L : c, 4, in ? 0 : 4);
        }
    };
    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) { if (st < ktiles) issue(st); __pipeline_commit(); }
    for (int kt = 0; kt < ktiles; ++kt) {
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();
        if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1);
        __pipeline_commit();
        const float* A = As + (kt % STAGES) * BK * APITCH;
        const float* B = Bs + (kt % STAGES) * BK * PITCH;
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM];
            #pragma unroll
            for (int q = 0; q < TM / 4; ++q) {
                const float4 a = *reinterpret_cast<const float4*>(A + kk * APITCH + 64 * q + ty * 4);
                av[4 * q] = a.x; av[4 * q + 1] = a.y; av[4 * q + 2] = a.z; av[4 * q + 3] = a.w;
            }
            const float4 c0 = *reinterpret_cast<const float4*>(B + kk * PITCH + tx * 4);
            const float4 c1 = *reinterpret_cast<const float4*>(B + kk * PITCH + 64 + tx * 4);
            const float bv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
    }
    __pipeline_wait_prior(0);
    __syncthreads();
#ifdef NO_EPILOGUE
    float sink = 0;
    #pragma unroll
    for (int a = 0; a < TM; ++a)
        #pragma unroll
        for (int q = 0; q < 8; ++q) sink += acc[a][q];
    if (sink == 12345.f) dx[t] = sink;
    return;
#endif
    // Epilogue in passes of EPI_ROWS rows of the W tile (aliasing the pipeline
    // buffers), with the pass's inputs x[rows, in0 .. in0+in_count) staged by
    // coalesced loads; then one (row, input) dot product per task.
    float* W = smem;
    float* X = smem + EPI_ROWS * PITCH;
    for (int pass = 0; pass < BM / EPI_ROWS; ++pass) {
        const int r0 = pass * EPI_ROWS;
        if (b0 + r0 >= batch) break;
        for (int e = t; e < EPI_ROWS * in_count; e += THREADS) {
            const int m = e / in_count, ii = e % in_count, b = b0 + r0 + m;
            X[e] = b < batch ? x[static_cast<size_t>(b) * inputs + in0 + ii] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int m = 64 * (i / 4) + ty * 4 + i % 4;
            if (m >= r0 && m < r0 + EPI_ROWS) {
                float* row = W + (m - r0) * PITCH;
                *reinterpret_cast<float4*>(row + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                *reinterpret_cast<float4*>(row + 64 + tx * 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            }
        }
        __syncthreads();
        for (int task = t; task < EPI_ROWS * in_count; task += THREADS) {
            const int m = task / in_count, ii = task % in_count, b = b0 + r0 + m;
            if (b < batch) {
                basis_terms_for<Kind>(basis, X[task], BasisRowOf<float>{scr, scr + K, nullptr, nullptr}, NoGuard{});
                const float* w = W + m * PITCH + ii * K;
                float sum = 0;
                for (int k = 0; k < K; ++k) sum += scr[K + k] * w[k];
                dx[static_cast<size_t>(b) * inputs + in0 + ii] = sum;
            }
        }
        __syncthreads();
    }
}

// Reference finish (library stage 1: staged W tile + recomputed Phi').
template<BasisKind Kind>
__global__ void finish_ref(const float* x, const float* w, float* dx, size_t rows, int terms, int tile_rows) {
    extern __shared__ __align__(16) float stage[];
    const size_t plane = size_t(tile_rows) * terms;
    BasisViewOf<float> basis{Kind, size_t(terms), 0, 0, 1, 1, nullptr, nullptr, nullptr, nullptr, 0, false};
    for (size_t base = size_t(blockIdx.x) * tile_rows; base < rows; base += size_t(gridDim.x) * tile_rows) {
        const size_t count = rows - base < size_t(tile_rows) ? rows - base : tile_rows;
        for (size_t i = threadIdx.x; i < count * terms; i += blockDim.x) stage[plane + i] = w[base * terms + i];
        if (threadIdx.x < count) {
            const size_t row = threadIdx.x * terms;
            basis_terms_for<Kind>(basis, x[base + threadIdx.x], BasisRowOf<float>{stage + 2 * plane + row, stage + row, nullptr, nullptr}, NoGuard{});
        }
        __syncthreads();
        if (threadIdx.x < count) {
            float sum = 0; const float* d = stage + threadIdx.x * terms;
            for (int k = 0; k < terms; ++k) sum += d[k] * d[plane + k];
            dx[base + threadIdx.x] = sum;
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    const int B = std::atoi(argv[1]), I = std::atoi(argv[2]), O = std::atoi(argv[3]), K = argc > 4 ? std::atoi(argv[4]) : 7;
    const int L = I * K;
    std::vector<float> hx(size_t(B) * I), hu(size_t(B) * O), hc(size_t(O) * L);
    unsigned s = 7;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) / 16777216.f * 2 - 1; };
    for (auto& v : hx) v = rnd();
    for (auto& v : hu) v = rnd();
    for (auto& v : hc) v = rnd() * 0.05f;
    float *x, *u, *c, *w, *dx1, *dx2;
    CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&u, hu.size() * 4)); CK(cudaMalloc(&c, hc.size() * 4));
    CK(cudaMalloc(&w, size_t(B) * L * 4)); CK(cudaMalloc(&dx1, hx.size() * 4)); CK(cudaMalloc(&dx2, hx.size() * 4));
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(u, hu.data(), hu.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    BasisViewOf<float> view{BasisKind::Chebyshev, size_t(K), 0, 0, 1, 1, nullptr, nullptr, nullptr, nullptr, 0, false};
    cublasHandle_t h; cublasCreate(&h);
    const int tile_rows = std::min<int>(256, 32768 / (K * 3 * 4)) / 32 * 32;
    const size_t rows = size_t(B) * I;
    auto ref = [&] {
        const float one = 1, zero = 0;
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, L, B, O, &one, c, L, u, O, &zero, w, L);
        const unsigned g = unsigned(std::min<size_t>((rows + tile_rows - 1) / tile_rows, 65535));
        finish_ref<BasisKind::Chebyshev><<<g, 256, size_t(tile_rows) * K * 3 * 4>>>(x, w, dx1, rows, K, tile_rows);
    };
    const int per_tile = BN / K;
    const size_t shared = (std::max(STAGES * BK * (APITCH + PITCH), EPI_ROWS * (PITCH + per_tile)) + THREADS * 2 * K) * 4;
    CK(cudaFuncSetAttribute(fused_dx<BasisKind::Chebyshev>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    auto fused = [&] {
        fused_dx<BasisKind::Chebyshev><<<dim3((I + per_tile - 1) / per_tile, (B + BM - 1) / BM), THREADS, shared>>>(u, c, x, dx2, B, I, O, view, per_tile);
    };
    auto time = [&](auto f) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < 3; ++i) f();
        std::vector<float> v;
        for (int r = 0; r < 5; ++r) { cudaEventRecord(e0); for (int i = 0; i < 20; ++i) f(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms / 20); }
        std::sort(v.begin(), v.end()); return v[2];
    };
    const float tr = time(ref), tf = time(fused);
    CK(cudaGetLastError());
    std::vector<float> a(rows), e(rows);
    CK(cudaMemcpy(a.data(), dx2, rows * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(e.data(), dx1, rows * 4, cudaMemcpyDeviceToHost));
    double md = 0, mx = 0; for (size_t i = 0; i < rows; ++i) { md = std::max(md, double(std::fabs(a[i] - e[i]))); mx = std::max(mx, double(std::fabs(e[i]))); }
    const double f = 2.0 * B * O * L / 1e9;
    std::printf("B=%d I=%d O=%d K=%d  ref %.1f us  fused %.1f us (%.1f TF)  ratio %.2f  maxdiff %.1e of %.1e\n", B, I, O, K, tr * 1e3, tf * 1e3, f / tf, tf / tr, md, mx);
}

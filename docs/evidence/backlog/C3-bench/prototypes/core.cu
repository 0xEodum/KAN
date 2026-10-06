// C3 prototype: hand-written SIMT SGEMM core (128x128 tile, 8x8 per thread,
// register-staged double buffer, BK = 8 or 16) with split-K slices, against
// cuBLAS on the forward shapes. Times the main kernel only (the split partials
// are not reduced). Usage: core <M> <N> <K>
// Pure SGEMM core: Y (M x N, row-major) = A (M x Kd, row-major) * B^T (B: N x Kd, row-major).
// cuBLAS "tn" equivalent. Variants of the tile engine to find the ceiling.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1);} } while (0)

constexpr int BM = 128, BN = 128, THREADS = 256, PAD = 4;

// Both operands K-contiguous; tile loads float4 along k, transposed stores.
template<int BK>
__global__ void __launch_bounds__(THREADS, 2)
core_tn(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int M, int N, int Kd) {
    __shared__ __align__(16) float As[2][BK][BM + PAD];
    __shared__ __align__(16) float Bs[2][BK][BN + PAD];
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int span = Kd / gridDim.z; a += blockIdx.z * span; b += blockIdx.z * span; y += static_cast<size_t>(blockIdx.z) * M * N;
    constexpr int Q = BK / 4;            // float4 per row of the tile
    constexpr int LOADS = BM * Q / THREADS; // float4 per thread per operand
    float4 ar[LOADS], br[LOADS];
    auto load = [&](int k0) {
        #pragma unroll
        for (int r = 0; r < LOADS; ++r) {
            const int e = t + THREADS * r, row = e / Q, q = e % Q;
            ar[r] = (m0 + row < M) ? *reinterpret_cast<const float4*>(a + static_cast<size_t>(m0 + row) * Kd + k0 + 4 * q) : make_float4(0, 0, 0, 0);
            br[r] = (n0 + row < N) ? *reinterpret_cast<const float4*>(b + static_cast<size_t>(n0 + row) * Kd + k0 + 4 * q) : make_float4(0, 0, 0, 0);
        }
    };
    auto store = [&](int s) {
        #pragma unroll
        for (int r = 0; r < LOADS; ++r) {
            const int e = t + THREADS * r, row = e / Q, q = e % Q;
            As[s][4 * q + 0][row] = ar[r].x; As[s][4 * q + 1][row] = ar[r].y; As[s][4 * q + 2][row] = ar[r].z; As[s][4 * q + 3][row] = ar[r].w;
            Bs[s][4 * q + 0][row] = br[r].x; Bs[s][4 * q + 1][row] = br[r].y; Bs[s][4 * q + 2][row] = br[r].z; Bs[s][4 * q + 3][row] = br[r].w;
        }
    };
    float acc[8][8] = {};
    load(0); store(0); __syncthreads();
    const int tiles = span / BK;
    for (int tile = 0; tile < tiles; ++tile) {
        const int s = tile & 1;
        if (tile + 1 < tiles) load((tile + 1) * BK);
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[s][kk][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[s][kk][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[s][kk][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[s][kk][64 + tx * 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        if (tile + 1 < tiles) store(s ^ 1);
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (m >= M) continue;
        float* row = y + static_cast<size_t>(m) * N + n0;
        if (n0 + 128 <= N) {
            *reinterpret_cast<float4*>(row + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(row + 64 + tx * 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        }
    }
}

int main(int argc, char** argv) {
    const int M = std::atoi(argv[1]), N = std::atoi(argv[2]), Kd = std::atoi(argv[3]);
    std::vector<float> ha(size_t(M) * Kd), hb(size_t(N) * Kd);
    unsigned s = 1;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) / 16777216.f * 2 - 1; };
    for (auto& v : ha) v = rnd();
    for (auto& v : hb) v = rnd();
    float *a, *b, *y1, *y2;
    CK(cudaMalloc(&a, ha.size() * 4)); CK(cudaMalloc(&b, hb.size() * 4));
    CK(cudaMalloc(&y1, size_t(M) * N * 4)); CK(cudaMalloc(&y2, size_t(M) * N * 4 * 8));
    CK(cudaMemcpy(a, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
    cublasHandle_t h; cublasCreate(&h);
    auto time = [&](auto f) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < 3; ++i) f();
        std::vector<float> v;
        for (int r = 0; r < 5; ++r) { cudaEventRecord(e0); for (int i = 0; i < 20; ++i) f(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms / 20); }
        std::sort(v.begin(), v.end()); return v[2];
    };
    const float one = 1, zero = 0;
    const float tc = time([&] { cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, Kd, &one, b, Kd, a, Kd, &zero, y1, N); });
    for (int split : {1, 2, 4, 7}) {
        const dim3 grid(N / BN, (M + BM - 1) / BM, split);
        const float t16 = time([&] { core_tn<16><<<grid, THREADS>>>(a, b, y2, M, N, Kd); });
        const float t8 = time([&] { core_tn<8><<<grid, THREADS>>>(a, b, y2, M, N, Kd); });
        const double f = 2.0 * M * N * Kd / 1e9;
        std::printf("M=%d N=%d K=%d split %d: cublas %.3f (%.1f TF) core8 %.3f (%.1f) core16 %.3f (%.1f)\n", M, N, Kd, split, tc, f / tc, t8, f / t8, t16, f / t16);
    }
    return 0;
}

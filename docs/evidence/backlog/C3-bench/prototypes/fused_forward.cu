// C3 prototype: fused basis expansion + SGEMM forward, Y = Phi(X) C^T + b,
// against a basis kernel + cuBLAS SGEMM + bias. Chebyshev, FP32.
// The fused kernel walks the contraction in 16-column windows; the inputs a
// window touches are evaluated in full (formula into a per-thread shared
// scratch row) and their in-window terms copied into the A tile. split > 1
// splits the windows over gridDim.z into partial outputs, summed in order by
// a reduction kernel (included in the timing).
// The reference basis kernel here is unstaged (slower than the library's
// staged basis_kernel); compare the fused time with the library's nsys times.
// Usage: fused_forward <batch> <inputs> <outputs> [terms] [split]
#include "detail/basis_formulas.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace kan::detail;
#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1);} } while (0)

struct NoGuard { template<class T> __device__ T operator()(T v) const { return v; } };

constexpr int BM = 128, BN = 128, BK = 16, PITCH = 132, THREADS = 256;

// Shared: As[2][BK][PITCH], Bs[2][BK][PITCH], scratch[THREADS][2K]
template<BasisKind Kind>
__global__ void __launch_bounds__(THREADS, 2)
fused_forward(const float* __restrict__ x, const float* __restrict__ c, const float* __restrict__ bias, float* __restrict__ y,
              int batch, int inputs, int outputs, BasisViewOf<float> basis, int unused) {
    extern __shared__ __align__(16) float smem[];
    float* As = smem;
    float* Bs = As + 2 * BK * PITCH;
    const int K = static_cast<int>(basis.terms), L = inputs * K;
    float* scr = Bs + 2 * BK * PITCH + threadIdx.x * 2 * K;
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    const int b0 = blockIdx.y * BM, o0 = blockIdx.x * BN;
    const int all = (L + BK - 1) / BK, per = (all + gridDim.z - 1) / gridDim.z, cbeg = blockIdx.z * per, chunks = min(all, cbeg + per) - cbeg;
    y += static_cast<size_t>(blockIdx.z) * batch * outputs;
    float acc[8][8] = {};
    // C tile: 128 rows x 16 columns = 512 float4, two per thread: row n = t/4 (+64), quad q = t%4.
    float4 breg[2];
    const int bq = t % 4, bn = t / 4;
    auto load_b = [&](int chunk) {
        const int l = chunk * BK + bq * 4;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int o = o0 + bn + 64 * r;
            breg[r] = (o < outputs && l < L) ? *reinterpret_cast<const float4*>(c + static_cast<size_t>(o) * L + l) : make_float4(0, 0, 0, 0);
        }
    };
    auto store_b = [&](int buf) {
        float* B = Bs + buf * BK * PITCH;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int n = bn + 64 * r;
            B[(bq * 4 + 0) * PITCH + n] = breg[r].x; B[(bq * 4 + 1) * PITCH + n] = breg[r].y;
            B[(bq * 4 + 2) * PITCH + n] = breg[r].z; B[(bq * 4 + 3) * PITCH + n] = breg[r].w;
        }
    };
    auto eval_a = [&](int chunk, int buf) {
        float* A = As + buf * BK * PITCH;
        const int l0 = chunk * BK;
        const int in0 = l0 / K, in1 = min(inputs - 1, (l0 + BK - 1) / K);
        const int tasks = BM * (in1 - in0 + 1);
        for (int task = t; task < tasks; task += THREADS) {
            const int m = task % BM, in = in0 + task / BM;
            const int first = in * K - l0; // window column of term 0
            if (b0 + m < batch) {
                basis_terms_for<Kind>(basis, x[static_cast<size_t>(b0 + m) * inputs + in], BasisRowOf<float>{scr, scr + K, nullptr, nullptr}, NoGuard{});
                for (int k = max(0, -first); k < K && first + k < BK; ++k) A[(first + k) * PITCH + m] = scr[k];
            } else {
                for (int k = max(0, -first); k < K && first + k < BK; ++k) A[(first + k) * PITCH + m] = 0.f;
            }
        }
        // Columns past L (last chunk): zero.
        if (l0 + BK > L) for (int e = t; e < BK * BM; e += THREADS) { const int col = e / BM; if (l0 + col >= L) A[col * PITCH + e % BM] = 0.f; }
    };
    load_b(cbeg);
    store_b(0); eval_a(cbeg, 0);
    __syncthreads();
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int buf = chunk & 1;
        const bool more = chunk + 1 < chunks;
        if (more) load_b(cbeg + chunk + 1);
        const float* A = As + buf * BK * PITCH;
        const float* B = Bs + buf * BK * PITCH;
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(A + kk * PITCH + ty * 4);
            const float4 a1 = *reinterpret_cast<const float4*>(A + kk * PITCH + 64 + ty * 4);
            const float4 c0 = *reinterpret_cast<const float4*>(B + kk * PITCH + tx * 4);
            const float4 c1 = *reinterpret_cast<const float4*>(B + kk * PITCH + 64 + tx * 4);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        if (more) { store_b(buf ^ 1); eval_a(cbeg + chunk + 1, buf ^ 1); }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = b0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (m >= batch) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int o = o0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
            if (o < outputs) y[static_cast<size_t>(m) * outputs + o] = acc[i][j] + (blockIdx.z == 0 ? bias[o] : 0.f);
        }
    }
}

template<BasisKind Kind>
__global__ void basis_rows(const float* x, float* v, float* d, size_t count, BasisViewOf<float> basis) {
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < count; idx += size_t(gridDim.x) * blockDim.x)
        basis_terms_for<Kind>(basis, x[idx], BasisRowOf<float>{v + idx * basis.terms, d + idx * basis.terms, nullptr, nullptr}, NoGuard{});
}
__global__ void reduce_split(float* y, size_t count, int split) {
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < count; idx += size_t(gridDim.x) * blockDim.x) {
        float sum = y[idx];
        for (int z = 1; z < split; ++z) sum += y[z * count + idx];
        y[idx] = sum;
    }
}
__global__ void add_bias(float* y, const float* b, size_t count, int outputs) {
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < count; idx += size_t(gridDim.x) * blockDim.x) y[idx] += b[idx % outputs];
}

int main(int argc, char** argv) {
    const int B = std::atoi(argv[1]), I = std::atoi(argv[2]), O = std::atoi(argv[3]), K = argc > 4 ? std::atoi(argv[4]) : 7;
    const int L = I * K;
    std::vector<float> hx(size_t(B) * I), hc(size_t(O) * L), hb(O);
    unsigned s = 1;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) / 16777216.f * 2 - 1; };
    for (auto& v : hx) v = rnd();
    for (auto& v : hc) v = rnd() * 0.05f;
    for (auto& v : hb) v = rnd();
    float *x, *c, *b, *y1, *y2, *phi, *dphi;
    CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&c, hc.size() * 4)); CK(cudaMalloc(&b, O * 4));
    CK(cudaMalloc(&y1, size_t(B) * O * 4)); CK(cudaMalloc(&y2, size_t(B) * O * 4 * 8));
    CK(cudaMalloc(&phi, size_t(B) * L * 4)); CK(cudaMalloc(&dphi, size_t(B) * L * 4));
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b, hb.data(), O * 4, cudaMemcpyHostToDevice));
    BasisViewOf<float> view{BasisKind::Chebyshev, size_t(K), 0, 0, 1, 1, nullptr, nullptr, nullptr, nullptr, 0, false};
    cublasHandle_t h; cublasCreate(&h);
    const int ic = argc > 5 ? std::atoi(argv[5]) : 1;
    const size_t shared = (4 * BK * PITCH + THREADS * 2 * K) * 4;
    CK(cudaFuncSetAttribute(fused_forward<BasisKind::Chebyshev>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    auto ref = [&] {
        basis_rows<BasisKind::Chebyshev><<<1024, 256>>>(x, phi, dphi, size_t(B) * I, view);
        const float one = 1, zero = 0;
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, O, B, L, &one, c, L, phi, L, &zero, y1, O);
        add_bias<<<1024, 256>>>(y1, b, size_t(B) * O, O);
    };
    auto fused = [&] {
        fused_forward<BasisKind::Chebyshev><<<dim3((O + BN - 1) / BN, (B + BM - 1) / BM, ic), THREADS, shared>>>(x, c, b, y2, B, I, O, view, ic);
        if (ic > 1) reduce_split<<<1024, 256>>>(y2, size_t(B) * O, ic);
    };
    auto time = [&](auto f, int reps) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < 3; ++i) f();
        cudaEventRecord(e0); for (int i = 0; i < reps; ++i) f(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / reps;
    };
    const int reps = 50;
    std::vector<float> ta, tb;
    for (int r = 0; r < 5; ++r) { ta.push_back(time(ref, reps)); tb.push_back(time(fused, reps)); }
    std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
    std::vector<float> h1(size_t(B) * O), h2(size_t(B) * O);
    CK(cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2.data(), y2, h2.size() * 4, cudaMemcpyDeviceToHost));
    double md = 0, mx = 0;
    for (size_t i = 0; i < h1.size(); ++i) { md = std::max(md, double(std::fabs(h1[i] - h2[i]))); mx = std::max(mx, double(std::fabs(h1[i]))); }
    const double flop = 2.0 * B * O * L;
    std::printf("B=%d I=%d O=%d K=%d split=%d  ref %.3f ms  fused %.3f ms (%.1f TFLOPS)  maxdiff %.2e of %.2e\n",
                B, I, O, K, ic, ta[2], tb[2], flop / tb[2] / 1e9, md, mx);
}

// M3 phase 2: are the residual GEMMs of the FP32 256-wide step library-bound?
// Times cublasSgemm (the executor's call) against every cuBLASLt heuristic
// algorithm (up to 32) for the three residual shapes at batch 8192, width 256
// (column-major, as the executor calls them):
//   forward  Y^T += W^T(op T) * S^T     m=256  n=8192 k=256   (T, N)
//   dW       dW^T = S^T * U(op T)       m=256  n=256  k=8192  (N, T)
//   T        T^T  = W^T * U^T           m=256  n=8192 k=256   (N, N)
// Usage: gemm_algos [width] [batch] [workspace MiB, default 4 as the executor]
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void ok(cudaError_t e) { if (e != cudaSuccess) { std::printf("cuda %s\n", cudaGetErrorString(e)); std::exit(1); } }
void ok(cublasStatus_t e) { if (e != CUBLAS_STATUS_SUCCESS) { std::printf("cublas %d\n", int(e)); std::exit(1); } }
template<class F> float time_ms(F f, int reps = 50) {
    cudaEvent_t a, b; ok(cudaEventCreate(&a)); ok(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) f();
    ok(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) f();
    ok(cudaEventRecord(b)); ok(cudaEventSynchronize(b));
    float ms = 0; ok(cudaEventElapsedTime(&ms, a, b));
    return ms/reps;
}
}

int main(int argc, char** argv) {
    const long width = argc > 1 ? std::atol(argv[1]) : 256, batch = argc > 2 ? std::atol(argv[2]) : 8192;
    struct Shape { const char* name; cublasOperation_t ta, tb; long m, n, k, lda, ldb, ldc; float beta; };
    const Shape shapes[] = {
        {"forward", CUBLAS_OP_T, CUBLAS_OP_N, width, batch, width, width, width, width, 1.0f},
        {"dW", CUBLAS_OP_N, CUBLAS_OP_T, width, width, batch, width, width, width, 0.0f},
        {"T", CUBLAS_OP_N, CUBLAS_OP_N, width, batch, width, width, width, width, 0.0f},
    };
    float *a, *b, *c, *workspace;
    const std::size_t big = std::size_t(width)*batch, bytes = std::size_t(argc > 3 ? std::atol(argv[3]) : 4) << 20;
    ok(cudaMalloc(&a, big*4)); ok(cudaMalloc(&b, big*4)); ok(cudaMalloc(&c, big*4)); ok(cudaMalloc(&workspace, bytes));
    ok(cudaMemset(a, 0, big*4)); ok(cudaMemset(b, 0, big*4)); ok(cudaMemset(c, 0, big*4));
    cublasHandle_t h; ok(cublasCreate(&h));
    ok(cublasSetWorkspace(h, workspace, bytes)); // as the executor: its own workspace of `bytes`
    cublasLtHandle_t lt; ok(cublasLtCreate(&lt));
    const float one = 1;
    std::printf("width %ld batch %ld workspace %zu MiB\n", width, batch, bytes >> 20);
    for (const auto& s : shapes) {
        const double gflop = 2.0*s.m*s.n*s.k*1e-9;
        const float ms = time_ms([&] { ok(cublasSgemm(h, s.ta, s.tb, s.m, s.n, s.k, &one, a, s.lda, b, s.ldb, &s.beta, c, s.ldc)); });
        std::printf("%-8s cublasSgemm      %8.1f us  %5.1f TFLOPS\n", s.name, ms*1e3, gflop/ms);
        cublasLtMatmulDesc_t op; ok(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        ok(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &s.ta, sizeof s.ta));
        ok(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &s.tb, sizeof s.tb));
        cublasLtMatrixLayout_t la, lb, lc;
        const long ar = s.ta == CUBLAS_OP_N ? s.m : s.k, ac = s.ta == CUBLAS_OP_N ? s.k : s.m;
        const long br = s.tb == CUBLAS_OP_N ? s.k : s.n, bc = s.tb == CUBLAS_OP_N ? s.n : s.k;
        ok(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, ar, ac, s.lda));
        ok(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, br, bc, s.ldb));
        ok(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, s.m, s.n, s.ldc));
        cublasLtMatmulPreference_t pref; ok(cublasLtMatmulPreferenceCreate(&pref));
        std::size_t ws = bytes;
        ok(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
        std::vector<cublasLtMatmulHeuristicResult_t> results(32);
        int found = 0;
        ok(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 32, results.data(), &found));
        float best = 1e9; int best_index = -1;
        for (int i = 0; i < found; ++i) {
            const float t = time_ms([&] {
                ok(cublasLtMatmul(lt, op, &one, a, la, b, lb, &s.beta, c, lc, c, lc, &results[i].algo, workspace, bytes, nullptr));
            });
            if (t < best) { best = t; best_index = i; }
        }
        std::printf("%-8s cuBLASLt best   %8.1f us  %5.1f TFLOPS  (heuristic #%d of %d; #0 %s)\n", s.name, best*1e3,
                    gflop/best, best_index, found, best_index == 0 ? "is best" : "is not best");
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulDescDestroy(op);
    }
}

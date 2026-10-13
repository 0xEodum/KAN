// Port diagnostic: cuBLAS DGEMM/SGEMM/TF32 throughput and device copy bandwidth.
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
template<class T> double gemm_tflops(cublasHandle_t h, int n, int reps) {
    T *a, *b, *c; cudaMalloc(&a, sizeof(T)*n*n); cudaMalloc(&b, sizeof(T)*n*n); cudaMalloc(&c, sizeof(T)*n*n);
    cudaMemset(a, 0, sizeof(T)*n*n); cudaMemset(b, 0, sizeof(T)*n*n);
    T one = 1, zero = 0; cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto run = [&] { if constexpr (sizeof(T) == 8) cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, (const double*)&one, (const double*)a, n, (const double*)b, n, (const double*)&zero, (double*)c, n);
                     else cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, (const float*)&one, (const float*)a, n, (const float*)b, n, (const float*)&zero, (float*)c, n); };
    run(); cudaDeviceSynchronize(); cudaEventRecord(e0); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); cudaFree(a); cudaFree(b); cudaFree(c);
    return 2.0*n*double(n)*n*reps/(ms*1e-3)/1e12;
}
int main() {
    cublasHandle_t h; cublasCreate(&h);
    for (int r = 0; r < 3; ++r) {
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
        double d = gemm_tflops<double>(h, 4096, 5), s = gemm_tflops<float>(h, 8192, 10);
        cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
        double t = gemm_tflops<float>(h, 8192, 10);
        size_t bytes = size_t(1) << 30; char *x, *y; cudaMalloc(&x, bytes); cudaMalloc(&y, bytes);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaMemcpy(y, x, bytes, cudaMemcpyDeviceToDevice);
        cudaEventRecord(e0); for (int i = 0; i < 10; ++i) cudaMemcpy(y, x, bytes, cudaMemcpyDeviceToDevice); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); cudaFree(x); cudaFree(y);
        std::printf("run %d: DGEMM 4096 %.3f TFLOP/s, SGEMM 8192 %.2f TFLOP/s, TF32 8192 %.2f TFLOP/s, D2D copy %.1f GB/s (read+write)\n",
                    r, d, s, t, 2.0*bytes*10/(ms*1e-3)/1e9);
    }
}

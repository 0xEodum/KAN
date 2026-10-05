// The three contraction GEMMs of the 256-wide and 1024-wide steps (backlog C1)
// in cuBLAS FP32 (default math) and TF32 tensor-op math, in TFLOPS.
// Build: nvcc -O3 -arch=sm_86 -std=c++20 tf32_gemm.cu -lcublas -o tf32_gemm.exe
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    // Row-major Phi (B x IK), C (O x IK), U (B x O); the resident executor's calls.
    struct S { const char* name; long o, b, ik; } shapes[] = {{"256-wide", 256, 8192, 1792}, {"1024-wide", 1024, 4096, 7168}};
    cublasHandle_t h; cublasCreate(&h);
    float *phi, *c, *u, *y; const long n = 1l<<28;
    cudaMalloc(&phi, 4*n); cudaMalloc(&c, 4*n); cudaMalloc(&u, 4*n); cudaMalloc(&y, 4*n);
    cudaMemset(phi, 0, 4*n); cudaMemset(c, 0, 4*n); cudaMemset(u, 0, 4*n);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::printf("shape,product,math,us,tflops\n");
    for (auto s : shapes)
        for (int math = 0; math < 2; ++math) {
            cublasSetMathMode(h, math ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
            const float one = 1, zero = 0;
            for (int product = 0; product < 3; ++product) {
                auto run = [&] {
                    if (product == 0) cublasSgemm_64(h, CUBLAS_OP_T, CUBLAS_OP_N, s.o, s.b, s.ik, &one, c, s.ik, phi, s.ik, &zero, y, s.o);
                    else if (product == 1) cublasSgemm_64(h, CUBLAS_OP_N, CUBLAS_OP_T, s.ik, s.o, s.b, &one, phi, s.ik, u, s.o, &zero, y, s.ik);
                    else cublasSgemm_64(h, CUBLAS_OP_N, CUBLAS_OP_N, s.ik, s.b, s.o, &one, c, s.ik, u, s.o, &zero, y, s.ik);
                };
                for (int i = 0; i < 3; ++i) run();
                const int reps = 10;
                cudaEventRecord(e0); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= reps;
                const char* names[] = {"forward Y=Phi*C^T", "coefficient dC=U^T*Phi", "input W=U*C"};
                std::printf("%s,%s,%s,%.1f,%.2f\n", s.name, names[product], math ? "tf32" : "fp32", ms*1e3,
                            2.0*s.o*s.b*s.ik/(ms*1e-3)/1e12);
            }
        }
}

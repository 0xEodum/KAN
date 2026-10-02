// Forward contraction Y[b,o] = dot(Phi[b,:], C[o,:]): warp-per-output kernel vs cuBLAS DGEMM.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
__global__ void dot_kernel(const double* v, const double* c, double* y, long batch, long outputs, long ik) {
    const long warp = (static_cast<long>(blockIdx.x)*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    const long warps = static_cast<long>(gridDim.x)*blockDim.x/32;
    for (long index = warp; index < batch*outputs; index += warps) {
        const double* row = v+(index/outputs)*ik; const double* col = c+(index%outputs)*ik;
        double sum = 0;
        for (long k = lane; k < ik; k += 32) sum += row[k]*col[k];
        for (int off = 16; off; off /= 2) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) y[index] = sum;
    }
}
int main() {
    struct S { long o, b, k; } shapes[] = {{24,32,112},{8,32,168},{24,1024,112},{8,1024,168},{64,32,448},{64,1024,448},{32,1024,448},{16,1024,224},{256,1024,1792},{256,8192,1792}};
    cublasHandle_t h; cublasCreate(&h); cudaStream_t st; cudaStreamCreate(&st); cublasSetStream(h, st);
    double *a, *b, *c; cudaMalloc(&a, 8l<<25); cudaMalloc(&b, 8l<<25); cudaMalloc(&c, 8l<<25);
    cudaMemset(a,0,8l<<25); cudaMemset(b,0,8l<<25);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (auto s : shapes) {
        const double one = 1, zero = 0; const int reps = s.o*s.b*s.k > 1e8 ? 10 : 200;
        float t[2];
        for (int which = 0; which < 2; ++which) {
            auto run = [&] {
                if (which == 0) cublasDgemm_64(h, CUBLAS_OP_T, CUBLAS_OP_N, s.o, s.b, s.k, &one, a, s.k, b, s.k, &zero, c, s.o);
                else { long warps = s.o*s.b; unsigned blocks = (unsigned)((warps*32+255)/256 < 65535 ? (warps*32+255)/256 : 65535);
                       dot_kernel<<<blocks, 256, 0, st>>>(b, a, c, s.b, s.o, s.k); }
            };
            for (int i = 0; i < 5; ++i) run();
            cudaEventRecord(e0, st); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t[which], e0, e1); t[which] *= 1000.0f/reps;
        }
        std::printf("O=%4ld B=%5ld IK=%5ld  FMA=%10ld  cublas %9.1f us  warp-dot %9.1f us\n", s.o, s.b, s.k, s.o*s.b*s.k, t[0], t[1]);
    }
}

// FP32 thresholds of the contraction engine (backlog C1): the two small
// kernels of src/resident.cu against cuBLAS SGEMM on the shapes around their
// FP64 thresholds (C2: forward 2^23 multiply-adds, parameter VJP 2^15
// coefficients+outputs). Same kernels as the resident executor, float only.
// Build: nvcc -O3 -arch=sm_86 -std=c++20 small_shapes_f32.cu -lcublas -o small_shapes_f32.exe
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

__global__ void dot_kernel(const float* v, const float* c, float* y, long count, long outputs, long length) {
    const long lane = threadIdx.x%32, warps = static_cast<long>(gridDim.x)*(blockDim.x/32);
    for (long index = (static_cast<long>(blockIdx.x)*blockDim.x+threadIdx.x)/32; index < count; index += warps) {
        const float* row = v+(index/outputs)*length; const float* col = c+(index%outputs)*length;
        float sum = 0;
        for (long k = lane; k < length; k += 32) sum += row[k]*col[k];
        for (int off = 16; off; off /= 2) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) y[index] = sum;
    }
}
__global__ void partial_kernel(const float* v, const float* u, float* partial, long batch, long outputs, long length, long chunk) {
    const long coefficients = outputs*length, count = coefficients+outputs;
    const long begin = static_cast<long>(blockIdx.y)*chunk, end = begin+chunk < batch ? begin+chunk : batch;
    for (long c = static_cast<long>(blockIdx.x)*blockDim.x+threadIdx.x; c < count; c += static_cast<long>(gridDim.x)*blockDim.x) {
        float sum = 0;
        if (c < coefficients) { const long o = c/length, ik = c%length; for (long r = begin; r < end; ++r) sum += u[r*outputs+o]*v[r*length+ik]; }
        else for (long r = begin; r < end; ++r) sum += u[r*outputs+c-coefficients];
        partial[blockIdx.y*count+c] = sum;
    }
}
__global__ void finish_kernel(const float* partial, float* g, long count, unsigned tiles) {
    for (long q = static_cast<long>(blockIdx.x)*blockDim.x+threadIdx.x; q < count; q += static_cast<long>(gridDim.x)*blockDim.x) {
        float sum = 0; for (unsigned t = 0; t < tiles; ++t) sum += partial[t*count+q]; g[q] = sum;
    }
}
unsigned grid(long count, long per) { return static_cast<unsigned>(std::min<long>((count-1)/per+1, 65535)); }

int main() {
    struct S { long o, b, ik; } shapes[] = {{24,32,112},{8,32,168},{24,1024,112},{8,1024,168},{64,32,448},{16,1024,224},
                                            {32,1024,448},{64,1024,448},{80,400,448},{64,4096,448},{128,1024,896},
                                            {256,1024,1792},{10,8192,1792},{256,8192,1792}};
    cublasHandle_t h; cublasCreate(&h); cudaStream_t st; cudaStreamCreate(&st); cublasSetStream(h, st);
    float *a, *b, *c, *p, *ones; const long n = 1l<<26;
    cudaMalloc(&a, 4*n); cudaMalloc(&b, 4*n); cudaMalloc(&c, 4*n); cudaMalloc(&p, 4*n); cudaMalloc(&ones, 4*8192);
    cudaMemset(a,0,4*n); cudaMemset(b,0,4*n); cudaMemset(ones,0,4*8192);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::printf("O,B,IK,forward_fma,sgemm_fwd_us,dot_us,param_count,sgemm_param_us,tiled_param_us\n");
    for (auto s : shapes) {
        const float one = 1, zero = 0; const int reps = s.o*s.b*s.ik > 1e8 ? 20 : 200;
        float t[4];
        for (int which = 0; which < 4; ++which) {
            auto run = [&] {
                if (which == 0) cublasSgemm_64(h, CUBLAS_OP_T, CUBLAS_OP_N, s.o, s.b, s.ik, &one, a, s.ik, b, s.ik, &zero, c, s.o);
                else if (which == 1) dot_kernel<<<grid(s.o*s.b, 8), 256, 0, st>>>(b, a, c, s.o*s.b, s.o, s.ik);
                else if (which == 2) {
                    cublasSgemm_64(h, CUBLAS_OP_N, CUBLAS_OP_T, s.ik, s.o, s.b, &one, b, s.ik, a, s.o, &zero, c, s.ik);
                    cublasSgemv_64(h, CUBLAS_OP_N, s.o, s.b, &one, a, s.o, ones, 1, &zero, c, 1);
                } else {
                    const long count = s.o*s.ik+s.o; const unsigned tiles = static_cast<unsigned>(std::min<long>(64, (s.b-1)/64+1));
                    const long chunk = (s.b-1)/tiles+1;
                    partial_kernel<<<dim3(grid(count, 256), tiles), 256, 0, st>>>(b, a, p, s.b, s.o, s.ik, chunk);
                    finish_kernel<<<grid(count, 256), 256, 0, st>>>(p, c, count, tiles);
                }
            };
            for (int i = 0; i < 5; ++i) run();
            cudaEventRecord(e0, st); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t[which], e0, e1); t[which] *= 1000.0f/reps;
        }
        std::printf("%ld,%ld,%ld,%ld,%.1f,%.1f,%ld,%.1f,%.1f\n", s.o, s.b, s.ik, s.o*s.b*s.ik, t[0], t[1], s.o*s.ik+s.o, t[2], t[3]);
    }
}

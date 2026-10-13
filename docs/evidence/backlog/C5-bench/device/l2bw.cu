#include <cuda_runtime.h>
#include <cstdio>
__global__ void rd(const float4* p, size_t n, int reps, float* out) {
    float4 acc = {0,0,0,0};
    for (int r = 0; r < reps; ++r)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
            float4 v = p[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    if (acc.x == 12345.f) *out = acc.y + acc.z + acc.w;
}
int main() {
    for (size_t mb : {2, 8, 16, 64, 512}) {
        size_t n = mb * (1 << 20) / 16; float4* p; float* o; cudaMalloc(&p, n * 16); cudaMalloc(&o, 4); cudaMemset(p, 0, n*16);
        int reps = mb <= 16 ? 50 : 4;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        rd<<<48*8, 256>>>(p, n, 1, o);
        cudaEventRecord(a); rd<<<48*8, 256>>>(p, n, reps, o); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("%zu MB: %.1f GB/s\n", mb, (double)n * 16 * reps / ms / 1e6);
        cudaFree(p); cudaFree(o);
    }
}

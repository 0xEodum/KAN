#include <cuda_runtime.h>
#include <cstdio>
template<int V>
__global__ void k(float* out, int iters, int mode) {
    __shared__ float s[8192];
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) s[i] = i * 0.001f;
    __syncthreads();
    float acc[8] = {0};
    int base = (mode == 0 ? threadIdx.x % 32 : (threadIdx.x * 7) % 64);
    for (int it = 0; it < iters; ++it) {
        int off = (it * 64) & 4095;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if constexpr (V == 1) acc[t] += s[base + off + t * 128] * 1.0001f;
            else { float4 v = *reinterpret_cast<float4*>(&s[(base * 4 + off * 4 + t * 512) & 8188]); acc[t] += v.x * v.y + v.z * v.w; }
        }
    }
    float r = 0; for (int t = 0; t < 8; ++t) r += acc[t];
    if (r == 1234.5f) *out = r;
}
int main() {
    float* o; cudaMalloc(&o, 4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int mode : {0, 1}) {
        int iters = 20000, blocks = 48 * 8;
        k<1><<<blocks, 256>>>(o, 10, mode);
        cudaEventRecord(a); k<1><<<blocks, 256>>>(o, iters, mode); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        double loads = (double)blocks * 256 * iters * 8;
        printf("LDS.32 mode %d: %.2f T lane-loads/s = %.1f lane-loads/clk/SM @2.4GHz\n", mode, loads / ms / 1e9, loads / ms / 1e9 * 1e12 / 48 / 2.4e9);
        k<4><<<blocks, 256>>>(o, 10, mode);
        cudaEventRecord(a); k<4><<<blocks, 256>>>(o, iters, mode); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
        printf("LDS.128 mode %d: %.2f T float-loads/s\n", mode, loads * 4 / ms / 1e9);
    }
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0); printf("clock attr %d kHz\n", clk);
}

#include <cuda_runtime.h>
#include <cstdio>
int main() {
    size_t f0, f1, t;
    void* p = nullptr;
    cudaMalloc(&p, 1); cudaFree(p);
    cudaMemGetInfo(&f0, &t);
    for (int i = 0; i < 120; ++i) { cudaMalloc(&p, 8u << 20); cudaMemset(p, 0, 8u << 20); cudaFree(p); }
    cudaDeviceSynchronize();
    cudaMemGetInfo(&f1, &t);
    std::printf("free drop after 120 x (8 MiB malloc+free): %lld MiB\n", (long long)((long long)f0 - (long long)f1) / (1 << 20));
}

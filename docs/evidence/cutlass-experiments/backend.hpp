#pragma once
#include <cuda_runtime.h>

// Research-only hooks. Mode is captured at executor construction from
// KAN_EXPERIMENT_MODE, never read on the GPU or during graph replay.
namespace kan::cuda::experiment {
int mode();
int tile();
void forward(const float* phi, const float* x, const float* c, float* y,
             int batch, int inputs, int outputs, bool virtual_phi, int tile, int* status, cudaStream_t stream);
void coefficient(const float* x, const float* u, const float* c, float* dc,
                 int batch, int inputs, int outputs, float lambda, int tile, int* status, cudaStream_t stream);
void input(const float* x, const float* u, const float* c, float* dx,
           int batch, int inputs, int outputs, int tile, int* status, cudaStream_t stream);
void residual(const float* derivative, const float* u, const float* w, float* dx,
              int batch, int inputs, int outputs, int tile, int* status, cudaStream_t stream);
}

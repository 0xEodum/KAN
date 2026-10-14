#include <cstdio>
#include <cuda_runtime.h>
template<class T> __device__ void report(T v, int* s) { if (!isfinite(v)) atomicOr(s, 1); }
template<class T>
__global__ void column(const T* dx, const T* dw, const T* w, T* partial, std::size_t rows, std::size_t terms, unsigned tiles, int* status) {
    __shared__ T centers[256], widths[256];
    const auto k=static_cast<std::size_t>(blockIdx.x)/tiles; const auto tile=blockIdx.x%tiles, lane=threadIdx.x;
    T center=0,width=0; const auto stride=static_cast<std::size_t>(tiles)*blockDim.x;
    for(auto r=static_cast<std::size_t>(tile)*blockDim.x+lane;r<rows;r+=stride){const T f=w[r*terms+k];center+=f*(-dx[r*terms+k]);width+=f*dw[r*terms+k];}
    report(center,status);report(width,status); centers[lane]=center;widths[lane]=width;__syncthreads();
    for(unsigned step=blockDim.x/2;step;step/=2){if(lane<step){centers[lane]+=centers[lane+step];widths[lane]+=widths[lane+step];}__syncthreads();}
    if(lane==0){partial[k*tiles+tile]=centers[0];partial[(terms+k)*tiles+tile]=widths[0];}
}
int main(){ int n64=0,n32=0,sm=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n64, column<double>, 256, 0); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n32, column<float>, 256, 0);
 cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0); printf("SMs %d, column<double> %d blocks/SM (%d resident), column<float> %d blocks/SM (%d resident)\n", sm, n64, n64*sm, n32, n32*sm); }

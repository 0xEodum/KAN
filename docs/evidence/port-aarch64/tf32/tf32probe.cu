#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <cmath>
int main(){
  for (int n : {16, 64, 256, 1024}) for (int mode=0; mode<2; ++mode) {
  std::vector<float> a(n*n, 1.0f + 3.0f*std::ldexp(1.0f,-12)), b(n*n, 1.0f), c(n*n);
  float *da,*db,*dc; cudaMalloc(&da,n*n*4); cudaMalloc(&db,n*n*4); cudaMalloc(&dc,n*n*4);
  cudaMemcpy(da,a.data(),n*n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,b.data(),n*n*4,cudaMemcpyHostToDevice);
  cublasHandle_t h; cublasCreate(&h); cublasSetMathMode(h, mode? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
  float one=1, zero=0; cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,da,n,db,n,&zero,dc,n);
  cudaMemcpy(c.data(),dc,n*n*4,cudaMemcpyDeviceToHost);
  double per = c[0]/n; printf("n=%d mode=%s c/n=%.9f  (fp32 1.000732422, tf32-round 1.000976562, tf32-trunc 1.0)\n", n, mode?"TF32":"FP32", per);
  cublasDestroy(h); cudaFree(da); cudaFree(db); cudaFree(dc);}
  int v; cublasGetVersion(nullptr,&v); printf("cublas %d\n", v);
}

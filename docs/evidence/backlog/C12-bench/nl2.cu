// Backlog C12 microbenchmark: trainable-RBF center/log-width partial reduction
// (sum over rows r of W[r,k]*(-dPhi/dx[r,k]) and W[r,k]*dPhi/dlogw[r,k]) on
// row-major (rows, terms) tensors, as in src/backends/cuda/resident.cu.
//   column : pre-C12 nonlinear_partial_kernel (block per (term, tile), lane reads column k, stride terms)
//   staged : C1 FP32 nonlinear_staged_kernel (coalesced stage, its own summation order)
//   tile   : C12 candidate (block per (tile, term group); coalesced chunk loads;
//            per-(lane, term) chains and 256-lane trees of `column`, i.e. bitwise `column`)
// Prints time per launch (best of 5 x 20 launches), effective bandwidth of the
// three input tensors and whether the partials are bitwise those of `column`.
// Build: nvcc -O3 -arch=native -std=c++17 --fmad=false nl2.cu -o nl2   (parity build flags; --fmad=true for the FMA build)
// Usage: nl2 <f64|f32> <rows> <terms> [group]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
template<class T> __device__ void report(T value, int* status) { if (!isfinite(value)) atomicOr(status, 1); }
constexpr unsigned stage_threads = 256;
constexpr std::size_t stage_bytes = std::size_t{32} << 10;
template<class T> __device__ T* stage_memory() { extern __shared__ __align__(16) unsigned char raw[]; return reinterpret_cast<T*>(raw); }

template<class T>
__global__ void column(const T* dx, const T* dw, const T* w, T* partial, std::size_t rows, std::size_t terms, unsigned tiles, int* status) {
    __shared__ T centers[256], widths[256];
    const auto k=static_cast<std::size_t>(blockIdx.x)/tiles;
    const auto tile=blockIdx.x%tiles, lane=threadIdx.x;
    T center=0,width=0;
    const auto stride=static_cast<std::size_t>(tiles)*blockDim.x;
    for(auto r=static_cast<std::size_t>(tile)*blockDim.x+lane;r<rows;r+=stride) {
        const T factor=w[r*terms+k];
        center+=factor*(-dx[r*terms+k]);width+=factor*dw[r*terms+k];
    }
    report(center,status);report(width,status);
    centers[lane]=center;widths[lane]=width;__syncthreads();
    for(unsigned step=blockDim.x/2;step;step/=2) {
        if(lane<step) {centers[lane]+=centers[lane+step];widths[lane]+=widths[lane+step];}
        __syncthreads();
    }
    if(lane==0) {
        partial[k*tiles+tile]=centers[0];partial[(terms+k)*tiles+tile]=widths[0];
        report(centers[0],status);report(widths[0],status);
    }
}
template<class T>
__global__ void staged(const T* dx, const T* dw, const T* w, T* partial, std::size_t rows,
                       std::size_t terms, unsigned tiles, unsigned tile_rows, int* status) {
    __shared__ T centers[stage_threads], widths[stage_threads];
    T* stage = stage_memory<T>();
    const auto plane = static_cast<std::size_t>(tile_rows)*terms;
    const auto per = blockDim.x/terms, k = threadIdx.x%terms, lane_row = threadIdx.x/terms;
    const bool active = lane_row < per;
    T center = 0, width = 0;
    for (auto base = static_cast<std::size_t>(blockIdx.x)*tile_rows; base < rows; base += static_cast<std::size_t>(tiles)*tile_rows) {
        const auto count = rows-base < tile_rows ? rows-base : static_cast<std::size_t>(tile_rows);
        const auto length = count*terms, offset = base*terms;
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < length; i += blockDim.x) {
            stage[i] = dx[offset+i]; stage[plane+i] = dw[offset+i]; stage[2*plane+i] = w[offset+i];
        }
        __syncthreads();
        if (active)
            for (std::size_t r = lane_row; r < count; r += per) {
                const auto i = r*terms+k;
                const T factor = stage[2*plane+i];
                center += factor*(-stage[i]); width += factor*stage[plane+i];
            }
        __syncthreads();
    }
    report(center, status); report(width, status);
    centers[threadIdx.x] = center; widths[threadIdx.x] = width;
    __syncthreads();
    if (threadIdx.x < terms) {
        T c = 0, v = 0;
        for (unsigned r = 0; r < per; ++r) { c += centers[r*terms+threadIdx.x]; v += widths[r*terms+threadIdx.x]; }
        partial[threadIdx.x*tiles+blockIdx.x] = c; partial[(terms+threadIdx.x)*tiles+blockIdx.x] = v;
        report(c, status); report(v, status);
    }
}
// Candidate: block (tile, group of `group` terms k0..k0+g). Item i = lane*g + j
// is the chain of `column`'s lane `lane` for term k0+j; chunk loads read g
// consecutive values of 256 consecutive rows (contiguous when g = terms).
template<class T>
__global__ void tile(const T* dx, const T* dw, const T* w, T* partial, std::size_t rows,
                     std::size_t terms, unsigned tiles, unsigned group, int* status) {
    T* centers = stage_memory<T>();
    T* widths = centers+256*static_cast<std::size_t>(group);
    const auto t = blockIdx.x%tiles;
    const auto k0 = static_cast<std::size_t>(blockIdx.x/tiles)*group;
    const auto g = terms-k0 < group ? terms-k0 : static_cast<std::size_t>(group);
    const auto items = 256*g;
    for (auto i = static_cast<std::size_t>(threadIdx.x); i < items; i += blockDim.x) { centers[i] = 0; widths[i] = 0; }
    const auto stride = static_cast<std::size_t>(tiles)*256;
    for (auto base = static_cast<std::size_t>(t)*256; base < rows; base += stride) {
        const auto count = (rows-base < 256 ? rows-base : std::size_t{256})*g;
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < count; i += blockDim.x) {
            const auto r = i/g, o = (base+r)*terms+k0+(i-r*g);
            const T factor = w[o];
            centers[i] += factor*(-dx[o]); widths[i] += factor*dw[o];
        }
    }
    __syncthreads();
    for (auto i = static_cast<std::size_t>(threadIdx.x); i < items; i += blockDim.x) { report(centers[i], status); report(widths[i], status); }
    for (std::size_t step = 128; step; step /= 2) {
        for (auto i = static_cast<std::size_t>(threadIdx.x); i < step*g; i += blockDim.x) {
            centers[i] += centers[i+step*g]; widths[i] += widths[i+step*g];
        }
        __syncthreads();
    }
    for (auto j = static_cast<std::size_t>(threadIdx.x); j < g; j += blockDim.x) {
        partial[(k0+j)*tiles+t] = centers[j]; partial[(terms+k0+j)*tiles+t] = widths[j];
        report(centers[j], status); report(widths[j], status);
    }
}
template<class T> int bench(std::size_t rows, std::size_t terms, unsigned group_arg) {
    const unsigned tiles = static_cast<unsigned>(std::min<std::size_t>(64, rows ? (rows-1)/256+1 : 1));
    const std::size_t n = rows*terms;
    std::vector<T> h(n*3); unsigned long long s = 1;
    for (auto& v : h) { s = s*6364136223846793005ULL+1442695040888963407ULL; v = static_cast<T>((s>>11)*(1.0/9007199254740992.0)-0.5); }
    T *dx, *dw, *w, *p; int* status;
    CK(cudaMalloc(&dx, n*sizeof(T))); CK(cudaMalloc(&dw, n*sizeof(T))); CK(cudaMalloc(&w, n*sizeof(T)));
    CK(cudaMalloc(&p, 2*terms*tiles*sizeof(T))); CK(cudaMalloc(&status, 4));
    CK(cudaMemcpy(dx, h.data(), n*sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, h.data()+n, n*sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w, h.data()+2*n, n*sizeof(T), cudaMemcpyHostToDevice));
    std::vector<T> ref(2*terms*tiles), got(ref.size());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    int optin = 0; CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CK(cudaFuncSetAttribute(tile<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
    auto run = [&](const char* name, auto launch) {
        CK(cudaMemset(p, 0, ref.size()*sizeof(T))); launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), p, got.size()*sizeof(T), cudaMemcpyDeviceToHost));
        const bool same = !std::memcmp(got.data(), ref.data(), got.size()*sizeof(T));
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {
            CK(cudaEventRecord(a)); for (int i = 0; i < 20; ++i) launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            CK(cudaGetLastError()); float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms/20);
        }
        std::printf("%s,%zu,%zu,%s,%.2f,%.1f,%d\n", sizeof(T) == 8 ? "f64" : "f32", rows, terms, name, best*1e3,
                    3.0*n*sizeof(T)/(best*1e-3)/1e9, same ? 1 : 0);
    };
    auto col = [&] { column<T><<<static_cast<unsigned>(terms)*tiles, 256>>>(dx, dw, w, p, rows, terms, tiles, status); };
    col(); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref.data(), p, ref.size()*sizeof(T), cudaMemcpyDeviceToHost));
    run("column", col);
    if (sizeof(T) == 4 && terms <= stage_threads) {
        const auto tile_rows = static_cast<unsigned>(std::min<std::size_t>(stage_threads, stage_bytes/(terms*3*sizeof(T)))/32*32);
        if (tile_rows) run("staged", [&] { staged<T><<<tiles, stage_threads, tile_rows*terms*3*sizeof(T)>>>(dx, dw, w, p, rows, terms, tiles, tile_rows, status); });
    }
    // Default group: the executor's rule (select_reduction): all terms when they
    // fit the opt-in shared memory, otherwise a multiple of 128 bytes of terms.
    const auto max_group = static_cast<std::size_t>(optin)/(2*256*sizeof(T));
    const auto line = 128/sizeof(T);
    const unsigned group = group_arg ? group_arg
                         : static_cast<unsigned>(terms <= max_group ? terms : max_group >= line ? max_group/line*line : max_group);
    if (group > max_group) { std::printf("group %u exceeds the shared-memory limit %zu\n", group, max_group); return 1; }
    const unsigned grid = static_cast<unsigned>((terms+group-1)/group)*tiles;
    char name[32]; std::snprintf(name, sizeof name, "tile-g%u", group);
    run(name, [&] { tile<T><<<grid, 256, 2*256*group*sizeof(T)>>>(dx, dw, w, p, rows, terms, tiles, group, status); });
    return 0;
}
int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: nl2 f64|f32 rows terms [group]\n"); return 2; }
    const std::size_t rows = std::atol(argv[2]), terms = std::atol(argv[3]);
    const unsigned group = argc > 4 ? std::atoi(argv[4]) : 0;
    return std::string(argv[1]) == "f32" ? bench<float>(rows, terms, group) : bench<double>(rows, terms, group);
}

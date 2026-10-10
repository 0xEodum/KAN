// CUTLASS 3.9.2 SIMT mainloops with KAN operand/epilogue extensions.
// These functions are linked only by KAN_CUTLASS_EXPERIMENT. No atomics are
// used for numerical reductions: each (sample,input) has exactly one owner.
#include "backend.hpp"
#include "detail/basis_formulas.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/epilogue/threadblock/default_epilogue_simt.h>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

namespace kan::cuda::experiment {
namespace {
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Op = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int terms = 7;
void checked(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Guard {
    int* status;
    __device__ float operator()(float v) const { if (!isfinite(v)) atomicOr(status, 1); return v; }
};
__device__ void basis(float x, float* v, float* d, int* status) {
    detail::BasisViewOf<float> view{};
    view.kind = detail::BasisKind::Chebyshev; view.terms = terms;
    detail::basis_terms_for<detail::BasisKind::Chebyshev>(view, x, detail::BasisRowOf<float>{v,d,nullptr,nullptr}, Guard{status});
}

// A virtual row-major [batch,inputs*7] matrix. CUTLASS owns warp tiling,
// register fragments and shared-memory pipeline; this iterator supplies
// Chebyshev values in the existing fragment's expected pitch-linear order.
template<class Base>
struct VirtualPhi {
    using Element = typename Base::Element;
    using Layout = typename Base::Layout;
    using TensorRef = typename Base::TensorRef;
    using ThreadMap = typename Base::ThreadMap;
    using Fragment = typename Base::Fragment;
    using Shape = typename Base::Shape;
    struct Params : Base::Params {
        int inputs = 0; int* status = nullptr;
        Params() = default;
        CUTLASS_HOST_DEVICE Params(Row layout) : Base::Params(layout), inputs(int(layout.stride(0))/terms) {}
    };
    const float* x; Params params; cutlass::MatrixCoord extent, origin;
    bool enabled = true, residue_tile = true;
    int residue_size, residue_end;
    __device__ VirtualPhi(Params p, const float* pointer, cutlass::MatrixCoord e, int thread,
                         cutlass::MatrixCoord offset, const int* = nullptr)
        : x(pointer), params(p), extent(e), origin(offset) {
        constexpr int step = Base::kAdvanceRank == 0 ? Shape::kRow : Shape::kColumn;
        const int limit = Base::kAdvanceRank == 0 ? e.row() : e.column();
        residue_size = limit%step; if (!residue_size) residue_size = step;
        residue_end = (Base::kAdvanceRank == 0 ? offset.row() : offset.column())+residue_size;
        auto t = ThreadMap::initial_offset(thread);
        origin += cutlass::MatrixCoord(t.strided(), t.contiguous());
    }
    __device__ VirtualPhi& operator++() {
        if constexpr (Base::kAdvanceRank == 0) origin.row() += residue_tile ? residue_size : Shape::kRow;
        else origin.column() += residue_tile ? residue_size : Shape::kColumn;
        residue_tile = false;
        return *this;
    }
    __device__ void clear_mask(bool clear = true) { if (clear) enabled = false; }
    __device__ void load(Fragment& f) {
        int last_row = -1, last_input = -1;
        float values[terms], derivatives[terms];
        #pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            #pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                #pragma unroll
                for (int a = 0; a < ThreadMap::kElementsPerAccess; ++a) {
                    const int r = origin.row() + s*ThreadMap::Delta::kStrided;
                    const int col = origin.column() + c*ThreadMap::Delta::kContiguous + a;
                    const int at = (s*ThreadMap::Iterations::kContiguous+c)*ThreadMap::kElementsPerAccess+a;
                    float value = 0;
                    const bool residue_valid = !residue_tile || (Base::kAdvanceRank == 0 ? r : col) < residue_end;
                    if (enabled && residue_valid && r < extent.row() && col < extent.column()) {
                        const int in = col/terms;
                        if (r != last_row || in != last_input) {
                            basis(x[static_cast<std::size_t>(r)*params.inputs+in], values, derivatives, params.status);
                            last_row = r; last_input = in;
                        }
                        value = values[col%terms];
                    }
                    f[at] = value;
                }
            }
        }
    }
};

template<int Tile, class LA, class LB>
struct Configuration {
    using Shape = cutlass::gemm::GemmShape<(Tile == 1 ? 128 : 64), (Tile == 2 ? 64 : 128), 8>;
    using Warp = cutlass::gemm::GemmShape<(Tile == 1 ? 64 : 32), (Tile == 2 ? 32 : 64), 8>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemm<float, LA, 1, float, LB, 1, float, Row,
        float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80, Shape, Warp,
        cutlass::gemm::GemmShape<1,1,1>, Op, Swizzle, 2, false, cutlass::arch::OpMultiplyAdd>::GemmKernel;
    using Mma = typename Kernel::Mma;
    using Epi = typename Kernel::Epilogue;
};

template<class Iterator, class Layout>
__device__ auto iterator_params(Layout layout, int* status) {
    typename Iterator::Params p(layout);
    if constexpr (requires { p.status; }) p.status = status;
    return p;
}

template<class Config, bool VirtualA, bool VirtualB>
__global__ void contract_kernel(const float* a, const float* b, const float* source, float* destination,
                                int m, int n, int k, int lda, int ldb, int ldc, float beta, int* status) {
    using Base = typename Config::Mma;
    using A = std::conditional_t<VirtualA, VirtualPhi<typename Base::IteratorA>, typename Base::IteratorA>;
    using B = std::conditional_t<VirtualB, VirtualPhi<typename Base::IteratorB>, typename Base::IteratorB>;
    using Mma = cutlass::gemm::threadblock::MmaPipelined<typename Base::Shape, A, typename Base::SmemIteratorA,
        B, typename Base::SmemIteratorB, float, Row, typename Base::Policy>;
    using Epi = typename Config::Epi;
    extern __shared__ __align__(16) unsigned char memory[];
    const int warp = cutlass::canonical_warp_idx_sync(), lane = threadIdx.x%32;
    A ia(iterator_params<A>(typename A::Layout(lda), status), const_cast<float*>(a), {m,k}, threadIdx.x,
         {int(blockIdx.x)*Config::Shape::kM,0});
    B ib(iterator_params<B>(typename B::Layout(ldb), status), const_cast<float*>(b), {k,n}, threadIdx.x,
         {0,int(blockIdx.y)*Config::Shape::kN});
    Mma mma(*reinterpret_cast<typename Mma::SharedStorage*>(memory), threadIdx.x, warp, lane);
    typename Mma::FragmentC accum; accum.clear();
    mma((k+Config::Shape::kK-1)/Config::Shape::kK, accum, ia, ib, accum);
    __syncthreads();
    using Out = typename Epi::OutputTileIterator;
    const cutlass::MatrixCoord offset(int(blockIdx.x)*Config::Shape::kM, int(blockIdx.y)*Config::Shape::kN);
    Out out(typename Out::Params(Row(ldc)), destination, {m,n}, threadIdx.x, offset);
    Out src(typename Out::Params(Row(ldc)), const_cast<float*>(source), {m,n}, threadIdx.x, offset);
    Epi epi(*reinterpret_cast<typename Epi::SharedStorage*>(memory), threadIdx.x, warp, lane);
    epi(Op(typename Op::Params(1.0f,beta)), out, accum, src);
}

// Reuse CUTLASS's accumulator-to-shared-memory epilogue mapping, using its
// generic CUDA store option rather than st.global. The following reduction
// consumes this tile in the same CTA; the full U*C/U*W result never reaches
// global memory. C3 tiles overlap by the unused tail (<7 columns), but each
// tile owns whole inputs, so no atomic sum or cross-CTA synchronization.
template<class Config, bool Residual>
__global__ void input_kernel(const float* x_or_derivative, const float* u, const float* c, float* dx,
                             int batch, int inputs, int outputs, int* status) {
    using Mma = typename Config::Mma;
    using Default = cutlass::epilogue::threadblock::DefaultEpilogueSimt<typename Config::Shape,
        typename Mma::Operator, Op, 1>;
    using Out = cutlass::epilogue::threadblock::PredicatedTileIterator<typename Default::OutputTileThreadMap,
        float, false, cutlass::layout::NoPermute, true>;
    using Epi = cutlass::epilogue::threadblock::Epilogue<typename Config::Shape, typename Mma::Operator, 1,
        Out, typename Default::AccumulatorFragmentIterator, typename Default::WarpTileIterator,
        typename Default::SharedLoadIterator, Op, typename Default::Padding>;
    constexpr int N = Config::Shape::kN, M = Config::Shape::kM;
    constexpr int per_input = Residual ? 1 : terms;
    constexpr int owned_inputs = N/per_input;
    constexpr int loop_bytes = sizeof(typename Mma::SharedStorage), epi_bytes = sizeof(typename Epi::SharedStorage);
    constexpr int shared_base = ((loop_bytes > epi_bytes ? loop_bytes : epi_bytes)+15)/16*16;
    extern __shared__ __align__(16) unsigned char memory[];
    float* tile = reinterpret_cast<float*>(memory+shared_base);
    const int row0 = int(blockIdx.x)*M, input0 = int(blockIdx.y)*owned_inputs;
    const int col0 = input0*per_input, length = inputs*per_input;
    const int warp = cutlass::canonical_warp_idx_sync(), lane = threadIdx.x%32;
    typename Mma::IteratorA ia(typename Mma::IteratorA::Params(Row(outputs)), const_cast<float*>(u), {batch,outputs},
                              threadIdx.x, {row0,0});
    typename Mma::IteratorB ib(typename Mma::IteratorB::Params(Row(length)), const_cast<float*>(c), {outputs,length},
                              threadIdx.x, {0,col0});
    Mma mma(*reinterpret_cast<typename Mma::SharedStorage*>(memory), threadIdx.x, warp, lane);
    typename Mma::FragmentC accum; accum.clear();
    mma((outputs+Config::Shape::kK-1)/Config::Shape::kK, accum, ia, ib, accum);
    __syncthreads();
    Out out(typename Out::Params(Row(N)), tile, {M,N}, threadIdx.x, {0,0});
    Epi epi(*reinterpret_cast<typename Epi::SharedStorage*>(memory), threadIdx.x, warp, lane);
    epi(Op(typename Op::Params(1.0f,0.0f)), out, accum, out);
    __syncthreads();
    for (int q = threadIdx.x; q < M*owned_inputs; q += blockDim.x) {
        const int row = q/owned_inputs, in = q%owned_inputs;
        if (row0+row >= batch || input0+in >= inputs) continue;
        const auto at = static_cast<std::size_t>(row0+row)*inputs+input0+in;
        float value;
        if constexpr (Residual) value = dx[at]+x_or_derivative[at]*tile[row*N+in];
        else {
            float v[terms], d[terms]; basis(x_or_derivative[at], v, d, status);
            value = 0;
            #pragma unroll
            for (int t = 0; t < terms; ++t) value += d[t]*tile[row*N+in*terms+t];
        }
        dx[at] = Guard{status}(value);
    }
}

template<int Tile, bool VirtualA, bool VirtualB, class LA, class LB>
void contract(const float* a, const float* b, const float* src, float* dst,
              int m, int n, int k, int lda, int ldb, float beta, int* status, cudaStream_t stream) {
    using C = Configuration<Tile,LA,LB>;
    using Base = typename C::Mma;
    constexpr int shared = std::max(sizeof(typename Base::SharedStorage),sizeof(typename C::Epi::SharedStorage));
    constexpr int threads = Base::WarpCount::kCount*32;
    contract_kernel<C,VirtualA,VirtualB><<<dim3((m+C::Shape::kM-1)/C::Shape::kM,(n+C::Shape::kN-1)/C::Shape::kN),
        threads, shared, stream>>>(a,b,src,dst,m,n,k,lda,ldb,n,beta,status);
    checked(cudaGetLastError());
}
template<int Tile, bool Residual>
void input_launch(const float* x, const float* u, const float* c, float* dx,
                  int batch, int inputs, int outputs, int* status, cudaStream_t stream) {
    using C = Configuration<Tile,Row,Row>;
    using Mma = typename C::Mma;
    constexpr int shared = (std::max(sizeof(typename Mma::SharedStorage),sizeof(typename C::Epi::SharedStorage))+15)/16*16
                         + C::Shape::kM*C::Shape::kN*sizeof(float);
    if constexpr (shared > 48*1024)
        checked(cudaFuncSetAttribute(input_kernel<C,Residual>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    constexpr int per_tile = C::Shape::kN/(Residual?1:terms);
    input_kernel<C,Residual><<<dim3((batch+C::Shape::kM-1)/C::Shape::kM,(inputs+per_tile-1)/per_tile),
        Mma::WarpCount::kCount*32,shared,stream>>>(x,u,c,dx,batch,inputs,outputs,status);
    checked(cudaGetLastError());
}
template<class F> void dispatch(int tile, F f) {
    if (tile == 0) f(std::integral_constant<int,0>{});
    else if (tile == 1) f(std::integral_constant<int,1>{});
    else f(std::integral_constant<int,2>{});
}
}
int mode() {
    const char* value = std::getenv("KAN_EXPERIMENT_MODE");
    const int result = value ? std::atoi(value) : 0;
    if (result < 0 || result > 5) throw std::invalid_argument("KAN_EXPERIMENT_MODE must be 0..5");
    return result;
}
int tile() {
    const char* value = std::getenv("KAN_EXPERIMENT_TILE");
    const int result = value ? std::atoi(value) : 0;
    if (result < 0 || result > 2) throw std::invalid_argument("KAN_EXPERIMENT_TILE must be 0..2");
    return result;
}
void forward(const float* phi,const float* x,const float* c,float* y,int batch,int inputs,int outputs,
             bool virtual_phi,int tile,int* status,cudaStream_t stream) {
    dispatch(tile,[&](auto t){
        if (virtual_phi) contract<t,true,false,Row,Col>(x,c,y,y,batch,outputs,inputs*terms,inputs*terms,inputs*terms,0,status,stream);
        else contract<t,false,false,Row,Col>(phi,c,y,y,batch,outputs,inputs*terms,inputs*terms,inputs*terms,0,status,stream);
    });
}
void coefficient(const float* x,const float* u,const float* c,float* dc,int batch,int inputs,int outputs,
                 float lambda,int tile,int* status,cudaStream_t stream) {
    dispatch(tile,[&](auto t){contract<t,false,true,Col,Row>(u,x,c,dc,outputs,inputs*terms,batch,outputs,inputs*terms,lambda,status,stream);});
}
void input(const float* x,const float* u,const float* c,float* dx,int batch,int inputs,int outputs,
           int tile,int* status,cudaStream_t stream) {
    dispatch(tile,[&](auto t){input_launch<t,false>(x,u,c,dx,batch,inputs,outputs,status,stream);});
}
void residual(const float* derivative,const float* u,const float* w,float* dx,int batch,int inputs,int outputs,
              int tile,int* status,cudaStream_t stream) {
    dispatch(tile,[&](auto t){input_launch<t,true>(derivative,u,w,dx,batch,inputs,outputs,status,stream);});
}
}

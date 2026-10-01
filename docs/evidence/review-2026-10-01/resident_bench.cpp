#include "kan/resident.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
using namespace kan;
int main(){
  struct C{std::vector<std::size_t> w; std::size_t B;};
  for (auto c : {C{{64,64,32,16},1024}, C{{256,256,256,10},8192}, C{{1024,1024,1024},4096}}) {
    std::vector<Layer> ls; std::size_t K=7;
    for (std::size_t j=0;j+1<c.w.size();++j){ BasisConfig b; b.kind=BasisKind::Chebyshev; b.size=K;
      Layer l(c.w[j],c.w[j+1],b); std::vector<double> co(l.coefficients().size()), bi(l.bias().size());
      for(std::size_t q=0;q<co.size();++q) co[q]=0.1*std::sin(q*0.37)/std::sqrt(double(c.w[j]*K));
      l.set_parameters(co,bi); ls.push_back(std::move(l)); }
    Network n(std::move(ls)); cuda::ResidentNetwork r(n,c.B);
    std::vector<double> x(c.B*c.w.front()), u(c.B*c.w.back());
    for(std::size_t q=0;q<x.size();++q) x[q]=std::sin(q*0.11)*0.9;
    for(std::size_t q=0;q<u.size();++q) u[q]=std::cos(q*0.07)/c.B;
    r.upload_input(x,c.B); r.upload_output_gradient(u);
    auto step=[&]{ r.forward(); r.backward(); r.sgd(1e-6); };
    for(int i=0;i<2;++i) step();
    int reps = c.w[0]>=1024 ? 3 : 10;
    auto t=std::chrono::steady_clock::now(); for(int i=0;i<reps;++i) step();
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/reps;
    std::printf("%zu-wide B=%zu resident fp64: %.3f ms/step\n", c.w[0], c.B, ms); std::fflush(stdout);
  }
}

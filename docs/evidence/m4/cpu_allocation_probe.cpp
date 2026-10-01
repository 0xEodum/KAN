// Frozen complete-call CPU fixture. Compile against baseline and final kan.lib.
// Usage: executable <warmups> <repeats> <binary-numerical-snapshot-path>
// The snapshot includes every final output/VJP and learned parameter, in order.
#include "kan/network.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <vector>
namespace { bool counting=false;std::size_t allocations=0,bytes=0; }
void* operator new(std::size_t size) {if(counting){++allocations;bytes+=size;}if(auto p=std::malloc(size?size:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept {std::free(p);}
void operator delete(void* p,std::size_t)noexcept {std::free(p);}
void dump(std::ofstream& stream,std::span<const double> values){
    const auto count=static_cast<std::uint64_t>(values.size());stream.write(reinterpret_cast<const char*>(&count),sizeof(count));
    stream.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size_bytes()));
}
int main(int argc,char** argv){
    if(argc!=4)return 2;const int warmups=std::stoi(argv[1]),repeats=std::stoi(argv[2]);
    if(warmups<0||repeats<1)return 2;
    kan::RationalConfig c;c.numerator_degree=6;c.denominator_degree=4;c.center=0.1;c.scale=1.2;
    const std::vector<std::size_t>w{64,64,32,16};std::vector<kan::Layer>layers;
    for(std::size_t k=1;k<w.size();++k){kan::Layer l(w[k-1],w[k],c);std::vector<double>a(l.coefficients().size()),b(l.denominators().size()),v(l.outputs());
        for(std::size_t j=0;j<a.size();++j)a[j]=.02*std::sin(double((j+1)*(k+1)))/(double(l.inputs())*double(1+j%(c.numerator_degree+1)));
        for(std::size_t j=0;j<b.size();++j)b[j]=.01*std::cos(double((j+3)*(k+1)));
        for(std::size_t j=0;j<v.size();++j)v[j]=.01*std::cos(double(j+k));l.set_rational_parameters(a,b,v);layers.push_back(std::move(l));}
    kan::Network n(std::move(layers));constexpr std::size_t batch=1024;
    std::vector<double>x(batch*64),up(batch*16);for(std::size_t j=0;j<x.size();++j)x[j]=.75*std::sin(double((j+1)%997)*.071);for(std::size_t j=0;j<up.size();++j)up[j]=(.02/batch)*std::sin(double((j+23)%997)*.071);
    std::vector<double> y;kan::NetworkGradients g;
    std::cout<<std::setprecision(17)<<"step,measured,full_call_ms,allocations,allocated_bytes\n";
    for(int step=0;step<warmups+repeats;++step){allocations=0;bytes=0;counting=true;
        auto start=std::chrono::steady_clock::now();y=n.forward(x,batch);g=n.backward(x,batch,up);n.sgd(g,.001);
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();counting=false;
        std::cout<<step<<','<<(step>=warmups)<<','<<elapsed<<','<<allocations<<','<<bytes<<'\n'<<std::flush;
    }
    std::ofstream snapshot(argv[3],std::ios::binary);if(!snapshot)return 3;
    dump(snapshot,y);dump(snapshot,g.input);
    for(std::size_t k=0;k<n.layers().size();++k){const auto& grad=g.layers[k];const auto& l=n.layers()[k];
        dump(snapshot,grad.input);dump(snapshot,grad.coefficients);dump(snapshot,grad.bias);dump(snapshot,grad.denominators);
        dump(snapshot,l.coefficients());dump(snapshot,l.bias());dump(snapshot,l.denominators());}
    return snapshot?0:4;
}

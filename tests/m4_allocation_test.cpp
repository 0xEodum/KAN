#include "kan/layer.hpp"
#include "support/test.hpp"
#include <cstdlib>
#include <new>
#include <iostream>

namespace { bool counting=false; std::size_t allocation_count=0; }
void* operator new(std::size_t size) {
    if(counting)++allocation_count;
    if(auto* p=std::malloc(size ? size : 1))return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}

TEST(complete_rational_layer_call_has_no_per_edge_heap_allocations) {
    kan::RationalConfig config;config.numerator_degree=6;config.denominator_degree=4;
    bool bounded=true;
    for(std::size_t width:{1u,8u,32u})for(std::size_t batch:{1u,64u}) {
        kan::Layer layer(width,width,config);
        std::vector<double> a(layer.coefficients().size(),0.01),b(layer.denominators().size(),0.01),bias(width);
        layer.set_rational_parameters(a,b,bias);
        std::vector<double> x(width*batch,0.2),up(width*batch,0.1);
        allocation_count=0;counting=true;
        auto y=layer.forward(x,batch);auto gradient=layer.backward(x,batch,up);layer.sgd(gradient,0.001);
        counting=false;
        std::cout<<"width="<<width<<" batch="<<batch<<" full_call_allocations="<<allocation_count<<'\n';
        bounded=bounded && allocation_count<256;
        REQUIRE(y.size()==width*batch);
    }
    REQUIRE(bounded);
}
int main(){return test::run();}

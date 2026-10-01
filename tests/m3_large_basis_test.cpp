// Manual resource-intensive regression: current capacity-bound scratch needs
// about 1 GiB device memory (the retained original RED used about 9 GiB).
// Excluded from default CTest so small GPUs retain the ordinary numerical suite.
#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        if(!kan::cuda::available())throw std::runtime_error("actual CUDA hardware required");
        std::size_t free_bytes=0,total_bytes=0;
        if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess || free_bytes<2ULL*1024*1024*1024)
            throw std::runtime_error("large-basis regression requires 2 GiB free GPU memory");
        constexpr std::size_t terms=65535ULL*256/2+1;
        kan::TrainableRbfConfig b{std::vector<double>(terms,1),std::vector<double>(terms,0)};
        kan::Layer layer(1,1,std::move(b));
        layer.set_parameters(std::vector<double>(terms,0.1),std::vector<double>{0});
        kan::cuda::ResidentNetwork gpu(kan::Network({std::move(layer)}),1);
        gpu.upload_input(std::vector<double>{0},1);gpu.upload_output_gradient(std::vector<double>{1});
        gpu.forward();gpu.backward();const auto g=gpu.download_gradients();
        const double expected=0.2*std::exp(-1.0);
        for(std::size_t k=terms-4;k<terms;++k) {
            const auto actual=g.layers[0].log_widths[k];
            if(!std::isfinite(actual)||std::abs(actual-expected)>1e-12) {
                std::cerr<<"FAIL unwritten nonlinear gradient at "<<k<<": actual="<<actual<<" expected="<<expected<<'\n';return 1;
            }
        }
        std::cout<<"PASS large RBF gradient tail, terms="<<terms<<", allocations="<<gpu.workspace_allocations()<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

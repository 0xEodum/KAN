#include <kan/network.hpp>
#ifdef KAN_CONSUMER_CUDA
#include <kan/cuda.hpp>
#include <kan/resident.hpp>
#endif
#include <cmath>
#include <iostream>

int main() {
    kan::BasisConfig basis; basis.size = 3;
    kan::Layer layer(1, 1, basis);
    layer.set_parameters(std::vector<double>{0, 0.7, -0.2}, std::vector<double>{0});
    kan::Network net({layer});
    const auto result = net.forward(std::vector<double>{-1, 0, 1}, 3);
    const std::vector<double> expected{-0.9, 0.2, 0.5};
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::abs(result[i] - expected[i]) > 1e-12) return 1;
#ifdef KAN_CONSUMER_CUDA
    if (!kan::cuda::available()) return 1;
    const auto gpu_result = kan::cuda::forward(layer, std::vector<double>{-1, 0, 1}, 3);
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::abs(gpu_result[i] - expected[i]) > 1e-12) return 1;
    std::cout << "Installed kan::cuda consumer passed\n";
    kan::cuda::ResidentNetwork resident(net, 3);
    resident.upload_input(std::vector<double>{-1, 0, 1}, 3);
    resident.upload_output_gradient(std::vector<double>{0.1, -0.2, 0.1});
    resident.forward();
    const auto resident_result = resident.download_output();
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::abs(resident_result[i] - expected[i]) > 1e-12) return 1;
    resident.backward(); resident.sgd(0.01);
    net.sgd(net.backward(std::vector<double>{-1, 0, 1}, 3,
                         std::vector<double>{0.1, -0.2, 0.1}), 0.01);
    const auto trained = resident.download_parameters().forward(std::vector<double>{-1, 0, 1}, 3);
    const auto reference = net.forward(std::vector<double>{-1, 0, 1}, 3);
    for (std::size_t i = 0; i < reference.size(); ++i)
        if (std::abs(trained[i] - reference[i]) > 1e-12) return 1;
    std::cout << "Installed resident CUDA/SGD consumer passed\n";
#endif
    std::cout << "Installed kan::kan consumer passed\n";
    return 0;
}

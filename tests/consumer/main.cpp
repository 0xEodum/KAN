#include <kan/families.hpp>
#include <kan/network.hpp>
#ifdef KAN_CONSUMER_CUDA
#include <kan/cuda.hpp>
#include <kan/resident.hpp>
#endif
#include <cmath>
#include <iostream>
#include <variant>

int main() {
    kan::ChebyshevConfig basis{3};
    kan::Layer layer(1, 1, basis);
    layer.set_parameters(std::vector<double>{0, 0.7, -0.2}, std::vector<double>{0});
    kan::Network net({layer});
    const auto result = net.forward(std::vector<double>{-1, 0, 1}, 3);
    const std::vector<double> expected{-0.9, 0.2, 0.5};
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::abs(result[i] - expected[i]) > 1e-12) return 1;
    // Explicit input map in front of the same layer: x -> x/100 then T(x).
    kan::Network mapped({kan::InputMap(1, kan::AffineMap{{0.01}, {0}}), layer});
    if (std::abs(mapped.forward(std::vector<double>{100}, 1)[0] - 0.5) > 1e-12) return 1;
    const kan::BSplineConfig spline{3,{0,0,0,0,1,1,1,1}};
    kan::Layer local(1,1,spline);
    local.set_parameters(std::vector<double>{0,0,1.0/3,1},std::vector<double>{0});
    kan::insert_knot(local,0.4);kan::adapt_grid(local,std::vector<double>{0.1,0.2,0.3});
    if(std::abs(local.forward(std::vector<double>{0.5},1)[0]-0.25)>1e-12)return 1;
    const kan::TrainableRbfConfig rbf{{0},{0}};
    kan::Layer nonlinear(1,1,rbf);nonlinear.set_parameters(std::vector<double>{0.2},std::vector<double>{0});
    kan::set_rbf_parameters(nonlinear,std::vector<double>{0.1},std::vector<double>{-0.1});
    nonlinear.sgd(nonlinear.regularization(0.1).gradients,0.01);
    const kan::MexicanHatConfig wave{{0},{1}};
    if(!std::isfinite(kan::evaluate_basis(wave,0).values[0]))return 1;
    kan::RationalConfig rational; rational.numerator_degree=1; rational.denominator_degree=1;
    kan::Layer pade(1,1,rational);
    kan::set_rational_parameters(pade,std::vector<double>{1,0.5},std::vector<double>{-0.5},std::vector<double>{0});
    const std::vector<double> rational_input{-0.4,0,0.7}, rational_upstream{0.1,-0.2,0.3};
    const auto rational_output=pade.forward(rational_input,3);
    for(std::size_t i=0;i<3;++i)
        if(std::abs(rational_output[i]-(1+0.5*rational_input[i])/(1-0.5*rational_input[i]))>1e-12)return 1;
    const auto rational_gradients=pade.backward(rational_input,3,rational_upstream);
    const auto& rational_denominators=std::get<kan::RationalGradients>(rational_gradients.nonlinear).denominators;
    if(rational_denominators.size()!=1)return 1;
#ifdef KAN_CONSUMER_CUDA
    if (!kan::cuda::available()) return 1;
    // The installed legacy API (deprecated since backlog R7) must still link and run.
#if defined(_MSC_VER)
#pragma warning(suppress : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    const auto gpu_result = kan::cuda::forward(layer, std::vector<double>{-1, 0, 1}, 3);
#if !defined(_MSC_VER) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
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
    kan::Network localized({local,nonlinear});
    kan::cuda::ResidentNetwork adaptive(localized,1);
    adaptive.upload_input(std::vector<double>{0.3},1);adaptive.upload_output_gradient(std::vector<double>{0.2});
    adaptive.forward();adaptive.backward(0.01);
    const auto gradients=adaptive.download_gradients();
    if(std::get<kan::TrainableRbfGradients>(std::get<kan::LayerGradients>(gradients.layers[1]).nonlinear).centers.size()!=1)return 1;
    adaptive.sgd(0.01);
    const auto snapshot=adaptive.download_parameters();
    if(std::get<kan::Layer>(snapshot.layers()[1]).carrier()==std::get<kan::Layer>(localized.layers()[1]).carrier())return 1;
    std::cout << "Installed M3 localized/nonlinear CUDA consumer passed\n";
    kan::Network rational_network({pade});
    kan::cuda::ResidentNetwork rational_gpu(rational_network,3);
    rational_gpu.upload_input(rational_input,3);rational_gpu.upload_output_gradient(rational_upstream);
    rational_gpu.forward();rational_gpu.backward();
    if(std::abs(std::get<kan::RationalGradients>(std::get<kan::LayerGradients>(rational_gpu.download_gradients().layers[0]).nonlinear).denominators[0]-rational_denominators[0])>1e-12)return 1;
    rational_gpu.sgd(0.001);rational_network.sgd(rational_network.backward(rational_input,3,rational_upstream),0.001);
    const auto denominator=[](const kan::Layer& l){return std::get<kan::RationalEdges>(l.carrier()).denominators[0];};
    if(std::abs(denominator(std::get<kan::Layer>(rational_gpu.download_parameters().layers()[0]))-denominator(std::get<kan::Layer>(rational_network.layers()[0])))>1e-12)return 1;
    std::cout << "Installed M4 rational CUDA consumer passed\n";
#endif
    std::cout << "Installed kan::kan consumer passed\n";
    return 0;
}

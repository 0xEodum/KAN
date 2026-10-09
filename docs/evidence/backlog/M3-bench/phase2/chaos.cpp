// M3 phase 2: build with cl_tool.cmd against a CUDA build; output in chaos.log.
// Sensitivity of the NoiseInit + branch trajectory: CPU vs CPU with one weight
// perturbed by 1 ulp, and CPU vs FP64 resident, at several learning rates.
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    std::vector<double> x, target;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            const double a = -0.9 + 1.8*i/7, b = -0.9 + 1.8*j/7;
            x.insert(x.end(), {a, b}); target.push_back(a*b);
        }
    auto make = [] {
        std::vector<kan::NetworkLayer> stages;
        std::size_t in = 2;
        for (int l = 0; l <= 4; ++l) {
            const std::size_t out = l == 4 ? 1 : 8;
            kan::Layer layer(in, out, kan::ChebyshevConfig{5});
            layer.set_residual(kan::SiluResidual{std::vector<double>(in*out, 0.0)});
            stages.emplace_back(kan::InputMap(in, kan::TanhMap{1.0}));
            stages.emplace_back(std::move(layer));
            in = out;
        }
        kan::Network n(std::move(stages));
        kan::initialize(n, kan::NoiseInit{0.3, kan::Distribution::Uniform, 1, {}});
        return n;
    };
    for (double rate : {0.03, 0.01}) {
        auto a = make(), b = make();
        {   // perturb one residual weight of b by one ulp
            std::vector<kan::NetworkLayer> layers(b.layers().begin(), b.layers().end());
            auto& l = std::get<kan::Layer>(layers[1]);
            auto w = l.residual()->weights; w[0] = std::nextafter(w[0], 1.0);
            l.set_residual(kan::SiluResidual{w});
            b = kan::Network(layers);
        }
        kan::cuda::ResidentNetwork gpu(a, 64);
        gpu.upload_input(x, 64); gpu.upload_target(target);
        for (int e = 0; e <= 1000; ++e) {
            auto loss = [&](kan::Network& n, bool step) {
                const auto y = n.forward(x, 64);
                std::vector<double> u(64); double s = 0;
                for (int k = 0; k < 64; ++k) { u[k] = 2*(y[k]-target[k])/64.0; s += (y[k]-target[k])*(y[k]-target[k]); }
                if (step) n.sgd(n.backward(x, 64, u), rate);
                return s/64;
            };
            const double la = loss(a, true), lb = loss(b, true);
            gpu.train_step(rate, 0.0, kan::cuda::Loss::MeanSquaredError);
            const double lg = gpu.download_loss();
            if (e % 50 == 0)
                std::printf("rate %.2f epoch %4d cpu %.12e  ulp-perturbed rel %.2e  gpu rel %.2e\n", rate, e, la,
                            std::abs(lb-la)/la, std::abs(lg-la)/la);
        }
    }
}

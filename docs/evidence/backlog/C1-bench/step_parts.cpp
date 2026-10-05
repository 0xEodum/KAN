// Wall time of each call of the 1024-wide step (large_step network); argument d = FP64, otherwise FP32.
// Build like large_step.cpp (build_large_step.cmd with this file).
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc, char** argv) {
    const auto precision = argc > 1 && argv[1][0] == 'd' ? kan::cuda::Precision::Float64 : kan::cuda::Precision::Float32;
    const std::size_t widths[] = {1024, 1024, 1024}, batch = 4096, K = 7;
    std::vector<kan::NetworkLayer> layers;
    for (int j = 0; j < 2; ++j) {
        kan::Layer layer(widths[j], widths[j+1], kan::ChebyshevConfig{K});
        std::vector<double> c(layer.coefficients().size()), b(layer.outputs());
        for (std::size_t q = 0; q < c.size(); ++q) c[q] = 0.1*std::sin(q*0.37)/std::sqrt(double(widths[j]*K));
        layer.set_parameters(c, b); layers.emplace_back(std::move(layer));
    }
    kan::cuda::ResidentNetwork r(kan::Network(std::move(layers)), batch, precision);
    std::vector<double> x(batch*1024), u(batch*1024);
    for (std::size_t q = 0; q < x.size(); ++q) { x[q] = std::sin(q*0.11)*0.9; u[q] = std::cos(q*0.07)/batch; }
    r.upload_input(x, batch);
    double t[4] = {1e9, 1e9, 1e9, 1e9};
    auto timed = [](double& best, auto f) {
        const auto s = std::chrono::steady_clock::now(); f();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-s).count());
    };
    for (int i = 0; i < 6; ++i) {
        timed(t[0], [&] { r.forward(); });
        timed(t[1], [&] { r.upload_output_gradient(u); });
        timed(t[2], [&] { r.backward(); });
        timed(t[3], [&] { r.sgd(1e-3); });
    }
    std::printf("forward %.2f  upload %.2f  backward %.2f  sgd %.2f  ms (best of 6)\n", t[0], t[1], t[2], t[3]);
}

// Backlog C13: full train_step time of a network of one basis family (tanh maps between
// layers), for the cost of computing the W = U*C input-VJP GEMM in FP32 instead of TF32
// (Option 1; the diagnostic library switches it with KAN_C13_W_FP32=1).
// Usage: family_step <f32|tf32> <family spec as dx_accuracy> <widths> <batch> [windows=3]
// Prints: precision,family,topology,batch,window,steps,step_ms
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
namespace {
using Clock = std::chrono::steady_clock;
std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out; std::stringstream in(s); std::string item;
    while (std::getline(in, item, c)) out.push_back(item);
    return out;
}
kan::BasisConfig family(const std::string& spec) {
    const auto f = split(spec, ':');
    const auto n = static_cast<std::size_t>(std::stoul(f.at(1)));
    if (f[0] == "cheb") return kan::ChebyshevConfig{n};
    if (f[0] == "legendre") return kan::LegendreConfig{n};
    if (f[0] == "jacobi") return kan::JacobiConfig{n, 0.3, -0.2};
    if (f[0] == "hermitef") return kan::HermiteConfig{n, kan::HermiteNormalization::Function};
    if (f[0] == "fourier") return kan::FourierConfig{n, std::stod(f.at(2))};
    std::vector<double> c(n);
    for (std::size_t k = 0; k < n; ++k) c[k] = -1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(n - 1);
    const double width = std::stod(f.at(2)) * 2.0 / static_cast<double>(n - 1);
    if (f[0] == "gauss") return kan::GaussianRbfConfig{c, width};
    return kan::MexicanHatConfig{c, std::vector<double>(n, width)};
}
}
int main(int argc, char** argv) {
    const std::string prec = argv[1], spec = argv[2];
    std::vector<std::size_t> widths;
    for (const auto& w : split(argv[3], 'x')) widths.push_back(std::stoul(w));
    const std::size_t batch = std::stoul(argv[4]);
    const int windows = argc > 5 ? std::atoi(argv[5]) : 3;
    std::vector<kan::NetworkLayer> layers;
    for (std::size_t j = 0; j + 1 < widths.size(); ++j) {
        if (j) layers.emplace_back(kan::InputMap(widths[j], kan::TanhMap{}));
        kan::Layer layer(widths[j], widths[j + 1], family(spec));
        kan::initialize(layer, kan::VarianceScaling{1.0, kan::Distribution::Uniform, 11 + j});
        layers.emplace_back(std::move(layer));
    }
    const kan::Network network(std::move(layers));
    std::vector<double> x(batch * widths.front()), dy(batch * widths.back());
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = 0.95 * std::sin(0.377 * static_cast<double>(k) + 0.2);
    for (std::size_t k = 0; k < dy.size(); ++k) dy[k] = 1e-3 * std::cos(0.613 * static_cast<double>(k));
    kan::cuda::ResidentNetwork gpu(network, batch, prec == "tf32" ? kan::cuda::Precision::TensorFloat32 : kan::cuda::Precision::Float32);
    gpu.upload_input(x, batch); gpu.upload_output_gradient(dy);
    gpu.train_step(1e-3); gpu.train_step(1e-3); gpu.synchronize();
    auto t = Clock::now(); gpu.train_step(1e-3); gpu.synchronize();
    const double probe = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    const int steps = static_cast<int>(std::clamp(400.0 / std::max(probe, 1e-3), 3.0, 200.0));
    for (int w = 0; w < windows; ++w) {
        t = Clock::now();
        for (int i = 0; i < steps; ++i) gpu.train_step(1e-3);
        gpu.synchronize();
        std::printf("%s,%s,%s,%zu,%d,%d,%.4f\n", prec.c_str(), spec.c_str(), argv[3], batch, w, steps,
                    std::chrono::duration<double, std::milli>(Clock::now() - t).count() / steps);
    }
}

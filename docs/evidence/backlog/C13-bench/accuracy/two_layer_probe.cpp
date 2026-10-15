// C13 probe: the C5 two-layer network (64->40->4, K=48 p=3, K=30 p=degree2) under TF32/FP32,
// per-layer input-gradient errors against the CPU at the executor's parameters, and the
// CPU sensitivity: layer-1 gradient re-evaluated at the GPU's layer-0 outputs.
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
float f32(double v) { return static_cast<float>(v); }
kan::BSplineConfig uniform(std::size_t degree, std::size_t terms) {
    kan::BSplineConfig b{degree, std::vector<double>(degree + 1, -1.0)};
    const auto interior = terms - degree - 1;
    for (std::size_t k = 1; k <= interior; ++k) b.knots.push_back(-1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(interior + 1));
    b.knots.insert(b.knots.end(), degree + 1, 1.0);
    return b;
}
bool c5 = false; // the C5 suite's exact data (no FP32 rounding of x, u on the CPU side)
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale * std::sin(frequency * static_cast<double>(i) + phase);
    return v;
}
kan::Layer seeded(kan::Layer l, double phase) {
    if (c5) {
        const auto scale = 1.0 / std::sqrt(static_cast<double>(l.inputs() * l.terms()));
        l.set_parameters(wave(l.coefficients().size(), 3 * scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
        return l;
    }
    const auto scale = 3.0 / std::sqrt(static_cast<double>(l.inputs() * l.terms()));
    std::vector<double> c(l.coefficients().size()), bias(l.outputs());
    for (std::size_t k = 0; k < c.size(); ++k) c[k] = f32(scale * std::sin(0.731 * static_cast<double>(k) + phase));
    for (std::size_t o = 0; o < bias.size(); ++o) bias[o] = f32(0.05 * std::sin(1.3 * static_cast<double>(o) + phase));
    l.set_parameters(c, bias);
    return l;
}
int main(int argc, char** argv) {
    const std::string prec = argv[1], path = argv[2];
    const std::size_t d2 = std::stoul(argv[3]);
    c5 = argc > 4 && std::string(argv[4]) == "c5";
    const kan::Network n({seeded(kan::Layer(64, 40, uniform(3, 48)), 0.2), seeded(kan::Layer(40, 4, uniform(d2, 30)), 0.6)});
    constexpr std::size_t batch = 1000;
    std::vector<double> x(batch * 64), u(batch * 4);
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = f32(0.9995 * std::sin(0.377 * static_cast<double>(k) + 0.9));
    for (std::size_t k = 0; k < u.size(); ++k) u[k] = f32(0.3 * std::sin(0.17 * static_cast<double>(k)));
    if (c5) {
        x = wave(batch * 64, 1.05, 0.377, 0.9);
        const double special[] = {-1, 1, 0, -0.5, 0.5, -2, 3, 1e3};
        for (std::size_t k = 0; k < x.size(); k += 7) x[k] = special[(k / 7) % 8];
        u = wave(batch * 4, 0.3, 0.17);
    }
    kan::cuda::testing::ResidentTestScope scope(path == "dense" ? kan::cuda::testing::SplinePath::Dense : kan::cuda::testing::SplinePath::Sparse);
    kan::cuda::ResidentNetwork gpu(n, batch, prec == "tf32" ? kan::cuda::Precision::TensorFloat32 : kan::cuda::Precision::Float32);
    gpu.upload_input(x, batch); gpu.forward(); gpu.upload_output_gradient(u); gpu.backward();
    const auto ref = gpu.download_parameters();
    const auto a = gpu.download_gradients();
    const auto e = ref.backward(x, batch, u);
    // Layer 0 output on the GPU: forward a one-layer executor of layer 0 with the same precision.
    const auto& l0 = std::get<kan::Layer>(ref.layers()[0]);
    const auto& l1 = std::get<kan::Layer>(ref.layers()[1]);
    kan::cuda::ResidentNetwork g0(kan::Network({l0}), batch, prec == "tf32" ? kan::cuda::Precision::TensorFloat32 : kan::cuda::Precision::Float32);
    g0.upload_input(x, batch); g0.forward();
    const auto y0_gpu = g0.download_output();
    const auto y0_cpu = kan::Network({l0}).forward(x, batch);
    const auto g1_at_gpu = kan::Network({l1}).backward(y0_gpu, batch, u).input; // CPU layer-1 dx at the GPU's activations
    const auto& a1 = std::get<kan::LayerGradients>(a.layers[1]).input;
    const auto& e1 = std::get<kan::LayerGradients>(e.layers[1]).input;
    double m1 = 0, m1s = 0, my = 0, m0 = 0, s0 = 0;
    std::size_t at = 0;
    for (std::size_t r = 0; r < a1.size(); ++r) { m1 = std::max(m1, std::abs(a1[r] - e1[r])); m1s = std::max(m1s, std::abs(a1[r] - g1_at_gpu[r])); my = std::max(my, std::abs(y0_gpu[r] - y0_cpu[r])); }
    for (std::size_t r = 0; r < a.input.size(); ++r) { if (std::abs(a.input[r] - e.input[r]) > m0) { m0 = std::abs(a.input[r] - e.input[r]); at = r; } s0 = std::max(s0, std::abs(e.input[r])); }
    const auto chain = kan::Network({l0}).backward(x, batch, g1_at_gpu).input; // CPU network dx at the GPU's activations
    double mc = 0, tol = 0;
    for (std::size_t r = 0; r < chain.size(); ++r) mc = std::max(mc, std::abs(a.input[r] - chain[r]));
    for (std::size_t r = 0; r < chain.size(); ++r) tol = std::max(tol, std::abs(a.input[r] - chain[r]) / (1e-2 * std::abs(chain[r]) + 1e-2 * s0));
    std::printf("  network dx err vs CPU at GPU activations=%.3g (C1 TF32 tolerance ratio %.3g)\n", mc, tol);
    std::printf("%s %s degree2=%zu: max|y0 gpu-cpu|=%.3g layer1 dx err vs CPU=%.3g vs CPU-at-GPU-activations=%.3g; network dx err=%.3g at %zu (actual %.4g expected %.4g, max|e| %.3g)\n",
                prec.c_str(), path.c_str(), d2, my, m1, m1s, m0, at, a.input[at], e.input[at], s0);
}

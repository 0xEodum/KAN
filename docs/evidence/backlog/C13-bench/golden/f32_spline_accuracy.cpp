// Backlog C13: the B-spline sections of the C3 FP32 dump (docs/evidence/backlog/C3-bench/f32_dump.cpp,
// family "bspline3": degree 3, knots(3, 5), networks 5->7->3 batch 33 and 96->80->10 batch 700, same
// seeds, inputs and upstream) against the FP64 CPU at the executor's (FP32-rounded) parameters.
// Per tensor: max |err| / max |e| and the C1 FP32 tolerance ratio max err/(2e-4|e| + 5e-5 max|e|).
// Build: scripts/link_harness.sh <build> f32_spline_accuracy.cpp <exe>
#include "kan/resident.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>
using namespace kan;
namespace {
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
std::vector<double> grid(std::size_t count, double low, double high) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = low + (high-low)*static_cast<double>(i)/static_cast<double>(count-1);
    return v;
}
std::vector<double> knots(std::size_t degree, std::size_t interior) {
    std::vector<double> k(degree, -1.0);
    for (double v : grid(interior+2, -1.0, 1.0)) k.push_back(v);
    k.insert(k.end(), degree, 1.0);
    return k;
}
Layer seeded(Layer l, double phase) {
    const auto scale = 1.0/std::sqrt(static_cast<double>(l.inputs()*l.terms()));
    l.set_parameters(wave(l.coefficients().size(), scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
    return l;
}
void report(const char* what, const std::vector<double>& a, const std::vector<double>& e) {
    double scale = 0, err = 0, tol = 0;
    for (double v : e) scale = std::max(scale, std::abs(v));
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::abs(a[i]-e[i]);
        err = std::max(err, d); tol = std::max(tol, d/(2e-4*std::abs(e[i]) + 5e-5*scale + 1e-37));
    }
    std::printf("  %-16s max|err|/max|e| %.3e  FP32 tolerance ratio %.3f\n", what, err/scale, tol);
}
void run(const char* name, const Network& network, std::size_t batch) {
    const auto& first = std::get<Layer>(network.layers().front());
    const auto& last = std::get<Layer>(network.layers().back());
    cuda::ResidentNetwork r(network, batch, cuda::Precision::Float32);
    const auto x = wave(batch*first.inputs(), 0.9, 0.377, 0.2), u = wave(batch*last.outputs(), 0.5, 0.613, 0.4);
    r.upload_input(x, batch);
    r.forward();
    r.upload_output_gradient(u);
    r.backward(1e-3);
    const auto g = r.download_gradients();
    const auto ref = r.download_parameters();
    // The executor sees FP32-rounded x and u.
    std::vector<double> xf(x), uf(u);
    for (auto& v : xf) v = static_cast<float>(v);
    for (auto& v : uf) v = static_cast<float>(v);
    auto e = ref.backward(xf, batch, uf);
    const auto penalty = ref.regularization(1e-3).gradients;
    std::printf("%s batch %zu\n", name, batch);
    report("input", g.input, e.input);
    for (std::size_t j = 0; j < g.layers.size(); ++j) {
        auto ec = std::get<LayerGradients>(e.layers[j]).coefficients;
        const auto& pc = std::get<LayerGradients>(penalty.layers[j]).coefficients;
        for (std::size_t k = 0; k < ec.size(); ++k) ec[k] += pc[k];
        report(("coefficients " + std::to_string(j)).c_str(), std::get<LayerGradients>(g.layers[j]).coefficients, ec);
        report(("bias " + std::to_string(j)).c_str(), std::get<LayerGradients>(g.layers[j]).bias, std::get<LayerGradients>(e.layers[j]).bias);
    }
}
}
int main() {
    const BasisConfig b = BSplineConfig{3, knots(3, 5)};
    run("bspline3 small", Network({seeded(Layer(5, 7, b), 0.1), seeded(Layer(7, 3, b), 0.2)}), 33);
    run("bspline3 wide", Network({seeded(Layer(96, 80, b), 0.3), seeded(Layer(80, 10, b), 0.4)}), 700);
}

// Backlog C3: bitwise dump of the FP32 resident executor for every fixed basis
// family (and the trainable RBF, whose path C3 leaves unchanged). Prints the
// forward output, the gradient with respect to the network input and every
// parameter gradient as exact hex floats, for a small network (warp-kernel
// contractions) and a wide one (cuBLAS contractions), plus a long-row layer
// (terms too long for the staged tile, which keeps stored derivative rows).
// Two builds whose dumps are byte-identical compute the same FP32 results.
// Usage: f32_dump > dump.txt
#include "kan/resident.hpp"
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
struct Family { const char* name; BasisConfig basis; };
std::vector<Family> families() {
    return {
        {"chebyshev7", ChebyshevConfig{7}},
        {"legendre6", LegendreConfig{6}},
        {"jacobi6", JacobiConfig{6, 0.5, -0.3}},
        {"hermite6", HermiteConfig{6}},
        {"fourier7", FourierConfig{7, 1.3}},
        {"rbf8", GaussianRbfConfig{grid(8, -1.2, 1.2), 0.4}},
        {"trainable_rbf8", TrainableRbfConfig{grid(8, -1.2, 1.2), std::vector<double>(8, std::log(0.4))}},
        {"bspline3", BSplineConfig{3, knots(3, 5)}},
        {"mexican_hat6", MexicanHatConfig{grid(6, -1.0, 1.0), std::vector<double>(6, 0.5)}},
    };
}
Layer seeded(Layer l, double phase) {
    const auto scale = 1.0/std::sqrt(static_cast<double>(l.inputs()*l.terms()));
    l.set_parameters(wave(l.coefficients().size(), scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
    return l;
}
void dump(const char* label, const std::vector<double>& values) {
    std::printf("%s %zu", label, values.size());
    for (double v : values) std::printf(" %a", static_cast<double>(static_cast<float>(v)));
    std::printf("\n");
}
void run(const std::string& name, const Network& network, std::size_t batch) {
    const auto& first = std::get<Layer>(network.layers().front());
    const auto& last = std::get<Layer>(network.layers().back());
    cuda::ResidentNetwork r(network, batch, cuda::Precision::Float32);
    r.upload_input(wave(batch*first.inputs(), 0.9, 0.377, 0.2), batch);
    r.forward();
    std::printf("== %s batch %zu\n", name.c_str(), batch);
    dump("output", r.download_output());
    r.upload_output_gradient(wave(batch*last.outputs(), 0.5, 0.613, 0.4));
    r.backward(1e-3);
    const auto g = r.download_gradients();
    dump("input", g.input);
    for (const auto& layer : g.layers) {
        const auto& l = std::get<LayerGradients>(layer);
        dump("coefficients", l.coefficients);
        dump("bias", l.bias);
        if (const auto* t = std::get_if<TrainableRbfGradients>(&l.nonlinear)) { dump("centers", t->centers); dump("log_widths", t->log_widths); }
    }
}
} // namespace

int main() {
    for (const auto& f : families()) {
        run(std::string(f.name) + " small", Network({seeded(Layer(5, 7, f.basis), 0.1), seeded(Layer(7, 3, f.basis), 0.2)}), 33);
        run(std::string(f.name) + " wide", Network({seeded(Layer(96, 80, f.basis), 0.3), seeded(Layer(80, 10, f.basis), 0.4)}), 700);
    }
    // 300 terms: rows too long for the three-plane stage (stored rows).
    run("chebyshev300 long rows", Network({seeded(Layer(4, 3, ChebyshevConfig{300}), 0.5)}), 65);
    return 0;
}

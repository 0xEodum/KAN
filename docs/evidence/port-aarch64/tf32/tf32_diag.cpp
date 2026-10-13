// Port diagnostic: TF32 layer-0 bias-gradient error relative to sum_b |U0[b,i]|.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "support/network.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
using kan::cuda::Precision; using kan::cuda::ResidentNetwork;
std::vector<double> wave(std::size_t n, double s, double f, double ph = 0) { std::vector<double> v(n); for (std::size_t i = 0; i < n; ++i) v[i] = s*std::sin(f*double(i)+ph); return v; }
kan::Layer layer(std::size_t in, std::size_t out, kan::BasisConfig b, double ph) {
    kan::Layer l(in, out, std::move(b)); const auto sc = 1.0/std::sqrt(double(l.inputs()*l.terms()));
    l.set_parameters(wave(l.coefficients().size(), sc, 0.731, ph), wave(l.outputs(), 0.05, 1.3, ph)); return l; }
int main() {
    kan::Network cpu({layer(64, 80, kan::ChebyshevConfig{7}, 0.7), layer(80, 3, kan::JacobiConfig{5, 0.5, -0.25}, 0.8)});
    const std::size_t B = 1000; const double lambda = 0.01, rate = 0.01;
    for (auto prec : {Precision::Float32, Precision::TensorFloat32}) {
        ResidentNetwork gpu(cpu, B, prec);
        const auto x = wave(B*64, 0.9, 0.37), dy = wave(B*3, 0.2, 0.53, 1);
        gpu.upload_input(x, B); gpu.upload_output_gradient(dy);
        for (int step = 0; step < 2; ++step) {
            const auto ref = gpu.download_parameters();
            gpu.forward(); gpu.backward(lambda);
            auto e = ref.backward(x, B, dy);
            { const auto pen = ref.regularization(lambda).gradients;
              for (std::size_t j = 0; j < 2; ++j) for (std::size_t k = 0; k < test::grad(e, j).coefficients.size(); ++k)
                  std::get<kan::LayerGradients>(e.layers[j]).coefficients[k] += test::grad(pen, j).coefficients[k]; } const auto a = gpu.download_gradients();
            const auto& eb = test::grad(e, 0).bias; const auto& ab = test::grad(a, 0).bias;
            const auto& u0 = test::grad(e, 1).input; // B x 80 upstream of layer 0
            double scale = 0; for (double v : eb) scale = std::max(scale, std::abs(v));
            double worst_ratio = 0, worst_bound = 0; std::size_t wi = 0;
            for (std::size_t i = 0; i < eb.size(); ++i) {
                double s = 0; for (std::size_t b = 0; b < B; ++b) s += std::abs(u0[b*80+i]);
                const double err = std::abs(ab[i]-eb[i]);
                const double r = err/s, bound = err/(1e-2*std::abs(eb[i]) + 1e-2*scale);
                if (r > worst_ratio) { worst_ratio = r; wi = i; }
                worst_bound = std::max(worst_bound, bound);
            }
            std::printf("%s step %d: layer0 bias grad max |err|/sum|U0| = %.3e (index %zu), max |err|/current-tf32-bound = %.3f\n",
                        prec == Precision::Float32 ? "FP32" : "TF32", step, worst_ratio, wi, worst_bound);
            auto ratio = [](const std::vector<double>& a, const std::vector<double>& e) {
                double sc = 0, w = 0; for (double v : e) sc = std::max(sc, std::abs(v));
                for (std::size_t i = 0; i < e.size(); ++i) w = std::max(w, std::abs(a[i]-e[i])/(1e-2*std::abs(e[i])+1e-2*sc));
                return w; };
            std::printf("   current-bound ratios: L0 coef %.3f, L1 coef %.3f, L1 bias %.3f, L0 input %.3f, L1 input %.3f\n",
                ratio(test::grad(a,0).coefficients, test::grad(e,0).coefficients), ratio(test::grad(a,1).coefficients, test::grad(e,1).coefficients),
                ratio(test::grad(a,1).bias, test::grad(e,1).bias), ratio(test::grad(a,0).input, test::grad(e,0).input), ratio(test::grad(a,1).input, test::grad(e,1).input));
            gpu.sgd(rate);
        }
    }
}

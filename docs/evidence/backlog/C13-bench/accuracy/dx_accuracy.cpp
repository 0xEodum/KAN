// Backlog C13: accuracy of the resident input VJP (dx) against the FP64 CPU
// reference, per basis family, precision and B-spline path.
//
// Network: one KAN layer inputs -> outputs with the given basis; the CPU
// reference is evaluated at the executor's downloaded (rounded) parameters, the
// upstream is uploaded exactly (FP32-representable values), so the only
// differences are the executor's arithmetic. Coefficients:
//   wave   - 3/sqrt(I*K) * sin(0.731*flat index), as the C5/C1 suites (rough in k)
//   smooth - edge functions smooth in the term index: c[o,i,k] = a_oi*sin(pi*s_k + phase_oi)
//            with s_k = k/(K-1), a_oi = 1/sqrt(I) (a trained KAN's edges vary slowly over the grid)
//   decay  - a_oi*sin(0.37*oi + 1.1*k)/(1+k) (decaying spectrum of a smooth function in a global basis)
//   fit    - least-squares fits of smooth shapes sin(w x + phi) over the domain (a trained layer's coefficients)
//   vs     - kan::initialize VarianceScaling (uniform, seed 7)
// Reported per run (one CSV row):
//   max|e| (reference dx), max|err|, max|err|/max|e|,
//   tf32_tol = max err/(1e-2|e| + 1e-2 max|e|)   (C1 TF32 tolerance; > 1 fails)
//   fp32_tol = max err/(2e-4|e| + 5e-5 max|e|)   (C1 FP32 tolerance)
//   amp      = max err/(2^-11 * sum_k |phi'_k| sum_o |U||C_k|)   (bound of the W = U*C formula)
//   diff     = max err/(2^-11 * sum_r b_r sum_o |U||C*_r|)       (bound of the coefficient-difference
//              formula, B-splines only; b_r the degree-(p-1) basis, C* = p(c_j - c_{j-1})/(t_{j+p} - t_j))
//   forward  = max |y - y_ref|/(1e-2|y_ref| + 1e-2 max|y_ref|) of the outputs; c_max = max |C|
//   ratio    = median over entries of (sum_k |phi'_k| sum_o |U||C_k|) / sum_o |U||f'_oi(x)|  (per-edge cancellation
//              factor; f'_oi = sum_k C_oik phi'_k in FP64)
//   edge     = max err/(2^-11 * (sum_o |U||f'_oi(x)| + 1e-2 * its largest value))  (error on the scale of the
//              true edge derivatives: K-independent if the formula does not amplify rounding)
// Usage: dx_accuracy <f64|f32|tf32> <auto|dense|sparse> <family spec> <inputs> <outputs> <batch> <wave|smooth|decay|fit|vs> [sin|uniform]
//   (inputs: 0.5 + 0.4999 sin(0.377 k + 0.2) mapped to the domain, or independent uniform draws)
// Family spec: bspline:p:K  cheb:N  legendre:N  jacobi:N  hermite:N  hermitef:N  fourier:N:w  gauss:K:f  mexhat:K:f
//   (gauss/mexhat: K centers uniform on [-1,1], width/scale f times the center spacing)
// Build: scripts/link_harness.sh <build> dx_accuracy.cpp <exe> -I src
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, c)) out.push_back(item);
    return out;
}
kan::BSplineConfig spline(std::size_t degree, std::size_t terms) {
    kan::BSplineConfig b{degree, std::vector<double>(degree + 1, -1.0)};
    const auto interior = terms - degree - 1;
    for (std::size_t k = 1; k <= interior; ++k) b.knots.push_back(-1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(interior + 1));
    b.knots.insert(b.knots.end(), degree + 1, 1.0);
    return b;
}
struct Family { kan::BasisConfig basis; double low, high; };
Family family(const std::string& spec) {
    const auto f = split(spec, ':');
    const auto n = static_cast<std::size_t>(std::stoul(f.at(1)));
    if (f[0] == "bspline") return {spline(n, std::stoul(f.at(2))), -1, 1};
    if (f[0] == "cheb") return {kan::ChebyshevConfig{n}, -1, 1};
    if (f[0] == "legendre") return {kan::LegendreConfig{n}, -1, 1};
    if (f[0] == "jacobi") return {kan::JacobiConfig{n, 0.3, -0.2}, -1, 1};
    if (f[0] == "hermite") return {kan::HermiteConfig{n}, -1, 1};
    if (f[0] == "hermitef") return {kan::HermiteConfig{n, kan::HermiteNormalization::Function}, -4, 4};
    if (f[0] == "fourier") { const double w = std::stod(f.at(2)); return {kan::FourierConfig{n, w}, -3.14159 / w, 3.14159 / w}; }
    if (f[0] == "gauss" || f[0] == "mexhat") {
        std::vector<double> c(n);
        for (std::size_t k = 0; k < n; ++k) c[k] = -1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(n - 1);
        const double width = std::stod(f.at(2)) * 2.0 / static_cast<double>(n - 1);
        if (f[0] == "gauss") return {kan::GaussianRbfConfig{c, width}, -1, 1};
        return {kan::MexicanHatConfig{c, std::vector<double>(n, width)}, -1, 1};
    }
    throw std::invalid_argument("unknown family " + spec);
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: dx_accuracy f64|f32|tf32 auto|dense|sparse <family> <inputs> <outputs> <batch> wave|smooth|decay|fit|vs\n");
        return 2;
    }
    const std::string precision_name = argv[1], path_name = argv[2], spec = argv[3], init = argv[7];
    const std::size_t inputs = std::stoul(argv[4]), outputs = std::stoul(argv[5]), batch = std::stoul(argv[6]);
    const auto precision = precision_name == "f64" ? kan::cuda::Precision::Float64
                         : precision_name == "f32" ? kan::cuda::Precision::Float32 : kan::cuda::Precision::TensorFloat32;
    const auto path = path_name == "dense" ? kan::cuda::testing::SplinePath::Dense
                    : path_name == "sparse" ? kan::cuda::testing::SplinePath::Sparse : kan::cuda::testing::SplinePath::Automatic;
    const auto fam = family(spec);
    kan::Layer layer(inputs, outputs, fam.basis);
    const auto terms = layer.terms();
    std::vector<double> c(layer.coefficients().size());
    if (init == "wave") {
        const double scale = 3.0 / std::sqrt(static_cast<double>(inputs * terms));
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = scale * std::sin(0.731 * static_cast<double>(k));
        layer.set_parameters(c, std::vector<double>(outputs, 0.01));
    } else if (init == "smooth") {
        const double a = 1.0 / std::sqrt(static_cast<double>(inputs));
        for (std::size_t oi = 0; oi < outputs * inputs; ++oi) {
            const double phase = 0.37 * static_cast<double>(oi), amp = a * (0.5 + 0.5 * std::cos(1.3 * static_cast<double>(oi)));
            for (std::size_t k = 0; k < terms; ++k)
                c[oi * terms + k] = amp * std::sin(3.14159 * static_cast<double>(k) / static_cast<double>(terms - 1) + phase);
        }
        layer.set_parameters(c, std::vector<double>(outputs, 0.01));
    } else if (init == "decay") {
        const double a = 1.0 / std::sqrt(static_cast<double>(inputs));
        for (std::size_t oi = 0; oi < outputs * inputs; ++oi)
            for (std::size_t k = 0; k < terms; ++k)
                c[oi * terms + k] = a * std::sin(0.37 * static_cast<double>(oi) + 1.1 * static_cast<double>(k)) / static_cast<double>(1 + k);
        layer.set_parameters(c, std::vector<double>(outputs, 0.01));
    } else if (init == "fit") {
        // Least-squares fits of 16 smooth target shapes sin(w_s x + phi_s) over the
        // input domain (w_s from 0.5 to 3.5 periods' worth of radians across the
        // domain half-width), assigned to the edges cyclically with amplitudes
        // a_oi: the coefficients a trained layer would hold for such edges.
        const std::size_t samples = std::max<std::size_t>(4 * terms, 64), shapes = 16;
        std::vector<std::vector<double>> rows(samples);
        std::vector<double> xs(samples);
        for (std::size_t m = 0; m < samples; ++m) {
            xs[m] = fam.low + (fam.high - fam.low) * (static_cast<double>(m) + 0.5) / static_cast<double>(samples);
            rows[m] = kan::evaluate_basis(fam.basis, xs[m]).values;
        }
        std::vector<double> gram(terms * terms, 0.0);
        for (std::size_t m = 0; m < samples; ++m)
            for (std::size_t a = 0; a < terms; ++a)
                for (std::size_t b = 0; b <= a; ++b) gram[a * terms + b] += rows[m][a] * rows[m][b];
        double trace = 0;
        for (std::size_t a = 0; a < terms; ++a) trace += gram[a * terms + a];
        for (std::size_t a = 0; a < terms; ++a) gram[a * terms + a] += 1e-12 * trace / static_cast<double>(terms);
        // Cholesky (lower) in place.
        for (std::size_t a = 0; a < terms; ++a) {
            for (std::size_t b = 0; b <= a; ++b) {
                double sum = gram[a * terms + b];
                for (std::size_t k = 0; k < b; ++k) sum -= gram[a * terms + k] * gram[b * terms + k];
                gram[a * terms + b] = a == b ? std::sqrt(sum) : sum / gram[b * terms + b];
            }
        }
        std::vector<std::vector<double>> fits(shapes, std::vector<double>(terms));
        const double half = 0.5 * (fam.high - fam.low), mid = 0.5 * (fam.high + fam.low);
        for (std::size_t sh = 0; sh < shapes; ++sh) {
            const double w = (0.5 + 0.2 * static_cast<double>(sh)) * 3.14159 / half, phi = 0.7 * static_cast<double>(sh);
            std::vector<double> rhs(terms, 0.0);
            for (std::size_t m = 0; m < samples; ++m)
                for (std::size_t a = 0; a < terms; ++a) rhs[a] += rows[m][a] * std::sin(w * (xs[m] - mid) + phi);
            for (std::size_t a = 0; a < terms; ++a) { // forward
                for (std::size_t k = 0; k < a; ++k) rhs[a] -= gram[a * terms + k] * rhs[k];
                rhs[a] /= gram[a * terms + a];
            }
            for (std::size_t a = terms; a-- > 0;) { // backward
                for (std::size_t k = a + 1; k < terms; ++k) rhs[a] -= gram[k * terms + a] * rhs[k];
                rhs[a] /= gram[a * terms + a];
            }
            fits[sh] = rhs;
        }
        const double amp = 1.0 / std::sqrt(static_cast<double>(inputs));
        for (std::size_t oi = 0; oi < outputs * inputs; ++oi) {
            const double a = amp * (0.5 + 0.5 * std::cos(1.3 * static_cast<double>(oi)));
            for (std::size_t k = 0; k < terms; ++k) c[oi * terms + k] = a * fits[(oi * 7) % shapes][k];
        }
        layer.set_parameters(c, std::vector<double>(outputs, 0.01));
    } else {
        kan::initialize(layer, kan::VarianceScaling{1.0, kan::Distribution::Uniform, 7});
    }
    const kan::Network network({layer});
    std::vector<double> x(batch * inputs), dy(batch * outputs);
    const bool uniform_x = argc > 8 && std::string(argv[8]) == "uniform";
    std::uint64_t lcg = 12345;
    for (std::size_t k = 0; k < x.size(); ++k) {
        double s = 0.5 + 0.4999 * std::sin(0.377 * static_cast<double>(k) + 0.2);
        if (uniform_x) { // independent uniform inputs on the domain
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            s = static_cast<double>(lcg >> 11) * 0x1p-53;
        }
        x[k] = fam.low + (fam.high - fam.low) * s;
    }
    for (std::size_t k = 0; k < dy.size(); ++k) dy[k] = static_cast<float>(0.3 * std::cos(0.613 * static_cast<double>(k)));
    kan::cuda::testing::ResidentLayout layout;
    kan::cuda::testing::ResidentTestScope scope(path, &layout);
    kan::cuda::ResidentNetwork gpu(network, batch, precision);
    // FP32 inputs: the reference sees the same rounded x.
    if (precision != kan::cuda::Precision::Float64)
        for (double& v : x) v = static_cast<float>(v);
    gpu.upload_input(x, batch);
    gpu.forward();
    const auto y_gpu = gpu.download_output();
    gpu.upload_output_gradient(dy);
    gpu.backward();
    const auto actual = gpu.download_gradients().input;
    const auto reference = gpu.download_parameters();
    const auto expected = reference.backward(x, batch, dy).input;
    const auto& ref_layer = std::get<kan::Layer>(reference.layers()[0]);
    const auto rc = ref_layer.coefficients();
    double scale = 0;
    for (double e : expected) scale = std::max(scale, std::abs(e));
    double max_err = 0, tf32 = 0, fp32 = 0, amp = 0, diff = 0;
    std::vector<double> ratios, errors(expected.size()), edge_scale(expected.size());
    const auto* bs = std::get_if<kan::BSplineConfig>(&fam.basis);
    // Degree-(p-1) basis on the knots without their first and last entry:
    // B_{j,p-1} for j = 1..K-1 of the full knot vector.
    kan::BSplineConfig lower;
    if (bs && bs->degree > 0) lower = {bs->degree - 1, std::vector<double>(bs->knots.begin() + 1, bs->knots.end() - 1)};
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < inputs; ++i) {
            const auto r = b * inputs + i;
            const double err = std::abs(actual[r] - expected[r]);
            errors[r] = err;
            max_err = std::max(max_err, err);
            tf32 = std::max(tf32, err / (1e-2 * std::abs(expected[r]) + 1e-2 * scale));
            fp32 = std::max(fp32, err / (2e-4 * std::abs(expected[r]) + 5e-5 * scale));
            const auto basis = kan::evaluate_basis(fam.basis, x[r]);
            double old_bound = 0, edge = 0;
            for (std::size_t o = 0; o < outputs; ++o) {
                double derivative = 0; // f'_oi(x) in FP64
                for (std::size_t k = 0; k < terms; ++k) derivative += rc[(o * inputs + i) * terms + k] * basis.derivatives[k];
                edge += std::abs(dy[b * outputs + o] * derivative);
            }
            edge_scale[r] = edge;
            for (std::size_t k = 0; k < terms; ++k) {
                double w = 0;
                for (std::size_t o = 0; o < outputs; ++o) w += std::abs(dy[b * outputs + o] * rc[(o * inputs + i) * terms + k]);
                old_bound += std::abs(basis.derivatives[k]) * w;
            }
            if (old_bound > 0) amp = std::max(amp, err / (0x1p-11 * old_bound));
            if (edge > 0) ratios.push_back(old_bound / edge);
            if (bs && bs->degree > 0) {
                const auto p = bs->degree;
                const auto& t = bs->knots;
                const auto lb = kan::evaluate_basis(lower, x[r]);
                double new_bound = 0;
                for (std::size_t j = 1; j < terms; ++j) {
                    if (lb.values[j - 1] == 0) continue;
                    const double gap = t[j + p] - t[j];
                    double w = 0;
                    for (std::size_t o = 0; o < outputs; ++o) {
                        const auto q = (o * inputs + i) * terms;
                        w += std::abs(dy[b * outputs + o] * static_cast<double>(p) * (rc[q + j] - rc[q + j - 1]) / gap);
                    }
                    new_bound += lb.values[j - 1] * w;
                }
                if (new_bound > 0) diff = std::max(diff, err / (0x1p-11 * new_bound));
            }
        }
    }
    // Error against the scale of the true edge derivatives: 2^-11 * (sum_o |U||f'_oi(x)|
    // + 1e-2 * the largest such sum), the floor covering entries where every f' crosses zero.
    double edge_max = 0, edge_ratio = 0;
    for (double e : edge_scale) edge_max = std::max(edge_max, e);
    for (std::size_t r = 0; r < errors.size(); ++r)
        if (edge_max > 0) edge_ratio = std::max(edge_ratio, errors[r] / (0x1p-11 * (edge_scale[r] + 1e-2 * edge_max)));
    // Forward outputs against the CPU within the C1 TF32 tolerance (conditioning of
    // the representation itself: an ill-conditioned coefficient vector shows here too).
    const auto y_cpu = reference.forward(x, batch);
    double y_scale = 0, forward = 0, c_max = 0;
    for (double v : y_cpu) y_scale = std::max(y_scale, std::abs(v));
    for (std::size_t r = 0; r < y_cpu.size(); ++r)
        forward = std::max(forward, std::abs(y_gpu[r] - y_cpu[r]) / (1e-2 * std::abs(y_cpu[r]) + 1e-2 * y_scale));
    for (double v : rc) c_max = std::max(c_max, std::abs(v));
    std::sort(ratios.begin(), ratios.end());
    const double median = ratios.empty() ? 0 : ratios[ratios.size() / 2];
    std::printf("%s,%s,%s,%zu,%zu,%zu,%s,%d,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g\n", precision_name.c_str(), path_name.c_str(),
                spec.c_str(), inputs, outputs, batch, init.c_str(), layout.sparse.empty() ? -1 : static_cast<int>(layout.sparse[0]),
                scale, max_err, max_err / scale, tf32, fp32, amp, diff, median, edge_ratio, forward, c_max);
    return 0;
}

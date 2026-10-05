// Full resident training step (forward + backward + SGD) on the large review
// topologies (BACKLOG.md, section C). Port of review-2026-10-01/resident_bench.cpp
// to the typed configuration API, plus a trainable-RBF case for the nonlinear
// reduction. Prints one CSV row per case: per-repeat step times and a checksum of
// the outputs after the timed steps, so two builds can be compared numerically.
// C1: optional precision (f64 default, f32) and the contraction rate: three
// GEMMs of 2*batch*outputs*inputs*K flops per layer per step.
// Usage: large_step [repeats] [case] [f64|f32|tf32]
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
struct Case {
    const char* name;
    std::vector<std::size_t> widths;
    std::size_t batch;
    bool trainable;
};

kan::Network network(const Case& c) {
    constexpr std::size_t K = 7;
    std::vector<kan::NetworkLayer> layers;
    for (std::size_t j = 0; j + 1 < c.widths.size(); ++j) {
        kan::BasisConfig basis = kan::ChebyshevConfig{K};
        if (c.trainable) {
            kan::TrainableRbfConfig rbf;
            for (std::size_t k = 0; k < K; ++k) {
                rbf.centers.push_back(-1.0 + 2.0*static_cast<double>(k)/(K-1));
                rbf.log_widths.push_back(std::log(0.4));
            }
            basis = rbf;
        }
        kan::Layer layer(c.widths[j], c.widths[j+1], basis);
        std::vector<double> coefficients(layer.coefficients().size()), bias(layer.outputs());
        for (std::size_t q = 0; q < coefficients.size(); ++q)
            coefficients[q] = 0.1*std::sin(static_cast<double>(q)*0.37)/std::sqrt(static_cast<double>(c.widths[j]*K));
        layer.set_parameters(coefficients, bias);
        layers.emplace_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
} // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 10;
    const std::string only = argc > 2 ? argv[2] : "";
    const std::string precision_name = argc > 3 ? argv[3] : "f64";
    const auto precision = precision_name == "f32"    ? kan::cuda::Precision::Float32
                           : precision_name == "tf32" ? kan::cuda::Precision::TensorFloat32
                                                      : kan::cuda::Precision::Float64;
    const Case cases[] = {
        {"cheb-64x64x32x16-b1024", {64, 64, 32, 16}, 1024, false},
        {"cheb-256x256x256x10-b8192", {256, 256, 256, 10}, 8192, false},
        {"cheb-1024x1024x1024-b4096", {1024, 1024, 1024}, 4096, false},
        {"trbf-256x256x256x10-b8192", {256, 256, 256, 10}, 8192, true},
    };
    std::printf("case,precision,repeats,median_ms,min_ms,gemm_gflop,gemm_tflops_median,output_checksum\n");
    for (const auto& c : cases) {
        if (!only.empty() && only != c.name) continue;
        kan::cuda::ResidentNetwork r(network(c), c.batch, precision);
        double gflop = 0;
        for (std::size_t j = 0; j + 1 < c.widths.size(); ++j)
            gflop += 3.0*2.0*static_cast<double>(c.batch)*static_cast<double>(c.widths[j+1]*c.widths[j]*7)/1e9;
        std::vector<double> x(c.batch*c.widths.front()), u(c.batch*c.widths.back());
        for (std::size_t q = 0; q < x.size(); ++q) x[q] = std::sin(static_cast<double>(q)*0.11)*0.9;
        for (std::size_t q = 0; q < u.size(); ++q) u[q] = std::cos(static_cast<double>(q)*0.07)/static_cast<double>(c.batch);
        r.upload_input(x, c.batch);
        auto step = [&] { r.forward(); r.upload_output_gradient(u); r.backward(); r.sgd(1e-3); };
        for (int i = 0; i < 2; ++i) step();
        const int reps = c.widths.front() >= 1024 ? std::max(1, repeats/3) : repeats;
        std::vector<double> samples;
        for (int i = 0; i < reps; ++i) {
            const auto t = std::chrono::steady_clock::now();
            step();
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count());
        }
        r.forward();
        double checksum = 0;
        for (double y : r.download_output()) checksum += y;
        std::sort(samples.begin(), samples.end());
        const double median = samples[samples.size()/2];
        std::printf("%s,%s,%d,%.3f,%.3f,%.2f,%.2f,%.17g\n", c.name, precision_name.c_str(), reps, median, samples.front(),
                    gflop, gflop/median, checksum);
        std::fflush(stdout);
    }
}

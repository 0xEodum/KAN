// Backlog C9: matched-protocol and realistic training-loop benchmark of the
// resident executor, the counterpart of torch_bench.py (same topologies,
// Chebyshev K=7, the same work per step).
//
// Protocols (one CSV row per timing window):
//   matched    x and the upstream gradient stay resident on the GPU; a step is
//              forward + backward(upstream) + SGD, exactly torch_reference.py.
//   realistic  every step trains on a new (input, target) batch taken from a
//              host-memory dataset (4 distinct batches, cycled), with the mean
//              squared error loss; nothing is resident between steps except
//              the model.
// Modes:
//   eager      the pre-C9 API: forward(); backward(); sgd() (three status
//              synchronizations per step). For `realistic` the loss gradient
//              is computed on the host from download_output().
//   step       C9 train_step (one CUDA-graph replay per step; status checked
//              every `interval` steps). For `realistic` the batch goes through
//              train_step(input, target, batch, ...), which stages it
//              asynchronously and computes the MSE gradient on the device.
// Usage: train_bench <precision f64|f32|tf32> <protocol> <mode> [interval] [windows] [case]
// Compile with -DKAN_C9_API for the `step` mode (the pre-C9 build has only `eager`).
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

namespace {
struct Case {
    const char* name;
    std::vector<std::size_t> widths;
    std::size_t batch;
    int steps; // timed steps per window
};
constexpr std::size_t K = 7;

// Coefficients as torch_reference.py draws them (normal * 0.1/sqrt(I*K)),
// from a fixed deterministic sequence; zero bias.
kan::Network network(const Case& c) {
    std::vector<kan::NetworkLayer> layers;
    std::uint64_t state = 12345;
    auto normal = [&] {
        auto uniform = [&] { state = state*6364136223846793005ULL + 1442695040888963407ULL; return ((state >> 11) + 0.5)/9007199254740992.0; };
        return std::sqrt(-2*std::log(uniform()))*std::cos(6.283185307179586*uniform());
    };
    for (std::size_t j = 0; j + 1 < c.widths.size(); ++j) {
        kan::Layer layer(c.widths[j], c.widths[j+1], kan::ChebyshevConfig{K});
        std::vector<double> coefficients(layer.coefficients().size()), bias(layer.outputs(), 0.0);
        const double scale = 0.1/std::sqrt(static_cast<double>(c.widths[j]*K));
        for (auto& v : coefficients) v = normal()*scale;
        layer.set_parameters(coefficients, bias);
        layers.emplace_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: train_bench f64|f32|tf32 matched|realistic eager|step [interval] [windows] [case]\n"); return 2; }
    const std::string precision_name = argv[1], protocol = argv[2], mode = argv[3];
    const std::size_t interval = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
    const int windows = argc > 5 ? std::atoi(argv[5]) : 3;
    const std::string only = argc > 6 ? argv[6] : "";
    const auto precision = precision_name == "f32"    ? kan::cuda::Precision::Float32
                           : precision_name == "tf32" ? kan::cuda::Precision::TensorFloat32
                                                      : kan::cuda::Precision::Float64;
    const bool f64 = precision == kan::cuda::Precision::Float64;
#ifndef KAN_C9_API
    if (mode == "step") { std::fprintf(stderr, "step mode needs a C9 build\n"); return 2; }
#endif
    const Case cases[] = {
        {"64x64x32x16-b1024", {64, 64, 32, 16}, 1024, 200},
        {"16x24x8-b1024", {16, 24, 8}, 1024, 200},
        {"256x256x256x10-b8192", {256, 256, 256, 10}, 8192, f64 ? 10 : 50},
        {"1024x1024x1024-b4096", {1024, 1024, 1024}, 4096, f64 ? 2 : 20},
    };
    constexpr double rate = 1e-3;
    std::printf("impl,precision,protocol,mode,interval,case,window,steps,ms_per_step,checksum\n");
    for (const auto& c : cases) {
        if (!only.empty() && only != c.name) continue;
        kan::cuda::ResidentNetwork r(network(c), c.batch, precision);
        const auto inputs = c.widths.front(), outputs = c.widths.back();
        constexpr int batches = 4;
        std::vector<std::vector<double>> xs, ts;
        for (int b = 0; b < batches; ++b) {
            // x uniform-like in [-1, 1], targets of the scale of the outputs.
            xs.push_back(wave(c.batch*inputs, 0.95, 0.113 + 0.017*b, 0.3*b));
            ts.push_back(wave(c.batch*outputs, 0.1, 0.071 + 0.013*b, 0.7*b));
        }
        const auto up = wave(c.batch*outputs, 1.0/static_cast<double>(c.batch), 0.07, 0.1);
        std::vector<double> gradient(c.batch*outputs);
        const double mse_scale = 2.0/static_cast<double>(c.batch*outputs);
#ifdef KAN_C9_API
        r.set_status_interval(interval);
#else
        (void)interval;
#endif
        if (protocol == "matched") { r.upload_input(xs[0], c.batch); r.upload_output_gradient(up); }
        int counter = 0;
        auto step = [&] {
            const int b = counter++ % batches;
            if (protocol == "matched") {
                if (mode == "eager") { r.forward(); r.backward(); r.sgd(rate); }
#ifdef KAN_C9_API
                else r.train_step(rate);
#endif
            } else {
                if (mode == "eager") {
                    r.upload_input(xs[b], c.batch); r.forward();
                    const auto y = r.download_output();
                    for (std::size_t i = 0; i < y.size(); ++i) gradient[i] = (y[i]-ts[b][i])*mse_scale;
                    r.upload_output_gradient(gradient); r.backward(); r.sgd(rate);
                }
#ifdef KAN_C9_API
                else r.train_step(xs[b], ts[b], c.batch, rate);
#endif
            }
        };
        const int warmup = std::max(3, c.steps/10);
        for (int i = 0; i < warmup; ++i) step();
        r.synchronize();
        for (int w = 0; w < windows; ++w) {
            const auto t = std::chrono::steady_clock::now();
            for (int i = 0; i < c.steps; ++i) step();
            r.synchronize(); // also reports any deferred status
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count()/c.steps;
            double checksum = 0;
            if (w + 1 == windows) {
                const auto trained = r.download_parameters();
                for (const auto& stage : trained.layers())
                    for (double v : std::get<kan::Layer>(stage).coefficients()) checksum += v;
            }
            std::printf("kan,%s,%s,%s,%zu,%s,%d,%d,%.4f,%.10g\n", precision_name.c_str(), protocol.c_str(), mode.c_str(),
                        interval, c.name, w, c.steps, ms, checksum);
            std::fflush(stdout);
        }
    }
}

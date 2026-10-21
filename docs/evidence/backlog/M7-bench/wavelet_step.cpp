// Backlog M7: full train_step of per-edge trainable wavelet networks
// (WaveletEdges) and of fixed Mexican-hat dictionary networks (BasisEdges with
// MexicanHatConfig) on the resident executor, for the cost of trainability.
// Protocol as C12's rbf_step (C9 matched protocol: input and output gradient
// uploaded once, a step is train_step(rate) = forward + backward + SGD as one
// graph replay, status interval 1). Both carriers start from the same
// functions: K wavelets per edge at tau_k = -1 + (2k+1)/K with scale 1/K (the
// WaveletEdges constructor placement) and the dictionary with those centers
// and scales; deterministic normal coefficients 0.1/sqrt(inputs*K), bias 0.
// The hash is FNV-1a over every downloaded parameter after 5 steps on a second
// executor.
// Usage: wavelet_step <f64|f32|tf32> <widths, e.g. 256x256x256x10> <batch> <wavelet|hat> <K> [windows=3]
//        [forward_samples backward_samples slots splits] (0 = the executor's choice; needs -DM7_HOOK)
// Build: scripts/link_harness.sh <build> docs/evidence/backlog/M7-bench/wavelet_step.cpp <exe>
//        (add -I src -DM7_HOOK for the launch arguments)
// One CSV row per window: precision,carrier,topology,batch,K,parameters,window,steps,step_ms,hash
#include "kan/families.hpp"
#include "kan/resident.hpp"
#ifdef M7_HOOK
#include "backends/cuda/resident_testing.hpp"
#endif
#include <chrono>
#include <memory>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
std::vector<std::size_t> widths_of(const std::string& text) {
    std::vector<std::size_t> widths;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, 'x')) widths.push_back(std::stoul(item));
    return widths;
}
kan::Network network(const std::vector<std::size_t>& widths, bool trainable, std::size_t K) {
    std::vector<kan::NetworkLayer> layers;
    std::uint64_t state = 12345;
    auto normal = [&] {
        auto uniform = [&] { state = state*6364136223846793005ULL + 1442695040888963407ULL; return ((state >> 11) + 0.5)/9007199254740992.0; };
        return std::sqrt(-2*std::log(uniform()))*std::cos(6.283185307179586*uniform());
    };
    std::vector<double> centers(K), scales(K, 1.0/static_cast<double>(K));
    for (std::size_t k = 0; k < K; ++k) centers[k] = -1.0 + static_cast<double>(2*k+1)/static_cast<double>(K);
    for (std::size_t j = 0; j + 1 < widths.size(); ++j) {
        const auto in = widths[j], out = widths[j+1];
        kan::Layer layer = trainable ? kan::Layer(in, out, kan::WaveletConfig{K}) : kan::Layer(in, out, kan::MexicanHatConfig{centers, scales});
        std::vector<double> coefficients(layer.coefficients().size()), bias(out, 0.0);
        const double scale = 0.1/std::sqrt(static_cast<double>(in*K));
        for (auto& v : coefficients) v = normal()*scale;
        layer.set_parameters(coefficients, bias);
        layers.emplace_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
struct Fnv {
    std::uint64_t h = 1469598103934665603ULL;
    void add(std::span<const double> v) {
        for (double x : v) {
            std::uint64_t bits; std::memcpy(&bits, &x, sizeof bits);
            for (int b = 0; b < 8; ++b) { h ^= (bits >> (8*b)) & 0xff; h *= 1099511628211ULL; }
        }
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: wavelet_step f64|f32|tf32 <widths> <batch> wavelet|hat <K> [windows]\n"); return 2; }
    const std::string precision_name = argv[1], topology = argv[2], carrier = argv[4];
    const std::size_t batch = std::stoul(argv[3]), K = std::stoul(argv[5]);
    const int windows = argc > 6 ? std::atoi(argv[6]) : 3;
    const auto precision = precision_name == "f32"    ? kan::cuda::Precision::Float32
                           : precision_name == "tf32" ? kan::cuda::Precision::TensorFloat32
                                                      : kan::cuda::Precision::Float64;
    const bool trainable = carrier == "wavelet";
    const auto widths = widths_of(topology);
    const auto net = network(widths, trainable, K);
    std::size_t parameters = 0;
    for (const auto& stage : net.layers()) {
        const auto& layer = std::get<kan::Layer>(stage);
        parameters += layer.coefficients().size()*(trainable ? 3 : 1);
    }
#ifdef M7_HOOK
    std::unique_ptr<kan::cuda::testing::ResidentTestScope> scope;
    if (argc > 10) {
        const kan::cuda::testing::WaveletLaunch launch{static_cast<unsigned>(std::atoi(argv[7])), static_cast<unsigned>(std::atoi(argv[8])),
                                                       static_cast<unsigned>(std::atoi(argv[9])), static_cast<unsigned>(std::atoi(argv[10]))};
        scope = std::make_unique<kan::cuda::testing::ResidentTestScope>(launch);
    }
#else
    if (argc > 7) { std::fprintf(stderr, "launch arguments need -DM7_HOOK\n"); return 2; }
#endif
    kan::cuda::ResidentNetwork r(net, batch, precision);
    kan::cuda::ResidentNetwork h(net, batch, precision);
#ifdef M7_HOOK
    scope.reset();
#endif
    const auto inputs = widths.front(), outputs = widths.back();
    r.upload_input(wave(batch*inputs, 0.95, 0.113, 0.0), batch);
    r.upload_output_gradient(wave(batch*outputs, 1.0/static_cast<double>(batch), 0.07, 0.1));
    const double rate = 1e-3;
    r.train_step(rate); r.train_step(rate); r.synchronize();
    auto t = std::chrono::steady_clock::now();
    r.train_step(rate); r.synchronize();
    const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count();
    const int steps = static_cast<int>(std::max(3.0, std::min(200.0, 400.0/std::max(one, 1e-3))));
    std::printf("precision,carrier,topology,batch,K,parameters,window,steps,step_ms,hash\n");
    std::vector<double> times;
    for (int w = 0; w < windows; ++w) {
        t = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) r.train_step(rate);
        r.synchronize();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count()/steps);
    }
    h.upload_input(wave(batch*inputs, 0.95, 0.113, 0.0), batch);
    h.upload_output_gradient(wave(batch*outputs, 1.0/static_cast<double>(batch), 0.07, 0.1));
    for (int i = 0; i < 5; ++i) h.train_step(rate);
    Fnv fnv;
    const auto trained = h.download_parameters();
    for (const auto& stage : trained.layers()) {
        const auto& layer = std::get<kan::Layer>(stage);
        fnv.add(layer.coefficients()); fnv.add(layer.bias());
        if (const auto* w = std::get_if<kan::WaveletEdges>(&layer.carrier())) { fnv.add(w->translations); fnv.add(w->log_scales); }
    }
    for (int w = 0; w < windows; ++w)
        std::printf("%s,%s,%s,%zu,%zu,%zu,%d,%d,%.4f,%016llx\n", precision_name.c_str(), carrier.c_str(), topology.c_str(), batch, K,
                    parameters, w, steps, times[w], static_cast<unsigned long long>(fnv.h));
}

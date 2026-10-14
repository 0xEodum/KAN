// Backlog C12: full train_step of trainable-RBF networks on the resident
// executor, for the before/after (ABBA) comparison of the center/log-width
// reduction. Network and protocol as the trainable-RBF case of
// M3-bench/phase2/residual_train_bench.cpp (C9 matched protocol: input and
// output gradient uploaded once, a step is train_step(rate) = forward +
// backward + SGD as one graph replay, status interval 1), generalized to any
// widths, batch and center count K: centers -1 + 2k/(K-1), log widths log(0.3),
// deterministic normal coefficients 0.1/sqrt(inputs*K), bias 0.
// The hash is FNV-1a over the bits of every downloaded parameter (coefficients,
// bias, centers, log widths of all layers) after warmup + all windows, so equal
// hashes mean bitwise-identical training trajectories.
// Usage: rbf_step <f64|f32|tf32> <widths, e.g. 256x256x256x10> <batch> <K> [windows=3] [path=auto|column|tile]
// Build: scripts/link_harness.sh <build> docs/evidence/backlog/C12-bench/rbf_step.cpp <exe>
//        (add -I src -DC12_HOOK for the path argument; needs a C12 build)
// One CSV row per window: precision,path,topology,batch,K,window,steps,step_ms,hash
#include "kan/families.hpp"
#include "kan/resident.hpp"
#ifdef C12_HOOK
#include "backends/cuda/resident_testing.hpp"
#endif
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
kan::Network network(const std::vector<std::size_t>& widths, std::size_t K) {
    std::vector<kan::NetworkLayer> layers;
    std::uint64_t state = 12345;
    auto normal = [&] {
        auto uniform = [&] { state = state*6364136223846793005ULL + 1442695040888963407ULL; return ((state >> 11) + 0.5)/9007199254740992.0; };
        return std::sqrt(-2*std::log(uniform()))*std::cos(6.283185307179586*uniform());
    };
    for (std::size_t j = 0; j + 1 < widths.size(); ++j) {
        const auto in = widths[j], out = widths[j+1];
        kan::TrainableRbfConfig rbf{std::vector<double>(K, 0.0), std::vector<double>(K, std::log(0.3))};
        for (std::size_t k = 0; k < K; ++k) rbf.centers[k] = K > 1 ? -1.0 + 2.0*static_cast<double>(k)/static_cast<double>(K-1) : 0.0;
        kan::Layer layer(in, out, rbf);
        std::vector<double> coefficients(layer.coefficients().size()), bias(out, 0.0);
        const double scale = 0.1/std::sqrt(static_cast<double>(in*layer.terms()));
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
    if (argc < 5) { std::fprintf(stderr, "usage: rbf_step f64|f32|tf32 <widths> <batch> <K> [windows] [auto|column|tile]\n"); return 2; }
    const std::string precision_name = argv[1], topology = argv[2];
    const std::size_t batch = std::stoul(argv[3]), K = std::stoul(argv[4]);
    const int windows = argc > 5 ? std::atoi(argv[5]) : 3;
    const std::string path = argc > 6 ? argv[6] : "auto";
    const auto precision = precision_name == "f32"    ? kan::cuda::Precision::Float32
                           : precision_name == "tf32" ? kan::cuda::Precision::TensorFloat32
                                                      : kan::cuda::Precision::Float64;
    const auto widths = widths_of(topology);
#ifdef C12_HOOK
    using kan::cuda::testing::NonlinearReduction;
    std::unique_ptr<kan::cuda::testing::ResidentTestScope> scope;
    if (path != "auto")
        scope = std::make_unique<kan::cuda::testing::ResidentTestScope>(path == "tile" ? NonlinearReduction::Tile : NonlinearReduction::Column);
#else
    if (path != "auto") { std::fprintf(stderr, "path needs -DC12_HOOK\n"); return 2; }
#endif
    kan::cuda::ResidentNetwork r(network(widths, K), batch, precision);
    // The timed trajectory depends on the step count (chosen from timing), so the
    // hash comes from a fixed-length trajectory on a second executor (same path).
    kan::cuda::ResidentNetwork h(network(widths, K), batch, precision);
#ifdef C12_HOOK
    scope.reset();
#endif
    const auto inputs = widths.front(), outputs = widths.back();
    r.upload_input(wave(batch*inputs, 0.95, 0.113, 0.0), batch);
    r.upload_output_gradient(wave(batch*outputs, 1.0/static_cast<double>(batch), 0.07, 0.1));
    const double rate = 1e-3;
    // Steps per window: about 0.4 s from one timed step (at least 3, at most 200).
    r.train_step(rate); r.train_step(rate); r.synchronize();
    auto t = std::chrono::steady_clock::now();
    r.train_step(rate); r.synchronize();
    const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count();
    const int steps = static_cast<int>(std::max(3.0, std::min(200.0, 400.0/std::max(one, 1e-3))));
    std::printf("precision,path,topology,batch,K,window,steps,step_ms,hash\n");
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
        if (const auto* t = std::get_if<kan::TrainableRbfEdges>(&layer.carrier())) { fnv.add(t->basis.centers); fnv.add(t->basis.log_widths); }
    }
    for (int w = 0; w < windows; ++w)
        std::printf("%s,%s,%s,%zu,%zu,%d,%d,%.4f,%016llx\n", precision_name.c_str(), path.c_str(), topology.c_str(), batch, K, w, steps,
                    times[w], static_cast<unsigned long long>(fnv.h));
}

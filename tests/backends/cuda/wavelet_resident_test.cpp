// Backlog M7: per-edge trainable Mexican-hat wavelets on the resident
// executor. Parity with the CPU in FP64 (1e-11) and within the C1 FP32/TF32
// tolerances (no cuBLAS contraction in the carrier: TF32 computes the carrier
// in FP32), the residual branch, captured training steps (bitwise the eager
// sequence) with atomic rollback of an invalid log-scale candidate, parameter
// upload (translations and log scales are state, the term count is
// structure), tails and nonfinite status.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <tuple>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

struct Tolerance {
    double relative, floor;
};
constexpr Tolerance fp64{1e-11, 1e-12}, fp32{2e-4, 5e-5}, tf32{1e-2, 1e-2};
Tolerance tolerance(Precision p) { return p == Precision::Float64 ? fp64 : p == Precision::Float32 ? fp32 : tf32; }

void close(std::span<const double> actual, std::span<const double> expected, Tolerance t, const char* what) {
    REQUIRE(actual.size() == expected.size());
    double scale = 0;
    for (double e : expected) scale = std::max(scale, std::abs(e));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > t.relative * std::abs(expected[i]) + t.floor * scale + 1e-37) {
            std::ostringstream out;
            out << std::setprecision(9) << what << " index " << i << " of " << actual.size() << " actual=" << actual[i]
                << " expected=" << expected[i] << " scale=" << scale;
            throw std::runtime_error(out.str());
        }
    }
}
std::vector<double> wave(std::size_t n, double scale, double frequency, double phase = 0) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * std::sin(frequency * static_cast<double>(k) + phase);
    return v;
}
kan::Layer wavelet(std::size_t in, std::size_t out, std::size_t size, double phase = 0) {
    kan::Layer layer(in, out, kan::WaveletConfig{size});
    const auto count = in * out * size;
    kan::set_wavelet_parameters(layer, wave(count, 0.5 / std::sqrt(double(in * size)) + 0.1, 0.9, 0.3 + phase),
                                wave(count, 0.8, 1.7, 0.1 + phase), wave(count, 0.4, 2.3, 0.2 + phase),
                                wave(out, 0.1, 1.1));
    return layer;
}
const kan::WaveletEdges& edges(const kan::Layer& l) { return std::get<kan::WaveletEdges>(l.carrier()); }
std::span<const double> translations(const kan::LayerGradients& g) {
    const auto* w = std::get_if<kan::WaveletGradients>(&g.nonlinear);
    return w ? std::span<const double>(w->translations) : std::span<const double>();
}
std::span<const double> log_scales(const kan::LayerGradients& g) {
    const auto* w = std::get_if<kan::WaveletGradients>(&g.nonlinear);
    return w ? std::span<const double>(w->log_scales) : std::span<const double>();
}

// Wavelet layers around a Chebyshev layer; the middle wavelet layer has the residual branch.
kan::Network mixed(std::size_t width = 4, std::size_t size = 2) {
    auto middle = wavelet(width, 3, 1, 0.4);
    middle.set_residual(kan::SiluResidual{wave(3 * width, 0.2, 0.6, 0.1)});
    kan::Layer basis(3, 3, kan::ChebyshevConfig{3});
    basis.set_parameters(wave(27, 0.2, 0.8), std::vector<double>(3, 0.01));
    return kan::Network({wavelet(3, width, size), middle, basis, wavelet(3, 2, 3, 0.9)});
}

void compare_gradients(const kan::NetworkGradients& a, const kan::NetworkGradients& e, Tolerance t) {
    close(a.input, e.input, t, "input");
    for (std::size_t j = 0; j < e.layers.size(); ++j) {
        const auto& ga = test::grad(a, j);
        const auto& ge = test::grad(e, j);
        REQUIRE(ga.nonlinear.index() == ge.nonlinear.index());
        close(ga.coefficients, ge.coefficients, t, "coefficients");
        close(ga.bias, ge.bias, t, "bias");
        close(translations(ga), translations(ge), t, "translations");
        close(log_scales(ga), log_scales(ge), t, "log scales");
        close(ga.residual, ge.residual, t, "residual");
    }
}
void compare_parameters(const kan::Network& a, const kan::Network& e, Tolerance t) {
    for (std::size_t j = 0; j < e.layers().size(); ++j) {
        const auto& la = test::layer(a, j);
        const auto& le = test::layer(e, j);
        close(la.coefficients(), le.coefficients(), t, "trained coefficients");
        close(la.bias(), le.bias(), t, "trained bias");
        if (const auto* w = std::get_if<kan::WaveletEdges>(&le.carrier())) {
            close(edges(la).translations, w->translations, t, "trained translations");
            close(edges(la).log_scales, w->log_scales, t, "trained log scales");
        }
    }
}

// forward, backward(l2) and sgd against the CPU for three steps.
void train_like_cpu(Precision precision, std::size_t width, std::size_t size, std::size_t batch) {
    const auto t = tolerance(precision);
    auto cpu = mixed(width, size);
    ResidentNetwork gpu(cpu, batch, precision);
    const auto allocations = gpu.workspace_allocations();
    const auto x = wave(batch * 3, 1.2, 0.37, 0.2), dy = wave(batch * 2, 0.5, 0.71, 0.3);
    gpu.upload_input(x, batch);
    gpu.upload_output_gradient(dy);
    for (int step = 0; step < 3; ++step) {
        gpu.forward();
        close(gpu.download_output(), cpu.forward(x, batch), t, "output");
        gpu.backward(0.1);
        auto expected = cpu.backward(x, batch, dy);
        const auto penalty = cpu.regularization(0.1).gradients;
        for (std::size_t j = 0; j < expected.layers.size(); ++j) {
            auto& g = test::grad(expected, j);
            for (std::size_t k = 0; k < g.coefficients.size(); ++k)
                g.coefficients[k] += test::grad(penalty, j).coefficients[k];
            for (std::size_t k = 0; k < g.residual.size(); ++k) g.residual[k] += test::grad(penalty, j).residual[k];
        }
        compare_gradients(gpu.download_gradients(), expected, t);
        gpu.sgd(0.05);
        cpu.sgd(expected, 0.05);
    }
    compare_parameters(gpu.download_parameters(), cpu, t);
    REQUIRE(gpu.workspace_allocations() == allocations);
}
} // namespace

TEST(wavelet_resident_matches_cpu_in_every_precision) {
    for (auto precision : {Precision::Float64, Precision::Float32, Precision::TensorFloat32}) {
        train_like_cpu(precision, 4, 2, 37);
        train_like_cpu(precision, 4, 1, 1);
    }
}

TEST(wavelet_resident_large_layers_match_cpu) {
    // Many samples, inputs, outputs and terms: every tile and reduction of the
    // kernels is partial somewhere (odd sizes) and spans several blocks.
    // TF32 computes this carrier in FP32 (no contraction): the FP32 tolerance applies.
    for (auto precision : {Precision::Float64, Precision::Float32, Precision::TensorFloat32}) {
        const auto t = precision == Precision::Float64 ? fp64 : fp32;
        for (const auto& shape :
             {std::array<std::size_t, 4>{37, 45, 3, 777}, std::array<std::size_t, 4>{70, 1, 2, 2600},
              std::array<std::size_t, 4>{1, 130, 5, 300}}) {
            const auto [in, out, size, batch] = shape;
            kan::Network cpu({wavelet(in, out, size)});
            ResidentNetwork gpu(cpu, batch + 5, precision);
            // FP64 checks the indexing of every partial sum to 1e-11; FP32 batch sums
            // of thousands of samples meet the C1 tolerance without heavy cancellation:
            // the upstream has a nonzero mean.
            const auto x = wave(batch * in, 1.1, 0.013, 0.4);
            auto dy = wave(batch * out, 0.1, 0.029, 0.1);
            for (auto& v : dy) v += 0.2;
            gpu.upload_input(x, batch);
            gpu.upload_output_gradient(dy);
            gpu.forward();
            close(gpu.download_output(), cpu.forward(x, batch), t, "large output");
            gpu.backward();
            try {
                compare_gradients(gpu.download_gradients(), cpu.backward(x, batch, dy), t);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string(precision == Precision::Float64 ? "FP64 " : "FP32 ") +
                                         std::to_string(in) + "x" + std::to_string(out) + "x" + std::to_string(size) +
                                         " batch " + std::to_string(batch) + ": " + e.what());
            }
        }
    }
}

TEST(wavelet_resident_every_launch_geometry_matches_cpu) {
    // Every kernel instantiation (samples per thread 1, 2, 4, 8 in both
    // kernels), several chunks per slot (grid-stride accumulation of the
    // partials) and output splits, forced through the private test hook so
    // that the coverage does not depend on the device's size.
    using kan::cuda::testing::ResidentTestScope;
    using kan::cuda::testing::WaveletLaunch;
    // Every third wavelet is narrow (s = exp(-3.5) = 0.03): its samples beyond
    // 13 (FP32) or 38 (FP64) scales take the log-space branch, so warps mix the
    // direct path and the warp-vote recompute.
    const std::size_t batch = 2100;
    auto layer = wavelet(5, 7, 2, 0.3);
    {
        auto e = edges(layer);
        for (std::size_t j = 0; j < e.log_scales.size(); j += 3) e.log_scales[j] = -3.5;
        layer.set_carrier(e);
    }
    kan::Network cpu({layer});
    const auto x = wave(batch * 5, 1.2, 0.011, 0.2);
    auto dy = wave(batch * 7, 0.1, 0.023, 0.4);
    for (auto& v : dy) v += 0.2;
    const auto y = cpu.forward(x, batch);
    const auto expected = cpu.backward(x, batch, dy);
    for (auto precision : {Precision::Float64, Precision::Float32}) {
        for (unsigned samples : {1u, 2u, 4u, 8u})
            for (const auto [slots, splits, input_splits] :
                 {std::tuple{1u, 1u, 1u}, std::tuple{2u, 3u, 2u}, std::tuple{3u, 7u, 5u}}) {
                // FP64 instantiates forward S = 1 and backward S <= 4 only.
                const bool fp64 = precision == Precision::Float64;
                const WaveletLaunch launch{fp64 ? 1u : samples, fp64 ? std::min(samples, 4u) : samples, slots, splits,
                                           input_splits};
                std::unique_ptr<ResidentNetwork> gpu;
                {
                    ResidentTestScope scope(launch);
                    gpu = std::make_unique<ResidentNetwork>(cpu, batch, precision);
                }
                gpu->upload_input(x, batch);
                gpu->upload_output_gradient(dy);
                gpu->forward();
                close(gpu->download_output(), y, tolerance(precision), "forced output");
                gpu->backward();
                compare_gradients(gpu->download_gradients(), expected, tolerance(precision));
            }
    }
}

TEST(wavelet_resident_narrow_layers_split_their_inputs) {
    // A forward of few outputs has few (output, chunk) blocks: the executor
    // splits the inputs over more blocks (how many is the cost model's choice
    // for the device) and adds the partial sums in a fixed order
    // (deterministic, close to the CPU). Wide layers keep one block per
    // (output, chunk).
    using kan::cuda::testing::ResidentLayout;
    using kan::cuda::testing::ResidentTestScope;
    using kan::cuda::testing::WaveletLaunch;
    const std::size_t batch = 2048;
    const kan::Network narrow({wavelet(256, 1, 1)}), wide({wavelet(4, 2048, 1)});
    for (auto precision : {Precision::Float64, Precision::Float32}) {
        ResidentLayout layout;
        std::unique_ptr<ResidentNetwork> gpu;
        {
            ResidentTestScope scope(WaveletLaunch{}, &layout);
            gpu = std::make_unique<ResidentNetwork>(narrow, batch, precision);
        }
        REQUIRE(layout.wavelet_forward.size() == 1 && layout.multiprocessors > 0);
        REQUIRE(layout.wavelet_forward[0].input_splits > 1);
        const auto x = wave(batch * 256, 1.1, 0.013, 0.3);
        gpu->upload_input(x, batch);
        gpu->forward();
        const auto first = gpu->download_output();
        close(first, narrow.forward(x, batch), tolerance(precision), "split output");
        gpu->forward();
        REQUIRE(gpu->download_output() == first);
        {
            ResidentTestScope scope(WaveletLaunch{}, &layout);
            ResidentNetwork single(wide, batch, precision);
        }
        REQUIRE(layout.wavelet_forward[0].input_splits == 1);
    }
}

TEST(wavelet_resident_test_hook_rejects_unsupported_launches) {
    using kan::cuda::testing::ResidentTestScope;
    using kan::cuda::testing::WaveletLaunch;
    const kan::Network net({wavelet(2, 2, 1)});
    for (const auto& launch : {WaveletLaunch{2, 0, 0, 0}, WaveletLaunch{0, 8, 0, 0}, WaveletLaunch{3, 0, 0, 0},
                               WaveletLaunch{0, 0, 0, 0, 70000}, WaveletLaunch{0, 0, 70000, 0}}) {
        ResidentTestScope scope(launch);
        test::throws<std::invalid_argument>([&] { ResidentNetwork(net, 4, Precision::Float64); });
    }
    ResidentTestScope scope(WaveletLaunch{8, 8, 0, 0});
    ResidentNetwork single(net, 4, Precision::Float32);
}

TEST(wavelet_resident_zero_batch_and_l2) {
    auto cpu = mixed();
    ResidentNetwork gpu(cpu, 4);
    gpu.upload_input({}, 0);
    gpu.upload_output_gradient({});
    gpu.forward();
    REQUIRE(gpu.download_output().empty());
    gpu.backward(0.25);
    const auto g = gpu.download_gradients();
    const auto expected = cpu.regularization(0.25).gradients;
    for (std::size_t j = 0; j < cpu.layers().size(); ++j) {
        close(test::grad(g, j).coefficients, test::grad(expected, j).coefficients, fp64, "l2 coefficients");
        for (double v : translations(test::grad(g, j))) REQUIRE(v == 0);
        for (double v : log_scales(test::grad(g, j))) REQUIRE(v == 0);
    }
}

TEST(wavelet_resident_training_steps_match_eager_and_roll_back) {
    for (auto precision : {Precision::Float64, Precision::Float32}) {
        const auto net = mixed();
        const std::size_t batch = 19;
        const auto x = wave(batch * 3, 1.0, 0.41), dy = wave(batch * 2, 0.4, 0.83);
        ResidentNetwork eager(net, batch, precision), graph(net, batch, precision);
        for (auto* r : {&eager, &graph}) {
            r->upload_input(x, batch);
            r->upload_output_gradient(dy);
        }
        for (int step = 0; step < 4; ++step) {
            eager.forward();
            eager.backward(0.01);
            eager.sgd(0.03);
            graph.train_step(0.03, 0.01);
        }
        const auto a = eager.download_parameters(), b = graph.download_parameters();
        for (std::size_t j = 0; j < net.layers().size(); ++j) {
            REQUIRE(test::layer(a, j).carrier() == test::layer(b, j).carrier());
            REQUIRE(std::ranges::equal(test::layer(a, j).bias(), test::layer(b, j).bias()));
        }
        REQUIRE(graph.trained_steps() == 4);
        // An upstream that drives a log scale out of range: the step fails and
        // the parameters stay at the last good step (eager SGD is atomic too).
        std::vector<double> huge(batch * 2, 1e30);
        graph.upload_output_gradient(huge);
        eager.upload_output_gradient(huge);
        test::throws<std::overflow_error>([&] { graph.train_step(1.0); });
        REQUIRE(graph.trained_steps() == 4);
        const auto kept = graph.download_parameters();
        for (std::size_t j = 0; j < net.layers().size(); ++j)
            REQUIRE(test::layer(kept, j).carrier() == test::layer(b, j).carrier());
        eager.forward();
        eager.backward();
        test::throws<std::overflow_error>([&] { eager.sgd(1.0); });
        const auto unchanged = eager.download_parameters();
        for (std::size_t j = 0; j < net.layers().size(); ++j)
            REQUIRE(test::layer(unchanged, j).carrier() == test::layer(a, j).carrier());
        // Training continues after the failure.
        graph.upload_output_gradient(dy);
        graph.train_step(0.03);
        REQUIRE(graph.trained_steps() == 5);
    }
}

TEST(wavelet_resident_out_of_range_log_scale_candidate_names_its_cause) {
    // A finite gradient that moves a log scale outside the valid range (exp(-l)
    // overflows) is not a nonfinite result: the eager SGD and the captured
    // step both report the cause the CPU names, and roll back.
    for (auto precision : {Precision::Float64, Precision::Float32}) {
        const auto net = mixed();
        const std::size_t batch = 19;
        const auto x = wave(batch * 3, 1.0, 0.41);
        const std::vector<double> large(batch * 2, 1e6);
        ResidentNetwork eager(net, batch, precision), graph(net, batch, precision);
        for (auto* r : {&eager, &graph}) {
            r->upload_input(x, batch);
            r->upload_output_gradient(large);
        }
        eager.forward();
        eager.backward();
        const auto g = eager.download_gradients();
        for (std::size_t j = 0; j < net.layers().size(); ++j)
            for (double v : log_scales(test::grad(g, j))) REQUIRE(std::isfinite(v));
        const auto before = eager.download_parameters();
        const auto what = [](auto fn) -> std::string {
            try {
                fn();
            } catch (const std::overflow_error& e) {
                return e.what();
            }
            return "no overflow_error";
        };
        const auto eager_text = what([&] { eager.sgd(1.0); });
        REQUIRE(eager_text.find("log-scale candidate leaves the valid range") != std::string::npos);
        const auto kept = eager.download_parameters();
        for (std::size_t j = 0; j < net.layers().size(); ++j)
            REQUIRE(test::layer(kept, j).carrier() == test::layer(before, j).carrier());
        const auto graph_text = what([&] { graph.train_step(1.0); });
        REQUIRE(graph_text.find("log-scale candidate leaves the valid range") != std::string::npos);
        REQUIRE(graph_text.find("training step 0") != std::string::npos);
        REQUIRE(graph.trained_steps() == 0);
    }
}

TEST(wavelet_resident_upload_parameters_checks_structure_and_values) {
    const auto net = mixed();
    ResidentNetwork gpu(net, 8);
    const auto x = wave(24, 0.9, 0.3);
    gpu.upload_input(x, 8);
    auto changed = net;
    std::vector<kan::NetworkLayer> layers(changed.layers().begin(), changed.layers().end());
    {
        auto& l = std::get<kan::Layer>(layers[0]);
        auto e = edges(l);
        for (auto& v : e.translations) v += 0.125;
        for (auto& v : e.log_scales) v -= 0.25;
        l.set_carrier(e);
    }
    changed = kan::Network(layers);
    gpu.upload_parameters(changed);
    gpu.forward();
    close(gpu.download_output(), changed.forward(x, 8), fp64, "uploaded output");
    REQUIRE(test::layer(gpu.download_parameters(), 0).carrier() == test::layer(changed, 0).carrier());
    // The term count is structure: a different size is rejected, executor unchanged.
    layers[0] = wavelet(3, 4, 3);
    test::throws<std::invalid_argument>([&] { gpu.upload_parameters(kan::Network(layers)); });
    layers[0] = kan::Layer(3, 4, kan::ChebyshevConfig{2});
    test::throws<std::invalid_argument>([&] { gpu.upload_parameters(kan::Network(layers)); });
    gpu.forward();
    close(gpu.download_output(), changed.forward(x, 8), fp64, "output after rejected upload");
    // FP32: a log scale whose exp leaves the float range is not representable.
    auto wide = wavelet(1, 1, 1);
    kan::set_wavelet_parameters(wide, std::vector<double>{1}, std::vector<double>{0}, std::vector<double>{100},
                                std::vector<double>{0});
    test::throws<std::invalid_argument>([&] { ResidentNetwork(kan::Network({wide}), 2, Precision::Float32); });
    ResidentNetwork single(kan::Network({wavelet(1, 1, 1)}), 2, Precision::Float32);
    test::throws<std::invalid_argument>([&] { single.upload_parameters(kan::Network({wide})); });
    ResidentNetwork(kan::Network({wide}), 2, Precision::Float64);
}

TEST(wavelet_resident_tails_and_nonfinite_status) {
    // Narrow wavelets far from their centers: FP64 keeps the derivatives that
    // log space preserves on the CPU.
    kan::Layer narrow(1, 1, kan::WaveletConfig{2});
    // Every sample is in the log-space tail (or the exact zero tail) of both
    // wavelets, where s^(-3/2) = exp(1050) or exp(975) would overflow a direct product.
    kan::set_wavelet_parameters(narrow, std::vector<double>{1, 0.5}, std::vector<double>{0, 2e-280},
                                std::vector<double>{-700, -650}, std::vector<double>{0});
    const std::vector<double> x{40 * std::exp(-700.0), 2e-280 - 38.9 * std::exp(-650.0), 3.0};
    kan::Network cpu({narrow});
    ResidentNetwork gpu(cpu, 3);
    gpu.upload_input(x, 3);
    gpu.upload_output_gradient(std::vector<double>{1, 1, 1});
    gpu.forward();
    close(gpu.download_output(), cpu.forward(x, 3), fp64, "tail output");
    gpu.backward();
    const auto g = gpu.download_gradients();
    const auto e = cpu.backward(x, 3, std::vector<double>{1, 1, 1});
    close(g.input, e.input, fp64, "tail input");
    REQUIRE(g.input[0] != 0 && g.input[1] != 0);
    close(log_scales(test::grad(g, 0)), log_scales(test::grad(e, 0)), fp64, "tail log scales");
    // An overflowing x - tau retried as x r - tau r in the direct range (q = 2.4).
    {
        kan::Layer wide(1, 1, kan::WaveletConfig{1});
        kan::set_wavelet_parameters(wide, std::vector<double>{1}, std::vector<double>{-1e308}, std::vector<double>{709},
                                    std::vector<double>{0});
        kan::Network net({wide});
        ResidentNetwork r(net, 1);
        r.upload_input(std::vector<double>{1e308}, 1);
        r.upload_output_gradient(std::vector<double>{1});
        r.forward();
        const auto expected = net.forward(std::vector<double>{1e308}, 1);
        REQUIRE(expected[0] != 0);
        close(r.download_output(), expected, fp64, "retried output");
        r.backward();
        close(log_scales(test::grad(r.download_gradients(), 0)),
              log_scales(test::grad(net.backward(std::vector<double>{1e308}, 1, std::vector<double>{1}), 0)), fp64,
              "retried log scale");
    }
    // Nonfinite results raise overflow_error and invalidate the state.
    kan::Layer big(1, 1, kan::WaveletConfig{2});
    kan::set_wavelet_parameters(big, std::vector<double>{1.5e308, 1.5e308}, std::vector<double>{0, 0},
                                std::vector<double>{0, 0}, std::vector<double>{0});
    ResidentNetwork over(kan::Network({big}), 1);
    over.upload_input(std::vector<double>{0}, 1);
    test::throws<std::overflow_error>([&] { over.forward(); });
    test::throws<std::logic_error>([&] { over.download_output(); });
    // Finite output (1e-80 * s^(-1/2) = 1.7e50), overflowing input derivative (times 1/s = 3.8e260).
    kan::set_wavelet_parameters(big, std::vector<double>{1e-80, 0}, std::vector<double>{0, 0},
                                std::vector<double>{-600, 0}, std::vector<double>{0});
    ResidentNetwork steep(kan::Network({big}), 1);
    steep.upload_input(std::vector<double>{1e-261}, 1);
    steep.upload_output_gradient(std::vector<double>{1});
    steep.forward();
    test::throws<std::overflow_error>([&] { steep.backward(); });
}

int main() { return test::run(); }

// Backlog C5: sparse B-spline path of the resident executor. A B-spline layer
// stores per (sample, input) the first possibly nonzero term and its degree+1
// values instead of all `terms` basis values, and contracts only those. These
// cases check the selection rule (through the private test hook, which reports
// each executor's layout and can force a path), parity with the CPU and with
// the dense path within the contract tolerances in every precision, and the
// executor features the path must keep: L2, residual branch, input maps, mixed
// networks, batch 0 and partial batches, graph-captured training steps,
// deferred status and rollback, parameter upload, determinism, and knot
// insertion that moves a layer across the selection threshold.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;
using kan::cuda::testing::ResidentLayout;
using kan::cuda::testing::ResidentTestScope;
using kan::cuda::testing::SplinePath;

// Clamped spline with `terms` terms on [low, high], uniform interior knots.
kan::BSplineConfig spline(std::size_t degree, std::size_t terms, double low = -1, double high = 1) {
    kan::BSplineConfig b{degree, std::vector<double>(degree + 1, low)};
    const auto interior = terms - degree - 1;
    for (std::size_t k = 1; k <= interior; ++k)
        b.knots.push_back(low + (high - low) * static_cast<double>(k) / static_cast<double>(interior + 1));
    b.knots.insert(b.knots.end(), degree + 1, high);
    return b;
}
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale * std::sin(frequency * static_cast<double>(i) + phase);
    return v;
}
kan::Layer seeded(kan::Layer l, double phase) {
    const auto scale = 1.0 / std::sqrt(static_cast<double>(l.inputs() * l.terms()));
    l.set_parameters(wave(l.coefficients().size(), 3 * scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
    return l;
}
// Inputs inside [-1, 1] with the endpoints, knots and points outside the
// domain (zero basis: dead edges) mixed in.
std::vector<double> inputs(std::size_t count, double phase = 0) {
    auto x = wave(count, 1.05, 0.377, phase);
    const double special[] = {-1, 1, 0, -0.5, 0.5, -2, 3, 1e3};
    for (std::size_t k = 0; k < count; k += 7) x[k] = special[(k / 7) % 8];
    return x;
}

struct Tolerance { double relative, floor; };
Tolerance tolerance(Precision p) {
    if (p == Precision::Float64) return {1e-11, 1e-12};
    if (p == Precision::Float32) return {2e-4, 5e-5};
    return {1e-2, 1e-2};
}
void compare(std::span<const double> actual, std::span<const double> expected, Tolerance t, const std::string& what) {
    REQUIRE(actual.size() == expected.size());
    double scale = 0;
    for (double e : expected) scale = std::max(scale, std::abs(e));
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (!(std::abs(actual[i] - expected[i]) <= t.relative * std::abs(expected[i]) + t.floor * scale + 1e-300))
            throw std::runtime_error(what + " index " + std::to_string(i) + " actual=" + std::to_string(actual[i]) +
                                     " expected=" + std::to_string(expected[i]));
}
bool same_bits(std::span<const double> a, std::span<const double> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}
ResidentLayout layout_of(const kan::Network& n, std::size_t capacity, Precision p, SplinePath path = SplinePath::Automatic) {
    ResidentLayout layout;
    ResidentTestScope scope(path, &layout);
    ResidentNetwork r(n, capacity, p);
    return layout;
}
kan::Network single(kan::Layer l) { return kan::Network({std::move(l)}); }

// Forward and backward (L2) on the GPU against the CPU network.
void check_against_cpu(const kan::Network& cpu, ResidentNetwork& gpu, std::size_t batch, double l2, Precision p, const std::string& what) {
    const auto x = inputs(batch * cpu.inputs(), 0.3);
    const auto dy = wave(batch * cpu.outputs(), 0.7, 0.613, 0.4);
    gpu.upload_input(x, batch);
    gpu.forward();
    compare(gpu.download_output(), cpu.forward(x, batch), tolerance(p), what + " output");
    gpu.upload_output_gradient(dy);
    gpu.backward(l2);
    const auto actual = gpu.download_gradients();
    auto expected = cpu.backward(x, batch, dy);
    const auto penalty = cpu.regularization(l2).gradients;
    compare(actual.input, expected.input, tolerance(p), what + " input gradient");
    for (std::size_t j = 0; j < actual.layers.size(); ++j) {
        if (!std::holds_alternative<kan::LayerGradients>(actual.layers[j])) continue;
        auto& e = test::grad(expected, j);
        const auto& r = test::grad(penalty, j);
        for (std::size_t k = 0; k < e.coefficients.size(); ++k) e.coefficients[k] += r.coefficients[k];
        for (std::size_t k = 0; k < e.residual.size(); ++k) e.residual[k] += r.residual[k];
        const auto& a = test::grad(actual, j);
        compare(a.coefficients, e.coefficients, tolerance(p), what + " coefficients " + std::to_string(j));
        compare(a.bias, e.bias, tolerance(p), what + " bias " + std::to_string(j));
        compare(a.residual, e.residual, tolerance(p), what + " residual " + std::to_string(j));
    }
}
// The CPU reference evaluated at the executor's (rounded) parameters.
kan::Network reference(ResidentNetwork& gpu) { return gpu.download_parameters(); }
constexpr Precision precisions[] = {Precision::Float64, Precision::Float32, Precision::TensorFloat32};
} // namespace

// Automatic selection: large-term-count spline layers with enough work take
// the sparse path in every precision; small term counts or work, latency-bound
// shapes, unsupported degrees and other families keep the dense rows. A sparse layer reserves no dense Phi/Phi'
// rows and no share of the W = U*C scratch.
// The large layer clears the rule on any device with up to about 150 SMs
// (1.6e9 multiply-adds against 2^23 per SM for TF32) and with a full-rate FP64
// unit (FP32 values), with its upstream (1 MiB FP32) well inside half the L2.
TEST(selection_rule_picks_sparse_for_large_spline_layers) {
    const auto large = single(seeded(kan::Layer(128, 64, spline(3, 48)), 0.1));
    for (auto p : precisions) REQUIRE(layout_of(large, 4096, p).sparse.at(0));
    REQUIRE(!layout_of(single(seeded(kan::Layer(64, 48, spline(3, 6)), 0.1)), 1024, Precision::Float32).sparse.at(0));
    REQUIRE(!layout_of(single(seeded(kan::Layer(64, 48, spline(3, 6)), 0.1)), 1024, Precision::TensorFloat32).sparse.at(0));
    REQUIRE(!layout_of(single(seeded(kan::Layer(3, 2, spline(3, 48)), 0.1)), 8, Precision::Float64).sparse.at(0));
    REQUIRE(!layout_of(single(seeded(kan::Layer(64, 48, spline(9, 60)), 0.1)), 1024, Precision::Float64).sparse.at(0));
    REQUIRE(!layout_of(single(seeded(kan::Layer(64, 48, kan::ChebyshevConfig{48}), 0.1)), 1024, Precision::Float64).sparse.at(0));
    // Mixed network: only the large spline layer is sparse.
    const kan::Network mixed({seeded(kan::Layer(128, 64, spline(3, 48)), 0.1), seeded(kan::Layer(64, 8, kan::ChebyshevConfig{5}), 0.2)});
    const auto layout = layout_of(mixed, 4096, Precision::Float32);
    REQUIRE(layout.sparse.size() == 2 && layout.sparse[0] && !layout.sparse[1]);
    // Memory: the dense layout holds capacity*inputs*terms values (Phi) plus the
    // W scratch, the sparse one about (2*(degree+1)+4) values per sample and input.
    for (auto p : {Precision::Float64, Precision::Float32}) {
        const auto sparse = layout_of(large, 4096, p, SplinePath::Automatic).arena_bytes;
        const auto dense = layout_of(large, 4096, p, SplinePath::Dense).arena_bytes;
        const auto rows = std::size_t{4096} * 128 * 48 * (p == Precision::Float64 ? 8 : 4);
        REQUIRE(dense > sparse);
        REQUIRE(dense - sparse >= rows);
    }
    // Forcing: Sparse applies to supported layers only.
    REQUIRE(layout_of(single(seeded(kan::Layer(3, 2, spline(2, 5)), 0.1)), 8, Precision::Float64, SplinePath::Sparse).sparse.at(0));
    REQUIRE(!layout_of(single(seeded(kan::Layer(3, 2, spline(9, 12)), 0.1)), 8, Precision::Float64, SplinePath::Sparse).sparse.at(0));
    REQUIRE(!layout_of(large, 4096, Precision::Float64, SplinePath::Dense).sparse.at(0));
}

// Every supported degree, forced sparse and automatic, against the CPU at the
// executor's parameters and against the forced dense path, with L2, partial
// batches and inputs at knots, endpoints and outside the domain.
TEST(sparse_matches_cpu_and_dense_for_every_supported_degree) {
    for (auto p : precisions) {
        for (std::size_t degree = 0; degree <= 3; ++degree) {
            const kan::Network n({seeded(kan::Layer(24, 20, spline(degree, 9 + 3 * degree)), 0.2),
                                  seeded(kan::Layer(20, 6, kan::ChebyshevConfig{4}), 0.4)});
            const auto what = "precision " + std::to_string(static_cast<int>(p)) + " degree " + std::to_string(degree);
            ResidentLayout layout;
            ResidentTestScope scope(SplinePath::Sparse, &layout);
            ResidentNetwork sparse(n, 64, p);
            REQUIRE(layout.sparse.at(0));
            const auto cpu = reference(sparse);
            for (std::size_t batch : {64u, 37u, 1u}) check_against_cpu(cpu, sparse, batch, 0.05, p, what);
        }
    }
    // Forced sparse, large term count, against the forced dense path of the same precision
    // (FP64, FP32) and the CPU (all). TF32 is compared with the CPU only: the
    // dense TF32 forward rounds layer 0's outputs by about 2^-11 of their terms
    // (2.6e-4 here), which moves some layer-1 inputs across a knot of the
    // degree-1 layer, whose derivative jumps there; the sparse kernels compute
    // in FP32. (Backlog C13 evidence: against the CPU evaluated at the dense
    // executor's own activations the dense TF32 input VJP is within the C1
    // tolerance; the case's 0.46 deviation, expected 0.398 against -0.060,
    // was this jump, not the input-VJP formula.)
    for (auto p : precisions) {
        const kan::Network n({seeded(kan::Layer(64, 40, spline(3, 48)), 0.2), seeded(kan::Layer(40, 4, spline(1, 30)), 0.6)});
        ResidentLayout layout;
        std::vector<double> sparse_out, dense_out;
        kan::NetworkGradients sparse_g, dense_g;
        for (auto path : {SplinePath::Sparse, SplinePath::Dense}) {
            ResidentTestScope scope(path, &layout);
            ResidentNetwork gpu(n, 1024, p);
            REQUIRE(layout.sparse.at(0) == (path == SplinePath::Sparse));
            const auto x = inputs(1000 * 64, 0.9);
            gpu.upload_input(x, 1000);
            gpu.forward();
            gpu.upload_output_gradient(wave(1000 * 4, 0.3, 0.17));
            gpu.backward(0.01);
            (path == SplinePath::Sparse ? sparse_out : dense_out) = gpu.download_output();
            (path == SplinePath::Sparse ? sparse_g : dense_g) = gpu.download_gradients();
            if (path == SplinePath::Sparse) check_against_cpu(reference(gpu), gpu, 1000, 0.01, p, "forced sparse");
        }
        if (p == Precision::TensorFloat32) continue;
        compare(sparse_out, dense_out, tolerance(p), "sparse vs dense output");
        compare(sparse_g.input, dense_g.input, tolerance(p), "sparse vs dense input gradient");
        for (std::size_t j = 0; j < 2; ++j) {
            compare(test::grad(sparse_g, j).coefficients, test::grad(dense_g, j).coefficients, tolerance(p), "sparse vs dense coefficients");
            compare(test::grad(sparse_g, j).bias, test::grad(dense_g, j).bias, tolerance(p), "sparse vs dense bias");
        }
    }
}

// Knots: repeated interior knots, a huge domain, a single interval (degree+1
// terms) and inputs exactly at every knot; batch zero gives L2-only gradients.
TEST(sparse_knot_contracts_batch_zero_and_dead_edges) {
    const double big = 1e300;
    const std::vector<kan::BSplineConfig> configs{
        {3, {-1, -1, -1, -1, 0, 0, 0, 0.5, 1, 1, 1, 1}}, {2, {-big, -big, -big, 0, big, big, big}},
        {3, {-1, -1, -1, -1, 1, 1, 1, 1}}, {1, {-1, -1, -0.5, -0.5, 0, 0.5, 1, 1}}, {0, {-1, -0.25, 0.5, 1}}};
    for (auto p : {Precision::Float64, Precision::Float32}) {
        for (const auto& b : configs) {
            if (p == Precision::Float32 && b.knots.front() < -1e30) continue; // not representable in FP32
            const auto n = single(seeded(kan::Layer(5, 3, b), 0.3));
            ResidentLayout layout;
            ResidentTestScope scope(SplinePath::Sparse, &layout);
            ResidentNetwork gpu(n, 32, p);
            REQUIRE(layout.sparse.at(0));
            const auto cpu = reference(gpu);
            check_against_cpu(cpu, gpu, 32, 0.1, p, "knots");
            std::vector<double> x;
            for (std::size_t k = 0; k < 6; ++k)
                for (double t : b.knots) x.push_back(t);
            x.resize(x.size() / 5 * 5);
            const auto batch = x.size() / 5;
            gpu.upload_input(x, batch);
            gpu.forward();
            compare(gpu.download_output(), cpu.forward(x, batch), tolerance(p), "at knots");
            gpu.upload_output_gradient(wave(batch * 3, 1, 0.5));
            gpu.backward();
            compare(gpu.download_gradients().input, cpu.backward(x, batch, wave(batch * 3, 1, 0.5)).input, tolerance(p), "at knots dx");
            // Batch zero: gradients are the L2 term only (zero bias VJP).
            gpu.upload_input({}, 0);
            gpu.forward();
            gpu.upload_output_gradient({});
            gpu.backward(0.25);
            const auto g = gpu.download_gradients();
            const auto penalty = cpu.regularization(0.25).gradients;
            compare(test::grad(g, 0).coefficients, test::grad(penalty, 0).coefficients, tolerance(p), "batch zero");
            for (double v : test::grad(g, 0).bias) REQUIRE(v == 0 && !std::signbit(v));
        }
    }
    // Every input outside the domain: zero output contribution and zero gradients.
    const auto n = single(seeded(kan::Layer(4, 2, spline(3, 20)), 0.1));
    ResidentTestScope scope(SplinePath::Sparse);
    ResidentNetwork gpu(n, 4, Precision::Float64);
    gpu.upload_input(std::vector<double>(16, 5.0), 4);
    gpu.forward();
    const auto y = gpu.download_output();
    for (std::size_t k = 0; k < y.size(); ++k) REQUIRE(y[k] == test::layer(n, 0).bias()[k % 2]);
    gpu.upload_output_gradient(std::vector<double>(8, 1.0));
    gpu.backward();
    const auto g = gpu.download_gradients();
    for (double v : g.input) REQUIRE(v == 0);
    for (double v : test::grad(g, 0).coefficients) REQUIRE(v == 0);
}

// Residual branch, input maps and every other carrier around a sparse layer.
TEST(sparse_layers_in_mixed_networks_with_residual_branch) {
    for (auto p : precisions) {
        kan::Layer first = seeded(kan::Layer(32, 24, spline(3, 40)), 0.1);
        first.set_residual(kan::SiluResidual{wave(32 * 24, 0.2, 0.3)});
        kan::Layer second = seeded(kan::Layer(24, 16, kan::TrainableRbfConfig{{-1, 0, 1}, {-0.5, 0, 0.3}}), 0.2);
        kan::Layer third = seeded(kan::Layer(16, 12, spline(2, 30)), 0.3);
        third.set_residual(kan::SiluResidual{wave(16 * 12, 0.1, 0.7)});
        kan::Layer fourth = seeded(kan::Layer(12, 3, kan::RationalConfig{}), 0.4);
        std::vector<kan::NetworkLayer> layers;
        layers.emplace_back(kan::InputMap(32, kan::TanhMap{}));
        layers.emplace_back(first);
        layers.emplace_back(kan::InputMap(24, kan::LayerNormMap{1e-5, std::vector<double>(24, 1.0), std::vector<double>(24, 0.0)}));
        layers.emplace_back(second);
        layers.emplace_back(third);
        layers.emplace_back(fourth);
        const kan::Network n(std::move(layers));
        ResidentLayout layout;
        ResidentTestScope scope(SplinePath::Sparse, &layout);
        ResidentNetwork gpu(n, 128, p);
        REQUIRE(layout.sparse.at(1) && layout.sparse.at(4) && !layout.sparse.at(3));
        const auto cpu = reference(gpu);
        check_against_cpu(cpu, gpu, 128, 0.02, p, "mixed");
        check_against_cpu(cpu, gpu, 77, 0, p, "mixed partial");
    }
}

// Captured training steps replay the eager sequence bitwise; repeated runs
// are deterministic; SGD trajectories follow the CPU.
TEST(sparse_training_steps_are_bitwise_eager_and_deterministic) {
    for (auto p : precisions) {
        const kan::Network n({seeded(kan::Layer(48, 64, spline(3, 48)), 0.2), seeded(kan::Layer(64, 64, spline(2, 40)), 0.5),
                              seeded(kan::Layer(64, 5, kan::ChebyshevConfig{4}), 0.7)});
        const auto x = inputs(512 * 48, 0.7);
        const auto dy = wave(512 * 5, 0.2, 0.31);
        ResidentLayout layout;
        ResidentTestScope scope(SplinePath::Sparse, &layout);
        ResidentNetwork eager(n, 512, p), graph(n, 512, p), again(n, 512, p);
        REQUIRE(layout.sparse.at(0) && layout.sparse.at(1));
        for (auto* r : {&eager, &graph, &again}) { r->upload_input(x, 512); r->upload_output_gradient(dy); }
        for (int step = 0; step < 4; ++step) {
            eager.forward(); eager.backward(0.01); eager.sgd(0.05);
            graph.train_step(0.05, 0.01);
            again.train_step(0.05, 0.01);
        }
        const auto a = eager.download_parameters(), b = graph.download_parameters(), c = again.download_parameters();
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(same_bits(test::layer(a, j).coefficients(), test::layer(b, j).coefficients()));
            REQUIRE(same_bits(test::layer(a, j).bias(), test::layer(b, j).bias()));
            REQUIRE(same_bits(test::layer(b, j).coefficients(), test::layer(c, j).coefficients()));
        }
        // MSE steps on staged batches against the CPU trajectory (FP64 only: SGD amplifies rounding).
        if (p == Precision::Float64) {
            auto cpu = n;
            ResidentNetwork mse(n, 512, p);
            const auto target = wave(512 * 5, 0.5, 0.09);
            for (int step = 0; step < 3; ++step) {
                mse.train_step(x, target, 512, 0.1);
                auto y = cpu.forward(x, 512);
                for (std::size_t k = 0; k < y.size(); ++k) y[k] = (y[k] - target[k]) * (2.0 / static_cast<double>(y.size()));
                cpu.sgd(cpu.backward(x, 512, y), 0.1);
            }
            const auto trained = mse.download_parameters();
            for (std::size_t j = 0; j < 3; ++j)
                compare(test::layer(trained, j).coefficients(), test::layer(cpu, j).coefficients(), tolerance(p), "mse trajectory");
        }
    }
}

// Nonfinite results of a sparse layer are reported like the dense path's:
// eagerly, and by a training step that rolls back to the last good step.
TEST(sparse_status_reports_overflow_and_rolls_back) {
    auto l = seeded(kan::Layer(16, 4, spline(3, 40)), 0.1);
    std::vector<double> c(l.coefficients().size(), 1e308);
    l.set_parameters(c, std::vector<double>(4, 0));
    ResidentTestScope scope(SplinePath::Sparse);
    {
        ResidentNetwork gpu(single(l), 16, Precision::Float64);
        gpu.upload_input(inputs(16 * 16), 16);
        test::throws<std::overflow_error>([&] { gpu.forward(); });
    }
    const auto good = single(seeded(kan::Layer(16, 4, spline(3, 40)), 0.1));
    ResidentNetwork gpu(good, 16, Precision::Float32);
    gpu.upload_input(inputs(16 * 16), 16);
    gpu.upload_output_gradient(wave(16 * 4, 0.1, 0.4));
    gpu.train_step(0.01);
    const auto before = gpu.download_parameters();
    gpu.upload_output_gradient(std::vector<double>(16 * 4, 3e38)); // dC = sum_b u*phi overflows FP32
    test::throws<std::overflow_error>([&] { gpu.train_step(0.01); });
    const auto after = gpu.download_parameters();
    REQUIRE(same_bits(test::layer(before, 0).coefficients(), test::layer(after, 0).coefficients()));
    gpu.upload_output_gradient(wave(16 * 4, 0.1, 0.4));
    gpu.train_step(0.01); // usable afterwards
}

// upload_parameters reaches the sparse kernels; knot insertion that raises the
// term count across the threshold gives a sparse executor computing the same
// function (refinement preserves it) as the dense executor before.
TEST(sparse_upload_parameters_and_knot_insertion_growth) {
    // 6 terms (ratio 1.5, below FP32's 4 on every device) grow to 30 (ratio 7.5;
    // 1.0e9 multiply-adds, above 2^22 per SM up to about 230 SMs).
    const auto n = single(seeded(kan::Layer(128, 64, spline(3, 6)), 0.2));
    ResidentLayout layout;
    ResidentTestScope scope(SplinePath::Automatic, &layout);
    ResidentNetwork before(n, 4096, Precision::Float32);
    REQUIRE(!layout.sparse.at(0));
    auto refined = before.download_parameters();
    for (int k = 0; k < 24; ++k) refined.insert_knot(0, -0.97 + 0.08 * k + 0.003);
    ResidentNetwork after(refined, 4096, Precision::Float32);
    REQUIRE(layout.sparse.at(0));
    const auto x = inputs(1000 * 128, 0.2);
    before.upload_input(x, 1000); before.forward();
    after.upload_input(x, 1000); after.forward();
    compare(after.download_output(), before.download_output(), tolerance(Precision::Float32), "refined output");
    // New parameters through upload_parameters.
    auto changed = refined;
    auto layer = test::layer(changed, 0);
    layer.set_parameters(wave(layer.coefficients().size(), 0.05, 0.11, 1.0), wave(64, 0.1, 0.2));
    changed = single(layer);
    after.upload_parameters(changed);
    check_against_cpu(after.download_parameters(), after, 1000, 0, Precision::Float32, "uploaded");
    after.upload_input(x, 1000); after.forward();
    compare(after.download_output(), after.download_parameters().forward(x, 1000), tolerance(Precision::Float32), "uploaded output");
}

// Long rows: 400 terms need fewer inputs per input-VJP block and more than
// the default 48 KiB of dynamic shared memory in both sparse kernels (FP64).
TEST(sparse_long_rows_use_smaller_input_groups) {
    for (auto p : {Precision::Float64, Precision::Float32}) {
        const auto n = single(seeded(kan::Layer(9, 5, spline(3, 400)), 0.3));
        ResidentLayout layout;
        ResidentTestScope scope(SplinePath::Sparse, &layout);
        ResidentNetwork gpu(n, 300, p);
        REQUIRE(layout.sparse.at(0));
        check_against_cpu(reference(gpu), gpu, 300, 0.01, p, "long rows");
    }
}

int main() {
    if (!kan::cuda::available()) { std::cout << "SKIP no CUDA device\n"; return 0; }
    return test::run();
}

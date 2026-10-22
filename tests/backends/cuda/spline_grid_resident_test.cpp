// Backlog M6: per-input B-spline grids on the resident executor (dense and
// sparse paths, every precision) and the device grid refit
// (ResidentNetwork::refit_grids) against the CPU refit.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;
using kan::cuda::testing::ResidentLayout;
using kan::cuda::testing::ResidentTestScope;
using kan::cuda::testing::SplinePath;

std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale * std::sin(frequency * static_cast<double>(i) + phase);
    return v;
}
// `inputs` different grids of degree p with `terms` terms: input i on
// [-1 - 0.1 i, 1 + 0.2 i] with interior knots bent by a per-input power.
kan::BSplineConfig grids(std::size_t p, std::size_t terms, std::size_t inputs) {
    kan::BSplineConfig c{p, {}, inputs};
    const auto intervals = terms - p;
    for (std::size_t i = 0; i < inputs; ++i) {
        const double lo = -1 - 0.1 * double(i % 5), hi = 1 + 0.2 * double(i % 3), power = 1 + 0.15 * double(i % 4);
        c.knots.insert(c.knots.end(), p + 1, lo);
        for (std::size_t j = 1; j < intervals; ++j) c.knots.push_back(lo + (hi - lo) * std::pow(double(j) / double(intervals), power));
        c.knots.insert(c.knots.end(), p + 1, hi);
    }
    return c;
}
kan::BSplineConfig shared_grid(std::size_t p, std::size_t terms) { return kan::spline_grid(grids(p, terms, 1), 0); }
kan::Layer seeded(kan::Layer l, double phase) {
    const auto scale = 1.0 / std::sqrt(static_cast<double>(l.inputs() * l.terms()));
    l.set_parameters(wave(l.coefficients().size(), 3 * scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
    return l;
}
// Inputs over the union of the domains, with knots, endpoints and outside points.
std::vector<double> inputs(std::size_t count, double phase = 0) {
    auto x = wave(count, 1.3, 0.377, phase);
    const double special[] = {-1, 1, 0, -0.5, 1.2, -2, 3, 1e3};
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
constexpr Precision precisions[] = {Precision::Float64, Precision::Float32, Precision::TensorFloat32};
const char* name(Precision p) { return p == Precision::Float64 ? "f64" : p == Precision::Float32 ? "f32" : "tf32"; }
const kan::BSplineConfig& spline_of(const kan::Network& n, std::size_t j) {
    return std::get<kan::BSplineConfig>(std::get<kan::BasisEdges>(std::get<kan::Layer>(n.layers()[j]).carrier()).basis);
}

// Forward and backward (L2) against the CPU at the executor's parameters.
void check_against_cpu(ResidentNetwork& gpu, std::size_t batch, double l2, Precision p, const std::string& what) {
    const auto cpu = gpu.download_parameters();
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
// Network inputs inside the first layer's domains, batch x inputs.
std::vector<double> refit_samples(std::size_t batch, std::size_t inputs, double phase = 0) {
    std::vector<double> x(batch * inputs);
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t i = 0; i < inputs; ++i)
            x[b * inputs + i] = 0.98 * std::sin(1.7 * double(b) + 2.1 * double(i) + phase + 0.4 * double(b * b % 7));
    return x;
}
} // namespace

// Every degree and path, with and without the residual branch, odd shapes,
// partial batches and batch 0.
TEST(per_input_layers_match_the_cpu) {
    for (auto p : precisions)
        for (std::size_t degree : {0u, 1u, 2u, 3u, 5u})
            for (auto path : {SplinePath::Dense, SplinePath::Sparse})
                for (bool residual : {false, true}) {
                    auto layer = seeded(kan::Layer(37, 9, grids(degree, degree + 7, 37)), 0.2);
                    if (residual) layer.set_residual(kan::SiluResidual{wave(37 * 9, 0.3, 0.91, 0.1)});
                    kan::Network network({seeded(kan::Layer(37, 37, grids(3, 12, 37)), 0.5), layer});
                    ResidentLayout layout;
                    ResidentTestScope scope(path, &layout);
                    ResidentNetwork gpu(network, 70, p);
                    REQUIRE(layout.sparse.at(1) == (path == SplinePath::Sparse && degree <= 3));
                    const auto what = std::string(name(p)) + " degree " + std::to_string(degree) +
                                      (path == SplinePath::Sparse ? " sparse" : " dense") + (residual ? " residual" : "");
                    check_against_cpu(gpu, 70, 0.01, p, what);
                    check_against_cpu(gpu, 33, 0.0, p, what + " partial");
                    gpu.upload_input({}, 0);
                    gpu.forward();
                    REQUIRE(gpu.download_output().empty());
                }
    // Rows too long for the FP32 stage (above 85 terms): the unstaged kernel path.
    for (auto p : precisions) {
        ResidentTestScope scope(SplinePath::Dense);
        ResidentNetwork gpu(kan::Network({seeded(kan::Layer(5, 3, grids(3, 120, 5)), 0.6)}), 64, p);
        check_against_cpu(gpu, 64, 0.01, p, std::string(name(p)) + " long rows");
    }
}

// Identical grids repeated per input give bitwise the shared layer's results.
TEST(identical_per_input_grids_match_the_shared_layer_bitwise) {
    for (auto p : precisions)
        for (auto path : {SplinePath::Dense, SplinePath::Sparse})
            for (std::size_t degree : {1u, 3u}) {
                const auto shared = shared_grid(degree, 11);
                const auto a = seeded(kan::Layer(19, 6, shared), 0.4);
                auto b = kan::Layer(19, 6, kan::per_input_grids(shared, 19));
                b.set_parameters(a.coefficients(), a.bias());
                ResidentTestScope scope(path);
                ResidentNetwork ga(kan::Network({a}), 40, p), gb(kan::Network({b}), 40, p);
                const auto x = inputs(40 * 19);
                const auto dy = wave(40 * 6, 0.5, 0.3);
                for (auto* g : {&ga, &gb}) { g->upload_input(x, 40); g->upload_output_gradient(dy); g->forward(); g->backward(0.02); }
                REQUIRE(same_bits(ga.download_output(), gb.download_output()));
                const auto da = ga.download_gradients(), db = gb.download_gradients();
                REQUIRE(same_bits(da.input, db.input));
                REQUIRE(same_bits(test::grad(da, 0).coefficients, test::grad(db, 0).coefficients));
            }
}

// The device refit against the CPU refit of the downloaded network on the
// same input: FP64 within the FP64 tolerance; FP32/TF32 within the FP32
// tolerance of the knots' span and of the refit functions.
TEST(device_refit_matches_the_cpu_refit) {
    for (auto p : precisions)
        for (auto path : {SplinePath::Dense, SplinePath::Sparse})
            for (std::size_t degree : {0u, 1u, 3u, 5u}) {
                kan::Network network({kan::NetworkLayer(kan::InputMap(13, kan::TanhMap{1.2})),
                                      kan::NetworkLayer(seeded(kan::Layer(13, 21, grids(degree, degree + 6, 13)), 0.3)),
                                      kan::NetworkLayer(seeded(kan::Layer(21, 7, kan::ChebyshevConfig{4}), 0.6)),
                                      kan::NetworkLayer(seeded(kan::Layer(7, 3, shared_grid(degree, degree + 9)), 0.1))});
                for (bool per_input : {true, false}) {
                    if (!per_input) { // a shared first layer refits to a shared grid
                        auto layers = std::vector<kan::NetworkLayer>(network.layers().begin(), network.layers().end());
                        std::get<kan::Layer>(layers[1]) = seeded(kan::Layer(13, 21, shared_grid(degree, degree + 6)), 0.3);
                        network = kan::Network(std::move(layers));
                    }
                    ResidentTestScope scope(path);
                    const std::size_t batch = 301;
                    ResidentNetwork gpu(network, 320, p);
                    const auto x = refit_samples(batch, 13);
                    gpu.upload_input(x, batch);
                    auto cpu = gpu.download_parameters();
                    const kan::GridRefit options{0, 0.05, 1e-8, per_input};
                    // The last layer is shared: per_input must match every layer's form.
                    if (per_input) test::throws<std::invalid_argument>([&] { gpu.refit_grids(options); });
                    auto layers = std::vector<kan::NetworkLayer>(cpu.layers().begin(), cpu.layers().end());
                    std::get<kan::Layer>(layers[3]) = [&] {
                        const auto& l = std::get<kan::Layer>(layers[3]);
                        kan::Layer r(7, 3, per_input ? kan::per_input_grids(spline_of(cpu, 3), 7) : spline_of(cpu, 3));
                        r.set_parameters(l.coefficients(), l.bias());
                        return r;
                    }();
                    cpu = kan::Network(std::move(layers));
                    ResidentNetwork device(cpu, 320, p);
                    device.upload_input(x, batch);
                    const auto allocations = device.workspace_allocations();
                    device.refit_grids(options);
                    REQUIRE(device.workspace_allocations() == allocations);
                    REQUIRE(cpu.refit_grids(x, batch, options) == 2);
                    const auto refit = device.download_parameters();
                    const auto what = std::string(name(p)) + " degree " + std::to_string(degree) +
                                      (path == SplinePath::Sparse ? " sparse" : " dense") + (per_input ? "" : " shared");
                    for (std::size_t j : {1u, 3u}) {
                        const auto& a = spline_of(refit, j);
                        const auto& e = spline_of(cpu, j);
                        REQUIRE(a.grids == e.grids && a.degree == e.degree && a.knots.size() == e.knots.size());
                        if (p == Precision::Float64) compare(a.knots, e.knots, {1e-14, 1e-15}, what + " knots");
                        else compare(a.knots, e.knots, {1e-5, 1e-5}, what + " knots");
                    }
                    const auto xa = refit_samples(97, 13, 0.7);
                    compare(refit.forward(xa, 97), cpu.forward(xa, 97),
                            p == Precision::Float64 ? Tolerance{1e-10, 1e-11} : Tolerance{2e-3, 2e-3}, what + " refit functions");
                }
            }
}

// After a device refit, captured training steps (graphs recorded before the
// refit) replay with the refit knots: bitwise the steps of an executor built
// from the downloaded refit network. Uploads see the refit structure.
TEST(refit_keeps_graphs_uploads_and_allocations) {
    for (auto p : precisions)
        for (auto path : {SplinePath::Dense, SplinePath::Sparse}) {
            kan::Network network({seeded(kan::Layer(24, 16, grids(3, 10, 24)), 0.3), seeded(kan::Layer(16, 4, grids(2, 9, 16)), 0.8)});
            ResidentTestScope scope(path);
            const std::size_t batch = 128;
            const auto x = refit_samples(batch, 24);
            const auto dy = wave(batch * 4, 0.01, 0.37);
            ResidentNetwork a(network, batch, p);
            a.upload_input(x, batch);
            a.upload_output_gradient(dy);
            for (int k = 0; k < 3; ++k) a.train_step(0.05);
            a.synchronize();
            const auto before = a.download_parameters();
            a.refit_grids();
            const auto refit = a.download_parameters();
            REQUIRE(!(spline_of(refit, 0) == spline_of(before, 0)));
            // The upload checks the refit structure: the old knots are rejected.
            test::throws<std::invalid_argument>([&] { a.upload_parameters(before); });
            a.upload_parameters(refit);
            ResidentNetwork b(refit, batch, p);
            b.upload_input(x, batch);
            for (auto* g : {&a, &b}) { g->upload_output_gradient(dy); for (int k = 0; k < 4; ++k) g->train_step(0.05); g->synchronize(); }
            const auto pa = a.download_parameters(), pb = b.download_parameters();
            for (std::size_t j = 0; j < 2; ++j) {
                REQUIRE(spline_of(pa, j) == spline_of(pb, j));
                REQUIRE(same_bits(std::get<kan::Layer>(pa.layers()[j]).coefficients(), std::get<kan::Layer>(pb.layers()[j]).coefficients()));
            }
            // Output and gradients are stale after a refit; input and upstream stay.
            a.forward();
            a.backward();
            a.download_output();
            a.refit_grids();
            test::throws<std::logic_error>([&] { a.download_output(); });
            test::throws<std::logic_error>([&] { a.download_gradients(); });
            a.forward();
            a.backward();
        }
}

TEST(refit_failures_leave_the_executor_unchanged) {
    for (auto p : precisions) {
        kan::Network network({seeded(kan::Layer(6, 5, grids(3, 8, 6)), 0.3), kan::Layer(5, 4, kan::ChebyshevConfig{3}),
                              seeded(kan::Layer(4, 2, grids(3, 8, 4)), 0.8)});
        const std::size_t batch = 50;
        auto x = refit_samples(batch, 6);
        ResidentNetwork gpu(network, batch, p);
        test::throws<std::logic_error>([&] { gpu.refit_grids(); }); // no input
        gpu.upload_input(x, batch);
        const auto before = gpu.download_parameters();
        const auto unchanged = [&] {
            const auto now = gpu.download_parameters();
            for (std::size_t j : {0u, 2u}) {
                REQUIRE(spline_of(now, j) == spline_of(before, j));
                REQUIRE(same_bits(std::get<kan::Layer>(now.layers()[j]).coefficients(), std::get<kan::Layer>(before.layers()[j]).coefficients()));
            }
        };
        // Constant activations of the last layer (zero Chebyshev layer): no interval.
        test::throws<std::invalid_argument>([&] { gpu.refit_grids(); });
        unchanged();
        // Options the device does not change: the term count and the grid form.
        auto ok = kan::Network({seeded(kan::Layer(6, 4, grids(3, 8, 6)), 0.3)});
        ResidentNetwork single(ok, batch, p);
        single.upload_input(x, batch);
        test::throws<std::invalid_argument>([&] { single.refit_grids(kan::GridRefit{7}); });
        test::throws<std::invalid_argument>([&] { single.refit_grids(kan::GridRefit{0, 0.02, 1e-8, false}); });
        test::throws<std::invalid_argument>([&] { single.refit_grids(kan::GridRefit{0, 2.0}); });
        single.refit_grids(kan::GridRefit{5, 0.02});
        // Repeated samples with grid_eps = 0: coinciding knots.
        for (std::size_t b = 0; b + 3 < batch; ++b) x[b * 6 + 2] = 0.5;
        single.upload_input(x, batch);
        const auto kept = single.download_parameters();
        test::throws<std::invalid_argument>([&] { single.refit_grids(kan::GridRefit{0, 0.0}); });
        REQUIRE(spline_of(single.download_parameters(), 0) == spline_of(kept, 0));
        // Fewer samples than intervals with ridge 0: a singular fit.
        single.upload_input(std::vector<double>(x.begin() + 47 * 6, x.begin() + 49 * 6), 2);
        test::throws<std::invalid_argument>([&] { single.refit_grids(kan::GridRefit{0, 0.5, 0.0}); });
        REQUIRE(spline_of(single.download_parameters(), 0) == spline_of(kept, 0));
        single.refit_grids(kan::GridRefit{0, 0.5, 1e-8});
        single.upload_input({}, 0);
        test::throws<std::invalid_argument>([&] { single.refit_grids(); });
        // A network without B-spline layers.
        ResidentNetwork none(kan::Network({kan::Layer(6, 2, kan::ChebyshevConfig{3})}), batch, p);
        none.upload_input(x, batch);
        test::throws<std::invalid_argument>([&] { none.refit_grids(); });
    }
}

int main() { return kan::cuda::available() ? test::run() : 0; }

// Backlog C12: center/log-width reduction of resident TrainableRbfEdges layers.
// The executor reduces sum_r W[r,k]*(-dPhi/dx) and W[r,k]*dPhi/dlog-width over
// the rows r = (sample, input) either with the column kernel (one block per
// term and tile, strided column reads) or with the tile kernel (one block per
// tile and term group, coalesced row reads, the same per-lane chains and
// trees), and FP32 layers that fit the C1 stage with the staged kernel. These
// cases force each path through the private test hook and require bitwise
// equal gradients and trajectories between the column and tile kernels in
// every precision and in shapes that exercise the kernel's edges (fewer rows
// than one chunk, a partial last chunk, several term groups with a short last
// one, partial and empty batches), parity of the tile path with the CPU, and
// nonfinite reporting.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;
using kan::cuda::testing::NonlinearReduction;
using kan::cuda::testing::ResidentLayout;
using kan::cuda::testing::ResidentTestScope;

std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale * std::sin(frequency * static_cast<double>(i) + phase);
    return v;
}
kan::TrainableRbfConfig rbf(std::size_t terms) {
    kan::TrainableRbfConfig c{std::vector<double>(terms), std::vector<double>(terms)};
    for (std::size_t k = 0; k < terms; ++k) {
        c.centers[k] = terms > 1 ? -1.1 + 2.2 * static_cast<double>(k) / static_cast<double>(terms - 1) : 0.1;
        c.log_widths[k] = std::log(0.35) + 0.2 * std::sin(0.7 * static_cast<double>(k));
    }
    return c;
}
kan::Layer seeded(kan::Layer l, double phase) {
    const auto scale = 1.0 / std::sqrt(static_cast<double>(l.inputs() * l.terms()));
    l.set_parameters(wave(l.coefficients().size(), 2 * scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
    return l;
}
// Two trainable-RBF layers (the second receives a network-internal input gradient).
kan::Network model(std::size_t inputs, std::size_t hidden, std::size_t terms) {
    return kan::Network({seeded(kan::Layer(inputs, hidden, rbf(terms)), 0.2), seeded(kan::Layer(hidden, 3, rbf(terms)), 0.5)});
}
// A forced tile executor must really have the tile kernel (otherwise the
// bitwise comparison would compare the column kernel with itself).
ResidentNetwork make(const kan::Network& n, std::size_t capacity, Precision p, NonlinearReduction reduction,
                     unsigned* group = nullptr) {
    ResidentLayout layout;
    ResidentTestScope scope(reduction, &layout);
    ResidentNetwork r(n, capacity, p);
    REQUIRE(layout.reduction_groups.size() == n.layers().size());
    for (auto g : layout.reduction_groups) REQUIRE(g > 0);
    if (group) *group = layout.reduction_groups.front();
    return r;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
void require_same(const kan::NetworkGradients& a, const kan::NetworkGradients& b, const std::string& what) {
    if (!same_bits(a.input, b.input)) throw std::runtime_error(what + ": input gradient differs");
    for (std::size_t j = 0; j < a.layers.size(); ++j) {
        const auto& x = test::grad(a, j);
        const auto& y = test::grad(b, j);
        const auto& u = std::get<kan::TrainableRbfGradients>(x.nonlinear);
        const auto& v = std::get<kan::TrainableRbfGradients>(y.nonlinear);
        if (!same_bits(x.coefficients, y.coefficients) || !same_bits(x.bias, y.bias) || !same_bits(u.centers, v.centers) ||
            !same_bits(u.log_widths, v.log_widths))
            throw std::runtime_error(what + ": layer " + std::to_string(j) + " gradients differ");
    }
}
void require_same(const kan::Network& a, const kan::Network& b, const std::string& what) {
    for (std::size_t j = 0; j < a.layers().size(); ++j) {
        const auto& x = test::layer(a, j);
        const auto& y = test::layer(b, j);
        const auto& u = std::get<kan::TrainableRbfEdges>(x.carrier()).basis;
        const auto& v = std::get<kan::TrainableRbfEdges>(y.carrier()).basis;
        const std::vector<double> xc(x.coefficients().begin(), x.coefficients().end()), yc(y.coefficients().begin(), y.coefficients().end());
        const std::vector<double> xb(x.bias().begin(), x.bias().end()), yb(y.bias().begin(), y.bias().end());
        if (!same_bits(xc, yc) || !same_bits(xb, yb) || !same_bits(u.centers, v.centers) || !same_bits(u.log_widths, v.log_widths))
            throw std::runtime_error(what + ": layer " + std::to_string(j) + " parameters differ");
    }
}
// Eager forward + backward (L2) + SGD, then two captured training steps, on
// both executors with the same data; gradients and parameters must agree bitwise.
void run_both(ResidentNetwork& a, ResidentNetwork& b, std::size_t inputs, std::size_t outputs, std::size_t batch, const std::string& what) {
    const auto x = wave(batch * inputs, 1.2, 0.377, 0.1);
    const auto dy = wave(batch * outputs, 0.7, 0.613, 0.4);
    for (auto* r : {&a, &b}) {
        r->upload_input(x, batch);
        r->forward();
        r->upload_output_gradient(dy);
        r->backward(1e-3);
    }
    REQUIRE(same_bits(a.download_output(), b.download_output()));
    require_same(a.download_gradients(), b.download_gradients(), what);
    for (auto* r : {&a, &b}) { r->sgd(0.05); r->train_step(0.05); r->train_step(0.05, 1e-3); }
    require_same(a.download_parameters(), b.download_parameters(), what);
}
constexpr Precision precisions[] = {Precision::Float64, Precision::Float32, Precision::TensorFloat32};
} // namespace

// Column and tile kernels in shapes covering the tile kernel's edges: 185 rows
// (less than one 256-row chunk), 16 800 rows (64 tiles, partial last chunk), a
// partial batch, an empty batch, and term counts with one group, and several
// groups with a short last one (70 and 90 terms exceed the per-block group on
// a 99 KiB device: FP64 16+16+16+16+6, FP32 32+32+6 / 32+32+26; 90 terms also
// takes FP32 past the C1 stage, so its automatic path is the column or tile kernel).
TEST(column_and_tile_reductions_are_bitwise_equal) {
    struct Shape { std::size_t inputs, hidden, terms, capacity, batch; };
    const Shape shapes[] = {{5, 4, 3, 64, 37}, {24, 6, 8, 800, 700}, {16, 5, 70, 160, 151}, {6, 4, 90, 96, 96}, {7, 3, 1, 300, 300}};
    for (auto p : precisions)
        for (const auto& s : shapes) {
            const auto what = "precision " + std::to_string(static_cast<int>(p)) + " terms " + std::to_string(s.terms) + " batch " + std::to_string(s.batch);
            const auto n = model(s.inputs, s.hidden, s.terms);
            auto column = make(n, s.capacity, p, NonlinearReduction::Column);
            unsigned group = 0;
            auto tile = make(n, s.capacity, p, NonlinearReduction::Tile, &group);
            // 70 and 90 terms need several groups on any device with at most 280 KiB per block.
            REQUIRE(group <= s.terms && (s.terms < 70 || group < s.terms));
            run_both(column, tile, s.inputs, 3, s.batch, what);
            // A smaller batch on the same executors, then an empty one (zero gradients from both).
            run_both(column, tile, s.inputs, 3, s.batch / 3 + 1, what + " partial");
            run_both(column, tile, s.inputs, 3, 0, what + " empty");
            // The automatic choice is one of the two kernels wherever the staged FP32 kernel does not apply.
            if (p == Precision::Float64 || s.terms > 85) {
                auto automatic = make(n, s.capacity, p, NonlinearReduction::Automatic);
                auto reference = make(n, s.capacity, p, NonlinearReduction::Column);
                run_both(automatic, reference, s.inputs, 3, s.batch, what + " automatic");
            }
        }
}

// The tile path against the FP64 CPU (tolerances of the FP64 resident suites:
// cuBLAS contractions sum in their own order).
TEST(tile_reduction_matches_cpu) {
    const auto n = model(24, 6, 8);
    auto gpu = make(n, 800, Precision::Float64, NonlinearReduction::Tile);
    const std::size_t batch = 700;
    const auto x = wave(batch * 24, 1.2, 0.377, 0.1);
    const auto dy = wave(batch * 3, 0.7, 0.613, 0.4);
    gpu.upload_input(x, batch);
    gpu.forward();
    gpu.upload_output_gradient(dy);
    gpu.backward();
    const auto actual = gpu.download_gradients();
    const auto expected = n.backward(x, batch, dy);
    for (std::size_t j = 0; j < 2; ++j) {
        const auto& a = std::get<kan::TrainableRbfGradients>(test::grad(actual, j).nonlinear);
        const auto& e = std::get<kan::TrainableRbfGradients>(test::grad(expected, j).nonlinear);
        for (const auto* pair : {&a.centers, &a.log_widths}) {
            const auto& ev = pair == &a.centers ? e.centers : e.log_widths;
            double scale = 0;
            for (double v : ev) scale = std::max(scale, std::abs(v));
            for (std::size_t k = 0; k < ev.size(); ++k)
                REQUIRE(std::abs((*pair)[k] - ev[k]) <= 1e-11 * std::abs(ev[k]) + 1e-12 * scale);
        }
    }
}

// An overflowing step is reported under either forced reduction and the
// executor recovers (other kernels of the step also report it, so this does
// not isolate the reduction's own reports).
TEST(overflow_is_reported_and_recovered_under_either_path) {
    for (auto reduction : {NonlinearReduction::Column, NonlinearReduction::Tile})
        for (auto p : {Precision::Float64, Precision::Float32}) {
            const auto n = model(24, 6, 8);
            auto gpu = make(n, 800, p, reduction);
            const double huge = p == Precision::Float64 ? 1e306 : 1e37;
            gpu.upload_input(wave(700 * 24, 1.2, 0.377, 0.1), 700);
            gpu.forward();
            gpu.upload_output_gradient(std::vector<double>(700 * 3, huge));
            test::throws<std::overflow_error>([&] { gpu.backward(); });
            gpu.upload_output_gradient(wave(700 * 3, 0.7, 0.613, 0.4));
            gpu.backward();
            for (std::size_t j = 0; j < 2; ++j)
                for (double v : std::get<kan::TrainableRbfGradients>(test::grad(gpu.download_gradients(), j).nonlinear).centers)
                    REQUIRE(std::isfinite(v));
        }
}

int main() {
    if (!kan::cuda::available()) { std::cout << "SKIP no CUDA device\n"; return 0; }
    return test::run();
}

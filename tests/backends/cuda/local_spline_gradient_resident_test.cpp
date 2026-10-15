// Backlog C13: input VJP of resident FP32/TF32 B-spline layers. The derivative
// of a clamped spline is a spline of one degree less,
//   d/dx sum_k c_k B_{k,p}(x) = sum_j C*_j B_{j,p-1}(x),  C*_j = p (c_j - c_{j-1}) / (t_{j+p} - t_j),
// whose weights B_{j,p-1} are nonnegative and sum to one. Computing dx from
// W = U*C and the signed derivatives B'_{k,p} (of order degree/knot spacing,
// summing to zero) amplifies the rounding of W by sum_k |B'_k|, which grows
// with the term count: under TF32 the error reached many times the result.
// These cases bound the error per entry by the rounding of the
// coefficient-difference form,
//   |dx - dx_ref| <= unit * sum_j B_{j,p-1}(x) sum_o |U[b,o]| |C*[o,i,j]|,
// unit = 2^-9 (TF32: operand rounding of U and C*, safe for truncating
// tensor-op kernels, plus FP32 accumulation) or (outputs + 8) * 2^-24 (FP32:
// any summation order of the output sum, the C* and basis rounding), which
// does not grow with the term count; and they check the cited two-layer case
// against the CPU within the C1 TF32 tolerance. Paths are forced through the
// private test hook so the cases do not depend on the device's selection rule.
// Knots are dyadic and inputs and upstreams FP32-representable, so the FP64
// CPU reference sees exactly the executor's operands.
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;
using kan::cuda::testing::ResidentLayout;
using kan::cuda::testing::ResidentTestScope;
using kan::cuda::testing::SplinePath;

// Clamped spline on [-1, 1] with `intervals` uniform intervals (dyadic knots
// for a power of two).
kan::BSplineConfig spline(std::size_t degree, std::size_t intervals) {
    kan::BSplineConfig b{degree, std::vector<double>(degree + 1, -1.0)};
    for (std::size_t k = 1; k < intervals; ++k) b.knots.push_back(-1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(intervals));
    b.knots.insert(b.knots.end(), degree + 1, 1.0);
    return b;
}
float f32(double v) { return static_cast<float>(v); }
// Edge functions smooth over the grid (as after training): c[o,i,k] =
// a_oi sin(pi k/(K-1) + phase_oi); the derivative stays O(1) as K grows.
kan::Layer smooth_layer(std::size_t in, std::size_t out, kan::BSplineConfig basis) {
    kan::Layer l(in, out, std::move(basis));
    const auto terms = l.terms();
    std::vector<double> c(l.coefficients().size());
    for (std::size_t oi = 0; oi < in * out; ++oi) {
        const double phase = 0.37 * static_cast<double>(oi), amp = (0.5 + 0.5 * std::cos(1.3 * static_cast<double>(oi))) / std::sqrt(static_cast<double>(in));
        for (std::size_t k = 0; k < terms; ++k)
            c[oi * terms + k] = f32(amp * std::sin(3.14159 * static_cast<double>(k) / static_cast<double>(terms - 1) + phase));
    }
    std::vector<double> bias(out);
    for (std::size_t o = 0; o < out; ++o) bias[o] = f32(0.01 * static_cast<double>(o));
    l.set_parameters(c, bias);
    return l;
}
// Inputs in the domain, the knots themselves, the endpoints and points outside.
std::vector<double> inputs(std::size_t count, std::size_t intervals) {
    std::vector<double> x(count);
    for (std::size_t k = 0; k < count; ++k) x[k] = f32(0.9999 * std::sin(0.377 * static_cast<double>(k) + 0.2));
    for (std::size_t k = 0; k < count; k += 11) x[k] = -1.0 + 2.0 * static_cast<double>((k / 11) % (intervals + 1)) / static_cast<double>(intervals);
    for (std::size_t k = 5; k < count; k += 97) x[k] = (k / 97) % 2 ? 1.5 : -3.0;
    return x;
}
std::vector<double> upstream(std::size_t count) {
    std::vector<double> u(count);
    for (std::size_t k = 0; k < count; ++k) u[k] = f32(0.3 * std::cos(0.613 * static_cast<double>(k)));
    return u;
}
// sum_j B_{j,p-1}(x) sum_o |U[b,o]| |C*[o,i,j]| per (b, i); the degree-(p-1)
// basis B_{j,p-1}, j = 1..K-1, is the clamped spline on the knots without the
// first and last.
std::vector<double> difference_scale(const kan::Layer& l, const kan::BSplineConfig& basis, std::span<const double> x,
                                     std::span<const double> u, std::size_t batch) {
    const auto p = basis.degree, terms = l.terms(), in = l.inputs(), out = l.outputs();
    const auto c = l.coefficients();
    const kan::BSplineConfig lower{p - 1, std::vector<double>(basis.knots.begin() + 1, basis.knots.end() - 1)};
    const auto& t = basis.knots;
    std::vector<double> scale(batch * in, 0.0);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < in; ++i) {
            const auto values = kan::evaluate_basis(lower, x[b * in + i]).values;
            double sum = 0;
            for (std::size_t j = 1; j < terms; ++j) {
                if (values[j - 1] == 0) continue;
                const double gap = t[j + p] - t[j];
                double w = 0;
                for (std::size_t o = 0; o < out; ++o) {
                    const auto q = (o * in + i) * terms;
                    w += std::abs(u[b * out + o]) * std::abs(static_cast<double>(p) * (c[q + j] - c[q + j - 1]) / gap);
                }
                sum += values[j - 1] * w;
            }
            scale[b * in + i] = sum;
        }
    }
    return scale;
}
std::string name(Precision p, SplinePath path, std::size_t degree, std::size_t intervals) {
    return std::string(p == Precision::Float32 ? "FP32" : "TF32") + (path == SplinePath::Dense ? " dense" : " sparse") +
           " degree " + std::to_string(degree) + " intervals " + std::to_string(intervals);
}
// The input VJP of a single smooth spline layer against the FP64 CPU, entry by
// entry within the coefficient-difference rounding bound. Prints the largest
// error/bound ratio of every configuration and fails after all of them ran.
struct BoundCheck {
    std::string failures;
    void operator()(Precision p, SplinePath path, std::size_t degree, std::size_t intervals, std::size_t in, std::size_t out) {
        const auto basis = spline(degree, intervals);
        const auto layer = smooth_layer(in, out, basis);
        const kan::Network network({layer});
        constexpr std::size_t batch = 256;
        const auto x = inputs(batch * in, intervals);
        const auto u = upstream(batch * out);
        ResidentLayout layout;
        ResidentTestScope scope(path, &layout);
        ResidentNetwork gpu(network, batch, p);
        REQUIRE(layout.sparse.at(0) == (path == SplinePath::Sparse));
        gpu.upload_input(x, batch);
        gpu.forward();
        gpu.upload_output_gradient(u);
        gpu.backward();
        const auto actual = gpu.download_gradients().input;
        const auto expected = network.backward(x, batch, u).input;
        const auto scale = difference_scale(layer, basis, x, u, batch);
        const double unit = p == Precision::TensorFloat32 ? 0x1p-9 : static_cast<double>(out + 8) * 0x1p-24;
        double worst = 0;
        std::size_t at = 0;
        for (std::size_t r = 0; r < actual.size(); ++r) {
            const double err = std::abs(actual[r] - expected[r]);
            const double ratio = scale[r] > 0 ? err / (unit * scale[r]) : (err > 0 ? INFINITY : 0.0);
            if (!(ratio <= worst)) { worst = ratio; at = r; }
        }
        const auto what = name(p, path, degree, intervals);
        std::cout << "  " << what << ": largest error/bound " << worst << " (entry " << at << ")\n";
        if (!(worst <= 1)) failures += " [" + what + ": " + std::to_string(worst) + "]";
        REQUIRE(worst > 0); // the bound is not vacuous: some entries carry rounding
    }
    void finish() const {
        if (!failures.empty()) throw std::runtime_error("error/bound > 1:" + failures);
    }
};
} // namespace

// Large term counts (129 and 257 terms at degree 3): the coefficient-difference
// bound holds for both paths in FP32 and TF32, entry by entry.
TEST(spline_input_gradient_is_bounded_by_the_difference_rounding) {
    BoundCheck check_bound;
    for (auto p : {Precision::TensorFloat32, Precision::Float32}) {
        for (auto path : {SplinePath::Dense, SplinePath::Sparse}) {
            check_bound(p, path, 3, 128, 32, 64);
            check_bound(p, path, 3, 256, 16, 4);
        }
    }
    check_bound.finish();
}

// Every sparse degree (1..3) and a dense-only one (5), smaller grids.
TEST(spline_input_gradient_bound_holds_for_every_degree) {
    BoundCheck check_bound;
    for (auto p : {Precision::TensorFloat32, Precision::Float32}) {
        for (std::size_t degree = 1; degree <= 3; ++degree)
            for (auto path : {SplinePath::Dense, SplinePath::Sparse}) check_bound(p, path, degree, 32, 24, 40);
        check_bound(p, SplinePath::Dense, 5, 64, 24, 40);
    }
    check_bound.finish();
}

// The case cited in the C5 evidence (64 -> 40 -> 4, K = 48 and 30, degrees 3
// and 1). Against the CPU at its own activations the dense TF32 network input
// VJP deviates by 0.46 (expected 0.398, got -0.060) before and after C13: the
// TF32 forward rounds layer 0's outputs by about 2^-11 of their terms, which
// moves some of them across a knot of the degree-1 layer, whose derivative
// jumps there (an ill-posed comparison, docs/evidence/backlog/C13.md). The
// well-posed check takes layer 1's input VJP U from the executor itself and
// compares the network input VJP with the CPU's layer-0 VJP at U: within the
// difference rounding bound and the C1 TF32 tolerance. With a degree-3 second
// layer (continuous derivative) the network agrees with the CPU end to end.
TEST(two_layer_tf32_input_gradient_matches_the_cpu_layer_by_layer) {
    const auto seeded = [](kan::Layer l, double phase) {
        const auto scale = 3.0 / std::sqrt(static_cast<double>(l.inputs() * l.terms()));
        std::vector<double> c(l.coefficients().size()), bias(l.outputs());
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = f32(scale * std::sin(0.731 * static_cast<double>(k) + phase));
        for (std::size_t o = 0; o < bias.size(); ++o) bias[o] = f32(0.05 * std::sin(1.3 * static_cast<double>(o) + phase));
        l.set_parameters(c, bias);
        return l;
    };
    const auto uniform = [](std::size_t degree, std::size_t terms) {
        kan::BSplineConfig b{degree, std::vector<double>(degree + 1, -1.0)};
        const auto interior = terms - degree - 1;
        for (std::size_t k = 1; k <= interior; ++k) b.knots.push_back(-1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(interior + 1));
        b.knots.insert(b.knots.end(), degree + 1, 1.0);
        return b;
    };
    constexpr std::size_t batch = 1000;
    std::vector<double> x(batch * 64), u(batch * 4);
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = f32(0.9995 * std::sin(0.377 * static_cast<double>(k) + 0.9));
    for (std::size_t k = 0; k < u.size(); ++k) u[k] = f32(0.3 * std::sin(0.17 * static_cast<double>(k)));
    const auto within_c1 = [](std::span<const double> actual, std::span<const double> expected, const std::string& what) {
        double scale = 0;
        for (double e : expected) scale = std::max(scale, std::abs(e));
        for (std::size_t r = 0; r < actual.size(); ++r)
            if (!(std::abs(actual[r] - expected[r]) <= 1e-2 * std::abs(expected[r]) + 1e-2 * scale))
                throw std::runtime_error(what + " entry " + std::to_string(r) + " actual=" + std::to_string(actual[r]) +
                                         " expected=" + std::to_string(expected[r]));
    };
    for (std::size_t second : {1u, 3u}) {
        const auto first = uniform(3, 48);
        const kan::Network n({seeded(kan::Layer(64, 40, first), 0.2), seeded(kan::Layer(40, 4, uniform(second, 30)), 0.6)});
        for (auto path : {SplinePath::Dense, SplinePath::Sparse}) {
            const auto what = std::string(path == SplinePath::Dense ? "dense" : "sparse") + " second degree " + std::to_string(second);
            ResidentTestScope scope(path);
            ResidentNetwork gpu(n, batch, Precision::TensorFloat32);
            gpu.upload_input(x, batch);
            gpu.forward();
            gpu.upload_output_gradient(u);
            gpu.backward();
            const auto reference = gpu.download_parameters(); // the CPU at the executor's parameters
            const auto actual = gpu.download_gradients();
            const auto& layer0 = test::layer(reference, 0);
            const auto& upstream0 = test::grad(actual, 1).input; // layer 1's input VJP, from the executor
            const auto expected = kan::Network({layer0}).backward(x, batch, upstream0).input;
            within_c1(actual.input, expected, what + " at the executor's upstream");
            const auto scale = difference_scale(layer0, first, x, upstream0, batch);
            for (std::size_t r = 0; r < actual.input.size(); ++r)
                if (!(std::abs(actual.input[r] - expected[r]) <= 0x1p-9 * scale[r] + 1e-6 * std::abs(expected[r])))
                    throw std::runtime_error(what + " entry " + std::to_string(r) + " outside the difference bound");
            if (second == 3) within_c1(actual.input, reference.backward(x, batch, u).input, what + " end to end");
        }
    }
}

// Degree 0: piecewise-constant edges have a zero input VJP on both paths.
TEST(degree_zero_input_gradient_is_zero) {
    for (auto p : {Precision::TensorFloat32, Precision::Float32}) {
        for (auto path : {SplinePath::Dense, SplinePath::Sparse}) {
            const kan::Network network({smooth_layer(16, 8, spline(0, 64))});
            constexpr std::size_t batch = 64;
            ResidentLayout layout;
            ResidentTestScope scope(path, &layout);
            ResidentNetwork gpu(network, batch, p);
            REQUIRE(layout.sparse.at(0) == (path == SplinePath::Sparse));
            gpu.upload_input(inputs(batch * 16, 64), batch);
            gpu.forward();
            gpu.upload_output_gradient(upstream(batch * 8));
            gpu.backward();
            for (double v : gpu.download_gradients().input) REQUIRE(v == 0);
        }
    }
}

// Status: an FP32 derivative coefficient C* = p (c_j - c_{j-1})/gap that
// overflows (adjacent coefficients of opposite sign near FLT_MAX: the true
// derivative is beyond the FP32 range) reaches the input VJP of every sample
// whose degree-(p-1) terms include it and is reported, on both paths: eager
// backward raises std::overflow_error, a training step raises and rolls back.
TEST(overflowing_difference_coefficients_are_reported) {
    for (auto path : {SplinePath::Dense, SplinePath::Sparse}) {
        kan::Layer spline_layer(1, 1, spline(3, 16));
        std::vector<double> c(spline_layer.coefficients().size());
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = k % 2 ? 3e38 : -3e38;
        spline_layer.set_parameters(c, std::vector<double>{0.0});
        constexpr std::size_t batch = 64;
        std::vector<double> x(batch), u(batch, 0.125);
        for (std::size_t b = 0; b < batch; ++b) x[b] = f32(-0.97 + 1.94 * static_cast<double>(b) / static_cast<double>(batch - 1));
        ResidentLayout layout;
        ResidentTestScope scope(path, &layout);
        {
            ResidentNetwork gpu(kan::Network({spline_layer}), batch, Precision::Float32);
            REQUIRE(layout.sparse.at(0) == (path == SplinePath::Sparse));
            gpu.upload_input(x, batch);
            gpu.forward(); // the values stay finite
            gpu.upload_output_gradient(u);
            test::throws<std::overflow_error>([&] { gpu.backward(); });
        }
        // Behind a map the spline layer's input VJP is part of a training step.
        const kan::Network n({kan::InputMap(1, kan::TanhMap{}), spline_layer});
        ResidentNetwork gpu(n, batch, Precision::Float32);
        gpu.upload_input(x, batch);
        gpu.upload_output_gradient(u);
        const auto before = gpu.download_parameters();
        test::throws<std::overflow_error>([&] { gpu.train_step(1e-30); });
        const auto after = gpu.download_parameters();
        const auto a = test::layer(before, 1).coefficients(), b = test::layer(after, 1).coefficients();
        REQUIRE(std::equal(a.begin(), a.end(), b.begin()));
    }
}

int main() { return test::run(); }

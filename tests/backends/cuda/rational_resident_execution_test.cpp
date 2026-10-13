// Backlog C6-C8: resident rational execution. Pins the behavior the kernel
// restructuring must preserve: the forward pass reports a nonfinite parameter
// VJP intermediate even when the forward result itself is finite (C7), exactly
// at the floating-point overflow boundary, also inside a captured training
// step; and launches larger than one grid (C6/C8 grid-stride bounds).
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using kan::DenominatorPolicy;
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

constexpr Precision precisions[] = {Precision::Float64, Precision::Float32};

kan::Layer edge(std::size_t m, std::size_t n, DenominatorPolicy policy, std::vector<double> a, std::vector<double> b) {
    kan::RationalConfig r;
    r.numerator_degree = m; r.denominator_degree = n; r.denominator_policy = policy;
    kan::Layer l(1, 1, r);
    kan::set_rational_parameters(l, a, b, std::vector<double>{0});
    return l;
}
// A one-edge network evaluated at x with upstream 1.
ResidentNetwork single(const kan::Layer& l, Precision p, double x) {
    ResidentNetwork gpu(kan::Network({l}), 1, p);
    gpu.upload_input(std::vector<double>{x}, 1);
    gpu.upload_output_gradient(std::vector<double>{1});
    return gpu;
}
bool equal(std::span<const double> a, std::span<const double> b) { return std::ranges::equal(a, b); }
std::vector<double> numerator16() { std::vector<double> a(17, 0.0); a[0] = 1; return a; }
// Largest finite exponent e with 2^e representable, per precision.
int top(Precision p) { return p == Precision::Float64 ? 1023 : 127; }
} // namespace

// z^16 overflows while P = a0 = 1 stays finite: the output is finite, the
// numerator VJP is not. Forward reports it and invalidates its output.
TEST(forward_reports_overflowing_powers_with_a_finite_output) {
    for (auto p : precisions) {
        const double x = p == Precision::Float64 ? 1e20 : 1e3;
        for (auto policy : {DenominatorPolicy::Guarded, DenominatorPolicy::Absolute, DenominatorPolicy::Smooth}) {
            auto gpu = single(edge(16, 0, policy, numerator16(), {}), p, x);
            test::throws<std::overflow_error>([&] { gpu.forward(); });
            test::throws<std::logic_error>([&] { gpu.download_output(); });
            test::throws<std::logic_error>([&] { gpu.backward(); });
            // One step below: every power is finite.
            gpu.upload_input(std::vector<double>{p == Precision::Float64 ? 1e19 : 2e2}, 1);
            gpu.upload_output_gradient(std::vector<double>{1});
            gpu.forward(); gpu.backward();
            test::near(gpu.download_output()[0], 1);
        }
    }
}

// dr/db = -r z / Q with r = 2^a, z = 2^e, Q = 1: finite up to the largest
// binade, overflowing one power of two later. Exact values on both sides.
TEST(forward_reports_denominator_vjp_overflow_exactly_at_the_boundary) {
    for (auto p : precisions) {
        const int a = p == Precision::Float64 ? 600 : 64, e = top(p) - a;
        for (auto policy : {DenominatorPolicy::Guarded, DenominatorPolicy::Absolute}) {
            // Absolute: S = 0 * z, g = sign(0) = 0, the guarded intermediates are still checked.
            auto gpu = single(edge(0, 1, policy, {std::ldexp(1.0, a)}, {0.0}), p, std::ldexp(1.0, e));
            gpu.forward(); gpu.backward();
            REQUIRE(gpu.download_output()[0] == std::ldexp(1.0, a));
            const auto g = gpu.download_gradients();
            const double expected = policy == DenominatorPolicy::Guarded ? -std::ldexp(1.0, top(p)) : 0.0;
            REQUIRE(test::denominators(test::grad(g, 0))[0] == expected);
            gpu.upload_input(std::vector<double>{std::ldexp(1.0, e + 1)}, 1);
            test::throws<std::overflow_error>([&] { gpu.forward(); });
            test::throws<std::logic_error>([&] { gpu.download_output(); });
        }
    }
}

// Smooth: dr/db = -r (2S) z / Q^2 overflows in r * (g z / Q) with Q = 1.
TEST(forward_reports_smooth_denominator_vjp_overflow) {
    for (auto p : precisions) {
        const bool f64 = p == Precision::Float64;
        const double x = f64 ? 1e150 : 1e15, b = f64 ? 1e-200 : 1e-20;
        auto gpu = single(edge(0, 1, DenominatorPolicy::Smooth, {f64 ? 1e150 : 1e10}, {b}), p, x);
        gpu.forward(); gpu.backward();
        const double expected = -(f64 ? 1e150 : 1e10) * 2 * (b * x) * x;
        test::near(test::denominators(test::grad(gpu.download_gradients(), 0))[0] / expected, 1, f64 ? 1e-12 : 1e-5);
        auto over = single(edge(0, 1, DenominatorPolicy::Smooth, {f64 ? 1e250 : 1e30}, {b}), p, x);
        test::throws<std::overflow_error>([&] { over.forward(); });
    }
}

// The same failure inside a captured training step is attributed to that
// step; the step commits nothing and training continues afterwards.
TEST(training_step_attributes_parameter_vjp_overflow_to_its_step) {
    for (auto p : precisions) {
        const int a = p == Precision::Float64 ? 600 : 64, e = top(p) - a;
        ResidentNetwork gpu(kan::Network({edge(0, 1, DenominatorPolicy::Guarded, {std::ldexp(1.0, a)}, {0.0})}), 1, p);
        // A zero upstream keeps the parameters (and so the boundary) fixed;
        // the forward check does not depend on the upstream.
        gpu.upload_input(std::vector<double>{0.5}, 1);
        gpu.upload_output_gradient(std::vector<double>{0});
        gpu.train_step(1e-3);
        const auto good = gpu.download_parameters();
        gpu.upload_input(std::vector<double>{std::ldexp(1.0, e + 1)}, 1);
        gpu.upload_output_gradient(std::vector<double>{0});
        std::string message;
        try { gpu.train_step(1e-3); } catch (const std::overflow_error& error) { message = error.what(); }
        REQUIRE(message.find("training step 1") != std::string::npos);
        REQUIRE(gpu.trained_steps() == 1);
        const auto after = gpu.download_parameters();
        REQUIRE(equal(test::layer(after, 0).coefficients(), test::layer(good, 0).coefficients()));
        REQUIRE(equal(test::denominators(test::layer(after, 0)), test::denominators(test::layer(good, 0))));
        gpu.upload_input(std::vector<double>{0.5}, 1);
        gpu.upload_output_gradient(std::vector<double>{0});
        gpu.train_step(1e-3);
        gpu.check_status();
        REQUIRE(gpu.trained_steps() == 2);
    }
}

// 600001 edges per layer: more threads (batch*outputs, batch*inputs) and more
// edge warps than one capped grid holds, so every kernel's grid-stride loop
// runs; the capacity exceeds the batch. All values match the CPU.
TEST(large_rational_launches_match_the_cpu) {
    constexpr std::size_t wide = 600001, batch = 29, capacity = 32;
    kan::RationalConfig r;
    r.numerator_degree = 2; r.denominator_degree = 1; r.center = 0.1; r.scale = 1.3;
    r.denominator_policy = DenominatorPolicy::Absolute;
    kan::Layer first(1, wide, r), second(wide, 1, r);
    for (auto* l : {&first, &second}) {
        std::vector<double> a(l->coefficients().size()), b(test::denominators(*l).size()), bias(l->outputs());
        for (std::size_t k = 0; k < a.size(); ++k) a[k] = 0.3 * std::sin(0.37 * static_cast<double>(k));
        for (std::size_t k = 0; k < b.size(); ++k) b[k] = 0.4 * std::cos(0.53 * static_cast<double>(k));
        for (std::size_t k = 0; k < bias.size(); ++k) bias[k] = 0.01 * std::sin(static_cast<double>(k));
        kan::set_rational_parameters(*l, a, b, bias);
    }
    kan::Network cpu({first, second});
    std::vector<double> x(batch), dy(batch);
    for (std::size_t s = 0; s < batch; ++s) { x[s] = 0.9 * std::sin(0.7 * static_cast<double>(s)); dy[s] = std::cos(0.3 * static_cast<double>(s)); }
    ResidentNetwork gpu(cpu, capacity);
    gpu.upload_input(x, batch); gpu.upload_output_gradient(dy);
    gpu.forward(); gpu.backward(0.25);
    const auto y = cpu.forward(x, batch);
    auto expected = cpu.backward(x, batch, dy);
    const auto reg = cpu.regularization(0.25).gradients;
    const auto actual = gpu.download_gradients();
    const auto output = gpu.download_output();
    for (std::size_t s = 0; s < batch; ++s) test::near(output[s], y[s], 1e-9);
    for (std::size_t s = 0; s < batch; ++s) test::near(actual.input[s], expected.input[s], 1e-9);
    for (std::size_t j = 0; j < 2; ++j) {
        const auto& e = test::grad(expected, j);
        const auto& g = test::grad(actual, j);
        for (std::size_t k = 0; k < e.coefficients.size(); ++k)
            test::near(g.coefficients[k], e.coefficients[k] + test::grad(reg, j).coefficients[k], 1e-9);
        for (std::size_t k = 0; k < e.bias.size(); ++k) test::near(g.bias[k], e.bias[k], 1e-9);
        const auto& de = test::denominators(e);
        const auto& dg = test::denominators(g);
        for (std::size_t k = 0; k < de.size(); ++k) test::near(dg[k], de[k], 1e-9);
    }
}

int main() {
    if (!kan::cuda::available()) { std::cout << "CUDA device unavailable\n"; return 0; }
    return test::run();
}

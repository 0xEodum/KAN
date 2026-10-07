// Backlog C6-C8: bitwise dump of the resident rational path (forward caches,
// input VJP and parameter VJPs) in FP64 and FP32, for every denominator policy
// and a range of degrees, small (5x7x3) and wide (96x80x10) networks, capacity
// above the batch, the empty batch, the eager calls and graph training steps
// (device MSE). Every value is printed as the bit pattern of the downloaded
// double (the format of golden.cpp, so golden_diff.py can compare two dumps);
// exceptions are printed with their messages. Two builds whose dumps are
// byte-identical compute the same results for these fixtures.
// Usage: rational_dump > dump.txt
#include "kan/families.hpp"
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace kan;

namespace {
void hex(const char* label, std::span<const double> values) {
    std::printf("%s[%zu]", label, values.size());
    for (double v : values) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        std::printf(" %016llx", static_cast<unsigned long long>(bits));
    }
    std::printf("\n");
}
void guarded(const std::string& label, const std::function<void()>& body) {
    std::printf("== %s\n", label.c_str());
    try {
        body();
    } catch (const std::domain_error& e) {
        std::printf("domain_error: %s\n", e.what());
    } catch (const std::overflow_error& e) {
        std::printf("overflow_error: %s\n", e.what());
    } catch (const std::invalid_argument& e) {
        std::printf("invalid_argument: %s\n", e.what());
    } catch (const std::logic_error& e) {
        std::printf("logic_error: %s\n", e.what());
    }
}
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
const char* name(DenominatorPolicy p) {
    return p == DenominatorPolicy::Guarded ? "guarded" : p == DenominatorPolicy::Absolute ? "absolute" : "smooth";
}
const char* name(cuda::Precision p) { return p == cuda::Precision::Float64 ? "f64" : "f32"; }

Layer rational(std::size_t in, std::size_t out, std::size_t m, std::size_t n, DenominatorPolicy policy, std::uint64_t seed) {
    RationalConfig c;
    c.numerator_degree = m; c.denominator_degree = n; c.center = 0.1; c.scale = 1.3; c.denominator_policy = policy;
    Layer l(in, out, c);
    VarianceScaling init;
    init.seed = seed;
    initialize(l, init);
    // Nonzero bias, so that the bias VJP and its update are visible.
    auto a = l.coefficients();
    set_rational_parameters(l, a, std::get<RationalEdges>(l.carrier()).denominators, wave(out, 0.02, 0.9, 0.1*static_cast<double>(seed)));
    return l;
}

void dump_gradients(const NetworkGradients& g) {
    hex("dx", g.input);
    for (const auto& stage : g.layers) {
        const auto& l = std::get<LayerGradients>(stage);
        hex("da", l.coefficients);
        hex("dbias", l.bias);
        hex("dden", std::get<RationalGradients>(l.nonlinear).denominators);
    }
}
void dump_parameters(const Network& n) {
    for (const auto& stage : n.layers()) {
        const auto& l = std::get<Layer>(stage);
        hex("a", l.coefficients());
        hex("bias", l.bias());
        hex("den", std::get<RationalEdges>(l.carrier()).denominators);
    }
}

// Eager forward/backward/SGD twice, then three graph MSE steps.
void run(const std::string& label, const Network& network, std::size_t batch, std::size_t capacity, cuda::Precision precision) {
    guarded(label, [&] {
        const auto inputs = std::get<Layer>(network.layers().front()).inputs();
        const auto outputs = std::get<Layer>(network.layers().back()).outputs();
        cuda::ResidentNetwork r(network, capacity, precision);
        r.upload_input(wave(batch*inputs, 0.9, 0.377, 0.2), batch);
        r.upload_output_gradient(wave(batch*outputs, 0.5, 0.613, 0.4));
        for (int step = 0; step < 2; ++step) {
            r.forward();
            hex("y", r.download_output());
            r.backward(1e-3);
            dump_gradients(r.download_gradients());
            r.sgd(0.01);
        }
        if (batch) {
            r.upload_target(wave(batch*outputs, 0.3, 0.29, 0.5));
            for (int step = 0; step < 3; ++step) r.train_step(0.01, 1e-3, cuda::Loss::MeanSquaredError);
            r.check_status();
            hex("loss", std::vector<double>{r.download_loss()});
        }
        dump_parameters(r.download_parameters());
    });
}

// Single-edge fixtures near the parameter-VJP overflow boundary (backlog C7):
// the forward result is finite in each, the parameter VJPs overflow in some.
void boundary(cuda::Precision precision) {
    const bool f64 = precision == cuda::Precision::Float64;
    struct Fixture { const char* name; std::size_t m, n; DenominatorPolicy policy; std::vector<double> a, b; double x; };
    const double big = f64 ? std::ldexp(1.0, 600) : std::ldexp(1.0, 64);
    const std::vector<Fixture> fixtures = {
        // z^16 overflows (P = a0 stays finite).
        {"power16", 16, 0, DenominatorPolicy::Guarded, [] { std::vector<double> a(17, 0.0); a[0] = 1; return a; }(), {}, f64 ? 1e20 : 1e3},
        {"power16-finite", 16, 0, DenominatorPolicy::Guarded, [] { std::vector<double> a(17, 0.0); a[0] = 1; return a; }(), {}, f64 ? 1e19 : 2e2},
        // r * z / Q just below and above the largest finite value.
        {"denominator-finite", 0, 1, DenominatorPolicy::Guarded, {big}, {0.0}, std::ldexp(1.0, f64 ? 423 : 63)},
        {"denominator-overflow", 0, 1, DenominatorPolicy::Guarded, {big}, {0.0}, std::ldexp(1.0, f64 ? 424 : 64)},
        {"absolute-finite", 0, 1, DenominatorPolicy::Absolute, {big}, {0.0}, std::ldexp(1.0, f64 ? 423 : 63)},
        {"absolute-overflow", 0, 1, DenominatorPolicy::Absolute, {big}, {0.0}, std::ldexp(1.0, f64 ? 424 : 64)},
        // Smooth: g z / Q and r g z / Q.
        {"smooth-overflow", 0, 1, DenominatorPolicy::Smooth, {f64 ? 1e250 : 1e30}, {f64 ? 1e-200 : 1e-20}, f64 ? 1e150 : 1e15},
        {"smooth-finite", 0, 1, DenominatorPolicy::Smooth, {f64 ? 1e150 : 1e10}, {f64 ? 1e-200 : 1e-20}, f64 ? 1e150 : 1e15},
    };
    for (const auto& f : fixtures) {
        guarded(std::string("boundary ") + name(precision) + " " + f.name, [&] {
            RationalConfig c;
            c.numerator_degree = f.m; c.denominator_degree = f.n; c.denominator_policy = f.policy;
            Layer l(1, 1, c);
            set_rational_parameters(l, f.a, f.b, std::vector<double>{0});
            cuda::ResidentNetwork r(Network({l}), 1, precision);
            r.upload_input(std::vector<double>{f.x}, 1);
            r.upload_output_gradient(std::vector<double>{1});
            r.forward();
            hex("y", r.download_output());
            r.backward();
            dump_gradients(r.download_gradients());
        });
    }
}
} // namespace

int main() {
    const DenominatorPolicy policies[] = {DenominatorPolicy::Guarded, DenominatorPolicy::Absolute, DenominatorPolicy::Smooth};
    const std::pair<std::size_t, std::size_t> degrees[] = {{0, 0}, {3, 2}, {4, 1}, {0, 3}, {7, 5}, {16, 16}};
    for (auto precision : {cuda::Precision::Float64, cuda::Precision::Float32}) {
        for (auto policy : policies) {
            for (const auto& [m, n] : degrees) {
                const auto tag = std::string(name(precision)) + " " + name(policy) + " " + std::to_string(m) + "/" + std::to_string(n);
                const Network small({rational(5, 7, m, n, policy, 1), rational(7, 3, 3, 2, policy, 2)});
                run("small " + tag, small, 33, 40, precision);
                run("empty " + tag, small, 0, 4, precision);
                if ((m == 3 && n == 2) || (m == 16 && n == 16))
                    run("wide " + tag, Network({rational(96, 80, m, n, policy, 3), rational(80, 10, 3, 2, policy, 4)}), 700, 701, precision);
            }
        }
        boundary(precision);
    }
    return 0;
}

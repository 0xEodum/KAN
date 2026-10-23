// M5 phase 1: CPU throughput of PqcEdges layer calls (forward; backward =
// Layer::backward, which recomputes the forward state per edge and sample).
// Prints CSV: qubits,repetitions,entangler,width,batch,forward_ms,backward_ms,
// forward_circuits_per_s,backward_circuits_per_s,checksum. Median of `repeats`
// timed calls after one warm-up; one thread.
// Build: g++ -std=c++20 -O3 -ffp-contract=off -Iinclude <this> build-cpu/libkan.a
#include "kan/families.hpp"
#include "kan/layer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
double median_ms(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
std::vector<double> wave(std::size_t n, double scale, double frequency, double phase) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * std::sin(frequency * double(k) + phase);
    return v;
}
} // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 5;
    std::printf("qubits,repetitions,entangler,width,batch,forward_ms,backward_ms,forward_circuits_per_s,"
                "backward_circuits_per_s,checksum\n");
    struct Case {
        std::size_t qubits, repetitions;
        kan::Entangler entangler;
        std::size_t width, batch;
    };
    std::vector<Case> cases;
    for (std::size_t width : {16u, 64u})
        for (std::size_t n : {1u, 2u, 4u})
            for (std::size_t r : {2u, 4u})
                cases.push_back({n, r, n == 1 ? kan::Entangler::None : kan::Entangler::CnotChain, width, 256});
    for (const auto& c : cases) {
        kan::PqcConfig config;
        config.qubits = c.qubits;
        config.repetitions = c.repetitions;
        config.entangler = c.entangler;
        kan::Layer layer(c.width, c.width, config);
        const auto& e = std::get<kan::PqcEdges>(layer.carrier());
        kan::set_pqc_parameters(layer, wave(e.coefficients.size(), 0.3, 0.7, 0.1), wave(e.angles.size(), 2.0, 1.3, 0.2),
                                wave(e.encoding_weights.size(), 1.0, 0.9, 1.4), wave(e.encoding_biases.size(), 0.5, 0.4, 0.3),
                                std::vector<double>(c.width, 0.0));
        const auto x = wave(c.width * c.batch, 0.9, 0.37, 0.5), u = wave(c.width * c.batch, 0.2, 0.71, 0.9);
        std::vector<double> forward, backward;
        double checksum = 0;
        for (int k = 0; k <= repeats; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto y = layer.forward(x, c.batch);
            const auto t1 = std::chrono::steady_clock::now();
            const auto g = layer.backward(x, c.batch, u);
            const auto t2 = std::chrono::steady_clock::now();
            if (k == 0) continue;
            forward.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            backward.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            checksum = y[0] + y.back() + g.input[0] + std::get<kan::PqcGradients>(g.nonlinear).angles.back();
        }
        const double circuits = double(c.width * c.width * c.batch);
        const double f = median_ms(forward), b = median_ms(backward);
        std::printf("%zu,%zu,%d,%zu,%zu,%.4f,%.4f,%.4e,%.4e,%.17g\n", c.qubits, c.repetitions, int(c.entangler), c.width,
                    c.batch, f, b, circuits / (f * 1e-3), circuits / (b * 1e-3), checksum);
    }
}

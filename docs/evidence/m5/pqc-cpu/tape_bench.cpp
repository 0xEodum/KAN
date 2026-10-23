// M5 phase 1, review fix R5: the cost of PQC tape preparation measured
// separately from complete Layer::forward / Layer::backward calls on the CPU.
// Per call the CPU loops (src/backends/cpu/pqc_edges.cpp) compile the tape
// once (compile_tape, structural) and, per edge, form the angle half-angles and
// block matrices once (prepare_edge, parametric; it cannot be hoisted across
// edges). prepare_edge is in an anonymous namespace, so `prepare_edge_replica`
// below repeats its two loops from the shared formulas (pqc_formulas.hpp).
// Prints CSV: qubits,repetitions,block,entangler,width,batch,compile_ns,
// prepare_edge_ns,forward_ms,backward_ms,forward_prep_fraction,
// backward_prep_fraction,checksum; the fractions are
// (compile + width^2 prepare_edge) / call. Medians of `repeats` timed calls
// (forward and backward) or batches of timed loops (compile, prepare) after
// one warm-up; one thread.
// Build: g++ -std=c++20 -O3 -ffp-contract=off -Iinclude -Isrc <this> build-cpu/libkan.a
#include "carriers/quantum/pqc_formulas.hpp"
#include "kan/families.hpp"
#include "kan/layer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using namespace kan::detail::pqc;
using Clock = std::chrono::steady_clock;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
std::vector<double> wave(std::size_t n, double scale, double frequency, double phase) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * std::sin(frequency * double(k) + phase);
    return v;
}
// Makes the object's memory observable (passes its address, not a copy), so
// the stores into it and the reads of it cannot be dropped or hoisted.
template<class T> void sink(const T& value) { asm volatile("" : : "r"(&value) : "memory"); }

// The two loops of prepare_edge (pqc_edges.cpp), unchanged.
struct EdgeWorkspace {
    HalfAngle<double> angles[max_angles];
    Matrix2<double> blocks[(max_repetitions + 1) * max_qubits];
};
void prepare_edge_replica(const Tape& tape, const double* theta, EdgeWorkspace& s) {
    for (std::size_t j = 0; j < tape.angles; ++j) s.angles[j] = half_angle(theta[j]);
    for (std::size_t g = 0; g * tape.block < tape.angles; ++g)
        s.blocks[g] = block_matrix(tape.axes, tape.block, s.angles + g * tape.block);
}

// ns per compile_tape call: median over `repeats` loops of `count` calls.
double compile_ns(const kan::PqcConfig& config, int repeats) {
    static Tape tape;
    const int count = 20000;
    std::vector<double> samples;
    for (int k = 0; k <= repeats; ++k) {
        const auto t0 = Clock::now();
        for (int i = 0; i < count; ++i) {
            sink(config);
            compile_tape(config, tape);
            sink(tape);
        }
        const auto t1 = Clock::now();
        if (k > 0) samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / count);
    }
    return median(samples);
}

// ns per edge: median over `repeats` sweeps over every edge of the layer
// (repeated to at least 20000 edges per sweep).
double prepare_ns(const kan::PqcConfig& config, const std::vector<double>& angles, std::size_t edges, int repeats) {
    static Tape tape;
    static EdgeWorkspace workspace;
    compile_tape(config, tape);
    const std::size_t sweeps = std::max<std::size_t>(1, 20000 / edges);
    std::vector<double> samples;
    for (int k = 0; k <= repeats; ++k) {
        const auto t0 = Clock::now();
        for (std::size_t s = 0; s < sweeps; ++s)
            for (std::size_t e = 0; e < edges; ++e) {
                sink(angles);
                prepare_edge_replica(tape, angles.data() + e * tape.angles, workspace);
                sink(workspace);
            }
        const auto t1 = Clock::now();
        if (k > 0) samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / double(sweeps * edges));
    }
    return median(samples);
}
} // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 7;
    std::printf("qubits,repetitions,block,entangler,width,batch,compile_ns,prepare_edge_ns,forward_ms,backward_ms,"
                "forward_prep_fraction,backward_prep_fraction,checksum\n");
    struct Shape {
        std::size_t qubits, repetitions;
        kan::Entangler entangler;
    };
    std::vector<Shape> shapes; // cpu-final.csv, then the D9 limit
    for (std::size_t n : {1u, 2u, 4u})
        for (std::size_t r : {2u, 4u}) shapes.push_back({n, r, n == 1 ? kan::Entangler::None : kan::Entangler::CnotChain});
    shapes.push_back({5, 32, kan::Entangler::CnotRing});
    for (const auto& shape : shapes)
        for (std::size_t width : {1u, 16u, 64u})
            for (std::size_t batch : {1u, 256u}) {
                kan::PqcConfig config; // default block {Z, Y, Z}, Y encoding
                config.qubits = shape.qubits;
                config.repetitions = shape.repetitions;
                config.entangler = shape.entangler;
                kan::Layer layer(width, width, config);
                const auto& e = std::get<kan::PqcEdges>(layer.carrier());
                kan::set_pqc_parameters(layer, wave(e.coefficients.size(), 0.3, 0.7, 0.1),
                                        wave(e.angles.size(), 2.0, 1.3, 0.2),
                                        wave(e.encoding_weights.size(), 1.0, 0.9, 1.4),
                                        wave(e.encoding_biases.size(), 0.5, 0.4, 0.3), std::vector<double>(width, 0.0));
                const auto x = wave(width * batch, 0.9, 0.37, 0.5), u = wave(width * batch, 0.2, 0.71, 0.9);
                std::vector<double> forward, backward;
                double checksum = 0;
                for (int k = 0; k <= repeats; ++k) {
                    const auto t0 = Clock::now();
                    const auto y = layer.forward(x, batch);
                    const auto t1 = Clock::now();
                    const auto g = layer.backward(x, batch, u);
                    const auto t2 = Clock::now();
                    if (k == 0) continue;
                    forward.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    backward.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
                    checksum = y[0] + y.back() + g.input[0] + std::get<kan::PqcGradients>(g.nonlinear).angles.back();
                }
                const double compile = compile_ns(config, repeats);
                const double prepare = prepare_ns(config, e.angles, width * width, repeats);
                const double f = median(forward), b = median(backward);
                const double preparation_ms = (compile + double(width * width) * prepare) * 1e-6;
                std::printf("%zu,%zu,%zu,%d,%zu,%zu,%.1f,%.1f,%.5f,%.5f,%.5f,%.5f,%.17g\n", shape.qubits,
                            shape.repetitions, config.block.size(), int(shape.entangler), width, batch, compile,
                            prepare, f, b, preparation_ms / f, preparation_ms / b, checksum);
                std::fflush(stdout);
            }
}

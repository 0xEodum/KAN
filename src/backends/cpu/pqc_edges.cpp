// PQC edges on the CPU (M5 phase 1): one re-uploading circuit per edge and
// sample from the shared tape and state-vector formulas. The tape is compiled
// once per call; per edge the angle half-angles, the block matrices W_(l,q)
// (and, backward, the diagonal readout observable) are formed once into stack
// arrays; the forward state is the fused form (one pass per qubit and layer),
// the reverse pass is gate by gate. Samples run in
// blocks of `lanes` as independent vector lanes (backends/cpu/lanes.hpp):
// every lane performs the scalar operation sequence, so results are bitwise
// those of one sample at a time, and lane results are accumulated in
// ascending sample order. No heap allocation per edge or sample. Loop order
// edge (o, then i), then sample block: each output and input-gradient element
// still accumulates in the documented order.
#include "backends/cpu/edge_ops.hpp"
#include "backends/cpu/lanes.hpp"
#include "carriers/quantum/pqc_formulas.hpp"
#include "core/checks.hpp"
#include <algorithm>
#include <stdexcept>

namespace kan::detail {
namespace {
using namespace pqc;
constexpr std::size_t lanes = 8;
using Vector = Lanes<lanes>;

struct Workspace {
    Tape tape;
    HalfAngle<double> angles[max_angles];
    Matrix2<double> blocks[(max_repetitions + 1) * max_qubits]; // W_(l,q), index l n + q
    HalfAngle<Vector> encodings[max_encodings];
    double x[lanes];
};

// The edge's angle half-angles (for the reverse pass) and block matrices (forward).
void prepare_edge(const PqcEdges& edges, std::size_t edge, Workspace& s) {
    const auto& tape = s.tape;
    const double* theta = edges.angles.data() + edge * tape.angles;
    for (std::size_t j = 0; j < tape.angles; ++j) s.angles[j] = half_angle(theta[j]);
    for (std::size_t g = 0; g * tape.block < tape.angles; ++g)
        s.blocks[g] = block_matrix(tape.axes, tape.block, s.angles + g * tape.block);
}

// The block's inputs (unused lanes: x = 0) and their encoding half-angles of w x + b.
void encode(std::span<const double> input, std::size_t inputs, std::size_t i, std::size_t b0, std::size_t count,
            const double* w, const double* b, Workspace& s) {
    for (std::size_t l = 0; l < lanes; ++l) s.x[l] = l < count ? input[(b0 + l) * inputs + i] : 0.0;
    for (std::size_t j = 0; j < s.tape.encodings; ++j)
        for (std::size_t l = 0; l < lanes; ++l) {
            const auto h = half_angle(w[j] * s.x[l] + b[j]);
            s.encodings[j].c.v[l] = h.c;
            s.encodings[j].s.v[l] = h.s;
        }
}
} // namespace

std::size_t terms(const PqcEdges& edges) noexcept { return pqc_terms(edges.config); }

void validate(const PqcEdges& edges, EdgeShape shape) {
    validate_pqc(edges.config);
    if (edges.shot_seed != 0 && edges.config.shots == 0)
        throw std::invalid_argument("a PQC shot seed requires finite shots");
    const auto count = checked_size(shape.inputs, shape.outputs);
    if (edges.coefficients.size() != checked_size(count, terms(edges)))
        throw std::invalid_argument("carrier coefficient shape mismatch");
    const auto encodings = checked_size(count, pqc_encodings(edges.config));
    if (edges.angles.size() != checked_size(count, pqc_angles(edges.config)) ||
        edges.encoding_weights.size() != encodings || edges.encoding_biases.size() != encodings)
        throw std::invalid_argument("PQC angle or encoding shape mismatch");
}

void require_finite_nonlinear(const PqcEdges& edges) {
    require_finite(edges.angles);
    require_finite(edges.encoding_weights);
    require_finite(edges.encoding_biases);
}

// output[b,o] = bias[o] + sum_i sum_t c[o,i,t] m_t(x[b,i]), accumulated in
// ascending i, then t. Layer checks the finished output.
void forward(const PqcEdges& edges, EdgeShape shape, std::span<const double> bias, std::span<const double> input,
             std::size_t batch, std::span<double> output) {
    const auto inputs = shape.inputs, outputs = shape.outputs;
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t o = 0; o < outputs; ++o) output[b * outputs + o] = bias[o];
    if (batch == 0) return;
    Workspace s;
    compile_tape(edges.config, s.tape);
    const auto& tape = s.tape;
    dispatch_qubits(tape.qubits, [&](auto qubits) {
        constexpr unsigned N = decltype(qubits)::value;
        const std::size_t T = tape.terms, E = tape.encodings;
        for (std::size_t o = 0; o < outputs; ++o)
            for (std::size_t i = 0; i < inputs; ++i) {
                const std::size_t edge = o * inputs + i;
                prepare_edge(edges, edge, s);
                const double* c = edges.coefficients.data() + edge * T;
                const double* w = edges.encoding_weights.data() + edge * E;
                const double* bb = edges.encoding_biases.data() + edge * E;
                for (std::size_t b0 = 0; b0 < batch; b0 += lanes) {
                    const std::size_t count = std::min(lanes, batch - b0);
                    encode(input, inputs, i, b0, count, w, bb, s);
                    Amplitude<Vector> psi[1u << N];
                    Vector m[max_qubits];
                    run_fused_forward<N>(tape, s.blocks, s.encodings, psi);
                    readout<N>(psi, T, tape.parity, m);
                    for (std::size_t l = 0; l < count; ++l) {
                        double& sum = output[(b0 + l) * outputs + o];
                        for (std::size_t t = 0; t < T; ++t) sum += c[t] * m[t].v[l];
                    }
                }
            }
    });
}

// Per sample and edge, with upstream u = u[b,o]: forward state and m_t;
// dc[t] += u m_t; lambda = (u o_k) psi_k with o_k = sum_t sign_t(k) c_t; the
// reverse pass (last gate first) adds d to the angle VJP, or for an encoding
// gate d to the bias VJP, x d to the weight VJP and w d to the edge's input
// VJP, which is added to dx[b,i] after the pass (ascending o per sample).
// Parameter VJPs accumulate in ascending b.
void backward(const PqcEdges& edges, EdgeShape shape, std::span<const double> input, std::size_t batch,
              std::span<const double> upstream, std::span<double> input_gradient,
              std::span<double> coefficient_gradient, PqcGradients& nonlinear) {
    if (batch == 0) return;
    const auto inputs = shape.inputs, outputs = shape.outputs;
    Workspace s;
    compile_tape(edges.config, s.tape);
    const auto& tape = s.tape;
    dispatch_qubits(tape.qubits, [&](auto qubits) {
        constexpr unsigned N = decltype(qubits)::value;
        const std::size_t T = tape.terms, A = tape.angles, E = tape.encodings;
        for (std::size_t o = 0; o < outputs; ++o)
            for (std::size_t i = 0; i < inputs; ++i) {
                const std::size_t edge = o * inputs + i;
                prepare_edge(edges, edge, s);
                const double* c = edges.coefficients.data() + edge * T;
                const double* w = edges.encoding_weights.data() + edge * E;
                const double* bb = edges.encoding_biases.data() + edge * E;
                double* dc = coefficient_gradient.data() + edge * T;
                double* dtheta = nonlinear.angles.data() + edge * A;
                double* dw = nonlinear.encoding_weights.data() + edge * E;
                double* db = nonlinear.encoding_biases.data() + edge * E;
                double diagonal[1u << N];
                observable<N>(c, T, tape.parity, diagonal);
                for (std::size_t b0 = 0; b0 < batch; b0 += lanes) {
                    const std::size_t count = std::min(lanes, batch - b0);
                    encode(input, inputs, i, b0, count, w, bb, s);
                    double u[lanes];
                    for (std::size_t l = 0; l < lanes; ++l) u[l] = l < count ? upstream[(b0 + l) * outputs + o] : 0.0;
                    Amplitude<Vector> psi[1u << N], lambda[1u << N];
                    Vector m[max_qubits];
                    run_fused_forward<N>(tape, s.blocks, s.encodings, psi);
                    readout<N>(psi, T, tape.parity, m);
                    for (std::size_t l = 0; l < count; ++l)
                        for (std::size_t t = 0; t < T; ++t) dc[t] += u[l] * m[t].v[l];
                    for (unsigned k = 0; k < (1u << N); ++k) {
                        Vector scale;
                        for (std::size_t l = 0; l < lanes; ++l) scale.v[l] = u[l] * diagonal[k];
                        lambda[k] = {scale * psi[k].re, scale * psi[k].im};
                    }
                    Vector dx(0.0);
                    run_adjoint<N>(tape, s.angles, s.encodings, psi, lambda, [&](const TapeEntry& e, const Vector& d) {
                        if (e.op == Op::Rotation) {
                            for (std::size_t l = 0; l < count; ++l) dtheta[e.slot] += d.v[l];
                        } else {
                            for (std::size_t l = 0; l < count; ++l) {
                                db[e.slot] += d.v[l];
                                dw[e.slot] += s.x[l] * d.v[l];
                            }
                            dx += w[e.slot] * d;
                        }
                    });
                    for (std::size_t l = 0; l < count; ++l) input_gradient[(b0 + l) * inputs + i] += dx.v[l];
                }
            }
    });
}

PqcGradients zero_nonlinear(const PqcEdges& edges) {
    return {std::vector<double>(edges.angles.size(), 0.0), std::vector<double>(edges.encoding_weights.size(), 0.0),
            std::vector<double>(edges.encoding_biases.size(), 0.0)};
}

void check_gradient(const PqcEdges& edges, const PqcGradients& gradient) {
    if (gradient.angles.size() != edges.angles.size() ||
        gradient.encoding_weights.size() != edges.encoding_weights.size() ||
        gradient.encoding_biases.size() != edges.encoding_biases.size())
        throw std::invalid_argument("PQC gradient shape mismatch");
    require_finite(gradient.angles);
    require_finite(gradient.encoding_weights);
    require_finite(gradient.encoding_biases);
}

// Every candidate is finite (std::overflow_error otherwise; the layer is unchanged).
void update_nonlinear(PqcEdges& candidate, const PqcGradients& gradient, double rate) {
    for (std::size_t j = 0; j < candidate.angles.size(); ++j) candidate.angles[j] -= rate * gradient.angles[j];
    for (std::size_t j = 0; j < candidate.encoding_weights.size(); ++j) {
        candidate.encoding_weights[j] -= rate * gradient.encoding_weights[j];
        candidate.encoding_biases[j] -= rate * gradient.encoding_biases[j];
    }
    result_finite(candidate.angles);
    result_finite(candidate.encoding_weights);
    result_finite(candidate.encoding_biases);
}

void check_result(const PqcGradients& gradient) {
    result_finite(gradient.angles);
    result_finite(gradient.encoding_weights);
    result_finite(gradient.encoding_biases);
}

} // namespace kan::detail

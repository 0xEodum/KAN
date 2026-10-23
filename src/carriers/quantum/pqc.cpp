// PQC carrier (M5 phase 1): configuration validation, the single-circuit
// evaluation of kan/pqc.hpp (exact expectations, adjoint and parameter-shift
// derivatives) and the family setter, from the shared tape and formulas
// (carriers/quantum/tape.hpp, pqc_formulas.hpp). Error messages are literals.
#include "carriers/quantum/pqc_formulas.hpp"
#include "core/checks.hpp"
#include "kan/families.hpp"
#include "kan/pqc.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace kan {
namespace {
using namespace detail::pqc;

bool valid(Rotation a) { return a == Rotation::X || a == Rotation::Y || a == Rotation::Z; }

// Per-slot half-angles of one circuit at x: angles, encodings w x + b.
struct Circuit {
    Tape tape;
    HalfAngle<double> angles[max_angles], encodings[max_encodings];
    Matrix2<double> blocks[(max_repetitions + 1) * max_qubits];
    double phi[max_encodings];
    void block(std::size_t g) { blocks[g] = block_matrix(tape.axes, tape.block, angles + g * tape.block); }
};

void prepare(const PqcConfig& config, double x, std::span<const double> angles,
             std::span<const double> encoding_weights, std::span<const double> encoding_biases, Circuit& circuit) {
    validate_pqc(config);
    if (angles.size() != pqc_angles(config) || encoding_weights.size() != pqc_encodings(config) ||
        encoding_biases.size() != pqc_encodings(config))
        throw std::invalid_argument("PQC parameter count does not match the configuration");
    if (!std::isfinite(x)) throw std::invalid_argument("PQC input must be finite");
    detail::require_finite(angles);
    detail::require_finite(encoding_weights);
    detail::require_finite(encoding_biases);
    compile_tape(config, circuit.tape);
    for (std::size_t j = 0; j < angles.size(); ++j) circuit.angles[j] = half_angle(angles[j]);
    for (std::size_t g = 0; g * circuit.tape.block < angles.size(); ++g) circuit.block(g);
    for (std::size_t j = 0; j < encoding_weights.size(); ++j) {
        circuit.phi[j] = encoding_weights[j] * x + encoding_biases[j];
        circuit.encodings[j] = half_angle(circuit.phi[j]);
    }
}
} // namespace

std::size_t pqc_terms(const PqcConfig& config) noexcept {
    return config.readout == PqcReadout::Parity ? 1 : config.qubits;
}
std::size_t pqc_angles(const PqcConfig& config) noexcept {
    return (config.repetitions + 1) * config.qubits * config.block.size();
}
std::size_t pqc_encodings(const PqcConfig& config) noexcept { return config.repetitions * config.qubits; }

void validate_pqc(const PqcConfig& config) {
    if (config.qubits < 1 || config.qubits > max_qubits) throw std::invalid_argument("PQC qubits must be in 1..5");
    if (config.repetitions < 1 || config.repetitions > max_repetitions)
        throw std::invalid_argument("PQC repetitions must be in 1..32");
    if (config.block.empty() || config.block.size() > max_block)
        throw std::invalid_argument("PQC block must have 1..3 rotation axes");
    if (!valid(config.encoding)) throw std::invalid_argument("invalid PQC encoding axis");
    for (std::size_t k = 0; k < config.block.size(); ++k) {
        if (!valid(config.block[k])) throw std::invalid_argument("invalid PQC block axis");
        if (k > 0 && config.block[k] == config.block[k - 1])
            throw std::invalid_argument("adjacent PQC block axes must differ (their angles add)");
    }
    switch (config.entangler) {
    case Entangler::None: break;
    case Entangler::CnotChain:
    case Entangler::CzChain:
        if (config.qubits < 2) throw std::invalid_argument("a PQC entangler needs at least two qubits");
        break;
    case Entangler::CnotRing:
        if (config.qubits < 3) throw std::invalid_argument("a PQC CNOT ring needs at least three qubits");
        break;
    default: throw std::invalid_argument("invalid PQC entangler");
    }
    if (config.readout != PqcReadout::PauliZ && config.readout != PqcReadout::Parity)
        throw std::invalid_argument("invalid PQC readout");
    if (config.shots != 0) throw std::invalid_argument("finite shots are not available");
}

PqcEvaluation evaluate_pqc(const PqcConfig& config, double x, std::span<const double> angles,
                           std::span<const double> encoding_weights, std::span<const double> encoding_biases,
                           GradientMethod method) {
    if (method != GradientMethod::Adjoint && method != GradientMethod::ParameterShift)
        throw std::invalid_argument("invalid PQC gradient method");
    Circuit c;
    prepare(config, x, angles, encoding_weights, encoding_biases, c);
    const auto& tape = c.tape;
    const std::size_t T = tape.terms, A = tape.angles, E = tape.encodings;
    PqcEvaluation r{std::vector<double>(T), std::vector<double>(T, 0.0), std::vector<double>(T * A, 0.0),
                    std::vector<double>(T * E, 0.0), std::vector<double>(T * E, 0.0)};
    dispatch_qubits(tape.qubits, [&](auto qubits) {
        constexpr unsigned N = decltype(qubits)::value;
        Amplitude<double> psi[1u << N], lambda[1u << N];
        run_fused_forward<N>(tape, c.blocks, c.encodings, psi);
        readout<N>(psi, T, tape.parity, r.values.data());
        if (method == GradientMethod::Adjoint) {
            // One reverse pass per term t, lambda = M_t psi.
            for (std::size_t t = 0; t < T; ++t) {
                Amplitude<double> state[1u << N];
                for (unsigned k = 0; k < (1u << N); ++k) {
                    state[k] = psi[k];
                    const double sign = negative<N>(k, unsigned(t), tape.parity) ? -1.0 : 1.0;
                    lambda[k] = {sign * psi[k].re, sign * psi[k].im};
                }
                double dx = 0;
                run_adjoint<N>(tape, c.angles, c.encodings, state, lambda, [&](const TapeEntry& e, double d) {
                    if (e.op == Op::Rotation) {
                        r.angle_derivatives[t * A + e.slot] = d;
                    } else {
                        r.encoding_bias_derivatives[t * E + e.slot] = d;
                        r.encoding_weight_derivatives[t * E + e.slot] = x * d;
                        dx += encoding_weights[e.slot] * d;
                    }
                });
                r.input_derivatives[t] = dx;
            }
            return;
        }
        // Parameter shift: d m / dp = (m(p + pi/2) - m(p - pi/2)) / 2 for every
        // non-inert slot (angles, then encodings with p = w x + b).
        constexpr double quarter = std::numbers::pi / 2;
        double plus[max_qubits], minus[max_qubits];
        const auto measure = [&](double* m) {
            run_fused_forward<N>(tape, c.blocks, c.encodings, psi);
            readout<N>(psi, T, tape.parity, m);
        };
        // A shifted angle changes the block matrix of its group j / B; a shifted
        // encoding only its half-angle.
        const auto shift_angle = [&](std::size_t j, double p, double* m) {
            const auto saved = c.angles[j];
            c.angles[j] = half_angle(p);
            c.block(j / tape.block);
            measure(m);
            c.angles[j] = saved;
            c.block(j / tape.block);
        };
        const auto shift_encoding = [&](std::size_t j, double p, double* m) {
            const auto saved = c.encodings[j];
            c.encodings[j] = half_angle(p);
            measure(m);
            c.encodings[j] = saved;
        };
        bool inert_angle[max_angles] = {}, inert_encoding[max_encodings] = {};
        for (std::size_t g = 0; g < tape.size; ++g) {
            const auto& e = tape.entries[g];
            if (e.op == Op::Rotation) inert_angle[e.slot] = e.inert;
            if (e.op == Op::Encoding) inert_encoding[e.slot] = e.inert;
        }
        for (std::size_t j = 0; j < A; ++j) {
            if (inert_angle[j]) continue;
            shift_angle(j, angles[j] + quarter, plus);
            shift_angle(j, angles[j] - quarter, minus);
            for (std::size_t t = 0; t < T; ++t) r.angle_derivatives[t * A + j] = (plus[t] - minus[t]) / 2;
        }
        for (std::size_t j = 0; j < E; ++j) {
            if (inert_encoding[j]) continue;
            shift_encoding(j, c.phi[j] + quarter, plus);
            shift_encoding(j, c.phi[j] - quarter, minus);
            for (std::size_t t = 0; t < T; ++t) {
                const double d = (plus[t] - minus[t]) / 2;
                r.encoding_bias_derivatives[t * E + j] = d;
                r.encoding_weight_derivatives[t * E + j] = x * d;
            }
        }
        for (std::size_t t = 0; t < T; ++t) {
            double dx = 0;
            for (std::size_t j = 0; j < E; ++j) dx += encoding_weights[j] * r.encoding_bias_derivatives[t * E + j];
            r.input_derivatives[t] = dx;
        }
    });
    // An encoding angle w x + b that overflows makes the circuit nonfinite.
    for (const auto* v : {&r.values, &r.input_derivatives, &r.angle_derivatives, &r.encoding_weight_derivatives,
                          &r.encoding_bias_derivatives})
        detail::result_finite(*v);
    return r;
}

std::vector<std::complex<double>> pqc_state(const PqcConfig& config, double x, std::span<const double> angles,
                                            std::span<const double> encoding_weights,
                                            std::span<const double> encoding_biases) {
    Circuit c;
    prepare(config, x, angles, encoding_weights, encoding_biases, c);
    std::vector<std::complex<double>> state(std::size_t{1} << config.qubits);
    dispatch_qubits(config.qubits, [&](auto qubits) {
        constexpr unsigned N = decltype(qubits)::value;
        Amplitude<double> psi[1u << N];
        run_fused_forward<N>(c.tape, c.blocks, c.encodings, psi);
        for (unsigned k = 0; k < (1u << N); ++k) {
            if (!std::isfinite(psi[k].re) || !std::isfinite(psi[k].im))
                throw std::overflow_error("nonfinite numerical result");
            state[k] = {psi[k].re, psi[k].im};
        }
    });
    return state;
}

void set_pqc_parameters(Layer& layer, std::span<const double> coefficients, std::span<const double> angles,
                        std::span<const double> encoding_weights, std::span<const double> encoding_biases,
                        std::span<const double> bias) {
    const auto* edges = std::get_if<PqcEdges>(&layer.carrier());
    if (!edges) throw std::invalid_argument("PQC parameters require a PQC layer");
    // set_carrier validates the configuration, the shapes for the layer and finiteness.
    layer.set_carrier(PqcEdges{edges->config,
                               {coefficients.begin(), coefficients.end()},
                               {angles.begin(), angles.end()},
                               {encoding_weights.begin(), encoding_weights.end()},
                               {encoding_biases.begin(), encoding_biases.end()},
                               edges->shot_seed},
                      bias);
}

} // namespace kan

#pragma once

// The PQC configuration compiled into a flat gate tape (M5 D3): one entry per
// rotation, encoding rotation and two-qubit gate, in application order. The
// tape is identical for every edge of a layer (per-edge parameters are read
// through `slot`), so a device kernel's opcode stream is warp-uniform. On the
// CPU it is compiled once per call into this bounded aggregate (at most 815
// entries at the D9 limits), never stored in the public carrier. Numerical
// code reads integers only: no strings, hashing or parsing.

#include "kan/pqc.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace kan::detail::pqc {

inline constexpr std::size_t max_qubits = 5;
inline constexpr std::size_t max_repetitions = 32;
inline constexpr std::size_t max_block = 3;
// (r+1) n B rotations + r n encodings + r n two-qubit gates (a CNOT ring) at the limits.
inline constexpr std::size_t max_angles = (max_repetitions + 1) * max_qubits * max_block; // 495
inline constexpr std::size_t max_encodings = max_repetitions * max_qubits;                // 160
inline constexpr std::size_t max_tape = max_angles + 2 * max_encodings;                   // 815

enum class Op : std::uint8_t { Rotation, Encoding, Cnot, Cz };

// Rotation: R_axis(angles[slot]) on `qubit`; Encoding: R_axis(w[slot] x + b[slot])
// on `qubit`; Cnot: control `qubit`, target `other`; Cz: `qubit` and `other`.
// `inert` marks a rotation whose parameter cannot change any readout: a Z
// rotation while its qubit is still |0> up to a phase (before it on that qubit
// only Z rotations, CZ, CNOTs it controls, and CNOTs on it as the target only
// of a control in |0>), or a Z rotation followed on its qubit only by Z
// rotations, CZ and CNOTs that it controls (Z_q commutes with all of them and
// with the diagonal Z and parity readouts). Its derivative is exactly zero and
// is not computed.
struct TapeEntry {
    Op op;
    std::uint8_t axis; // static_cast<std::uint8_t>(Rotation) for Rotation and Encoding
    std::uint8_t qubit, other;
    std::uint16_t slot;
    bool inert;
};

struct Tape {
    TapeEntry entries[max_tape];
    std::size_t size = 0;
    // The first entry with a derivative (a non-inert rotation or encoding): the
    // reverse pass ends there (earlier entries only restore a discarded state).
    std::size_t first_derivative = 0;
    // The two-qubit gates of one entangler layer E (identical for every l), in order.
    TapeEntry entangler[max_qubits];
    std::size_t entangler_size = 0;
    std::uint8_t encoding_axis = 0;
    std::uint8_t axes[max_block] = {}; // block axes a_0..a_(B-1)
    std::size_t repetitions = 0, block = 0;
    std::size_t qubits = 0, terms = 0, angles = 0, encodings = 0;
    bool parity = false;
};

// Compiles a validated configuration (validate_pqc).
inline void compile_tape(const PqcConfig& config, Tape& tape) {
    const auto n = config.qubits, B = config.block.size();
    tape.size = 0;
    tape.entangler_size = 0;
    tape.encoding_axis = static_cast<std::uint8_t>(config.encoding);
    tape.repetitions = config.repetitions;
    tape.block = B;
    for (std::size_t k = 0; k < B; ++k) tape.axes[k] = static_cast<std::uint8_t>(config.block[k]);
    tape.qubits = n;
    tape.parity = config.readout == PqcReadout::Parity;
    tape.terms = tape.parity ? 1 : n;
    tape.angles = (config.repetitions + 1) * n * B;
    tape.encodings = config.repetitions * n;
    const auto push = [&](Op op, std::uint8_t axis, std::size_t qubit, std::size_t other, std::size_t slot) {
        tape.entries[tape.size++] = {op,
                                     axis,
                                     static_cast<std::uint8_t>(qubit),
                                     static_cast<std::uint8_t>(other),
                                     static_cast<std::uint16_t>(slot),
                                     false};
    };
    const auto block = [&](std::size_t l) {
        for (std::size_t q = 0; q < n; ++q)
            for (std::size_t k = 0; k < B; ++k)
                push(Op::Rotation, static_cast<std::uint8_t>(config.block[k]), q, q, (l * n + q) * B + k);
    };
    block(0);
    for (std::size_t l = 1; l <= config.repetitions; ++l) {
        switch (config.entangler) {
        case Entangler::None: break;
        case Entangler::CnotChain:
        case Entangler::CnotRing:
            for (std::size_t q = 0; q + 1 < n; ++q) push(Op::Cnot, 0, q, q + 1, 0);
            if (config.entangler == Entangler::CnotRing) push(Op::Cnot, 0, n - 1, 0, 0);
            break;
        case Entangler::CzChain:
            for (std::size_t q = 0; q + 1 < n; ++q) push(Op::Cz, 0, q, q + 1, 0);
            break;
        }
        for (std::size_t q = 0; q < n; ++q)
            push(Op::Encoding, static_cast<std::uint8_t>(config.encoding), q, q, (l - 1) * n + q);
        block(l);
    }
    for (std::size_t g = 0; g < tape.size; ++g) {
        const auto& e = tape.entries[g];
        if (e.op == Op::Encoding) break; // the first layer's entangler precedes its encodings
        if (e.op == Op::Cnot || e.op == Op::Cz) tape.entangler[tape.entangler_size++] = e;
    }
    // Structurally inert Z rotations, forward (leading) and backward (trailing).
    constexpr auto z = static_cast<std::uint8_t>(Rotation::Z);
    bool ground[max_qubits], commutes[max_qubits];
    for (std::size_t q = 0; q < n; ++q) ground[q] = commutes[q] = true;
    for (std::size_t g = 0; g < tape.size; ++g) {
        auto& e = tape.entries[g];
        if (e.op == Op::Rotation || e.op == Op::Encoding) {
            if (e.axis == z)
                e.inert = e.inert || ground[e.qubit];
            else
                ground[e.qubit] = false;
        } else if (e.op == Op::Cnot && !ground[e.qubit]) {
            ground[e.other] = false; // a control off |0> entangles the target
        }
    }
    for (std::size_t g = tape.size; g-- > 0;) {
        auto& e = tape.entries[g];
        if (e.op == Op::Rotation || e.op == Op::Encoding) {
            if (e.axis == z)
                e.inert = e.inert || commutes[e.qubit];
            else
                commutes[e.qubit] = false;
        } else if (e.op == Op::Cnot) {
            commutes[e.other] = false; // Z on the target does not commute with CNOT
        }
    }
    tape.first_derivative = tape.size;
    for (std::size_t g = 0; g < tape.size && tape.first_derivative == tape.size; ++g) {
        const auto& e = tape.entries[g];
        if ((e.op == Op::Rotation || e.op == Op::Encoding) && !e.inert) tape.first_derivative = g;
    }
}

// Calls f(std::integral_constant<unsigned, N>{}) for N = qubits (1..5, validated):
// the simulator is compiled per qubit count, so its loops have constant trip counts.
template<class F> decltype(auto) dispatch_qubits(std::size_t qubits, F&& f) {
    switch (qubits) {
    case 1: return f(std::integral_constant<unsigned, 1>{});
    case 2: return f(std::integral_constant<unsigned, 2>{});
    case 3: return f(std::integral_constant<unsigned, 3>{});
    case 4: return f(std::integral_constant<unsigned, 4>{});
    default: return f(std::integral_constant<unsigned, 5>{});
    }
}

} // namespace kan::detail::pqc

#pragma once

// Shared host/device state-vector formulas of the PQC carrier (M5): in-place
// gate application on 2^N amplitudes (N qubits, a compile-time constant), the
// fused forward (one 2x2 matrix per qubit and layer), Born readouts and the
// gate-by-gate adjoint step. The amplitude scalar S is double (CPU
// reference, FP64 resident), float (FP32 resident) or a vector of independent
// samples (the CPU's sample lanes, backends/cpu/lanes.hpp: every lane runs this
// exact operation sequence); a half-angle scalar H is S or a shared scalar.
// Little-endian: bit q of a basis index is qubit q. Every summation runs in
// ascending pair or basis index.

#include "carriers/quantum/tape.hpp"
#include "core/host_device.hpp"

// Gate kernels: out of line in host code (one compact, vectorized function per
// gate kind, qubit count and scalar; inlined into one interpreter they compile
// to slower code), inlined in device code.
#if defined(__CUDA_ARCH__)
#define KAN_PQC_GATE KAN_FORCE_INLINE
#else
#define KAN_PQC_GATE KAN_NOINLINE
#endif

namespace kan::detail::pqc {

template<class S> struct Amplitude {
    S re, im;
};
// cos(p/2) and sin(p/2) of a rotation angle p; R_a(p) = c I - i s sigma_a.
template<class S> struct HalfAngle {
    S c, s;
};

template<class S> KAN_HOST_DEVICE inline HalfAngle<S> half_angle(S p) {
    const S h = p * S(0.5); // exact
    return {math::cos(h), math::sin(h)};
}

// The pair (i0, i0 | 2^q) of pair index j < 2^(N-1).
KAN_HOST_DEVICE inline unsigned pair_low(unsigned j, unsigned q) {
    return ((j >> q) << (q + 1)) | (j & ((1u << q) - 1u));
}

// psi <- R_axis(p) psi on qubit q, from h = (cos p/2, sin p/2); the inverse
// R_axis(-p) is h = (c, -s). X: [[c, -is], [-is, c]]; Y: [[c, -s], [s, c]];
// Z: diag(c - is, c + is). One function per axis (Axis = 0, 1, 2 for X, Y, Z).
template<unsigned N, unsigned Axis, class S, class H>
KAN_HOST_DEVICE KAN_PQC_GATE void rotate_axis(Amplitude<S>* psi, unsigned q, HalfAngle<H> h) {
    const H c = h.c, s = h.s;
    const unsigned bit = 1u << q;
    for (unsigned j = 0; j < (1u << (N - 1)); ++j) {
        const unsigned i0 = pair_low(j, q), i1 = i0 | bit;
        const auto a0 = psi[i0], a1 = psi[i1];
        if constexpr (Axis == 0) {
            psi[i0] = {c * a0.re + s * a1.im, c * a0.im - s * a1.re};
            psi[i1] = {c * a1.re + s * a0.im, c * a1.im - s * a0.re};
        } else if constexpr (Axis == 1) {
            psi[i0] = {c * a0.re - s * a1.re, c * a0.im - s * a1.im};
            psi[i1] = {s * a0.re + c * a1.re, s * a0.im + c * a1.im};
        } else {
            psi[i0] = {c * a0.re + s * a0.im, c * a0.im - s * a0.re};
            psi[i1] = {c * a1.re - s * a1.im, c * a1.im + s * a1.re};
        }
    }
}
template<unsigned N, class S, class H>
KAN_HOST_DEVICE KAN_FORCE_INLINE void rotate(Amplitude<S>* psi, unsigned q, std::uint8_t axis, HalfAngle<H> h) {
    if (axis == static_cast<std::uint8_t>(Rotation::X))
        rotate_axis<N, 0>(psi, q, h);
    else if (axis == static_cast<std::uint8_t>(Rotation::Y))
        rotate_axis<N, 1>(psi, q, h);
    else
        rotate_axis<N, 2>(psi, q, h);
}

// CNOT(control -> target) and CZ(a, b); both are their own inverses.
template<unsigned N, class S>
KAN_HOST_DEVICE KAN_FORCE_INLINE void cnot(Amplitude<S>* psi, unsigned control, unsigned target) {
    for (unsigned k = 0; k < (1u << N); ++k)
        if (((k >> control) & 1u) && !((k >> target) & 1u)) {
            const auto t = psi[k];
            psi[k] = psi[k | (1u << target)];
            psi[k | (1u << target)] = t;
        }
}
template<unsigned N, class S> KAN_HOST_DEVICE KAN_FORCE_INLINE void cz(Amplitude<S>* psi, unsigned a, unsigned b) {
    for (unsigned k = 0; k < (1u << N); ++k)
        if ((k >> a) & (k >> b) & 1u) psi[k] = {-psi[k].re, -psi[k].im};
}

// A 2x2 complex matrix, row-major.
template<class T> struct Matrix2 {
    Amplitude<T> m00, m01, m10, m11;
};

// (a.re + i a.im)(b.re + i b.im); A and B may differ (a shared scalar times a lane vector).
template<class A, class B> KAN_HOST_DEVICE KAN_FORCE_INLINE auto multiply(Amplitude<A> a, Amplitude<B> b) {
    using R = decltype(a.re * b.re);
    return Amplitude<R>{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

// R_axis(p) as a matrix from h = (cos p/2, sin p/2).
template<class T> KAN_HOST_DEVICE inline Matrix2<T> rotation_matrix(std::uint8_t axis, HalfAngle<T> h) {
    const T zero = T(0);
    if (axis == static_cast<std::uint8_t>(Rotation::X)) return {{h.c, zero}, {zero, -h.s}, {zero, -h.s}, {h.c, zero}};
    if (axis == static_cast<std::uint8_t>(Rotation::Y)) return {{h.c, zero}, {-h.s, zero}, {h.s, zero}, {h.c, zero}};
    return {{h.c, -h.s}, {zero, zero}, {zero, zero}, {h.c, h.s}};
}

// a b for 2x2 complex matrices (each entry: the k = 0 product plus the k = 1 product).
template<class A, class B> KAN_HOST_DEVICE KAN_FORCE_INLINE auto product(const Matrix2<A>& a, const Matrix2<B>& b) {
    const auto add = [](auto x, auto y) { return decltype(x){x.re + y.re, x.im + y.im}; };
    return Matrix2<decltype(a.m00.re * b.m00.re)>{
        add(multiply(a.m00, b.m00), multiply(a.m01, b.m10)), add(multiply(a.m00, b.m01), multiply(a.m01, b.m11)),
        add(multiply(a.m10, b.m00), multiply(a.m11, b.m10)), add(multiply(a.m10, b.m01), multiply(a.m11, b.m11))};
}

// The block W_(l,q) = R_(a_(B-1))(theta_(B-1)) ... R_(a_0)(theta_0) from the half-angles of
// its B consecutive angle slots (built left to right: W <- R_k W).
template<class T>
KAN_HOST_DEVICE inline Matrix2<T> block_matrix(const std::uint8_t* axes, std::size_t B, const HalfAngle<T>* angles) {
    Matrix2<T> w = rotation_matrix(axes[0], angles[0]);
    for (std::size_t k = 1; k < B; ++k) w = product(rotation_matrix(axes[k], angles[k]), w);
    return w;
}

// W R_axis(p) from h = (cos p/2, sin p/2), row by row: (w0, w1) R. Y: R real,
// (w0 c + w1 s, w1 c - w0 s); X: R = [[c, -is], [-is, c]], (w0 c - i s w1,
// w1 c - i s w0); Z: R = diag(c - is, c + is), (w0 (c - is), w1 (c + is)).
template<class A, class H>
KAN_HOST_DEVICE KAN_FORCE_INLINE auto times_rotation(const Matrix2<A>& w, std::uint8_t axis, HalfAngle<H> h) {
    using R = decltype(w.m00.re * h.c);
    const H c = h.c, s = h.s;
    const auto row = [&](Amplitude<A> w0, Amplitude<A> w1, Amplitude<R>& m0, Amplitude<R>& m1) {
        if (axis == static_cast<std::uint8_t>(Rotation::Y)) {
            m0 = {w0.re * c + w1.re * s, w0.im * c + w1.im * s};
            m1 = {w1.re * c - w0.re * s, w1.im * c - w0.im * s};
        } else if (axis == static_cast<std::uint8_t>(Rotation::X)) {
            m0 = {w0.re * c + w1.im * s, w0.im * c - w1.re * s};
            m1 = {w1.re * c + w0.im * s, w1.im * c - w0.re * s};
        } else {
            m0 = {w0.re * c + w0.im * s, w0.im * c - w0.re * s};
            m1 = {w1.re * c - w1.im * s, w1.im * c + w1.re * s};
        }
    };
    Matrix2<R> m;
    row(w.m00, w.m01, m.m00, m.m01);
    row(w.m10, w.m11, m.m10, m.m11);
    return m;
}

// psi <- M psi on qubit q for a general 2x2 complex M.
template<unsigned N, class S, class H>
KAN_HOST_DEVICE KAN_PQC_GATE void apply_matrix(Amplitude<S>* psi, unsigned q, const Matrix2<H>& m) {
    const unsigned bit = 1u << q;
    for (unsigned j = 0; j < (1u << (N - 1)); ++j) {
        const unsigned i0 = pair_low(j, q), i1 = i0 | bit;
        const auto a0 = psi[i0], a1 = psi[i1];
        const auto p00 = multiply(m.m00, a0), p01 = multiply(m.m01, a1);
        const auto p10 = multiply(m.m10, a0), p11 = multiply(m.m11, a1);
        psi[i0] = {p00.re + p01.re, p00.im + p01.im};
        psi[i1] = {p10.re + p11.re, p10.im + p11.im};
    }
}

// The circuit applied to |0...0> one fused single-qubit group at a time:
// W_(0,q), then per layer l the entangler and M_(l,q) = W_(l,q) S_(l,q)(x),
// from the per-edge block matrices blocks[l n + q] and the encoding
// half-angles of the sample (mathematically the tape's product: gates on
// different qubits commute; rounding differs from the gate-by-gate form).
template<unsigned N, class S, class H, class HE>
KAN_HOST_DEVICE inline void run_fused_forward(const Tape& tape, const Matrix2<H>* blocks,
                                              const HalfAngle<HE>* encodings, Amplitude<S>* psi) {
    for (unsigned k = 0; k < (1u << N); ++k) psi[k] = {S(0), S(0)};
    psi[0].re = S(1);
    for (unsigned q = 0; q < N; ++q) apply_matrix<N>(psi, q, blocks[q]);
    for (std::size_t l = 1; l <= tape.repetitions; ++l) {
        for (std::size_t g = 0; g < tape.entangler_size; ++g) {
            const auto& e = tape.entangler[g];
            if (e.op == Op::Cnot)
                cnot<N>(psi, e.qubit, e.other);
            else
                cz<N>(psi, e.qubit, e.other);
        }
        for (unsigned q = 0; q < N; ++q) {
            const auto& h = encodings[(l - 1) * N + q];
            apply_matrix<N>(psi, q, times_rotation(blocks[l * N + q], tape.encoding_axis, h));
        }
    }
}

// Sign of basis state k in readout term t: Z_t (PauliZ) or Z...Z (parity).
template<unsigned N> KAN_HOST_DEVICE inline bool negative(unsigned k, unsigned t, bool parity) {
    if (!parity) return (k >> t) & 1u;
    unsigned ones = 0;
    for (unsigned q = 0; q < N; ++q) ones += (k >> q) & 1u;
    return ones & 1u;
}

// m_t = sum_k sign_t(k) |psi_k|^2 (ascending k), t < terms.
template<unsigned N, class S>
KAN_HOST_DEVICE inline void readout(const Amplitude<S>* psi, std::size_t terms, bool parity, S* m) {
    S p[1u << N];
    for (unsigned k = 0; k < (1u << N); ++k) p[k] = psi[k].re * psi[k].re + psi[k].im * psi[k].im;
    for (unsigned t = 0; t < terms; ++t) {
        S sum = S(0);
        for (unsigned k = 0; k < (1u << N); ++k) sum += negative<N>(k, t, parity) ? -p[k] : p[k];
        m[t] = sum;
    }
}

// The diagonal observable O = sum_t c_t M_t: o_k = sum_t sign_t(k) c_t (ascending t).
template<unsigned N, class S> KAN_HOST_DEVICE inline void observable(const S* c, std::size_t terms, bool parity, S* o) {
    for (unsigned k = 0; k < (1u << N); ++k) {
        S sum = S(0);
        for (unsigned t = 0; t < terms; ++t) sum += negative<N>(k, t, parity) ? -c[t] : c[t];
        o[k] = sum;
    }
}

// One adjoint step of a parametrized gate on qubit q, psi and lambda after it:
// returns d = Im <lambda|sigma_axis|psi> (= d<psi|O|psi>/dp for
// U = exp(-i p sigma/2), lambda = O psi), summed over the pairs in ascending
// order, then applies U^dagger (h = (c, s) of U) to psi and lambda. X:
// Im(l0* p1) + Im(l1* p0); Y: Re(l1* p0) - Re(l0* p1); Z: Im(l0* p0) -
// Im(l1* p1). Without `derivative` d is 0. (Three passes: on the CPU's
// 8-lane vectors one fused pass over psi and lambda spills registers and was
// measured slower.)
template<unsigned N, unsigned Axis, class S>
KAN_HOST_DEVICE KAN_PQC_GATE S generator(const Amplitude<S>* psi, const Amplitude<S>* lambda, unsigned q) {
    const unsigned bit = 1u << q;
    S d = S(0);
    for (unsigned j = 0; j < (1u << (N - 1)); ++j) {
        const unsigned i0 = pair_low(j, q), i1 = i0 | bit;
        const auto l0 = lambda[i0], l1 = lambda[i1], p0 = psi[i0], p1 = psi[i1];
        if constexpr (Axis == 0)
            d += (l0.re * p1.im - l0.im * p1.re) + (l1.re * p0.im - l1.im * p0.re);
        else if constexpr (Axis == 1)
            d += (l1.re * p0.re + l1.im * p0.im) - (l0.re * p1.re + l0.im * p1.im);
        else
            d += (l0.re * p0.im - l0.im * p0.re) - (l1.re * p1.im - l1.im * p1.re);
    }
    return d;
}
template<unsigned N, class S, class H>
KAN_HOST_DEVICE KAN_FORCE_INLINE S adjoint_rotate(Amplitude<S>* psi, Amplitude<S>* lambda, unsigned q,
                                                  std::uint8_t axis, HalfAngle<H> h, bool derivative) {
    const HalfAngle<H> inverse{h.c, -h.s};
    S d = S(0);
    if (axis == static_cast<std::uint8_t>(Rotation::X)) {
        if (derivative) d = generator<N, 0>(psi, lambda, q);
        rotate_axis<N, 0>(psi, q, inverse);
        rotate_axis<N, 0>(lambda, q, inverse);
    } else if (axis == static_cast<std::uint8_t>(Rotation::Y)) {
        if (derivative) d = generator<N, 1>(psi, lambda, q);
        rotate_axis<N, 1>(psi, q, inverse);
        rotate_axis<N, 1>(lambda, q, inverse);
    } else {
        if (derivative) d = generator<N, 2>(psi, lambda, q);
        rotate_axis<N, 2>(psi, q, inverse);
        rotate_axis<N, 2>(lambda, q, inverse);
    }
    return d;
}

// Reverse pass from psi = U|0...0> and lambda = O psi: calls
// sink(entry, d) for every parametrized, non-inert gate, from the last gate to
// the first one with a derivative (tape.first_derivative); psi and lambda are
// left in an unspecified state.
template<unsigned N, class S, class HA, class HE, class Sink>
KAN_HOST_DEVICE inline void run_adjoint(const Tape& tape, const HalfAngle<HA>* angles, const HalfAngle<HE>* encodings,
                                        Amplitude<S>* psi, Amplitude<S>* lambda, Sink&& sink) {
    for (std::size_t g = tape.size; g-- > tape.first_derivative;) {
        const auto& e = tape.entries[g];
        switch (e.op) {
        case Op::Rotation: {
            const S d = adjoint_rotate<N>(psi, lambda, e.qubit, e.axis, angles[e.slot], !e.inert);
            if (!e.inert) sink(e, d);
            break;
        }
        case Op::Encoding: {
            const S d = adjoint_rotate<N>(psi, lambda, e.qubit, e.axis, encodings[e.slot], !e.inert);
            if (!e.inert) sink(e, d);
            break;
        }
        case Op::Cnot:
            cnot<N>(psi, e.qubit, e.other);
            cnot<N>(lambda, e.qubit, e.other);
            break;
        case Op::Cz:
            cz<N>(psi, e.qubit, e.other);
            cz<N>(lambda, e.qubit, e.other);
            break;
        }
    }
}

} // namespace kan::detail::pqc

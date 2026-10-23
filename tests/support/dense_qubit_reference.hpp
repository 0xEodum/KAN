#pragma once
// Independent dense reference for the PQC carrier (M5 phase 1). Every gate is
// a full 2^n x 2^n matrix built by Kronecker products of explicitly written
// 2x2 matrices (two-qubit gates as sums of such products: CNOT = P0 (x) I +
// P1 (x) X, CZ = P0 (x) I + P1 (x) Z), applied by dense matrix-vector
// products; observables are dense diagonal matrices of the same form. A
// different algorithm from the library's in-place tape simulator; this header
// includes nothing from src/ or kan/ (checked by the quantum source check).
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace dense_qubit {

using complex = std::complex<double>;
using Matrix = std::vector<complex>; // row-major, square

// Plain description (independent of the library's types): axes 0 = X, 1 = Y,
// 2 = Z; entangler 0 none, 1 CNOT chain, 2 CNOT ring, 3 CZ chain; readout 0
// PauliZ per qubit, 1 parity.
struct Circuit {
    std::size_t qubits = 1, repetitions = 1;
    int encoding = 1;
    std::vector<int> block;
    int entangler = 0;
    int readout = 0;
};

inline std::size_t dimension(const Matrix& m) {
    std::size_t d = 0;
    while (d * d < m.size()) ++d;
    return d;
}

inline Matrix kron(const Matrix& a, const Matrix& b) {
    const std::size_t da = dimension(a), db = dimension(b), d = da * db;
    Matrix r(d * d);
    for (std::size_t i = 0; i < da; ++i)
        for (std::size_t j = 0; j < da; ++j)
            for (std::size_t k = 0; k < db; ++k)
                for (std::size_t l = 0; l < db; ++l) r[(i * db + k) * d + (j * db + l)] = a[i * da + j] * b[k * db + l];
    return r;
}

inline Matrix add(const Matrix& a, const Matrix& b) {
    Matrix r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

inline Matrix identity2() { return {1.0, 0.0, 0.0, 1.0}; }
inline Matrix pauli(int axis) {
    const complex i(0, 1);
    if (axis == 0) return {0.0, 1.0, 1.0, 0.0};
    if (axis == 1) return {0.0, -i, i, 0.0};
    return {1.0, 0.0, 0.0, -1.0};
}
inline Matrix projector(int bit) { return bit == 0 ? Matrix{1.0, 0.0, 0.0, 0.0} : Matrix{0.0, 0.0, 0.0, 1.0}; }

// R_a(p) = exp(-i p sigma_a / 2) = cos(p/2) I - i sin(p/2) sigma_a, written out.
inline Matrix rotation(int axis, double p) {
    const double c = std::cos(p / 2), s = std::sin(p / 2);
    const complex i(0, 1);
    if (axis == 0) return {c, -i * s, -i * s, c};
    if (axis == 1) return {c, -s, s, c};
    return {std::exp(-i * (p / 2)), 0.0, 0.0, std::exp(i * (p / 2))};
}

// M_(n-1) (x) ... (x) M_0, factors[q] acting on qubit q (little-endian index).
inline Matrix product(const std::vector<Matrix>& factors) {
    Matrix r{1.0};
    for (std::size_t q = factors.size(); q-- > 0;) r = kron(r, factors[q]);
    return r;
}
inline Matrix on_qubit(std::size_t n, std::size_t q, const Matrix& m) {
    std::vector<Matrix> f(n, identity2());
    f[q] = m;
    return product(f);
}
inline Matrix controlled(std::size_t n, std::size_t control, std::size_t target, const Matrix& m) {
    std::vector<Matrix> off(n, identity2()), on(n, identity2());
    off[control] = projector(0);
    on[control] = projector(1);
    on[target] = m;
    return add(product(off), product(on));
}

inline std::vector<complex> multiply(const Matrix& m, const std::vector<complex>& v) {
    const std::size_t d = v.size();
    std::vector<complex> r(d);
    for (std::size_t i = 0; i < d; ++i) {
        complex s = 0;
        for (std::size_t j = 0; j < d; ++j) s += m[i * d + j] * v[j];
        r[i] = s;
    }
    return r;
}

// |psi(x)> = W_r S_r E W_(r-1) ... S_1 E W_0 |0...0>; angles (l n + q) B + k,
// encodings (l-1) n + q.
inline std::vector<complex> state(const Circuit& c, double x, const std::vector<double>& angles,
                                  const std::vector<double>& weights, const std::vector<double>& biases) {
    const std::size_t n = c.qubits, B = c.block.size();
    std::vector<complex> psi(std::size_t{1} << n, 0.0);
    psi[0] = 1.0;
    const auto block = [&](std::size_t l) {
        for (std::size_t q = 0; q < n; ++q)
            for (std::size_t k = 0; k < B; ++k)
                psi = multiply(on_qubit(n, q, rotation(c.block[k], angles[(l * n + q) * B + k])), psi);
    };
    block(0);
    for (std::size_t l = 1; l <= c.repetitions; ++l) {
        if (c.entangler == 1 || c.entangler == 2)
            for (std::size_t q = 0; q + 1 < n; ++q) psi = multiply(controlled(n, q, q + 1, pauli(0)), psi);
        if (c.entangler == 2) psi = multiply(controlled(n, n - 1, 0, pauli(0)), psi);
        if (c.entangler == 3)
            for (std::size_t q = 0; q + 1 < n; ++q) psi = multiply(controlled(n, q, q + 1, pauli(2)), psi);
        for (std::size_t q = 0; q < n; ++q) {
            const std::size_t j = (l - 1) * n + q;
            psi = multiply(on_qubit(n, q, rotation(c.encoding, weights[j] * x + biases[j])), psi);
        }
        block(l);
    }
    return psi;
}

// <psi|O|psi> for O = Z_t (readout 0) or Z (x) ... (x) Z (readout 1).
inline std::vector<double> expectations(const Circuit& c, const std::vector<complex>& psi) {
    std::vector<Matrix> observables;
    if (c.readout == 0)
        for (std::size_t q = 0; q < c.qubits; ++q) observables.push_back(on_qubit(c.qubits, q, pauli(2)));
    else
        observables.push_back(product(std::vector<Matrix>(c.qubits, pauli(2))));
    std::vector<double> m;
    for (const auto& o : observables) {
        const auto v = multiply(o, psi);
        complex s = 0;
        for (std::size_t k = 0; k < psi.size(); ++k) s += std::conj(psi[k]) * v[k];
        m.push_back(s.real());
    }
    return m;
}

} // namespace dense_qubit

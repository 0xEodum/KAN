#pragma once

// A vector of L independent double lanes for the CPU's sample-blocked loops
// (M5 PQC): element-wise +, -, * on fixed-size arrays, which the compiler
// vectorizes. Each lane performs exactly the IEEE operations of the scalar
// code in the same order (no reassociation, no contraction), so a lane's
// result is bitwise the scalar result.

#include <cstddef>

namespace kan::detail {

template<std::size_t L> struct alignas(sizeof(double) * L) Lanes {
    double v[L];
    Lanes() = default;
    Lanes(double value) { // broadcast
        for (std::size_t l = 0; l < L; ++l) v[l] = value;
    }
    Lanes& operator+=(const Lanes& a) {
        for (std::size_t l = 0; l < L; ++l) v[l] += a.v[l];
        return *this;
    }
};

template<std::size_t L> inline Lanes<L> operator+(const Lanes<L>& a, const Lanes<L>& b) {
    Lanes<L> r;
    for (std::size_t l = 0; l < L; ++l) r.v[l] = a.v[l] + b.v[l];
    return r;
}
template<std::size_t L> inline Lanes<L> operator-(const Lanes<L>& a, const Lanes<L>& b) {
    Lanes<L> r;
    for (std::size_t l = 0; l < L; ++l) r.v[l] = a.v[l] - b.v[l];
    return r;
}
template<std::size_t L> inline Lanes<L> operator-(const Lanes<L>& a) {
    Lanes<L> r;
    for (std::size_t l = 0; l < L; ++l) r.v[l] = -a.v[l];
    return r;
}
template<std::size_t L> inline Lanes<L> operator*(const Lanes<L>& a, const Lanes<L>& b) {
    Lanes<L> r;
    for (std::size_t l = 0; l < L; ++l) r.v[l] = a.v[l] * b.v[l];
    return r;
}
template<std::size_t L> inline Lanes<L> operator*(double a, const Lanes<L>& b) {
    Lanes<L> r;
    for (std::size_t l = 0; l < L; ++l) r.v[l] = a * b.v[l];
    return r;
}

} // namespace kan::detail

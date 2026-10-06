#pragma once

// Platform-independent random draws and elementary functions for the
// initializers (backlog M4). Everything here uses integer arithmetic, IEEE
// basic operations (+ - * /), sqrt, frexp, ldexp and nearbyint, which are
// exactly specified, so results are bitwise identical on MSVC and GCC (the
// library is compiled without FMA contraction). std::log, std::exp and
// std::normal_distribution are not: their results are implementation-defined.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kan::init {

inline constexpr std::uint64_t splitmix_gamma = 0x9E3779B97F4A7C15ull;

inline std::uint64_t splitmix_mix(std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// ln 2 split for Cody-Waite reduction (fdlibm): k * ln2_hi is exact for |k| < 2^11.
inline constexpr double ln2_hi = 0x1.62e42fee00000p-1;
inline constexpr double ln2_lo = 0x1.a39ef35793c76p-33;

// Natural logarithm of a positive finite x, about 1 ulp.
inline double portable_log(double x) noexcept {
    int e = 0;
    double m = std::frexp(x, &e); // x = m * 2^e, m in [0.5, 1)
    if (m < 0x1.6a09e667f3bcdp-1) { // sqrt(1/2)
        m *= 2;
        --e;
    }
    // log(m) = 2 atanh(t), t = (m-1)/(m+1), |t| <= 0.1716: series to t^25.
    const double t = (m - 1) / (m + 1), t2 = t * t;
    double sum = 0;
    for (int k = 25; k >= 1; k -= 2) sum = sum * t2 + 1.0 / k;
    const double de = static_cast<double>(e);
    return de * ln2_hi + (de * ln2_lo + 2 * t * sum);
}

// e^x for finite x, about 1 ulp; 0 below -745.2 and +inf above 709.79.
inline double portable_exp(double x) noexcept {
    if (x > 709.782712893384) return std::numeric_limits<double>::infinity();
    if (x < -745.2) return 0;
    const double k = std::nearbyint(x * 0x1.71547652b82fep0); // x / ln 2
    const double r = (x - k * ln2_hi) - k * ln2_lo;            // |r| <= 0.35
    double p = 1;
    for (int n = 17; n >= 1; --n) p = 1 + (r / n) * p; // Taylor series by Horner
    return std::ldexp(p, static_cast<int>(k));
}

// SplitMix64 (Steele, Lea, Flood 2014): draw j is mix(seed + (j+1) * gamma).
class Generator {
public:
    explicit Generator(std::uint64_t seed) noexcept : state_(seed) {}
    std::uint64_t next() noexcept {
        state_ += splitmix_gamma;
        return splitmix_mix(state_);
    }
    // Uniform on [-1, 1): 2u - 1 with u = (draw >> 11) * 2^-53 (exact).
    double symmetric() noexcept { return 2 * (static_cast<double>(next() >> 11) * 0x1p-53) - 1; }
    // Standard normal by the Marsaglia polar method; values come in pairs.
    double normal() noexcept {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        for (;;) {
            const double v1 = symmetric(), v2 = symmetric(), s = v1 * v1 + v2 * v2;
            if (s >= 1 || s == 0) continue;
            const double f = std::sqrt(-2 * portable_log(s) / s);
            spare_ = v2 * f;
            has_spare_ = true;
            return v1 * f;
        }
    }

private:
    std::uint64_t state_;
    double spare_ = 0;
    bool has_spare_ = false;
};

// n-point Gauss-Legendre rule on [-1, 1]: roots of P_n by bisection on a
// fixed bracket grid, weights 2 / ((1 - x^2) P_n'(x)^2). Deterministic.
struct Quadrature {
    std::vector<double> nodes, weights;
};

inline void legendre(std::size_t n, double x, double& p, double& previous) noexcept {
    p = 1;
    previous = 0;
    for (std::size_t k = 1; k <= n; ++k) {
        const double next = ((2 * double(k) - 1) * x * p - (double(k) - 1) * previous) / double(k);
        previous = p;
        p = next;
    }
}

inline Quadrature gauss_legendre(std::size_t n) {
    Quadrature q;
    // An odd grid never contains 0, the only rational root; the other roots
    // are irrational, so every root lies strictly inside one grid cell.
    const std::size_t grid = 64 * n + 1;
    double left = -1, p_left = 0, unused = 0;
    legendre(n, left, p_left, unused);
    for (std::size_t j = 1; j <= grid; ++j) {
        const double right = -1 + 2 * double(j) / double(grid);
        double p_right = 0;
        legendre(n, right, p_right, unused);
        if ((p_left < 0) != (p_right < 0)) {
            double a = left, b = right;
            const bool negative_left = p_left < 0;
            for (;;) { // bisection to adjacent doubles
                const double mid = a + (b - a) / 2;
                if (mid <= a || mid >= b) break;
                double pm = 0;
                legendre(n, mid, pm, unused);
                if ((pm < 0) == negative_left) a = mid;
                else b = mid;
            }
            const double x = a + (b - a) / 2;
            double p = 0, previous = 0;
            legendre(n, x, p, previous);
            const double derivative = double(n) * (previous - x * p) / (1 - x * x);
            q.nodes.push_back(x);
            q.weights.push_back(2 / ((1 - x * x) * derivative * derivative));
        }
        left = right;
        p_left = p_right;
    }
    return q;
}

} // namespace kan::init

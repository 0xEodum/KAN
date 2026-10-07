// Backlog C7: randomized check of rational_parameter_vjps_bounded against the
// full rational_parameter_vjps_check it lets the forward pass skip. For
// random degrees, policies, inputs and coefficients spread over the whole
// exponent range (FP64 and FP32, host build of the shared formulas), every
// sample whose forward evaluation succeeds is tested: whenever the bound
// holds, the full check must find every intermediate finite. Also counts how
// often the bound fails although the check passes (the fallback rate).
// Build (Developer cmd, repository root):
//   cl /nologo /std:c++20 /O2 /EHsc /I include /I src docs\evidence\backlog\C6-C8-bench\c7_bound_fuzz.cpp /Fe:%TEMP%\c7_bound_fuzz.exe
// Usage: c7_bound_fuzz [samples per precision and policy]
#include "detail/rational_formulas.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
using kan::DenominatorPolicy;
using namespace kan::detail;

struct Nonfinite {};
template<class Scalar> struct Throwing {
    Scalar operator()(Scalar v) const { if (!std::isfinite(v)) throw Nonfinite{}; return v; }
};
struct Counts { long long evaluated = 0, bounded = 0, violations = 0, fallback_clean = 0, fallback_reports = 0; };

// log-uniform magnitude with random sign over [2^lo, 2^hi], sometimes zero.
template<class Scalar> Scalar draw(std::mt19937_64& rng, int lo, int hi) {
    std::uniform_int_distribution<int> e(lo, hi);
    std::uniform_real_distribution<double> m(1.0, 2.0);
    if (rng() % 16 == 0) return Scalar(0);
    const double v = std::ldexp(m(rng), e(rng));
    return static_cast<Scalar>(rng() % 2 ? v : -v);
}

template<class Scalar, DenominatorPolicy Policy> Counts run(long long samples, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int top = std::is_same_v<Scalar, double> ? 1023 : 127;
    Counts c;
    for (long long s = 0; s < samples; ++s) {
        RationalScalars<Scalar> config{rng() % 17, rng() % 17, Scalar(0), Scalar(1), Scalar(std::is_same_v<Scalar, double> ? 1e-8 : 1e-6)};
        // Spread the scales so that some samples land near the overflow threshold.
        const int span = static_cast<int>(rng() % 4);
        const int xe = span == 0 ? 8 : span == 1 ? top / 4 : top / 2 + 8, ce = span == 3 ? top - 8 : top / 3;
        std::vector<Scalar> a(config.numerator_degree + 1), b(config.denominator_degree);
        for (auto& v : a) v = draw<Scalar>(rng, -ce, ce);
        for (auto& v : b) v = draw<Scalar>(rng, -ce, ce);
        const Scalar x = draw<Scalar>(rng, -xe, xe);
        const Throwing<Scalar> guard;
        RationalHornerOf<Scalar> h;
        RationalEdgeOf<Scalar> e;
        try {
            h = rational_horner<Policy>(config, x, a.data(), b.data(), guard);
            if (rational_pole<Policy>(config, h)) continue;
            e = rational_edge<Policy>(config, h, guard);
        } catch (const Nonfinite&) { continue; }
        ++c.evaluated;
        bool clean = true;
        try { rational_parameter_vjps_check<Policy>(config, h, e.value, guard); } catch (const Nonfinite&) { clean = false; }
        if (rational_parameter_vjps_bounded<Policy>(config, h, e.value)) {
            ++c.bounded;
            if (!clean) {
                ++c.violations;
                std::printf("VIOLATION m=%zu n=%zu x=%a\n", config.numerator_degree, config.denominator_degree, static_cast<double>(x));
            }
        } else {
            ++(clean ? c.fallback_clean : c.fallback_reports);
        }
    }
    return c;
}
template<class Scalar, DenominatorPolicy Policy> bool report(const char* name, long long samples, std::uint64_t seed) {
    const auto c = run<Scalar, Policy>(samples, seed);
    std::printf("%-16s evaluated %10lld  bounded %10lld  bound fails: check clean %8lld, check reports %8lld  violations %lld\n",
                name, c.evaluated, c.bounded, c.fallback_clean, c.fallback_reports, c.violations);
    return c.violations == 0;
}
} // namespace

int main(int argc, char** argv) {
    const long long samples = argc > 1 ? std::atoll(argv[1]) : 2000000;
    bool ok = true;
    ok &= report<double, DenominatorPolicy::Guarded>("f64 guarded", samples, 1);
    ok &= report<double, DenominatorPolicy::Absolute>("f64 absolute", samples, 2);
    ok &= report<double, DenominatorPolicy::Smooth>("f64 smooth", samples, 3);
    ok &= report<float, DenominatorPolicy::Guarded>("f32 guarded", samples, 4);
    ok &= report<float, DenominatorPolicy::Absolute>("f32 absolute", samples, 5);
    ok &= report<float, DenominatorPolicy::Smooth>("f32 smooth", samples, 6);
    std::printf(ok ? "no violations\n" : "VIOLATIONS FOUND\n");
    return ok ? 0 : 1;
}

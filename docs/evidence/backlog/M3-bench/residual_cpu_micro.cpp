// Backlog M3 CPU timing: Layer::forward and Layer::backward of mid-size layers,
// branch disabled (comparable with the pre-M3 library) and, when compiled
// with /DKAN_M3_RESIDUAL against the M3 headers, branch enabled.
// Prints the median time per call over `repeats` samples of `calls` calls.
// Build: build_harness.cmd <build-dir> residual_cpu_micro.cpp <exe> [<tree root>]
#include "kan/families.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

namespace {
volatile double sink = 0;

std::vector<double> pattern(std::size_t n, double scale) {
    std::vector<double> v(n);
    for (std::size_t j = 0; j < n; ++j) v[j] = scale * std::sin(0.37 * double(j) + 0.1);
    return v;
}

template<class F> double median_us(F&& f, std::size_t calls, std::size_t repeats) {
    std::vector<double> t;
    for (std::size_t r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t c = 0; c < calls; ++c) f();
        t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / double(calls));
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

void measure(const char* name, kan::Layer layer, bool branch) {
    const std::size_t batch = 256, inputs = layer.inputs(), outputs = layer.outputs();
#ifdef KAN_M3_RESIDUAL
    if (branch) layer.set_residual(kan::SiluResidual{pattern(inputs * outputs, 0.2)});
#else
    if (branch) return;
#endif
    const auto x = pattern(batch * inputs, 0.9), u = pattern(batch * outputs, 0.5);
    const double forward = median_us([&] { sink = sink + layer.forward(x, batch)[0]; }, 20, 31);
    const double backward = median_us([&] { sink = sink + layer.backward(x, batch, u).input[0]; }, 10, 31);
    std::printf("%-22s branch=%d forward %9.1f us  backward %9.1f us\n", name, int(branch), forward, backward);
}

kan::BSplineConfig spline() {
    kan::BSplineConfig c{3, {}};
    c.knots.assign(3, -1);
    for (int j = 0; j <= 8; ++j) c.knots.push_back(-1 + 0.25 * j);
    c.knots.insert(c.knots.end(), 3, 1);
    return c;
}
} // namespace

int main() {
    for (bool branch : {false, true}) {
        kan::Layer chebyshev(64, 64, kan::ChebyshevConfig{8});
        chebyshev.set_parameters(pattern(chebyshev.coefficients().size(), 0.05), pattern(64, 0.1));
        measure("chebyshev 64x64 K=8", chebyshev, branch);
        kan::Layer bspline(64, 64, spline());
        bspline.set_parameters(pattern(bspline.coefficients().size(), 0.05), pattern(64, 0.1));
        measure("bspline 64x64 G=8", bspline, branch);
        kan::RationalConfig rc;
        rc.denominator_policy = kan::DenominatorPolicy::Smooth;
        kan::Layer rational(64, 64, rc);
        kan::set_rational_parameters(rational, pattern(rational.coefficients().size(), 0.05),
                                     pattern(64 * 64 * rc.denominator_degree, 0.1), pattern(64, 0.1));
        measure("rational[3/2] 64x64", rational, branch);
    }
    return 0;
}

// Backlog R7: wall time of the legacy kan::cuda::forward/backward calls.
// Diagnostic only (shared GPU, WDDM); not part of the frozen benchmark protocol.
// Usage: legacy_timing [repeat-scale [case-index]]   CSV: case,call,median_ms,min_ms,max_ms
#include "kan/cuda.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable : 4996) // the legacy API is measured on purpose
#endif

namespace {
struct Case { const char* name; std::size_t inputs, outputs, terms, batch; int repeats; };
std::vector<double> data(std::size_t n, double scale, int seed) {
    std::vector<double> v(n);
    for (std::size_t j = 0; j < n; ++j)
        v[j] = scale * (static_cast<double>((j * 7 + seed) % 19) - 9.0) / 9.5;
    return v;
}
template<class F> void time(const Case& c, const char* call, int repeats, F&& f) {
    for (int w = 0; w < 2; ++w) f();
    std::vector<double> ms;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        f();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(ms.begin(), ms.end());
    std::cout << c.name << ',' << call << ',' << ms[ms.size() / 2] << ',' << ms.front() << ',' << ms.back() << '\n';
}
}

int main(int argc, char** argv) {
    const int scale = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    if (!kan::cuda::available()) { std::cerr << "CUDA device required\n"; return 1; }
    const Case cases[] = {{"cheb-16x8-K8-b32", 16, 8, 8, 32, 21 * scale},
                          {"cheb-128x128-K8-b4096", 128, 128, 8, 4096, 5 * scale}};
    const int only = argc > 2 ? std::atoi(argv[2]) : -1; // optional case index
    std::cout << "case,call,median_ms,min_ms,max_ms\n";
    for (int index = 0; index < 2; ++index) {
        if (only != -1 && only != index) continue;
        const auto& c = cases[index];
        kan::Layer layer(c.inputs, c.outputs, kan::ChebyshevConfig{c.terms});
        layer.set_parameters(data(layer.coefficients().size(), 0.05, 3), data(c.outputs, 0.1, 5));
        const auto x = data(c.batch * c.inputs, 0.9, 1), dy = data(c.batch * c.outputs, 0.01, 2);
        double sink = 0;
        time(c, "forward", c.repeats, [&] { sink += kan::cuda::forward(layer, x, c.batch)[0]; });
        time(c, "backward", c.repeats, [&] { sink += kan::cuda::backward(layer, x, c.batch, dy).coefficients[0]; });
        if (sink == 12345.678) std::cout << sink << '\n';
    }
    return 0;
}

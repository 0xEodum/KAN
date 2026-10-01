// CPU rational-layer micro benchmark used for R3 profiling: best-of-7 forward
// and backward of a 64x64 rational layer over 256 samples, degrees [1/1],
// [3/2], [6/4]. Build: cl /std:c++20 /O2 /EHsc /MD /I include <this> <build>\kan.lib
#include "kan/layer.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{1, 1}, {3, 2}, {6, 4}}) {
        kan::RationalConfig config{m, n, 0.1, 1.3, 1e-8};
        kan::Layer layer(64, 64, config);
        std::vector<double> a(layer.coefficients().size()), b(layer.denominators().size());
        for (std::size_t k = 0; k < a.size(); ++k) a[k] = 0.04 * std::sin(double(k + 1));
        for (std::size_t k = 0; k < b.size(); ++k) b[k] = 0.03 * std::cos(double(k + 1));
        layer.set_rational_parameters(a, b, std::vector<double>(64, 0.01));
        std::vector<double> x(64 * 256), u(64 * 256);
        for (std::size_t k = 0; k < x.size(); ++k) { x[k] = 0.8 * std::sin(0.3 * double(k)); u[k] = 0.01; }
        double best_f = 1e300, best_b = 1e300;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            auto y = layer.forward(x, 256);
            auto t1 = std::chrono::steady_clock::now();
            auto g = layer.backward(x, 256, u);
            auto t2 = std::chrono::steady_clock::now();
            best_f = std::min(best_f, std::chrono::duration<double, std::milli>(t1 - t0).count());
            best_b = std::min(best_b, std::chrono::duration<double, std::milli>(t2 - t1).count());
            if (y.empty() || g.input.empty()) return 1;
        }
        std::printf("[%zu/%zu] forward %.3f ms backward %.3f ms\n", m, n, best_f, best_b);
    }
}

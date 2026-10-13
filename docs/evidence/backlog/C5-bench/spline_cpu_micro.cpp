// Backlog C5: CPU B-spline layer forward + backward (Network) across term counts.
// Usage: spline_cpu_micro [repeats]; prints case,ms per forward+backward (median).
#include "kan/network.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 7;
    struct Case { std::size_t width, batch, degree, terms; };
    std::vector<Case> cases;
    for (std::size_t terms : {7u, 12u, 24u, 48u, 96u}) cases.push_back({64, 256, 3, terms});
    cases.push_back({64, 256, 1, 24}); cases.push_back({16, 1024, 3, 7});
    double check = 0;
    for (const auto& c : cases) {
        kan::BSplineConfig b{c.degree, std::vector<double>(c.degree + 1, -1.0)};
        const std::size_t interior = c.terms - c.degree - 1;
        for (std::size_t k = 1; k <= interior; ++k) b.knots.push_back(-1.0 + 2.0*static_cast<double>(k)/static_cast<double>(interior + 1));
        b.knots.insert(b.knots.end(), c.degree + 1, 1.0);
        std::vector<kan::Layer> layers;
        for (int l = 0; l < 2; ++l) {
            kan::Layer layer(c.width, c.width, b);
            std::vector<double> coeff(layer.coefficients().size());
            for (std::size_t k = 0; k < coeff.size(); ++k) coeff[k] = 0.3*std::sin(0.7*static_cast<double>(k + l))/static_cast<double>(c.width);
            layer.set_parameters(coeff, std::vector<double>(c.width, 0.01));
            layers.push_back(std::move(layer));
        }
        kan::Network net(std::move(layers));
        std::vector<double> x(c.batch*c.width), u(c.batch*c.width);
        for (std::size_t k = 0; k < x.size(); ++k) { x[k] = 0.95*std::sin(0.013*static_cast<double>(k)); u[k] = std::cos(0.07*static_cast<double>(k)); }
        std::vector<double> times;
        for (int r = 0; r < repeats + 1; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto y = net.forward(x, c.batch);
            const auto g = net.backward(x, c.batch, u);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r) times.push_back(ms);
            check += y[7] + std::get<kan::LayerGradients>(g.layers[0]).coefficients[3];
        }
        std::sort(times.begin(), times.end());
        std::printf("width%zu_b%zu_p%zu_K%zu,%.3f\n", c.width, c.batch, c.degree, c.terms, times[times.size()/2]);
    }
    std::printf("checksum,%.17g\n", check);
}

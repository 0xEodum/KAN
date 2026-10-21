// Backlog M7 demonstration: trainable per-edge translations and scales
// against the fixed Mexican-hat dictionary at an equal parameter count.
//
// Protocol (fixed before running): a single KAN layer 2 -> 1 (two edges) fits
// f(x1, x2) = g1(x1) + g2(x2) on 256 training points (a 16 x 16 grid of
// [-1, 1]^2) and is evaluated on 4096 held-out points (a shifted 64 x 64
// grid). Per edge both models have 3K parameters:
//   wavelet K:  WaveletEdges with K wavelets (w, tau, l each), constructor
//               placement (tau_k = -1 + (2k+1)/K, s = 1/K), zero coefficients;
//   fixed 3K:   BasisEdges MexicanHatConfig with 3K terms at the same rule
//               (centers -1 + (2k+1)/(3K), scales 1/(3K)), zero coefficients.
// Both train by full-batch gradient descent on the mean squared error for the
// same number of steps at every learning rate of a fixed grid; all runs are
// reported. The dictionary is linear in its coefficients, so its global
// least-squares optimum (normal equations, long double) is reported too: the
// best any optimizer can reach with the fixed placement, and with the same
// centers at 2, 4 and 8 times the scale (the best single dictionary scale).
// Targets: "sharp": off-grid narrow Gaussian bumps of different widths;
// "smooth": low-frequency sinusoids that a fixed grid represents well.
// Build: g++ -std=c++20 -O2 -ffp-contract=off -Iinclude wavelet_demo.cpp build-cpu/libkan.a
#include "kan/families.hpp"
#include "kan/network.hpp"
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace {
struct Data {
    std::vector<double> x, y;
    std::size_t count;
};
Data grid(std::size_t n, double shift, const std::function<double(double, double)>& f) {
    Data d{{}, {}, n * n};
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) {
            const double x1 = -1 + (2 * (static_cast<double>(a) + shift)) / static_cast<double>(n);
            const double x2 = -1 + (2 * (static_cast<double>(b) + shift)) / static_cast<double>(n);
            d.x.push_back(x1); d.x.push_back(x2); d.y.push_back(f(x1, x2));
        }
    return d;
}
double mse(const kan::Layer& layer, const Data& d) {
    const auto out = layer.forward(d.x, d.count);
    double s = 0;
    for (std::size_t j = 0; j < d.count; ++j) s += (out[j] - d.y[j]) * (out[j] - d.y[j]);
    return s / static_cast<double>(d.count);
}
// Full-batch gradient descent; stops (reports the last finite state) if a step fails.
double train(kan::Layer& layer, const Data& d, double rate, int steps) {
    for (int step = 0; step < steps; ++step) {
        const auto out = layer.forward(d.x, d.count);
        std::vector<double> u(d.count);
        for (std::size_t j = 0; j < d.count; ++j) u[j] = 2 * (out[j] - d.y[j]) / static_cast<double>(d.count);
        try {
            layer.sgd(layer.backward(d.x, d.count, u), rate);
        } catch (const std::exception&) {
            return step;
        }
    }
    return steps;
}
// `terms` hats at -1 + (2k+1)/terms with scale widen/terms.
kan::MexicanHatConfig dictionary(std::size_t terms, double widen = 1) {
    kan::MexicanHatConfig c;
    for (std::size_t k = 0; k < terms; ++k) {
        c.centers.push_back(-1.0 + static_cast<double>(2 * k + 1) / static_cast<double>(terms));
        c.scales.push_back(widen / static_cast<double>(terms));
    }
    return c;
}
// Least-squares coefficients and bias of a fixed-dictionary layer (Gaussian
// elimination with partial pivoting on the normal equations, long double).
void least_squares(kan::Layer& layer, const kan::MexicanHatConfig& c, const Data& d) {
    const std::size_t K = c.centers.size(), n = 2 * K + 1; // coefficients of both edges, then the bias
    std::vector<long double> A(n * n, 0), r(n, 0);
    const kan::BasisConfig config = c;
    for (std::size_t j = 0; j < d.count; ++j) {
        std::vector<long double> row(n);
        for (std::size_t i = 0; i < 2; ++i) {
            const auto v = kan::evaluate_basis(config, d.x[2 * j + i]).values;
            for (std::size_t k = 0; k < K; ++k) row[i * K + k] = v[k];
        }
        row[2 * K] = 1;
        for (std::size_t a = 0; a < n; ++a) {
            r[a] += row[a] * d.y[j];
            for (std::size_t b = 0; b < n; ++b) A[a * n + b] += row[a] * row[b];
        }
    }
    for (std::size_t a = 0; a < n; ++a) A[a * n + a] += 1e-14L; // the two edges share the constant direction
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t a = col + 1; a < n; ++a)
            if (std::fabs(A[a * n + col]) > std::fabs(A[pivot * n + col])) pivot = a;
        for (std::size_t b = 0; b < n; ++b) std::swap(A[col * n + b], A[pivot * n + b]);
        std::swap(r[col], r[pivot]);
        for (std::size_t a = col + 1; a < n; ++a) {
            const long double f = A[a * n + col] / A[col * n + col];
            for (std::size_t b = col; b < n; ++b) A[a * n + b] -= f * A[col * n + b];
            r[a] -= f * r[col];
        }
    }
    std::vector<long double> z(n);
    for (std::size_t a = n; a-- > 0;) {
        long double s = r[a];
        for (std::size_t b = a + 1; b < n; ++b) s -= A[a * n + b] * z[b];
        z[a] = s / A[a * n + a];
    }
    std::vector<double> coefficients(2 * K), bias{static_cast<double>(z[2 * K])};
    for (std::size_t k = 0; k < 2 * K; ++k) coefficients[k] = static_cast<double>(z[k]);
    layer.set_parameters(coefficients, bias);
}
} // namespace

int main() {
    const auto bump = [](double x, double c, double w) { return std::exp(-((x - c) / w) * ((x - c) / w)); };
    struct Target { const char* name; std::function<double(double, double)> f; };
    const std::vector<Target> targets{
        {"sharp", [&](double a, double b) { return bump(a, 0.31, 0.06) - 0.6 * bump(a, -0.47, 0.15) + 0.8 * bump(b, -0.13, 0.09); }},
        {"smooth", [](double a, double b) { return std::sin(2.0 * a) + 0.5 * std::cos(3.0 * b); }},
    };
    const std::vector<double> rates{0.3, 0.1, 0.03, 0.01};
    const int steps = 20000;
    std::printf("target,model,K,parameters_per_edge,rate,steps_completed,train_mse,heldout_mse\n");
    for (const auto& t : targets) {
        const auto train_data = grid(16, 0.5, t.f), heldout = grid(64, 0.25, t.f);
        for (std::size_t K : {1u, 2u, 3u}) {
            for (double rate : rates) {
                kan::Layer wavelet(2, 1, kan::WaveletConfig{K});
                const auto done = train(wavelet, train_data, rate, steps);
                std::printf("%s,wavelet,%zu,%zu,%g,%d,%.6e,%.6e\n", t.name, K, 3 * K, rate, static_cast<int>(done),
                            mse(wavelet, train_data), mse(wavelet, heldout));
            }
            const auto hats = dictionary(3 * K);
            for (double rate : rates) {
                kan::Layer fixed(2, 1, hats);
                const auto done = train(fixed, train_data, rate, steps);
                std::printf("%s,fixed,%zu,%zu,%g,%d,%.6e,%.6e\n", t.name, 3 * K, 3 * K, rate, static_cast<int>(done),
                            mse(fixed, train_data), mse(fixed, heldout));
            }
            // Global optimum of the fixed placement, and of wider hats (scale
            // widen/(3K)): the fixed dictionary's best single scale.
            for (double widen : {1.0, 2.0, 4.0, 8.0}) {
                const auto wide = dictionary(3 * K, widen);
                kan::Layer optimum(2, 1, wide);
                least_squares(optimum, wide, train_data);
                std::printf("%s,fixed-lstsq-x%g,%zu,%zu,-,-,%.6e,%.6e\n", t.name, widen, 3 * K, 3 * K, mse(optimum, train_data),
                            mse(optimum, heldout));
            }
        }
    }
}

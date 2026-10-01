#include "kan/network.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    kan::ChebyshevConfig basis{3};
    kan::Network network({kan::Layer(1, 1, basis)});
    std::vector<double> x(65), target(65);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(x.size() - 1);
        target[i] = 0.2 + 0.7 * x[i] - 0.4 * x[i] * x[i];
    }
    for (int epoch = 0; epoch < 400; ++epoch) {
        auto gradient = network.forward(x, x.size());
        for (std::size_t i = 0; i < gradient.size(); ++i)
            gradient[i] = 2.0 * (gradient[i] - target[i]) / static_cast<double>(gradient.size());
        network.sgd(network.backward(x, x.size(), gradient), 0.1);
    }
    // Independent holdout grid, offset from the training samples.
    std::vector<double> holdout(64);
    for (std::size_t i = 0; i < holdout.size(); ++i)
        holdout[i] = -1.0 + 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(holdout.size());
    const auto prediction = network.forward(holdout, holdout.size());
    double mse = 0.0;
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        const double truth = 0.2 + 0.7 * holdout[i] - 0.4 * holdout[i] * holdout[i];
        mse += (prediction[i] - truth) * (prediction[i] - truth);
    }
    mse /= static_cast<double>(prediction.size());
    std::cout << std::setprecision(12) << "Chebyshev KAN holdout MSE: " << mse << '\n';
    return std::isfinite(mse) && mse < 1e-12 ? 0 : 1;
}

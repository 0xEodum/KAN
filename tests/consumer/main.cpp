#include <kan/network.hpp>
#include <cmath>
#include <iostream>

int main() {
    kan::BasisConfig basis; basis.size = 3;
    kan::Layer layer(1, 1, basis);
    layer.set_parameters(std::vector<double>{0, 0.7, -0.2}, std::vector<double>{0});
    kan::Network net({layer});
    const auto result = net.forward(std::vector<double>{-1, 0, 1}, 3);
    const std::vector<double> expected{-0.9, 0.2, 0.5};
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::abs(result[i] - expected[i]) > 1e-12) return 1;
    std::cout << "Installed kan::kan consumer passed\n";
    return 0;
}

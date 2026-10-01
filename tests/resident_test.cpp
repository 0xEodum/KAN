#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include "support/test.hpp"
#include <future>
#include <limits>

namespace {
void compare(std::span<const double> actual, std::span<const double> expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) test::near(actual[i], expected[i], 4e-10);
}
kan::Network network(kan::BasisKind kind) {
    kan::BasisConfig basis{kind, 5};
    basis.alpha = 0.3; basis.beta = -0.2; basis.frequency = 1.7;
    basis.centers = {-1, -0.5, 0, 0.5, 1}; basis.width = 0.8;
    kan::Layer first(2, 3, basis), second(3, 1, {kan::BasisKind::Chebyshev, 3});
    for (auto* layer : {&first, &second}) {
        std::vector<double> c(layer->coefficients().size()), b(layer->outputs(), 0.02);
        for (std::size_t i = 0; i < c.size(); ++i) c[i] = (static_cast<double>(i % 7) - 3) / 80;
        layer->set_parameters(c, b);
    }
    return kan::Network({first, second});
}
void parity(kan::BasisKind kind) {
    auto cpu = network(kind);
    kan::cuda::ResidentNetwork gpu(cpu, 8);
    const auto allocations = gpu.workspace_allocations();
    const std::vector<double> x{-1, 1, -0.3, 0.7, 1.2, -1.2}, dy{0.2, -0.1, 0.4};
    gpu.upload_input(x, 3); gpu.upload_output_gradient(dy);
    for (int iteration = 0; iteration < 3; ++iteration) {
        gpu.forward(); compare(gpu.download_output(), cpu.forward(x, 3));
        gpu.backward();
        const auto actual = gpu.download_gradients(), expected = cpu.backward(x, 3, dy);
        compare(actual.input, expected.input);
        REQUIRE(actual.layers.size() == expected.layers.size());
        for (std::size_t j = 0; j < actual.layers.size(); ++j) {
            compare(actual.layers[j].input, expected.layers[j].input);
            compare(actual.layers[j].coefficients, expected.layers[j].coefficients);
            compare(actual.layers[j].bias, expected.layers[j].bias);
        }
        gpu.sgd(0.01); cpu.sgd(expected, 0.01);
    }
    const auto downloaded = gpu.download_parameters();
    compare(downloaded.forward(x, 3), cpu.forward(x, 3));
    REQUIRE(gpu.workspace_allocations() == allocations);
    REQUIRE(gpu.capacity() == 8); REQUIRE(gpu.batch() == 3);
}
}
TEST(resident_all_families_mixed_topology_and_repeated_gpu_sgd) {
    for (auto kind : {kan::BasisKind::Chebyshev, kan::BasisKind::Legendre, kan::BasisKind::Jacobi,
                      kan::BasisKind::Hermite, kan::BasisKind::Fourier, kan::BasisKind::GaussianRbf}) parity(kind);
}
TEST(resident_states_upload_validation_and_move) {
    kan::cuda::ResidentNetwork gpu(network(kan::BasisKind::Legendre), 4);
    test::throws<std::logic_error>([&] { gpu.forward(); });
    test::throws<std::logic_error>([&] { gpu.backward(); });
    test::throws<std::logic_error>([&] { gpu.sgd(0.1); });
    test::throws<std::logic_error>([&] { gpu.download_output(); });
    test::throws<std::invalid_argument>([&] { gpu.upload_input({}, 1); });
    test::throws<std::invalid_argument>([&] { gpu.upload_input(std::vector<double>(10), 5); });
    test::throws<std::invalid_argument>([&] { gpu.upload_input(std::vector<double>{0, std::numeric_limits<double>::infinity()}, 1); });
    gpu.upload_input(std::vector<double>{0.1, 0.2}, 1); gpu.forward();
    test::throws<std::logic_error>([&] { gpu.backward(); });
    test::throws<std::invalid_argument>([&] { gpu.upload_output_gradient({}); });
    gpu.upload_output_gradient(std::vector<double>{1}); gpu.backward();
    test::throws<std::invalid_argument>([&] { gpu.sgd(-1); });
    gpu.upload_input(std::vector<double>{0.2, 0.3}, 1);
    test::throws<std::logic_error>([&] { gpu.backward(); });
    test::throws<std::logic_error>([&] { gpu.download_gradients(); });
    kan::cuda::ResidentNetwork moved(std::move(gpu));
    test::throws<std::logic_error>([&] { gpu.capacity(); });
    REQUIRE(moved.capacity() == 4);
}
TEST(resident_zero_batch_and_independent_instances) {
    auto cpu = network(kan::BasisKind::Hermite);
    kan::cuda::ResidentNetwork gpu(cpu, 0);
    gpu.upload_input({}, 0); gpu.upload_output_gradient({}); gpu.forward(); gpu.backward();
    REQUIRE(gpu.download_output().empty());
    const auto gradients = gpu.download_gradients(); REQUIRE(gradients.input.empty());
    for (const auto& layer : gradients.layers) {
        compare(layer.coefficients, std::vector<double>(layer.coefficients.size(), 0));
        compare(layer.bias, std::vector<double>(layer.bias.size(), 0));
    }
    gpu.sgd(0.1);
    auto first = std::async(std::launch::async, [] { parity(kan::BasisKind::Jacobi); });
    auto second = std::async(std::launch::async, [] { parity(kan::BasisKind::Fourier); });
    first.get(); second.get();
}
TEST(resident_numerical_overflow_and_atomic_network_sgd) {
    const auto maximum = std::numeric_limits<double>::max();
    kan::Layer first(1, 1, {kan::BasisKind::Chebyshev, 1});
    kan::Layer second(1, 1, {kan::BasisKind::Chebyshev, 2});
    first.set_parameters(std::vector<double>{0.1}, std::vector<double>{0});
    // The nonzero linear edge sends a real gradient into the earlier layer.
    // Its finite candidate must remain hidden when the later constant overflows.
    second.set_parameters(std::vector<double>{maximum, 1}, std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({first, second}), 1);
    gpu.upload_input(std::vector<double>{0}, 1); gpu.upload_output_gradient(std::vector<double>{-1});
    gpu.forward(); gpu.backward();
    test::throws<std::overflow_error>([&] { gpu.sgd(maximum); });
    const auto unchanged = gpu.download_parameters();
    compare(unchanged.layers()[0].coefficients(), first.coefficients());
    compare(unchanged.layers()[1].coefficients(), second.coefficients());
    gpu.sgd(0.01); // failed candidate validation must retain usable gradients
    test::near(gpu.download_parameters().layers()[0].coefficients()[0], 0.11);
    kan::Layer high_degree(1, 1, {kan::BasisKind::Chebyshev, 539});
    kan::cuda::ResidentNetwork high(kan::Network({high_degree}), 1);
    high.upload_input(std::vector<double>{2}, 1);
    test::throws<std::overflow_error>([&] { high.forward(); });
    test::throws<std::logic_error>([&] { high.download_output(); });
    kan::Layer cancel(1, 1, {kan::BasisKind::Chebyshev, 2});
    cancel.set_parameters(std::vector<double>{-maximum, maximum}, std::vector<double>{0});
    kan::cuda::ResidentNetwork cancellation(kan::Network({cancel}), 1);
    cancellation.upload_input(std::vector<double>{2}, 1);
    test::throws<std::overflow_error>([&] { cancellation.forward(); });
    test::throws<std::overflow_error>([&] {
        kan::cuda::ResidentNetwork huge(network(kan::BasisKind::Legendre), std::numeric_limits<std::size_t>::max());
    });
}
TEST(resident_jacobi_endpoints_and_gaussian_extreme_tail) {
    for (double endpoint : {-1.0, 1.0}) {
        kan::BasisConfig basis{kan::BasisKind::Jacobi, 6};
        basis.alpha = std::nextafter(-1.0, 0.0); basis.beta = 0.3;
        kan::Layer layer(1, 1, basis);
        layer.set_parameters(std::vector<double>(6, 0.05), std::vector<double>{0});
        kan::cuda::ResidentNetwork gpu(kan::Network({layer}), 1);
        gpu.upload_input(std::vector<double>{endpoint}, 1); gpu.upload_output_gradient(std::vector<double>{1});
        gpu.forward(); gpu.backward();
        compare(gpu.download_output(), layer.forward(std::vector<double>{endpoint}, 1));
        compare(gpu.download_gradients().input, layer.backward(std::vector<double>{endpoint}, 1, std::vector<double>{1}).input);
    }
    kan::BasisConfig tiny{kan::BasisKind::GaussianRbf, 1};
    tiny.centers = {0}; tiny.width = 1e-300;
    kan::Layer layer(1, 1, tiny);
    layer.set_parameters(std::vector<double>{1}, std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({layer}), 1);
    const std::vector<double> input{28e-300};
    gpu.upload_input(input, 1); gpu.upload_output_gradient(std::vector<double>{1});
    gpu.forward(); gpu.backward();
    compare(gpu.download_gradients().input, layer.backward(input, 1, std::vector<double>{1}).input);
    REQUIRE(gpu.download_gradients().input[0] != 0);
}
int main() {
    if (!kan::cuda::available()) { std::cerr << "real CUDA hardware required\n"; return 1; }
    return test::run();
}

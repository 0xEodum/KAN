#include "kan/cuda.hpp"
#include "kan/resident.hpp"
#include "support/families.hpp"
#include "support/test.hpp"
#include <limits>

namespace {
kan::Layer layer(test::Family kind) {
    test::FamilyParameters basis{5};
    basis.alpha = -0.4; basis.beta = 0.9; basis.frequency = 2.3;
    basis.centers = {-0.7, -0.2, 0.1, 0.4, 0.8}; basis.width = 0.6;
    kan::Layer result(2, 2, test::basis(kind, basis));
    std::vector<double> c(result.coefficients().size());
    for (std::size_t j = 0; j < c.size(); ++j)
        c[j] = (static_cast<double>(j % 9) - 4.0) / 37.0;
    result.set_parameters(c, std::vector<double>{0.04, -0.03});
    return result;
}
double objective(kan::cuda::ResidentNetwork& gpu, const std::vector<double>& x,
                 const std::vector<double>& dy) {
    gpu.upload_input(x, 2); gpu.forward();
    const auto y = gpu.download_output();
    double loss = 0;
    for (std::size_t j = 0; j < y.size(); ++j) loss += y[j] * dy[j];
    return loss;
}
}

TEST(resident_input_and_parameter_gradients_match_independent_finite_differences) {
    for (auto kind : {test::Family::Chebyshev, test::Family::Legendre,
                      test::Family::Jacobi, test::Family::Hermite,
                      test::Family::Fourier, test::Family::GaussianRbf}) {
        const auto model = layer(kind);
        const std::vector<double> x{-0.63, 0.19, 0.72, -0.28}, dy{0.3, -0.7, 0.9, 0.2};
        kan::cuda::ResidentNetwork gpu(kan::Network({model}), 2);
        gpu.upload_input(x, 2); gpu.upload_output_gradient(dy); gpu.forward(); gpu.backward();
        const auto gradients = gpu.download_gradients();
        constexpr double h = 1e-6;
        for (std::size_t j = 0; j < x.size(); ++j) {
            auto plus = x, minus = x; plus[j] += h; minus[j] -= h;
            const auto slope = (objective(gpu, plus, dy) - objective(gpu, minus, dy)) / (2*h);
            test::near(gradients.input[j], slope, 2e-7);
        }
        for (std::size_t j : {std::size_t{0}, std::size_t{7}, std::size_t{19}}) {
            auto plus = model, minus = model;
            std::vector<double> cp(model.coefficients().begin(), model.coefficients().end()), cm = cp;
            cp[j] += h; cm[j] -= h;
            plus.set_parameters(cp, model.bias()); minus.set_parameters(cm, model.bias());
            kan::cuda::ResidentNetwork gp(kan::Network({plus}), 2), gm(kan::Network({minus}), 2);
            const auto slope = (objective(gp, x, dy) - objective(gm, x, dy)) / (2*h);
            test::near(gradients.layers[0].coefficients[j], slope, 2e-7);
        }
    }
}

TEST(resident_failed_uploads_preserve_current_results_and_gradients) {
    kan::cuda::ResidentNetwork gpu(kan::Network({layer(test::Family::Fourier)}), 2);
    gpu.upload_input(std::vector<double>{0.1, 0.2}, 1);
    gpu.upload_output_gradient(std::vector<double>{0.3, 0.4}); gpu.forward(); gpu.backward();
    const auto output = gpu.download_output(), gradient = gpu.download_gradients().input;
    test::throws<std::invalid_argument>([&] { gpu.upload_input({}, 1); });
    test::throws<std::invalid_argument>([&] { gpu.upload_output_gradient(std::vector<double>{1}); });
    test::throws<std::invalid_argument>([&] {
        gpu.upload_output_gradient(std::vector<double>{0, std::numeric_limits<double>::quiet_NaN()});
    });
    REQUIRE(gpu.download_output() == output);
    REQUIRE(gpu.download_gradients().input == gradient);
    gpu.sgd(0.01);
    test::throws<std::logic_error>([&] { gpu.download_output(); });
    test::throws<std::logic_error>([&] { gpu.download_gradients(); });
    gpu.forward(); gpu.backward(); // uploaded input/upstream survive SGD
    const auto allocations = gpu.workspace_allocations();
    for (std::size_t batch : {2u, 0u, 1u, 2u}) {
        gpu.upload_input(std::vector<double>(2*batch, 0.3), batch);
        gpu.upload_output_gradient(std::vector<double>(2*batch, 0.2));
        gpu.forward(); gpu.backward(); gpu.sgd(0.01);
        REQUIRE(gpu.workspace_allocations() == allocations);
    }
}

TEST(resident_gaussian_underflow_tail_has_relative_accuracy) {
    const kan::GaussianRbfConfig basis{{0}, 1e-300};
    kan::Layer model(1, 1, basis);
    model.set_parameters(std::vector<double>{1}, std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({model}), 1);
    gpu.upload_input(std::vector<double>{28e-300}, 1);
    gpu.upload_output_gradient(std::vector<double>{1}); gpu.forward(); gpu.backward();
    // Decimal.from_float inputs, 100 digits: -2*q*exp(-q*q)/width.
    const double expected = -1.82521578010036018009053376422967315034882158760868e-39;
    test::near(gpu.download_gradients().input[0] / expected, 1.0, 2e-11);
}

TEST(resident_move_assignment_and_moved_cpu_rejection) {
    auto cpu = kan::Network({layer(test::Family::Legendre)});
    kan::cuda::ResidentNetwork first(cpu, 2), second(cpu, 1);
    first.upload_input(std::vector<double>{0.1, 0.2}, 1); first.forward();
    const auto expected = first.download_output();
    second = std::move(first);
    REQUIRE(second.capacity() == 2); REQUIRE(second.download_output() == expected);
    test::throws<std::logic_error>([&] { first.synchronize(); });
    auto moved = std::move(cpu);
    test::throws<std::invalid_argument>([&] { kan::cuda::ResidentNetwork rejected(cpu, 1); });
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--expect-no-device") {
        if (kan::cuda::available()) return 1;
        test::throws<std::runtime_error>([] {
            kan::cuda::ResidentNetwork gpu(kan::Network({layer(test::Family::Legendre)}), 1);
        });
        std::cout << "PASS resident construction without device fails explicitly\n";
        return 0;
    }
    if (argc != 1 || !kan::cuda::available()) return 1;
    return test::run();
}

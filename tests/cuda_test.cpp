#include "kan/cuda.hpp"
#include "support/test.hpp"
#include <future>
#include <limits>

namespace {
kan::Layer make_layer(std::size_t inputs, std::size_t outputs, std::size_t terms) {
    kan::Layer layer(inputs, outputs, {kan::BasisKind::Chebyshev, terms});
    std::vector<double> coefficients(layer.coefficients().size()), bias(outputs);
    for (std::size_t j = 0; j < coefficients.size(); ++j)
        coefficients[j] = static_cast<double>(static_cast<int>(j % 13) - 6) / 31.0;
    for (std::size_t j = 0; j < outputs; ++j) bias[j] = static_cast<double>(j + 1) / 7.0;
    layer.set_parameters(coefficients, bias);
    return layer;
}
void compare(std::span<const double> actual, std::span<const double> expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t j = 0; j < actual.size(); ++j) test::near(actual[j], expected[j], 2e-11);
}
void parity(std::size_t inputs, std::size_t outputs, std::size_t terms, std::size_t batch) {
    auto layer = make_layer(inputs, outputs, terms);
    std::vector<double> input(batch * inputs), upstream(batch * outputs);
    for (std::size_t j = 0; j < input.size(); ++j)
        input[j] = (static_cast<double>(j % 19) - 9.0) / 7.0;
    for (std::size_t j = 0; j < upstream.size(); ++j)
        upstream[j] = (static_cast<double>(j % 11) - 5.0) / 9.0;
    compare(kan::cuda::forward(layer, input, batch), layer.forward(input, batch));
    const auto actual = kan::cuda::backward(layer, input, batch, upstream);
    const auto expected = layer.backward(input, batch, upstream);
    compare(actual.input, expected.input);
    compare(actual.coefficients, expected.coefficients);
    compare(actual.bias, expected.bias);
}
}

TEST(cuda_forward_and_backward_match_cpu_across_shapes) {
    parity(1, 1, 1, 1);
    parity(2, 3, 2, 5);
    parity(5, 2, 8, 37);
    parity(3, 7, 13, 129);
    parity(7, 5, 4, 513);
}

TEST(cuda_endpoint_derivatives_and_unclipped_values) {
    kan::Layer layer(1, 1, {kan::BasisKind::Chebyshev, 9});
    std::vector<double> coefficients(9, 0.0);
    coefficients[8] = 1.0;
    layer.set_parameters(coefficients, std::vector<double>{0.0});
    const std::vector<double> input{-1.0, 1.0, -1.25, 1.25};
    const auto gradient = kan::cuda::backward(layer, input, 4, std::vector<double>(4, 1.0));
    test::near(gradient.input[0], -64.0);
    test::near(gradient.input[1], 64.0);
    compare(gradient.input, layer.backward(input, 4, std::vector<double>(4, 1.0)).input);
    compare(kan::cuda::forward(layer, input, 4), layer.forward(input, 4));
}

TEST(cuda_empty_batch_has_zero_parameter_gradients) {
    auto layer = make_layer(3, 2, 7);
    REQUIRE(kan::cuda::forward(layer, {}, 0).empty());
    const auto gradient = kan::cuda::backward(layer, {}, 0, {});
    REQUIRE(gradient.input.empty());
    compare(gradient.coefficients, std::vector<double>(layer.coefficients().size(), 0.0));
    compare(gradient.bias, std::vector<double>(layer.outputs(), 0.0));
}

TEST(cuda_rejects_invalid_shapes_and_nonfinite_data) {
    auto layer = make_layer(2, 3, 4);
    test::throws<std::invalid_argument>([&] { kan::cuda::forward(layer, {}, 1); });
    test::throws<std::invalid_argument>([&] { kan::cuda::forward(layer, std::vector<double>{1.0}, 0); });
    test::throws<std::invalid_argument>([&] { kan::cuda::backward(layer, std::vector<double>(2), 1, {}); });
    test::throws<std::invalid_argument>([&] { kan::cuda::backward(layer, {}, 0, std::vector<double>{1.0}); });
    test::throws<std::invalid_argument>([&] {
        kan::cuda::forward(layer, std::vector<double>{0.0, std::numeric_limits<double>::quiet_NaN()}, 1);
    });
    test::throws<std::invalid_argument>([&] {
        kan::cuda::backward(layer, std::vector<double>(2), 1,
                            std::vector<double>{0.0, std::numeric_limits<double>::infinity(), 0.0});
    });
}

TEST(cuda_checks_dimension_multiplication_before_allocation) {
    auto layer = make_layer(2, 3, 4);
    test::throws<std::overflow_error>([&] {
        kan::cuda::forward(layer, {}, std::numeric_limits<std::size_t>::max());
    });
    test::throws<std::overflow_error>([&] {
        kan::cuda::backward(layer, {}, std::numeric_limits<std::size_t>::max(), {});
    });
}

TEST(cuda_rejects_unsupported_families) {
    const kan::Layer layer(1, 1, {kan::BasisKind::Legendre, 3});
    test::throws<std::invalid_argument>([&] { kan::cuda::forward(layer, std::vector<double>{0.0}, 1); });
    test::throws<std::invalid_argument>([&] { kan::cuda::backward(layer, std::vector<double>{0.0}, 1, std::vector<double>{1.0}); });
    test::throws<std::invalid_argument>([&] { kan::cuda::forward(layer, {}, 0); });
}

TEST(cuda_reports_nonfinite_computed_results) {
    kan::Layer layer(1, 1, {kan::BasisKind::Chebyshev, 4});
    layer.set_parameters(std::vector<double>{1.0, 1.0, 1.0, 1.0}, std::vector<double>{0.0});
    const std::vector<double> input{1e200}, upstream{1.0};
    test::throws<std::overflow_error>([&] { kan::cuda::forward(layer, input, 1); });
    test::throws<std::overflow_error>([&] { kan::cuda::backward(layer, input, 1, upstream); });
    kan::Layer linear(1, 1, {kan::BasisKind::Chebyshev, 2});
    linear.set_parameters(std::vector<double>{0.0, 1e308}, std::vector<double>{0.0});
    test::throws<std::overflow_error>([&] { kan::cuda::forward(linear, std::vector<double>{2.0}, 1); });
    test::throws<std::overflow_error>([&] { kan::cuda::backward(linear, std::vector<double>{0.0}, 1, std::vector<double>{2.0}); });
    kan::Layer zero(1, 1, {kan::BasisKind::Chebyshev, 2});
    test::throws<std::overflow_error>([&] {
        kan::cuda::backward(zero, std::vector<double>{2.0}, 1, std::vector<double>{1e308});
    });
    test::throws<std::overflow_error>([&] {
        kan::cuda::backward(zero, std::vector<double>{0.0, 0.0}, 2, std::vector<double>{1e308, 1e308});
    });
    // At x=2 these terms have finite values but high-order derivatives
    // overflow. Forward must retain the CPU evaluator's derivative checks.
    const kan::Layer high_degree(1, 1, {kan::BasisKind::Chebyshev, 539});
    test::throws<std::overflow_error>([&] { high_degree.forward(std::vector<double>{2.0}, 1); });
    test::throws<std::overflow_error>([&] { kan::cuda::forward(high_degree, std::vector<double>{2.0}, 1); });
}

TEST(cuda_repeated_and_concurrent_calls_have_independent_storage) {
    for (int repeat = 0; repeat < 3; ++repeat) parity(3, 2, 6, 41);
    auto first = std::async(std::launch::async, [] { parity(2, 3, 5, 79); });
    auto second = std::async(std::launch::async, [] { parity(5, 1, 9, 53); });
    first.get();
    second.get();
}

void cuda_no_device_failure_is_explicit() {
    const auto layer = make_layer(1, 1, 2);
    test::throws<std::runtime_error>([&] { kan::cuda::forward(layer, std::vector<double>{0.0}, 1); });
    test::throws<std::runtime_error>([&] { kan::cuda::backward(layer, std::vector<double>{0.0}, 1, std::vector<double>{1.0}); });
}

int main(int argc, char** argv) {
    const bool expect_no_device = argc == 2 && std::string(argv[1]) == "--expect-no-device";
    if (argc != 1 && !expect_no_device) {
        std::cerr << "usage: cuda_test [--expect-no-device]\n";
        return 2;
    }
    const bool has_device = kan::cuda::available();
    std::cout << "CUDA device available: " << (has_device ? "yes" : "no") << '\n';
    if (expect_no_device) {
        if (has_device) {
            std::cerr << "FAIL no-device mode requires a hidden or absent CUDA device\n";
            return 1;
        }
        try {
            cuda_no_device_failure_is_explicit();
            std::cout << "PASS cuda_no_device_failure_is_explicit\n1/1 passed\n";
            return 0;
        } catch (const std::exception& ex) {
            std::cerr << "FAIL cuda_no_device_failure_is_explicit: " << ex.what() << '\n';
            return 1;
        }
    }
    if (!has_device) {
        std::cerr << "FAIL real CUDA hardware is required for the GPU parity suite\n";
        return 1;
    }
    return test::run();
}

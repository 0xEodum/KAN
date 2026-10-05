// Legacy M1 synchronous API (kan::cuda::forward/backward), deprecated since
// backlog R7 and routed through the resident executor. This suite exercises
// it on purpose, so the deprecation warning is suppressed for the whole file.
#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "kan/cuda.hpp"
#include "kan/families.hpp"
#include "support/test.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>

namespace {
kan::Layer make_layer(std::size_t inputs, std::size_t outputs, std::size_t terms) {
    kan::Layer layer(inputs, outputs, kan::ChebyshevConfig{terms});
    std::vector<double> coefficients(layer.coefficients().size()), bias(outputs);
    for (std::size_t j = 0; j < coefficients.size(); ++j)
        coefficients[j] = static_cast<double>(static_cast<int>(j % 13) - 6) / 31.0;
    for (std::size_t j = 0; j < outputs; ++j) bias[j] = static_cast<double>(j + 1) / 7.0;
    layer.set_parameters(coefficients, bias);
    return layer;
}
// Accepted legacy/CPU tolerance since R7 (the resident contract of C2), per entry:
// |a-e| <= 1e-12|e| + 1e-13 max|e|. The floor covers entries that cancel in long
// reductions; the CPU stays the FP64 reference.
void compare(std::span<const double> actual, std::span<const double> expected) {
    REQUIRE(actual.size() == expected.size());
    double scale = 0;
    for (double value : expected) scale = std::max(scale, std::abs(value));
    for (std::size_t j = 0; j < actual.size(); ++j) {
        if (!std::isfinite(actual[j]) ||
            std::abs(actual[j] - expected[j]) > 1e-12 * std::abs(expected[j]) + 1e-13 * scale + 1e-300)
            throw std::runtime_error("entry " + std::to_string(j) + ": actual=" + std::to_string(actual[j]) +
                                     " expected=" + std::to_string(expected[j]));
    }
}
void compare(const kan::NonlinearGradients& actual, const kan::NonlinearGradients& expected) {
    REQUIRE(actual.index() == expected.index());
    if (const auto* rbf = std::get_if<kan::TrainableRbfGradients>(&expected)) {
        compare(std::get<kan::TrainableRbfGradients>(actual).centers, rbf->centers);
        compare(std::get<kan::TrainableRbfGradients>(actual).log_widths, rbf->log_widths);
    } else if (const auto* rational = std::get_if<kan::RationalGradients>(&expected)) {
        compare(std::get<kan::RationalGradients>(actual).denominators, rational->denominators);
    }
}
void compare(const kan::LayerGradients& actual, const kan::LayerGradients& expected) {
    compare(actual.input, expected.input);
    compare(actual.coefficients, expected.coefficients);
    compare(actual.bias, expected.bias);
    compare(actual.nonlinear, expected.nonlinear);
}
std::vector<double> data(std::size_t count, double scale, int seed) {
    std::vector<double> values(count);
    for (std::size_t j = 0; j < count; ++j)
        values[j] = scale * (static_cast<double>((j * 7 + static_cast<std::size_t>(seed)) % 19) - 9.0) / 9.5;
    return values;
}
void parity(const kan::Layer& layer, std::size_t batch, double input_scale = 0.9) {
    const auto input = data(batch * layer.inputs(), input_scale, 1);
    const auto upstream = data(batch * layer.outputs(), 0.5, 4);
    compare(kan::cuda::forward(layer, input, batch), layer.forward(input, batch));
    compare(kan::cuda::backward(layer, input, batch, upstream), layer.backward(input, batch, upstream));
}
void parity(std::size_t inputs, std::size_t outputs, std::size_t terms, std::size_t batch) {
    auto layer = make_layer(inputs, outputs, terms);
    std::vector<double> input(batch * inputs), upstream(batch * outputs);
    for (std::size_t j = 0; j < input.size(); ++j)
        input[j] = (static_cast<double>(j % 19) - 9.0) / 7.0;
    for (std::size_t j = 0; j < upstream.size(); ++j)
        upstream[j] = (static_cast<double>(j % 11) - 5.0) / 9.0;
    compare(kan::cuda::forward(layer, input, batch), layer.forward(input, batch));
    compare(kan::cuda::backward(layer, input, batch, upstream), layer.backward(input, batch, upstream));
}
kan::Layer with_parameters(kan::Layer layer) {
    layer.set_parameters(data(layer.coefficients().size(), 0.3, 3), data(layer.outputs(), 0.2, 5));
    return layer;
}
kan::Layer rational_layer(std::size_t inputs, std::size_t outputs, kan::DenominatorPolicy policy) {
    kan::RationalConfig config;
    config.numerator_degree = 3; config.denominator_degree = 2; config.center = 0.1; config.scale = 1.3;
    config.denominator_policy = policy;
    kan::Layer layer(inputs, outputs, config);
    const auto denominators = inputs * outputs * config.denominator_degree;
    kan::set_rational_parameters(layer, data(layer.coefficients().size(), 0.2, 3), data(denominators, 0.05, 6),
                                 data(outputs, 0.1, 5));
    return layer;
}
// Every carrier the resident executor supports, with nontrivial parameters.
std::vector<kan::Layer> every_carrier(std::size_t inputs, std::size_t outputs) {
    std::vector<kan::Layer> layers;
    layers.push_back(with_parameters(kan::Layer(inputs, outputs, kan::ChebyshevConfig{6})));
    layers.push_back(with_parameters(kan::Layer(inputs, outputs, kan::LegendreConfig{5})));
    layers.push_back(with_parameters(kan::Layer(inputs, outputs, kan::JacobiConfig{5, 0.5, -0.3})));
    layers.push_back(with_parameters(kan::Layer(inputs, outputs, kan::HermiteConfig{4})));
    layers.push_back(with_parameters(kan::Layer(inputs, outputs, kan::FourierConfig{5, 1.5})));
    layers.push_back(with_parameters(kan::Layer(inputs, outputs, kan::GaussianRbfConfig{{-0.8, -0.2, 0.3, 0.9}, 0.6})));
    layers.push_back(with_parameters(kan::Layer(inputs, outputs,
                                                kan::BSplineConfig{3, {-1, -1, -1, -1, -0.4, 0.2, 0.7, 1, 1, 1, 1}})));
    layers.push_back(with_parameters(kan::Layer(inputs, outputs, kan::MexicanHatConfig{{-0.5, 0.0, 0.6}, {0.4, 0.7, 0.5}})));
    auto rbf = with_parameters(kan::Layer(inputs, outputs, kan::TrainableRbfConfig{{-0.6, 0.1, 0.7}, {-0.5, -0.2, 0.1}}));
    kan::set_rbf_parameters(rbf, std::vector<double>{-0.55, 0.15, 0.65}, std::vector<double>{-0.4, -0.3, 0.2});
    layers.push_back(std::move(rbf));
    for (const auto policy : {kan::DenominatorPolicy::Guarded, kan::DenominatorPolicy::Absolute,
                              kan::DenominatorPolicy::Smooth})
        layers.push_back(rational_layer(inputs, outputs, policy));
    return layers;
}
std::size_t free_device_bytes() {
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    return free;
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
    kan::Layer layer(1, 1, kan::ChebyshevConfig{9});
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

// R7 contract change: the legacy API runs every carrier through the resident
// executor (M1 accepted Chebyshev only and rejected the rest).
TEST(cuda_legacy_api_supports_every_resident_carrier) {
    for (const auto& layer : every_carrier(3, 2)) {
        parity(layer, 1);
        parity(layer, 37);
    }
    for (const auto& layer : every_carrier(5, 4)) parity(layer, 129);
}

TEST(cuda_legacy_empty_batch_shapes_every_carrier) {
    for (const auto& layer : every_carrier(3, 2)) {
        REQUIRE(kan::cuda::forward(layer, {}, 0).empty());
        compare(kan::cuda::backward(layer, {}, 0, {}), layer.backward({}, 0, {}));
    }
}

TEST(cuda_legacy_reports_unsafe_rational_denominator) {
    kan::RationalConfig config;
    config.numerator_degree = 1; config.denominator_degree = 1;
    kan::Layer layer(1, 1, config);
    kan::set_rational_parameters(layer, std::vector<double>{1, -1}, std::vector<double>{-1}, std::vector<double>{0});
    const std::vector<double> pole{1.0}, upstream{1.0};
    test::throws<std::domain_error>([&] { layer.forward(pole, 1); });
    test::throws<std::domain_error>([&] { kan::cuda::forward(layer, pole, 1); });
    test::throws<std::domain_error>([&] { kan::cuda::backward(layer, pole, 1, upstream); });
}

TEST(cuda_reports_nonfinite_computed_results) {
    kan::Layer layer(1, 1, kan::ChebyshevConfig{4});
    layer.set_parameters(std::vector<double>{1.0, 1.0, 1.0, 1.0}, std::vector<double>{0.0});
    const std::vector<double> input{1e200}, upstream{1.0};
    test::throws<std::overflow_error>([&] { kan::cuda::forward(layer, input, 1); });
    test::throws<std::overflow_error>([&] { kan::cuda::backward(layer, input, 1, upstream); });
    kan::Layer linear(1, 1, kan::ChebyshevConfig{2});
    linear.set_parameters(std::vector<double>{0.0, 1e308}, std::vector<double>{0.0});
    test::throws<std::overflow_error>([&] { kan::cuda::forward(linear, std::vector<double>{2.0}, 1); });
    test::throws<std::overflow_error>([&] { kan::cuda::backward(linear, std::vector<double>{0.0}, 1, std::vector<double>{2.0}); });
    kan::Layer zero(1, 1, kan::ChebyshevConfig{2});
    test::throws<std::overflow_error>([&] {
        kan::cuda::backward(zero, std::vector<double>{2.0}, 1, std::vector<double>{1e308});
    });
    test::throws<std::overflow_error>([&] {
        kan::cuda::backward(zero, std::vector<double>{0.0, 0.0}, 2, std::vector<double>{1e308, 1e308});
    });
    // At x=2 these terms have finite values but high-order derivatives
    // overflow. Forward must retain the CPU evaluator's derivative checks.
    const kan::Layer high_degree(1, 1, kan::ChebyshevConfig{539});
    test::throws<std::overflow_error>([&] { high_degree.forward(std::vector<double>{2.0}, 1); });
    test::throws<std::overflow_error>([&] { kan::cuda::forward(high_degree, std::vector<double>{2.0}, 1); });
}

TEST(cuda_repeated_and_concurrent_calls_have_independent_storage) {
    for (int repeat = 0; repeat < 3; ++repeat) parity(3, 2, 6, 41);
    auto first = std::async(std::launch::async, [] { parity(2, 3, 5, 79); });
    auto second = std::async(std::launch::async, [] { parity(5, 1, 9, 53); });
    auto third = std::async(std::launch::async, [] { for (const auto& l : every_carrier(2, 3)) parity(l, 17); });
    first.get();
    second.get();
    third.get();
}

// Each call owns its device storage (now one resident executor) and releases
// it on return and on exceptions. A leak of the ~8 MiB per call would lose
// hundreds of MiB here; the bound tolerates other processes on a shared GPU.
TEST(cuda_legacy_calls_release_device_storage) {
    const auto layer = make_layer(64, 64, 8);
    const auto input = data(256 * 64, 0.9, 1), upstream = data(256 * 64, 0.1, 2);
    kan::Layer overflow(1, 1, kan::ChebyshevConfig{2});
    overflow.set_parameters(std::vector<double>{0.0, 1e308}, std::vector<double>{0.0});
    const auto cycle = [&] {
        for (int call = 0; call < 40; ++call) {
            (void)kan::cuda::forward(layer, input, 256);
            (void)kan::cuda::backward(layer, input, 256, upstream);
            test::throws<std::overflow_error>([&] { kan::cuda::forward(overflow, std::vector<double>{2.0}, 1); });
        }
    };
    cycle(); // first use loads the runtime and cuBLAS
    constexpr std::size_t bound = 64u << 20;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const auto before = free_device_bytes();
        cycle();
        const auto after = free_device_bytes();
        if (after + bound >= before) return;
    }
    throw std::runtime_error("legacy calls did not release device storage");
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

// Backlog C2: the resident contraction engine (Y = Phi*C^T + b and its VJPs)
// on shapes that exercise GEMM tiling and leading dimensions: inputs*terms and
// outputs not multiples of any tile, batch below capacity, several expansion
// layers sharing scratch, repeated backward after one forward, the coefficient
// L2, empty batches and the nonfinite-result checks.
#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
template<class... Parts> std::string precise(const Parts&... parts) {
    std::ostringstream out;
    out << std::setprecision(17);
    (out << ... << parts);
    return out.str();
}
// Per entry: relative error, plus a much smaller floor tied to the largest
// magnitude of the tensor, because cancellation in long reductions makes
// small entries carry absolute rather than relative error.
void compare(std::span<const double> actual, std::span<const double> expected, double tolerance = 1e-12) {
    REQUIRE(actual.size() == expected.size());
    double scale = 0;
    for (double e : expected) scale = std::max(scale, std::abs(e));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::isfinite(actual[i]));
        if (std::abs(actual[i]-expected[i]) > tolerance*std::abs(expected[i]) + 1e-1*tolerance*scale + 1e-300)
            throw std::runtime_error(precise("index ", i, " of ", actual.size(), " actual=", actual[i],
                                             " expected=", expected[i], " scale=", scale));
    }
}
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
kan::Layer layer(std::size_t in, std::size_t out, kan::BasisConfig basis, double phase) {
    kan::Layer l(in, out, std::move(basis));
    const auto scale = 1.0/std::sqrt(static_cast<double>(in*l.terms()));
    l.set_parameters(wave(l.coefficients().size(), scale, 0.731, phase), wave(out, 0.05, 1.3, phase));
    return l;
}
kan::TrainableRbfConfig trainable() {
    return {{-1, -0.6, -0.2, 0.2, 0.6, 1}, {-1, -0.9, -0.8, -0.8, -0.9, -1}};
}
// Mixed expansion layers of different I*K, so shared scratch must fit each.
kan::Network wide() {
    return kan::Network({layer(37, 23, kan::ChebyshevConfig{7}, 0.1),
                         layer(23, 41, trainable(), 0.2),
                         layer(41, 5, kan::BSplineConfig{3, {-2,-2,-2,-2,-1,0,0.5,1,2,2,2,2}}, 0.3),
                         layer(5, 3, kan::FourierConfig{5, 1.1}, 0.4)});
}
void gradients(const kan::NetworkGradients& actual, const kan::NetworkGradients& expected,
               const kan::NetworkGradients* penalty = nullptr) {
    compare(actual.input, expected.input);
    for (std::size_t j = 0; j < actual.layers.size(); ++j) {
        auto coefficients = test::grad(expected, j).coefficients;
        if (penalty)
            for (std::size_t k = 0; k < coefficients.size(); ++k) coefficients[k] += test::grad(*penalty, j).coefficients[k];
        compare(test::grad(actual, j).coefficients, coefficients);
        compare(test::grad(actual, j).bias, test::grad(expected, j).bias);
        compare(test::centers(test::grad(actual, j)), test::centers(test::grad(expected, j)));
        compare(test::log_widths(test::grad(actual, j)), test::log_widths(test::grad(expected, j)));
    }
}
} // namespace

TEST(contraction_wide_mixed_network_trains_like_cpu) {
    auto cpu = wide();
    kan::cuda::ResidentNetwork gpu(cpu, 200);
    const auto allocations = gpu.workspace_allocations();
    const std::size_t batch = 129;
    const auto x = wave(batch*37, 0.95, 0.37), dy = wave(batch*3, 0.02, 0.53, 1);
    gpu.upload_input(x, batch);
    for (int step = 0; step < 3; ++step) {
        gpu.forward(); compare(gpu.download_output(), cpu.forward(x, batch));
        gpu.upload_output_gradient(dy); gpu.backward(0.05);
        const auto expected = cpu.backward(x, batch, dy);
        const auto penalty = cpu.regularization(0.05).gradients;
        gradients(gpu.download_gradients(), expected, &penalty);
        auto total = expected;
        for (std::size_t j = 0; j < total.layers.size(); ++j)
            for (std::size_t k = 0; k < test::grad(total, j).coefficients.size(); ++k)
                test::grad(total, j).coefficients[k] += test::grad(penalty, j).coefficients[k];
        gpu.sgd(0.2); cpu.sgd(total, 0.2);
    }
    const auto trained = gpu.download_parameters();
    for (std::size_t j = 0; j < cpu.layers().size(); ++j) {
        compare(test::layer(trained, j).coefficients(), test::layer(cpu, j).coefficients());
        compare(test::layer(trained, j).bias(), test::layer(cpu, j).bias());
    }
    compare(test::trainable(test::layer(trained, 1)).centers, test::trainable(test::layer(cpu, 1)).centers);
    compare(test::trainable(test::layer(trained, 1)).log_widths, test::trainable(test::layer(cpu, 1)).log_widths);
    REQUIRE(allocations == gpu.workspace_allocations());
}

// Both execution paths of the engine on one network. The first layer has
// 80*448+80 = 35 920 > 2^15 coefficients and outputs (cuBLAS parameter VJP)
// and a forward of 400*80*448 = 14.3M > 2^23 multiply-adds at batch 400
// (cuBLAS) but 0.25M at batch 7 (warp kernel); the second layer always takes
// the small parameter-VJP kernel, over 7 batch tiles at batch 400.
TEST(contraction_engine_paths_agree_with_cpu) {
    kan::Network cpu({layer(64, 80, kan::ChebyshevConfig{7}, 0.7), layer(80, 3, kan::JacobiConfig{5, 0.5, -0.25}, 0.8)});
    kan::cuda::ResidentNetwork gpu(cpu, 400);
    for (std::size_t batch : {400u, 7u}) {
        const auto x = wave(batch*64, 0.9, 0.23), dy = wave(batch*3, 0.05, 0.61);
        gpu.upload_input(x, batch); gpu.forward(); compare(gpu.download_output(), cpu.forward(x, batch));
        gpu.upload_output_gradient(dy); gpu.backward(0.01);
        const auto penalty = cpu.regularization(0.01).gradients;
        gradients(gpu.download_gradients(), cpu.backward(x, batch, dy), &penalty);
    }
}

TEST(contraction_repeated_backward_after_one_forward) {
    auto cpu = wide();
    kan::cuda::ResidentNetwork gpu(cpu, 64);
    const std::size_t batch = 64;
    const auto x = wave(batch*37, 0.8, 0.29);
    gpu.upload_input(x, batch); gpu.forward();
    for (double phase : {0.0, 2.0, 0.0}) {
        const auto dy = wave(batch*3, 0.1, 0.41, phase);
        gpu.upload_output_gradient(dy); gpu.backward();
        gradients(gpu.download_gradients(), cpu.backward(x, batch, dy));
    }
}

TEST(contraction_single_sample_and_single_edge) {
    for (std::size_t batch : {1u, 2u}) {
        kan::Network cpu({layer(1, 1, kan::LegendreConfig{1}, 0.5), layer(1, 2, kan::HermiteConfig{3}, 0.6)});
        kan::cuda::ResidentNetwork gpu(cpu, 3);
        const auto x = wave(batch, 0.7, 1.0, 0.3), dy = wave(batch*2, 1.0, 0.9);
        gpu.upload_input(x, batch); gpu.upload_output_gradient(dy); gpu.forward(); gpu.backward(0.5);
        compare(gpu.download_output(), cpu.forward(x, batch));
        const auto penalty = cpu.regularization(0.5).gradients;
        gradients(gpu.download_gradients(), cpu.backward(x, batch, dy), &penalty);
    }
}

TEST(contraction_empty_batch_gives_penalty_only) {
    auto cpu = wide();
    kan::cuda::ResidentNetwork gpu(cpu, 16);
    gpu.upload_input({}, 0); gpu.upload_output_gradient({}); gpu.forward();
    // lambda != 0 scales a copy of C; lambda == 0 clears stale gradients.
    for (double lambda : {0.25, 0.0}) {
        gpu.backward(lambda);
        const auto g = gpu.download_gradients(), e = cpu.regularization(lambda).gradients;
        REQUIRE(g.input.empty());
        for (std::size_t j = 0; j < g.layers.size(); ++j) {
            compare(test::grad(g, j).coefficients, test::grad(e, j).coefficients);
            compare(test::grad(g, j).bias, test::grad(e, j).bias);
            compare(test::centers(test::grad(g, j)), test::centers(test::grad(e, j)));
            compare(test::log_widths(test::grad(g, j)), test::log_widths(test::grad(e, j)));
        }
    }
    // A smaller batch after a larger one must not read stale rows.
    const auto x = wave(5*37, 0.6, 0.77), dy = wave(5*3, 0.3, 0.19);
    gpu.upload_input(x, 5); gpu.upload_output_gradient(dy); gpu.forward(); gpu.backward();
    compare(gpu.download_output(), cpu.forward(x, 5));
    gradients(gpu.download_gradients(), cpu.backward(x, 5, dy));
    gpu.upload_input(std::span<const double>(x).first(2*37), 2); gpu.forward();
    compare(gpu.download_output(), cpu.forward(std::span<const double>(x).first(2*37), 2));
}

TEST(contraction_nonfinite_results_are_reported) {
    const double maximum = std::numeric_limits<double>::max();
    // Output overflow: two edges whose finite contributions sum past the range.
    kan::Layer sum(2, 1, kan::ChebyshevConfig{1});
    sum.set_parameters(std::vector<double>{maximum, maximum}, std::vector<double>{0});
    kan::cuda::ResidentNetwork forward(kan::Network({sum}), 1);
    forward.upload_input(std::vector<double>{0.5, -0.5}, 1);
    test::throws<std::overflow_error>([&] { forward.forward(); });
    test::throws<std::logic_error>([&] { forward.download_output(); });
    // Coefficient-VJP overflow over the batch; the forward itself is finite.
    kan::Layer grad(1, 1, kan::ChebyshevConfig{2});
    grad.set_parameters(std::vector<double>{0, 1e-300}, std::vector<double>{0});
    kan::cuda::ResidentNetwork backward(kan::Network({grad}), 2);
    backward.upload_input(std::vector<double>{1, 1}, 2); backward.forward();
    backward.upload_output_gradient(std::vector<double>{maximum, maximum});
    test::throws<std::overflow_error>([&] { backward.backward(); });
    test::throws<std::logic_error>([&] { backward.download_gradients(); });
    // The overflowed coefficient VJP is stale device data; a lambda = 0
    // backward must overwrite it without reading it.
    backward.upload_output_gradient(std::vector<double>{0.5, 0.25}); backward.backward();
    const auto recovered = backward.download_gradients();
    compare(test::grad(recovered, 0).coefficients, std::vector<double>{0.75, 0.75});
    compare(test::grad(recovered, 0).bias, std::vector<double>{0.75});
    // Input-VJP overflow: u*C*Phi' leaves the range while Phi*u stays finite.
    kan::Layer input(1, 2, kan::ChebyshevConfig{2});
    input.set_parameters(std::vector<double>{0, 1e300, 0, 1e300}, std::vector<double>{0, 0});
    kan::cuda::ResidentNetwork chain(kan::Network({input}), 1);
    chain.upload_input(std::vector<double>{1e-300}, 1); chain.forward();
    chain.upload_output_gradient(std::vector<double>{maximum, maximum});
    test::throws<std::overflow_error>([&] { chain.backward(); });
    // The network stays usable after a reported failure.
    chain.upload_output_gradient(std::vector<double>{1, 1}); chain.backward();
    test::near(chain.download_gradients().input[0], 2e300);
}

int main() {
    if (!kan::cuda::available()) { std::cerr << "real CUDA hardware required\n"; return 1; }
    return test::run();
}

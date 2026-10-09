// Backlog M3 phase 1: until the resident executor implements the SiLU residual
// branch (phase 2), construction and upload_parameters reject a network with
// the branch instead of silently dropping it. Replace with parity tests in phase 2.
#include "kan/resident.hpp"
#include "support/test.hpp"
#include <cstdio>
#include <string>

namespace {
kan::Network network(bool residual) {
    kan::Layer first(2, 3, kan::ChebyshevConfig{3}), second(3, 1, kan::ChebyshevConfig{3});
    first.set_parameters(std::vector<double>(18, 0.1), std::vector<double>{0.1, 0.2, 0.3});
    if (residual) second.set_residual(kan::SiluResidual{{0.5, -0.25, 0.125}});
    return kan::Network({first, second});
}

template<class Fn> void rejects_branch(Fn fn) {
    try {
        fn();
    } catch (const std::invalid_argument& error) {
        REQUIRE(std::string(error.what()).find("residual branch") != std::string::npos);
        return;
    }
    throw std::runtime_error("the residual branch was not rejected");
}
} // namespace

TEST(construction_rejects_the_residual_branch_in_every_precision) {
    if (!kan::cuda::available()) { std::puts("no CUDA device: skipped"); return; }
    for (auto precision : {kan::cuda::Precision::Float64, kan::cuda::Precision::Float32, kan::cuda::Precision::TensorFloat32})
        rejects_branch([&] { kan::cuda::ResidentNetwork gpu(network(true), 4, precision); });
}

TEST(upload_rejects_the_residual_branch_and_keeps_the_executor) {
    if (!kan::cuda::available()) { std::puts("no CUDA device: skipped"); return; }
    const auto plain = network(false);
    kan::cuda::ResidentNetwork gpu(plain, 4);
    const std::vector<double> x{0.1, -0.2, 0.3, 0.4};
    gpu.upload_input(x, 2);
    gpu.forward();
    const auto before = gpu.download_output();
    rejects_branch([&] { gpu.upload_parameters(network(true)); });
    gpu.forward();
    REQUIRE(gpu.download_output() == before);
    const auto expected = plain.forward(x, 2);
    for (std::size_t j = 0; j < expected.size(); ++j) test::near(before[j], expected[j], 1e-12);
}

int main() { return test::run(); }

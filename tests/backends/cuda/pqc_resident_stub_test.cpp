// M5 phase 1 (D13): until the resident PQC phase (2a), constructing a resident
// executor for a network with a PQC layer, in any precision, raises
// std::invalid_argument; uploading a PQC network into an executor built for
// another carrier is a structure mismatch. Executors of other networks are
// unaffected.
#include "kan/resident.hpp"
#include "support/test.hpp"
#include <cmath>
#include <cstdio>
#include <string>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

kan::Network pqc_network() {
    kan::PqcConfig config;
    config.qubits = 2;
    config.entangler = kan::Entangler::CnotChain;
    return kan::Network({kan::Layer(2, 3, kan::ChebyshevConfig{3}), kan::Layer(3, 1, config)});
}
} // namespace

TEST(resident_construction_rejects_pqc_layers) {
    if (!kan::cuda::available()) {
        std::printf("no CUDA device: skipped\n");
        return;
    }
    for (auto precision : {Precision::Float64, Precision::Float32, Precision::TensorFloat32}) {
        try {
            ResidentNetwork resident(pqc_network(), 8, precision);
            throw std::runtime_error("a PQC layer was accepted");
        } catch (const std::invalid_argument& e) {
            REQUIRE(std::string(e.what()) == "resident execution of PQC layers is not available");
        }
    }
}

TEST(other_networks_are_unaffected_and_uploads_reject_pqc_layers) {
    if (!kan::cuda::available()) return;
    kan::Network chebyshev({kan::Layer(2, 3, kan::ChebyshevConfig{3}), kan::Layer(3, 1, kan::ChebyshevConfig{2})});
    ResidentNetwork resident(chebyshev, 8);
    const std::vector<double> x{0.1, 0.2, -0.3, 0.4};
    const auto expected = chebyshev.forward(x, 2);
    const auto run = [&] {
        resident.upload_input(x, 2);
        resident.forward();
        return resident.download_output();
    };
    const auto before = run();
    for (std::size_t j = 0; j < expected.size(); ++j)
        REQUIRE(std::abs(before[j] - expected[j]) <= 1e-12 * (1 + std::abs(expected[j])));
    test::throws<std::invalid_argument>([&] { resident.upload_parameters(pqc_network()); });
    REQUIRE(run() == before);
}

int main() { return test::run(); }

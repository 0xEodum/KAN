// Backlog C3: the FP32 resident executor no longer keeps the basis-derivative
// rows Phi' (batch*inputs*terms per fixed-basis layer) between forward and
// backward; the backward pass recomputes them from the layer input. Results
// are covered by the FP32 parity suites (resident_precision, every family);
// this suite checks the reservation.
#include "kan/resident.hpp"
#include "support/test.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

std::size_t free_device_bytes() {
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    return free;
}

// Device bytes taken by constructing the executor (the arena; the executor's
// cuBLAS handle is created by an identical warm-up executor first).
std::size_t construction_bytes(const kan::Network& network, std::size_t capacity, Precision precision) {
    ResidentNetwork warm(network, capacity, precision);
    const auto before = free_device_bytes();
    ResidentNetwork measured(network, capacity, precision);
    const auto after = free_device_bytes();
    return before > after ? before - after : 0;
}

std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
} // namespace

kan::Network chebyshev(std::size_t width, std::size_t terms) {
    std::vector<kan::NetworkLayer> layers;
    layers.emplace_back(kan::Layer(width, width, kan::ChebyshevConfig{terms}));
    return kan::Network(std::move(layers));
}

// One 256->256 Chebyshev layer at capacity 8192, 7 and 14 terms. Every term
// adds one batch*inputs column (8.4 MB in FP32) to each per-term row tensor:
// Phi and the shared W = U*C scratch, plus Phi' when it is stored. The
// parameters grow by outputs*inputs per term and region (32x smaller).
TEST(float32_fixed_basis_layer_reserves_no_derivative_rows) {
    constexpr std::size_t capacity = 8192, width = 256, terms = 7;
    const auto narrow = construction_bytes(chebyshev(width, terms), capacity, Precision::Float32);
    const auto wide = construction_bytes(chebyshev(width, 2*terms), capacity, Precision::Float32);
    const auto column = capacity*width*terms*sizeof(float);
    const auto growth = wide > narrow ? wide - narrow : 0;
    if (growth >= 5*column/2)
        throw std::runtime_error("7 more terms took " + std::to_string(growth) + " bytes, " +
                                 std::to_string(growth/static_cast<double>(column)) + " row tensors");
    REQUIRE(growth >= 2*column);
}

int main() {
    if (!kan::cuda::available()) { std::cout << "SKIP no CUDA device\n"; return 0; }
    return test::run();
}

// Backlog R7/R8 compile probe, built with deprecation warnings as errors.
// cuda_deprecation_available (no macro) must compile: available() and the
// resident executor are not deprecated. cuda_deprecation_resident_only
// (KAN_PROBE_RESIDENT_ONLY) must compile with kan/resident.hpp as the only
// CUDA header: the device query is declared in kan/cuda_runtime.hpp, which the
// resident header includes (R8). cuda_deprecation_legacy (KAN_PROBE_LEGACY)
// must fail with the deprecation message pointing to kan::cuda::ResidentNetwork.
#ifndef KAN_PROBE_RESIDENT_ONLY
#include "kan/cuda.hpp"
#endif
#include "kan/resident.hpp"

int main() {
    const kan::Layer layer(1, 1, kan::ChebyshevConfig{2});
    if (!kan::cuda::available()) return 0;
    kan::cuda::ResidentNetwork resident(kan::Network({layer}), 1);
#ifdef KAN_PROBE_LEGACY
    (void)kan::cuda::forward(layer, std::vector<double>{0.0}, 1);
    (void)kan::cuda::backward(layer, std::vector<double>{0.0}, 1, std::vector<double>{1.0});
#endif
    return static_cast<int>(resident.capacity()) - 1;
}

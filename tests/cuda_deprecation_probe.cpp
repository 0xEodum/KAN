// Backlog R7 compile probe, built with deprecation warnings as errors.
// cuda_deprecation_available (no KAN_PROBE_LEGACY) must compile: available()
// and the resident executor are not deprecated. cuda_deprecation_legacy must
// fail with the deprecation message pointing to kan::cuda::ResidentNetwork.
#include "kan/cuda.hpp"
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

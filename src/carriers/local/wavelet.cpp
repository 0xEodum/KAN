// Per-edge trainable Mexican-hat wavelets (backlog M7): configuration and
// log-scale validation and the single-wavelet evaluation of kan/wavelet.hpp,
// from the shared host/device formulas (carriers/local/local_formulas.hpp).
#include "kan/wavelet.hpp"
#include "backends/cpu/basis_host.hpp"
#include "carriers/local/local_formulas.hpp"
#include <cmath>
#include <stdexcept>

namespace kan {

void validate_wavelet(const WaveletConfig& config) {
    if (config.size == 0) throw std::invalid_argument("wavelet edges need at least one wavelet per edge");
}

bool valid_log_scale(double log_scale) noexcept {
    if (!std::isfinite(log_scale)) return false;
    const double scale = std::exp(log_scale), inverse = std::exp(-log_scale);
    return std::isfinite(scale) && scale > 0 && std::isfinite(inverse) && inverse > 0;
}

WaveletEvaluation evaluate_wavelet(double x, double translation, double log_scale) {
    if (!std::isfinite(x) || !std::isfinite(translation) || !valid_log_scale(log_scale))
        throw std::invalid_argument("wavelet input, translation and log scale must be finite and valid");
    const auto t =
        detail::wavelet_terms<true>(x, translation, detail::wavelet_scales(log_scale), detail::FiniteBasisGuard{});
    return {t.value, t.input_derivative, -t.input_derivative, t.log_scale_derivative};
}

} // namespace kan

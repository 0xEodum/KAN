// Input maps (backlog M1): validation, forward, VJPs and SGD per map kind.
// InputMap dispatches on the kind once per call; the loops below are compiled
// per kind and share their formulas with the resident kernels.
#include "kan/input_map.hpp"
#include "detail/checks.hpp"
#include "detail/input_map_formulas.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kan {
namespace {
using detail::checked_size;
using detail::require_finite;
using detail::result_finite;

void validate(const AffineMap& map, std::size_t features) {
    if (map.scale.size() != features || map.shift.size() != features)
        throw std::invalid_argument("affine map scale and shift need one value per feature");
    require_finite(map.scale);
    require_finite(map.shift);
}
void validate(const TanhMap& map, std::size_t) {
    if (!std::isfinite(map.scale) || map.scale <= 0)
        throw std::invalid_argument("tanh map scale must be finite and positive");
}
void validate(const LayerNormMap& map, std::size_t features) {
    if (!std::isfinite(map.epsilon) || map.epsilon <= 0)
        throw std::invalid_argument("layer norm epsilon must be finite and positive");
    const bool none = map.gain.empty() && map.bias.empty();
    if (!none && (map.gain.size() != features || map.bias.size() != features))
        throw std::invalid_argument("layer norm gain and bias must both be empty or have one value per feature");
    require_finite(map.gain);
    require_finite(map.bias);
}

// Trainable parameter count of each kind (gain and bias each).
std::size_t trainable(const AffineMap&) noexcept { return 0; }
std::size_t trainable(const TanhMap&) noexcept { return 0; }
std::size_t trainable(const LayerNormMap& map) noexcept { return map.gain.size(); }

void forward(const AffineMap& map, std::size_t n, std::span<const double> x, std::size_t batch, std::span<double> y) {
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t i = 0; i < n; ++i) y[b * n + i] = detail::affine_value(map.scale[i], map.shift[i], x[b * n + i]);
}
void forward(const TanhMap& map, std::size_t n, std::span<const double> x, std::size_t batch, std::span<double> y) {
    for (std::size_t k = 0; k < batch * n; ++k) y[k] = detail::tanh_value(map.scale, x[k]);
}

struct RowMoments {
    double mean, rstd;
};
// Two-pass population moments of one row; nonfinite moments overflow.
RowMoments moments(const double* row, std::size_t n, double epsilon) {
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += row[i];
    const double inverse = 1.0 / static_cast<double>(n);
    const double mean = detail::layer_norm_mean(sum, inverse);
    double squares = 0;
    for (std::size_t i = 0; i < n; ++i) squares += (row[i] - mean) * (row[i] - mean);
    const double variance = detail::layer_norm_mean(squares, inverse);
    if (!std::isfinite(mean) || !std::isfinite(variance)) throw std::overflow_error("nonfinite numerical result");
    return {mean, detail::layer_norm_rstd(variance, epsilon)};
}

void forward(const LayerNormMap& map, std::size_t n, std::span<const double> x, std::size_t batch, std::span<double> y) {
    const bool affine = !map.gain.empty();
    for (std::size_t b = 0; b < batch; ++b) {
        const double* row = x.data() + b * n;
        const auto m = moments(row, n, map.epsilon);
        for (std::size_t i = 0; i < n; ++i) {
            const double xhat = detail::layer_norm_normalized(row[i], m.mean, m.rstd);
            y[b * n + i] = affine ? map.gain[i] * xhat + map.bias[i] : xhat;
        }
    }
}

void backward(const AffineMap& map, std::size_t n, std::span<const double>, std::size_t batch,
              std::span<const double> u, InputMapGradients& g) {
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t i = 0; i < n; ++i) g.input[b * n + i] = map.scale[i] * u[b * n + i];
}
void backward(const TanhMap& map, std::size_t n, std::span<const double> x, std::size_t batch,
              std::span<const double> u, InputMapGradients& g) {
    for (std::size_t k = 0; k < batch * n; ++k)
        g.input[k] = u[k] * detail::tanh_derivative(map.scale, detail::tanh_value(map.scale, x[k]));
}
void backward(const LayerNormMap& map, std::size_t n, std::span<const double> x, std::size_t batch,
              std::span<const double> u, InputMapGradients& g) {
    const bool affine = !map.gain.empty();
    const double inverse = 1.0 / static_cast<double>(n);
    for (std::size_t b = 0; b < batch; ++b) {
        const double* row = x.data() + b * n;
        const double* up = u.data() + b * n;
        const auto m = moments(row, n, map.epsilon);
        double sum_w = 0, sum_wx = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double w = affine ? up[i] * map.gain[i] : up[i];
            sum_w += w;
            sum_wx += w * detail::layer_norm_normalized(row[i], m.mean, m.rstd);
        }
        const double mean_w = detail::layer_norm_mean(sum_w, inverse);
        const double mean_wx = detail::layer_norm_mean(sum_wx, inverse);
        for (std::size_t i = 0; i < n; ++i) {
            const double xhat = detail::layer_norm_normalized(row[i], m.mean, m.rstd);
            const double w = affine ? up[i] * map.gain[i] : up[i];
            g.input[b * n + i] = detail::layer_norm_input_vjp(m.rstd, w, mean_w, xhat, mean_wx);
            if (affine) {
                g.gain[i] += up[i] * xhat;
                g.bias[i] += up[i];
            }
        }
    }
}

// candidate -= rate * gradient for the trainable parameters of each kind.
void update(AffineMap&, const InputMapGradients&, double) {}
void update(TanhMap&, const InputMapGradients&, double) {}
void update(LayerNormMap& map, const InputMapGradients& g, double rate) {
    for (std::size_t i = 0; i < map.gain.size(); ++i) {
        map.gain[i] -= rate * g.gain[i];
        map.bias[i] -= rate * g.bias[i];
    }
    result_finite(map.gain);
    result_finite(map.bias);
}
} // namespace

InputMap::InputMap(std::size_t features, InputMapKind map) : features_(features), map_(std::move(map)) {
    if (features == 0) throw std::invalid_argument("input map features must be positive");
    std::visit([&](const auto& m) { validate(m, features_); }, map_);
}

InputMap::InputMap(InputMap&& other) noexcept
    : features_(std::exchange(other.features_, 0)), map_(std::move(other.map_)) {}

InputMap& InputMap::operator=(InputMap&& other) noexcept {
    if (this != &other) {
        features_ = std::exchange(other.features_, 0);
        map_ = std::move(other.map_);
    }
    return *this;
}

void InputMap::validate_state() const {
    if (features_ == 0) throw std::invalid_argument("input map is uninitialized or moved from");
    std::visit([&](const auto& m) { validate(m, features_); }, map_);
}

void InputMap::set_map(InputMapKind map) {
    validate_state();
    std::visit([&](const auto& m) { validate(m, features_); }, map);
    map_ = std::move(map);
}

std::vector<double> InputMap::forward(std::span<const double> input, std::size_t batch) const {
    validate_state();
    const auto size = checked_size(batch, features_);
    if (input.size() != size) throw std::invalid_argument("input shape mismatch");
    require_finite(input);
    std::vector<double> output(size);
    std::visit([&](const auto& m) { kan::forward(m, features_, input, batch, output); }, map_);
    result_finite(output);
    return output;
}

InputMapGradients InputMap::backward(std::span<const double> input, std::size_t batch,
                                     std::span<const double> output_gradient) const {
    validate_state();
    const auto size = checked_size(batch, features_);
    if (input.size() != size || output_gradient.size() != size)
        throw std::invalid_argument("backward shape mismatch");
    require_finite(input);
    require_finite(output_gradient);
    InputMapGradients gradient;
    gradient.input.assign(size, 0.0);
    std::visit([&](const auto& m) {
        gradient.gain.assign(trainable(m), 0.0);
        gradient.bias.assign(trainable(m), 0.0);
        kan::backward(m, features_, input, batch, output_gradient, gradient);
    }, map_);
    result_finite(gradient.input);
    result_finite(gradient.gain);
    result_finite(gradient.bias);
    return gradient;
}

void InputMap::sgd(const InputMapGradients& gradients, double learning_rate) {
    validate_state();
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
        throw std::invalid_argument("learning rate must be finite and positive");
    auto next = std::visit([&](const auto& m) -> InputMapKind {
        if (gradients.gain.size() != trainable(m) || gradients.bias.size() != trainable(m))
            throw std::invalid_argument("parameter gradient shape mismatch");
        require_finite(gradients.gain);
        require_finite(gradients.bias);
        auto candidate = m;
        update(candidate, gradients, learning_rate);
        return candidate;
    }, map_);
    map_ = std::move(next);
}

namespace {
struct Samples {
    std::span<const double> data;
    std::size_t batch, features;
    double at(std::size_t b, std::size_t i) const { return data[b * features + i]; }
};
Samples checked_samples(std::span<const double> samples, std::size_t batch, std::size_t features) {
    if (batch == 0 || features == 0) throw std::invalid_argument("samples need a positive batch and feature count");
    if (samples.size() != checked_size(batch, features)) throw std::invalid_argument("sample shape mismatch");
    require_finite(samples);
    return {samples, batch, features};
}
} // namespace

AffineMap affine_from_range(std::span<const double> samples, std::size_t batch, std::size_t features,
                            double lower, double upper) {
    const auto s = checked_samples(samples, batch, features);
    if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper))
        throw std::invalid_argument("target interval must be finite with lower < upper");
    AffineMap map{std::vector<double>(features), std::vector<double>(features)};
    for (std::size_t i = 0; i < features; ++i) {
        double low = s.at(0, i), high = low;
        for (std::size_t b = 1; b < batch; ++b) {
            low = std::min(low, s.at(b, i));
            high = std::max(high, s.at(b, i));
        }
        if (high > low) {
            map.scale[i] = (upper - lower) / (high - low);
            map.shift[i] = lower - map.scale[i] * low;
        } else {
            map.scale[i] = 1.0;
            map.shift[i] = (0.5 * lower + 0.5 * upper) - low;
        }
    }
    result_finite(map.scale);
    result_finite(map.shift);
    return map;
}

AffineMap affine_from_moments(std::span<const double> samples, std::size_t batch, std::size_t features) {
    const auto s = checked_samples(samples, batch, features);
    AffineMap map{std::vector<double>(features), std::vector<double>(features)};
    const double count = static_cast<double>(batch);
    for (std::size_t i = 0; i < features; ++i) {
        double sum = 0;
        for (std::size_t b = 0; b < batch; ++b) sum += s.at(b, i);
        const double mean = sum / count;
        double squares = 0;
        for (std::size_t b = 0; b < batch; ++b) squares += (s.at(b, i) - mean) * (s.at(b, i) - mean);
        const double deviation = std::sqrt(squares / count);
        if (!std::isfinite(mean) || !std::isfinite(deviation)) throw std::overflow_error("nonfinite numerical result");
        map.scale[i] = deviation > 0 ? 1.0 / deviation : 1.0;
        map.shift[i] = -mean * map.scale[i];
    }
    result_finite(map.scale);
    result_finite(map.shift);
    return map;
}

} // namespace kan

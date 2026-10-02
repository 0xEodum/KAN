#pragma once

#include <cstddef>
#include <span>
#include <variant>
#include <vector>

namespace kan {

// Explicit typed input maps (backlog M1). An input map is a network layer
// kind of shape I -> I placed before a KAN layer, so that inputs reach the
// basis domain: polynomials grow like (2|x|)^n outside [-1, 1] and localized
// bases are zero, with zero gradient, outside their support. Nothing is
// normalized implicitly; a map is a declared part of the model.

// y[b,i] = scale[i] * x[b,i] + shift[i]. Fixed: scale and shift are not
// trained (training could move inputs out of the domain again). Build one
// from data with affine_from_range or affine_from_moments.
struct AffineMap {
    std::vector<double> scale; // (features), finite
    std::vector<double> shift; // (features), finite
    bool operator==(const AffineMap&) const = default;
};

// y = tanh(scale * x) in (-1, 1). Fixed, scale finite and positive.
struct TanhMap {
    double scale = 1.0;
    bool operator==(const TanhMap&) const = default;
};

// Per-sample normalization over the features:
// xhat = (x - mean) / sqrt(var + epsilon) with the population variance, and
// y = gain * xhat + bias. Epsilon is fixed, finite and positive. Gain and
// bias are either both empty (y = xhat) or both of length features and then
// trainable.
struct LayerNormMap {
    double epsilon = 1e-5;
    std::vector<double> gain;
    std::vector<double> bias;
    bool operator==(const LayerNormMap&) const = default;
};

using InputMapKind = std::variant<AffineMap, TanhMap, LayerNormMap>;

// VJPs of an input map, summed over the batch for the parameters. gain/bias
// have the length of the map's trainable gain/bias (empty for fixed maps).
struct InputMapGradients {
    std::vector<double> input; // (batch, features)
    std::vector<double> gain;
    std::vector<double> bias;
    bool operator==(const InputMapGradients&) const = default;
};

// A network layer kind of shape features -> features holding one map.
// Every operation dispatches on the map kind once per call.
class InputMap {
public:
    InputMap(std::size_t features, InputMapKind map);
    InputMap(const InputMap&) = default;
    InputMap& operator=(const InputMap&) = default;
    // A moved-from map remains assignable; its operations raise invalid_argument.
    InputMap(InputMap&& other) noexcept;
    InputMap& operator=(InputMap&& other) noexcept;
    std::size_t features() const noexcept { return features_; }
    std::size_t inputs() const noexcept { return features_; }
    std::size_t outputs() const noexcept { return features_; }
    const InputMapKind& map() const noexcept { return map_; }
    // Validates the map for this layer's features and replaces it atomically.
    void set_map(InputMapKind map);
    std::vector<double> forward(std::span<const double> input, std::size_t batch) const;
    InputMapGradients backward(std::span<const double> input, std::size_t batch,
                               std::span<const double> output_gradient) const;
    // Updates the trainable parameters (LayerNorm gain and bias); validates the
    // gradient shapes and the finite candidates before committing.
    void sgd(const InputMapGradients& gradients, double learning_rate);

private:
    friend class Network;
    void validate_state() const;
    std::size_t features_;
    InputMapKind map_;
};

// Fixed affine map sending each feature's sample range [min, max] onto
// [lower, upper] (up to rounding: choose a slightly narrower interval when the
// endpoints must stay inside a closed domain). A constant feature is shifted
// to (lower + upper) / 2 with scale 1. Samples are (batch, features).
AffineMap affine_from_range(std::span<const double> samples, std::size_t batch, std::size_t features,
                            double lower = -1.0, double upper = 1.0);
// Fixed standardizing map: (x - mean) / std with the population standard
// deviation per feature; a constant feature gets scale 1 and is centered.
AffineMap affine_from_moments(std::span<const double> samples, std::size_t batch, std::size_t features);

} // namespace kan

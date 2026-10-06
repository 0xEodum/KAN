// Explicit typed initializers (backlog M4). Each initializer builds a complete
// replacement carrier and bias and commits it through Layer::set_carrier, so a
// failure leaves the layer unchanged. Random draws and every value derived
// from them use only portable arithmetic (init/portable_math.hpp).
#include "kan/initializers.hpp"
#include "init/moments.hpp"
#include "init/portable_math.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace kan {
namespace {
using init::Generator;

struct Common {
    Distribution distribution;
    std::uint64_t seed;
    DenominatorInit denominators;
};

void require_positive(double value, const char* message) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(message);
}

Common validate(const Initializer& initializer) {
    return std::visit([](const auto& i) {
        if constexpr (std::is_same_v<std::decay_t<decltype(i)>, VarianceScaling>)
            require_positive(i.gain, "variance scaling gain must be finite and positive");
        else
            require_positive(i.scale, "noise scale must be finite and positive");
        if (i.distribution != Distribution::Uniform && i.distribution != Distribution::Normal)
            throw std::invalid_argument("invalid initializer distribution");
        const auto& d = i.denominators;
        if (!(d.bound > 0 && d.bound < 1)) throw std::invalid_argument("denominator bound must be in (0, 1)");
        require_positive(d.radius, "denominator radius must be finite and positive");
        return Common{i.distribution, i.seed, d};
    }, initializer);
}

// Raw draws: U[-1, 1) for Uniform, N(0, 1) for Normal. Every parameter is a
// per-term factor times one raw draw.
double raw_draw(Generator& generator, Distribution distribution) {
    return distribution == Distribution::Uniform ? generator.symmetric() : generator.normal();
}

void finite_parameters(const std::vector<double>& values) {
    for (double v : values)
        if (!std::isfinite(v)) throw std::overflow_error("initialized parameters are not finite");
}

// Linear-in-parameter carriers, per-term factors:
//   VarianceScaling: sigma_k = sqrt(gain^2 variance / (inputs terms m_k)),
//     times sqrt(3) for Uniform (U[-1,1) has variance 1/3); m_k = 0 gives 0;
//   NoiseInit: a = scale / (G sqrt(inputs)), a/2 for Uniform (pykan's
//     U(-a/2, a/2)) and a/sqrt(12) for Normal (the same variance).
std::vector<double> term_factors(const Initializer& initializer, const BasisConfig& basis, std::size_t inputs,
                                 Distribution distribution) {
    const auto terms = basis_size(basis);
    const bool uniform = distribution == Distribution::Uniform;
    std::vector<double> factor(terms);
    if (const auto* v = std::get_if<VarianceScaling>(&initializer)) {
        const auto m = reference_moments(basis);
        for (std::size_t k = 0; k < terms; ++k) {
            const double sigma = m.second_moments[k] == 0 ? 0
                : std::sqrt(v->gain * v->gain * m.variance / (double(inputs) * double(terms) * m.second_moments[k]));
            factor[k] = uniform ? std::sqrt(3.0) * sigma : sigma;
        }
    } else {
        const auto* spline = std::get_if<BSplineConfig>(&basis);
        const double grid = double(spline ? terms - spline->degree : terms);
        const double a = std::get<NoiseInit>(initializer).scale / (grid * std::sqrt(double(inputs)));
        for (auto& f : factor) f = uniform ? 0.5 * a : a / std::sqrt(12.0);
    }
    return factor;
}

template<class Edges>
Edges linear_edges(const Edges& edges, const Initializer& initializer, std::size_t inputs, std::size_t outputs,
                   Generator& generator, Distribution distribution) {
    const auto factor = term_factors(initializer, BasisConfig(edges.basis), inputs, distribution);
    const auto terms = factor.size();
    Edges next{edges.basis, std::vector<double>(inputs * outputs * terms)};
    for (std::size_t j = 0; j < next.coefficients.size(); ++j)
        next.coefficients[j] = factor[j % terms] * raw_draw(generator, distribution);
    return next;
}

// Rational: raw numerator draws first (layout order), then the denominators
// beta_k = +-(bound/n) * [1/2, 1] (from one U[-1,1) draw each) and
// b_k = beta_k / radius^k. Under VarianceScaling the numerator factors use the
// moments of the edge's own denominator.
RationalEdges rational_edges(const RationalEdges& edges, const Initializer& initializer, std::size_t inputs,
                             std::size_t outputs, Generator& generator, const Common& common) {
    const auto& config = edges.config;
    const auto& d = common.denominators;
    const std::size_t m = config.numerator_degree + 1, n = config.denominator_degree, count = inputs * outputs;
    if (config.denominator_policy == DenominatorPolicy::Guarded && !(config.epsilon * (1 + d.bound) < 1 - d.bound))
        throw std::invalid_argument("denominator bound conflicts with the rational pole guard epsilon");
    const bool uniform = common.distribution == Distribution::Uniform;
    std::vector<double> raw(count * m), beta(count * n), b(count * n);
    for (auto& v : raw) v = raw_draw(generator, common.distribution);
    std::vector<double> inverse_power(std::max(m, n + 1), 1.0); // radius^-k
    for (std::size_t k = 1; k < inverse_power.size(); ++k) inverse_power[k] = inverse_power[k - 1] / d.radius;
    const double magnitude = d.bound / double(std::max<std::size_t>(n, 1));
    for (std::size_t j = 0; j < beta.size(); ++j) {
        const double xi = generator.symmetric();
        beta[j] = xi >= 0 ? magnitude * ((1 + xi) / 2) : -(magnitude * ((1 - xi) / 2));
        b[j] = beta[j] * inverse_power[j % n + 1];
        if (!std::isfinite(b[j]) || b[j] == 0)
            throw std::overflow_error("denominator radius is out of range for the denominator degree");
    }
    RationalEdges next{config, std::vector<double>(count * m), std::move(b)};
    if (const auto* v = std::get_if<VarianceScaling>(&initializer)) {
        // x uniform on center +- radius*scale has variance (radius*scale)^2 / 3.
        const double extent = d.radius * config.scale;
        const double target = v->gain * v->gain * (extent * extent / 3);
        for (std::size_t e = 0; e < count; ++e) {
            const auto mu = init::rational_moments(config.denominator_policy, config.numerator_degree,
                                                   std::span<const double>(beta).subspan(e * n, n));
            for (std::size_t k = 0; k < m; ++k) {
                const double sigma = std::sqrt(target / (double(inputs) * double(m) * mu[k])) * inverse_power[k];
                next.coefficients[e * m + k] = (uniform ? std::sqrt(3.0) * sigma : sigma) * raw[e * m + k];
            }
        }
    } else {
        const double a = std::get<NoiseInit>(initializer).scale / (double(m) * std::sqrt(double(inputs)));
        const double factor = uniform ? 0.5 * a : a / std::sqrt(12.0);
        for (std::size_t j = 0; j < raw.size(); ++j) next.coefficients[j] = factor * raw[j];
    }
    return next;
}
} // namespace

void initialize(Layer& layer, const Initializer& initializer) {
    const auto common = validate(initializer);
    if (layer.inputs() == 0 || layer.outputs() == 0 || layer.bias().size() != layer.outputs())
        throw std::invalid_argument("layer is uninitialized or moved from");
    Generator generator(common.seed);
    Carrier next = std::visit([&](const auto& edges) -> Carrier {
        using Edges = std::decay_t<decltype(edges)>;
        if constexpr (std::is_same_v<Edges, RationalEdges>)
            return rational_edges(edges, initializer, layer.inputs(), layer.outputs(), generator, common);
        else
            return linear_edges(edges, initializer, layer.inputs(), layer.outputs(), generator, common.distribution);
    }, layer.carrier());
    std::visit([](const auto& edges) {
        finite_parameters(edges.coefficients);
        if constexpr (requires { edges.denominators; }) finite_parameters(edges.denominators);
    }, next);
    const std::vector<double> bias(layer.outputs(), 0.0);
    layer.set_carrier(std::move(next), bias);
}

void initialize(Network& network, const Initializer& initializer) {
    validate(initializer);
    std::vector<NetworkLayer> layers(network.layers().begin(), network.layers().end());
    for (std::size_t p = 0; p < layers.size(); ++p) {
        if (auto* layer = std::get_if<Layer>(&layers[p])) {
            auto per_layer = initializer;
            std::visit([&](auto& i) { i.seed = layer_seed(i.seed, p); }, per_layer);
            initialize(*layer, per_layer);
        } else {
            auto& map = std::get<InputMap>(layers[p]);
            if (const auto* norm = std::get_if<LayerNormMap>(&map.map()); norm && !norm->gain.empty())
                map.set_map(LayerNormMap{norm->epsilon, std::vector<double>(map.features(), 1.0),
                                         std::vector<double>(map.features(), 0.0)});
        }
    }
    network = Network(std::move(layers));
}

std::uint64_t layer_seed(std::uint64_t seed, std::size_t position) noexcept {
    return init::splitmix_mix(seed + (static_cast<std::uint64_t>(position) + 1) * init::splitmix_gamma);
}

} // namespace kan

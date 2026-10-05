// Backlog R9: ResidentNetwork::upload_parameters. An executor built once is
// reused after its parameters are replaced from a host Network of the same
// structure (CPU training, restoring a model). Contract: every trainable state
// is uploaded (coefficients, biases, trainable RBF centers/log widths, rational
// denominators, LayerNorm gain/bias); any difference in structure or fixed
// configuration (layer kinds, dimensions, carriers, basis configuration
// including knots and fixed centers, rational configuration, fixed maps)
// raises std::invalid_argument and leaves the executor unchanged; FP32
// executors round and validate exactly as at construction; a successful upload
// invalidates the forward/backward state but keeps the uploaded input and
// upstream.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

// Per-entry tolerances against the FP64 CPU reference evaluated at the
// executor's parameters: the resident FP64 tolerance (R7/C2) and the FP32 one (C1).
struct Tolerance { double relative, floor; };
constexpr Tolerance fp64{1e-12, 1e-13}, fp32{2e-4, 5e-5}, tf32{1e-2, 1e-2};
Tolerance tolerance(Precision p) {
    return p == Precision::Float64 ? fp64 : p == Precision::Float32 ? fp32 : tf32;
}
void close(std::span<const double> actual, std::span<const double> expected, Tolerance t) {
    REQUIRE(actual.size() == expected.size());
    double scale = 0;
    for (double e : expected) scale = std::max(scale, std::abs(e));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::isfinite(actual[i]));
        if (std::abs(actual[i]-expected[i]) > t.relative*std::abs(expected[i]) + t.floor*scale + 1e-37) {
            std::ostringstream out;
            out << std::setprecision(17) << "index " << i << " of " << actual.size() << " actual=" << actual[i]
                << " expected=" << expected[i] << " scale=" << scale;
            throw std::runtime_error(out.str());
        }
    }
}
std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
kan::Layer seeded(kan::Layer l, double phase) {
    const auto scale = 1.0/std::sqrt(static_cast<double>(l.inputs()*l.terms()));
    l.set_parameters(wave(l.coefficients().size(), scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
    return l;
}
kan::BasisConfig family(test::Family kind) {
    test::FamilyParameters p{5};
    p.alpha = 0.3; p.beta = -0.2; p.frequency = 1.7;
    p.centers = {-1, -0.5, 0, 0.5, 1}; p.width = 0.8;
    p.log_widths = {-0.3, -0.2, -0.1, -0.2, -0.3};
    p.scales = {0.6, 0.7, 0.8, 0.7, 0.6};
    p.degree = 3; p.knots = {-1.5, -1.5, -1.5, -1.5, -0.5, 0.25, 1.5, 1.5, 1.5, 1.5};
    return test::basis(kind, p);
}
constexpr test::Family families[] = {test::Family::Chebyshev, test::Family::Legendre, test::Family::Jacobi,
                                     test::Family::Hermite, test::Family::Fourier, test::Family::GaussianRbf,
                                     test::Family::TrainableRbf, test::Family::BSpline, test::Family::MexicanHat};
kan::RationalConfig rational_config(kan::DenominatorPolicy policy) {
    kan::RationalConfig r;
    r.numerator_degree = 3; r.denominator_degree = 2; r.center = 0.1; r.scale = 1.3; r.denominator_policy = policy;
    return r;
}
kan::Layer rational(std::size_t in, std::size_t out, kan::DenominatorPolicy policy) {
    kan::Layer l(in, out, rational_config(policy));
    kan::set_rational_parameters(l, wave(l.coefficients().size(), 0.3, 1.0, 1.0),
                                 wave(test::denominators(l).size(), 0.2, 2.0, 2.6), std::vector<double>(out, 0.01));
    return l;
}
const std::vector<double> spline_knots{-1.5, -1.5, -1.5, -1.5, -0.5, 0.25, 1.5, 1.5, 1.5, 1.5};

// A network holding every kind of trainable state and every fixed map:
// affine -> Chebyshev -> LayerNorm(gain, bias) -> trainable RBF -> tanh ->
// rational (safe policy) -> B-spline.
kan::Network mixed() {
    std::vector<kan::NetworkLayer> layers;
    layers.emplace_back(kan::InputMap(3, kan::AffineMap{{0.5, 0.4, 0.3}, {0.1, -0.1, 0.0}}));
    layers.emplace_back(seeded(kan::Layer(3, 4, kan::ChebyshevConfig{5}), 0.1));
    layers.emplace_back(kan::InputMap(4, kan::LayerNormMap{1e-3, {1.0, 0.9, 1.1, 1.0}, {0.0, 0.05, -0.05, 0.1}}));
    kan::Layer rbf = seeded(kan::Layer(4, 4, family(test::Family::TrainableRbf)), 0.7);
    layers.emplace_back(std::move(rbf));
    layers.emplace_back(kan::InputMap(4, kan::TanhMap{0.8}));
    layers.emplace_back(rational(4, 3, kan::DenominatorPolicy::Absolute));
    layers.emplace_back(seeded(kan::Layer(3, 2, kan::BSplineConfig{3, spline_knots}), 1.9));
    return kan::Network(std::move(layers));
}
constexpr std::size_t batch = 9;
std::vector<double> input_for(const kan::Network& n, std::size_t rows = batch) {
    return wave(rows*n.inputs(), 0.9, 0.37, 0.2);
}
std::vector<double> upstream_for(const kan::Network& n, std::size_t rows = batch) {
    return wave(rows*n.outputs(), 1.0, 0.53, 0.4);
}
// `steps` CPU SGD steps: the "trained elsewhere" weights.
kan::Network cpu_trained(kan::Network n, int steps, double rate = 0.05) {
    const auto x = input_for(n), u = upstream_for(n);
    for (int s = 0; s < steps; ++s) n.sgd(n.backward(x, batch, u), rate);
    return n;
}
// Every trainable parameter of a network in a fixed order.
std::vector<double> flat(const kan::Network& network) {
    std::vector<double> v;
    auto add = [&](std::span<const double> s) { v.insert(v.end(), s.begin(), s.end()); };
    for (std::size_t j = 0; j < network.layers().size(); ++j) {
        if (std::holds_alternative<kan::InputMap>(network.layers()[j])) {
            if (const auto* n = std::get_if<kan::LayerNormMap>(&test::input_map(network, j).map())) { add(n->gain); add(n->bias); }
            continue;
        }
        const auto& l = test::layer(network, j);
        add(l.coefficients()); add(l.bias());
        if (const auto* t = std::get_if<kan::TrainableRbfEdges>(&l.carrier())) { add(t->basis.centers); add(t->basis.log_widths); }
        add(test::denominators(l));
    }
    return v;
}
std::vector<double> rounded(std::vector<double> v, Precision p) {
    if (p != Precision::Float64) for (auto& x : v) x = static_cast<double>(static_cast<float>(x));
    return v;
}
void gradients(const kan::NetworkGradients& actual, const kan::NetworkGradients& expected, Tolerance t) {
    close(actual.input, expected.input, t);
    REQUIRE(actual.layers.size() == expected.layers.size());
    for (std::size_t j = 0; j < actual.layers.size(); ++j) {
        if (std::holds_alternative<kan::InputMapGradients>(expected.layers[j])) {
            close(test::map_grad(actual, j).input, test::map_grad(expected, j).input, t);
            close(test::map_grad(actual, j).gain, test::map_grad(expected, j).gain, t);
            close(test::map_grad(actual, j).bias, test::map_grad(expected, j).bias, t);
            continue;
        }
        close(test::grad(actual, j).input, test::grad(expected, j).input, t);
        close(test::grad(actual, j).coefficients, test::grad(expected, j).coefficients, t);
        close(test::grad(actual, j).bias, test::grad(expected, j).bias, t);
        close(test::centers(test::grad(actual, j)), test::centers(test::grad(expected, j)), t);
        close(test::log_widths(test::grad(actual, j)), test::log_widths(test::grad(expected, j)), t);
        close(test::denominators(test::grad(actual, j)), test::denominators(test::grad(expected, j)), t);
    }
}
// Forward and backward of the executor against the FP64 CPU at the
// executor's (downloaded) parameters.
void matches_cpu(ResidentNetwork& gpu, const kan::Network& reference_shape, Precision p) {
    const auto x = input_for(reference_shape), u = upstream_for(reference_shape);
    const auto reference = gpu.download_parameters();
    gpu.upload_input(x, batch); gpu.upload_output_gradient(u);
    gpu.forward();
    close(gpu.download_output(), reference.forward(x, batch), tolerance(p));
    gpu.backward();
    gradients(gpu.download_gradients(), reference.backward(x, batch, u), tolerance(p));
}
constexpr Precision precisions[] = {Precision::Float64, Precision::Float32, Precision::TensorFloat32};

TEST(upload_after_cpu_training_matches_cpu) {
    for (const auto p : precisions) {
        const auto initial = mixed();
        ResidentNetwork gpu(initial, batch, p);
        const auto trained = cpu_trained(initial, 4);
        REQUIRE(flat(trained) != flat(initial));
        gpu.upload_parameters(trained);
        // Every trainable state arrived, rounded exactly like at construction.
        REQUIRE(flat(gpu.download_parameters()) == rounded(flat(trained), p));
        REQUIRE(flat(gpu.download_parameters()) == flat(ResidentNetwork(trained, batch, p).download_parameters()));
        matches_cpu(gpu, trained, p);
        // Training continues on the GPU from the uploaded weights.
        const auto before = gpu.download_parameters();
        gpu.sgd(0.05);
        auto expected = before;
        const auto x = input_for(trained), u = upstream_for(trained);
        expected.sgd(before.backward(x, batch, u), 0.05);
        close(flat(gpu.download_parameters()), flat(expected), p == Precision::Float64 ? fp64 : Tolerance{1e-3, 1e-4});
    }
}

TEST(upload_matches_cpu_for_every_family_and_policy) {
    for (const auto p : precisions) {
        std::vector<kan::Network> networks;
        for (const auto kind : families) networks.push_back(kan::Network({seeded(kan::Layer(3, 2, family(kind)), 0.4)}));
        for (const auto policy : {kan::DenominatorPolicy::Guarded, kan::DenominatorPolicy::Absolute, kan::DenominatorPolicy::Smooth})
            networks.push_back(kan::Network({rational(3, 2, policy)}));
        for (const auto& initial : networks) {
            ResidentNetwork gpu(initial, batch, p);
            const auto trained = cpu_trained(initial, 3, 0.02);
            REQUIRE(flat(trained) != flat(initial));
            gpu.upload_parameters(trained);
            REQUIRE(flat(gpu.download_parameters()) == rounded(flat(trained), p));
            matches_cpu(gpu, trained, p);
        }
    }
}

TEST(upload_round_trips_with_download) {
    for (const auto p : precisions) {
        const auto initial = mixed();
        const auto x = input_for(initial), u = upstream_for(initial);
        // A: trained on the GPU; B: a fresh executor of the initial network.
        ResidentNetwork a(initial, batch, p);
        a.upload_input(x, batch); a.upload_output_gradient(u);
        for (int s = 0; s < 3; ++s) { a.forward(); a.backward(0.01); a.sgd(0.05); }
        const auto saved = a.download_parameters();
        a.forward();
        const auto output = a.download_output();
        ResidentNetwork b(initial, batch, p);
        b.upload_parameters(saved);
        // Downloads are exact widenings, so the round trip is exact in every precision.
        REQUIRE(flat(b.download_parameters()) == flat(saved));
        REQUIRE(b.download_parameters().layers().size() == saved.layers().size());
        for (std::size_t j = 0; j < saved.layers().size(); ++j) REQUIRE(b.download_parameters().layers()[j].index() == saved.layers()[j].index());
        b.upload_input(x, batch);
        b.forward();
        REQUIRE(b.download_output() == output); // same executor kind, parameters and input
        // Uploading an executor's own download is a no-op on its state.
        a.upload_parameters(a.download_parameters());
        REQUIRE(flat(a.download_parameters()) == flat(saved));
        a.forward();
        REQUIRE(a.download_output() == output);
    }
}

// Structural variants of mixed(), each differing in exactly one aspect that
// upload_parameters must reject.
std::vector<std::pair<std::string, kan::Network>> mismatches() {
    std::vector<std::pair<std::string, kan::Network>> result;
    const auto base = mixed();
    const auto replaced = [&](std::size_t j, kan::NetworkLayer stage) {
        std::vector<kan::NetworkLayer> layers(base.layers().begin(), base.layers().end());
        layers[j] = std::move(stage);
        return kan::Network(std::move(layers));
    };
    const auto layer = [&](std::size_t j) { return test::layer(base, j); };
    {
        std::vector<kan::NetworkLayer> layers(base.layers().begin(), base.layers().end());
        layers.emplace_back(kan::InputMap(2, kan::TanhMap{1.0}));
        result.emplace_back("extra layer", kan::Network(std::move(layers)));
    }
    {
        std::vector<kan::NetworkLayer> layers(base.layers().begin(), base.layers().end() - 1);
        result.emplace_back("missing layer", kan::Network(std::move(layers)));
    }
    {
        std::vector<kan::NetworkLayer> layers(base.layers().begin()+1, base.layers().end());
        result.emplace_back("leading layer removed", kan::Network(std::move(layers)));
    }
    result.emplace_back("layer instead of input map", replaced(4, seeded(kan::Layer(4, 4, kan::ChebyshevConfig{2}), 0.3)));
    result.emplace_back("input map instead of layer", replaced(6, kan::InputMap(3, kan::TanhMap{1.0})));
    {
        // Different hidden width 3->5->... (dimensions).
        std::vector<kan::NetworkLayer> layers(base.layers().begin(), base.layers().end());
        layers[1] = seeded(kan::Layer(3, 5, kan::ChebyshevConfig{5}), 0.1);
        layers[2] = kan::InputMap(5, kan::LayerNormMap{1e-3, std::vector<double>(5, 1.0), std::vector<double>(5, 0.0)});
        layers[3] = seeded(kan::Layer(5, 4, family(test::Family::TrainableRbf)), 0.7);
        result.emplace_back("hidden dimensions", kan::Network(std::move(layers)));
    }
    result.emplace_back("basis size", replaced(1, seeded(kan::Layer(3, 4, kan::ChebyshevConfig{6}), 0.1)));
    result.emplace_back("basis family, same size", replaced(1, seeded(kan::Layer(3, 4, kan::LegendreConfig{5}), 0.1)));
    result.emplace_back("Jacobi family, same size", replaced(1, seeded(kan::Layer(3, 4, kan::JacobiConfig{5, 0.5, 0.0}), 0.1)));
    result.emplace_back("fixed Gaussian RBF instead of trainable", replaced(3, seeded(kan::Layer(4, 4, kan::GaussianRbfConfig{{-1, -0.5, 0, 0.5, 1}, 0.8}), 0.7)));
    result.emplace_back("trainable RBF term count", replaced(3, seeded(kan::Layer(4, 4, kan::TrainableRbfConfig{{-1, 0, 1}, {0, 0, 0}}), 0.7)));
    result.emplace_back("rational instead of basis carrier", replaced(1, rational(3, 4, kan::DenominatorPolicy::Absolute)));
    for (const auto& [name, change] : std::vector<std::pair<std::string, std::function<void(kan::RationalConfig&)>>>{
             {"rational numerator degree", [](kan::RationalConfig& c) { c.numerator_degree = 2; }},
             {"rational denominator degree", [](kan::RationalConfig& c) { c.denominator_degree = 3; }},
             {"rational center", [](kan::RationalConfig& c) { c.center = 0.2; }},
             {"rational scale", [](kan::RationalConfig& c) { c.scale = 1.4; }},
             {"rational epsilon", [](kan::RationalConfig& c) { c.epsilon = 1e-7; }},
             {"rational policy", [](kan::RationalConfig& c) { c.denominator_policy = kan::DenominatorPolicy::Smooth; }}}) {
        auto config = rational_config(kan::DenominatorPolicy::Absolute);
        change(config);
        kan::Layer l(4, 3, config);
        result.emplace_back(name, replaced(5, std::move(l)));
    }
    {
        // Knots moved at the same term count (set_carrier): fixed configuration.
        auto l = layer(6);
        auto knots = spline_knots; knots[5] = 0.3;
        l.set_carrier(kan::BasisEdges{kan::BSplineConfig{3, knots}, std::vector<double>(l.coefficients().begin(), l.coefficients().end())});
        result.emplace_back("B-spline knots, same term count", replaced(6, std::move(l)));
    }
    {
        auto l = layer(6);
        kan::insert_knot(l, 0.6);
        result.emplace_back("B-spline insert_knot", replaced(6, std::move(l)));
    }
    {
        auto l = layer(6);
        kan::adapt_grid(l, std::vector<double>{-0.2, -0.1, 0.0, 0.1});
        result.emplace_back("B-spline adapt_grid", replaced(6, std::move(l)));
    }
    result.emplace_back("affine scale", replaced(0, kan::InputMap(3, kan::AffineMap{{0.5, 0.4, 0.35}, {0.1, -0.1, 0.0}})));
    result.emplace_back("affine shift", replaced(0, kan::InputMap(3, kan::AffineMap{{0.5, 0.4, 0.3}, {0.1, -0.1, 0.01}})));
    result.emplace_back("map kind", replaced(0, kan::InputMap(3, kan::TanhMap{1.0})));
    result.emplace_back("tanh scale", replaced(4, kan::InputMap(4, kan::TanhMap{0.9})));
    result.emplace_back("LayerNorm epsilon", replaced(2, kan::InputMap(4, kan::LayerNormMap{1e-4, {1.0, 0.9, 1.1, 1.0}, {0.0, 0.05, -0.05, 0.1}})));
    result.emplace_back("LayerNorm without gain/bias", replaced(2, kan::InputMap(4, kan::LayerNormMap{1e-3, {}, {}})));
    return result;
}

TEST(upload_rejects_every_structure_mismatch_and_keeps_the_executor_unchanged) {
    for (const auto p : precisions) {
        const auto initial = mixed();
        const auto x = input_for(initial), u = upstream_for(initial);
        ResidentNetwork gpu(initial, batch, p);
        gpu.upload_input(x, batch); gpu.upload_output_gradient(u);
        gpu.forward();
        const auto parameters = gpu.download_parameters();
        const auto output = gpu.download_output();
        const auto allocations = gpu.workspace_allocations();
        const auto cases = mismatches();
        REQUIRE(cases.size() == 27);
        for (const auto& [name, network] : cases) {
            try {
                gpu.upload_parameters(network);
            } catch (const std::invalid_argument&) {
                // Unchanged: parameters, the current forward state and the batch.
                REQUIRE(flat(gpu.download_parameters()) == flat(parameters));
                REQUIRE(gpu.download_output() == output);
                REQUIRE(gpu.batch() == batch);
                continue;
            }
            throw std::runtime_error("mismatch accepted: " + name);
        }
        {
            // A moved-from network has no layers.
            auto moved = initial;
            const auto target = std::move(moved);
            test::throws<std::invalid_argument>([&] { gpu.upload_parameters(moved); });
            REQUIRE(target.layers().size() == initial.layers().size());
        }
        // The forward state survived every rejection: backward still runs.
        gpu.backward();
        gradients(gpu.download_gradients(), parameters.backward(x, batch, u), tolerance(p));
        REQUIRE(gpu.workspace_allocations() == allocations);
    }
}

TEST(upload_rejects_fixed_basis_configuration_changes) {
    // Same family and term count, different fixed configuration: structure.
    const std::vector<std::pair<kan::BasisConfig, kan::BasisConfig>> pairs{
        {kan::JacobiConfig{5, 0.3, -0.2}, kan::JacobiConfig{5, 0.3, -0.1}},
        {kan::FourierConfig{5, 1.7}, kan::FourierConfig{5, 1.8}},
        {kan::GaussianRbfConfig{{-1, 0, 1}, 0.8}, kan::GaussianRbfConfig{{-1, 0.1, 1}, 0.8}},
        {kan::GaussianRbfConfig{{-1, 0, 1}, 0.8}, kan::GaussianRbfConfig{{-1, 0, 1}, 0.9}},
        {kan::MexicanHatConfig{{-1, 0, 1}, {0.5, 0.6, 0.7}}, kan::MexicanHatConfig{{-1, 0, 1}, {0.5, 0.6, 0.8}}},
        {kan::MexicanHatConfig{{-1, 0, 1}, {0.5, 0.6, 0.7}}, kan::MexicanHatConfig{{-1, 0, 0.9}, {0.5, 0.6, 0.7}}},
        {kan::BSplineConfig{1, {0, 0, 0.5, 1, 1}}, kan::BSplineConfig{1, {0, 0, 0.4, 1, 1}}},
        {kan::BSplineConfig{1, {0, 0, 0.5, 1, 1}}, kan::BSplineConfig{2, {0, 0, 0, 1, 1, 1}}},
    };
    for (const auto& [built, other] : pairs) {
        const kan::Network initial({seeded(kan::Layer(2, 2, built), 0.2)});
        const kan::Network changed({seeded(kan::Layer(2, 2, other), 0.5)});
        REQUIRE(test::layer(initial, 0).terms() == test::layer(changed, 0).terms());
        ResidentNetwork gpu(initial, 3);
        const auto before = flat(gpu.download_parameters());
        test::throws<std::invalid_argument>([&] { gpu.upload_parameters(changed); });
        REQUIRE(flat(gpu.download_parameters()) == before);
        // The same family and configuration with new coefficients is accepted.
        const kan::Network retrained({seeded(kan::Layer(2, 2, built), 0.9)});
        gpu.upload_parameters(retrained);
        REQUIRE(flat(gpu.download_parameters()) == flat(retrained));
    }
}

TEST(float32_upload_rejects_unrepresentable_parameters_without_partial_upload) {
    // Each variant changes the first layer's coefficients (a partial upload
    // would show) and puts one FP32-unrepresentable value into a later state.
    const auto initial = mixed();
    const auto changed_first = [&] {
        std::vector<kan::NetworkLayer> layers(initial.layers().begin(), initial.layers().end());
        auto first = std::get<kan::Layer>(layers[1]);
        first.set_parameters(wave(first.coefficients().size(), 0.4, 0.3, 0.0), wave(first.outputs(), 0.1, 0.2));
        layers[1] = std::move(first);
        return layers;
    };
    std::vector<std::pair<std::string, kan::Network>> cases;
    const auto add = [&](const std::string& name, const std::function<void(std::vector<kan::NetworkLayer>&)>& edit) {
        auto layers = changed_first();
        edit(layers);
        cases.emplace_back(name, kan::Network(std::move(layers)));
    };
    add("coefficient beyond FLT_MAX", [](auto& layers) {
        auto& l = std::get<kan::Layer>(layers[6]);
        std::vector<double> c(l.coefficients().begin(), l.coefficients().end()); c.back() = 1e39;
        l.set_parameters(c, std::vector<double>(l.bias().begin(), l.bias().end()));
    });
    add("bias beyond FLT_MAX", [](auto& layers) {
        auto& l = std::get<kan::Layer>(layers[6]);
        std::vector<double> b(l.bias().begin(), l.bias().end()); b[0] = -1e39;
        l.set_parameters(std::vector<double>(l.coefficients().begin(), l.coefficients().end()), b);
    });
    add("trainable RBF center beyond FLT_MAX", [](auto& layers) {
        auto& l = std::get<kan::Layer>(layers[3]);
        auto c = test::trainable(l);
        c.centers[2] = 1e39;
        kan::set_rbf_parameters(l, c.centers, c.log_widths);
    });
    add("trainable RBF width underflows FP32", [](auto& layers) {
        auto& l = std::get<kan::Layer>(layers[3]);
        auto c = test::trainable(l);
        c.log_widths[1] = -110; // exp(-110) = 1.7e-48: positive in FP64, zero in FP32
        kan::set_rbf_parameters(l, c.centers, c.log_widths);
    });
    add("rational denominator beyond FLT_MAX", [](auto& layers) {
        auto& l = std::get<kan::Layer>(layers[5]);
        const auto& e = std::get<kan::RationalEdges>(l.carrier());
        auto b = e.denominators; b[0] = 1e39;
        kan::set_rational_parameters(l, e.coefficients, b, std::vector<double>(l.bias().begin(), l.bias().end()));
    });
    add("LayerNorm gain beyond FLT_MAX", [](auto& layers) {
        auto& m = std::get<kan::InputMap>(layers[2]);
        auto norm = std::get<kan::LayerNormMap>(m.map());
        norm.gain[3] = 1e39;
        m.set_map(norm);
    });
    add("LayerNorm bias beyond FLT_MAX", [](auto& layers) {
        auto& m = std::get<kan::InputMap>(layers[2]);
        auto norm = std::get<kan::LayerNormMap>(m.map());
        norm.bias[0] = 1e39;
        m.set_map(norm);
    });
    for (const auto p : {Precision::Float32, Precision::TensorFloat32}) {
        ResidentNetwork gpu(initial, batch, p);
        const auto before = flat(gpu.download_parameters());
        for (const auto& [name, network] : cases) {
            try {
                gpu.upload_parameters(network);
            } catch (const std::invalid_argument&) {
                REQUIRE(flat(gpu.download_parameters()) == before);
                // The same value is rejected at construction.
                test::throws<std::invalid_argument>([&] { (void)ResidentNetwork(network, batch, p); });
                continue;
            }
            throw std::runtime_error("unrepresentable upload accepted: " + name);
        }
    }
    // FP64 accepts every one of them (finite double parameters).
    for (const auto& [name, network] : cases) {
        ResidentNetwork gpu(initial, batch);
        gpu.upload_parameters(network);
        REQUIRE(flat(gpu.download_parameters()) == flat(network));
    }
    // Values below the FP32 range round, as at construction.
    {
        auto layers = changed_first();
        auto& l = std::get<kan::Layer>(layers[1]);
        std::vector<double> c(l.coefficients().begin(), l.coefficients().end()); c[0] = 1e-50;
        l.set_parameters(c, std::vector<double>(l.bias().begin(), l.bias().end()));
        const kan::Network tiny(std::move(layers));
        ResidentNetwork gpu(initial, batch, Precision::Float32);
        gpu.upload_parameters(tiny);
        REQUIRE(test::layer(gpu.download_parameters(), 1).coefficients()[0] == 0.0);
    }
}

TEST(upload_invalidates_forward_state_and_keeps_input_and_upstream) {
    for (const auto p : precisions) {
        const auto initial = mixed();
        const auto trained = cpu_trained(initial, 2);
        const auto x = input_for(initial), u = upstream_for(initial);
        ResidentNetwork gpu(initial, batch, p);
        // Before any input: allowed, and nothing to invalidate.
        gpu.upload_parameters(initial);
        gpu.upload_input(x, batch); gpu.upload_output_gradient(u);
        gpu.forward(); gpu.backward();
        gpu.upload_parameters(trained);
        test::throws<std::logic_error>([&] { gpu.backward(); });
        test::throws<std::logic_error>([&] { gpu.download_output(); });
        test::throws<std::logic_error>([&] { gpu.download_gradients(); });
        test::throws<std::logic_error>([&] { gpu.sgd(0.1); });
        REQUIRE(gpu.batch() == batch);
        // Input and upstream are kept: forward and backward run without re-upload.
        const auto reference = gpu.download_parameters();
        gpu.forward();
        close(gpu.download_output(), reference.forward(x, batch), tolerance(p));
        gpu.backward();
        gradients(gpu.download_gradients(), reference.backward(x, batch, u), tolerance(p));
        // After forward only, upload invalidates the output as well.
        gpu.forward();
        gpu.upload_parameters(initial);
        test::throws<std::logic_error>([&] { gpu.download_output(); });
        test::throws<std::logic_error>([&] { gpu.backward(); });
        // An empty batch still works after an upload.
        gpu.upload_input({}, 0);
        gpu.upload_parameters(trained);
        gpu.forward();
        REQUIRE(gpu.download_output().empty());
    }
}

TEST(upload_of_parameter_regions_above_one_mebibyte) {
    // FP64 regions above 1 MiB are copied tensor by tensor without host
    // staging; FP32 always stages. Offsets of every block kind must hold.
    std::vector<kan::NetworkLayer> layers;
    layers.emplace_back(seeded(kan::Layer(48, 64, kan::ChebyshevConfig{48}), 0.3)); // 147456 coefficients
    layers.emplace_back(kan::InputMap(64, kan::LayerNormMap{1e-3, wave(64, 0.1, 0.3, 1.0), wave(64, 0.05, 0.7)}));
    layers.emplace_back(seeded(kan::Layer(64, 8, family(test::Family::TrainableRbf)), 0.6));
    layers.emplace_back(rational(8, 2, kan::DenominatorPolicy::Smooth));
    const kan::Network initial(std::move(layers));
    REQUIRE(flat(initial).size()*sizeof(double) > (std::size_t{1} << 20));
    const auto trained = cpu_trained(initial, 2, 0.01);
    for (const auto p : precisions) {
        ResidentNetwork gpu(initial, batch, p);
        gpu.upload_parameters(trained);
        REQUIRE(flat(gpu.download_parameters()) == rounded(flat(trained), p));
        matches_cpu(gpu, trained, p);
        // Rejections still leave everything unchanged.
        std::vector<kan::NetworkLayer> stages(trained.layers().begin(), trained.layers().end());
        stages[1] = kan::InputMap(64, kan::LayerNormMap{1e-2, wave(64, 0.1, 0.3, 1.0), wave(64, 0.05, 0.7)});
        const auto before = flat(gpu.download_parameters());
        test::throws<std::invalid_argument>([&] { gpu.upload_parameters(kan::Network(std::move(stages))); });
        REQUIRE(flat(gpu.download_parameters()) == before);
    }
    // Validation precedes every copy: an invalid tensor after valid ones (FP32
    // representability; FP64 values of a Network are always finite) uploads nothing.
    auto huge = trained;
    {
        std::vector<kan::NetworkLayer> stages(huge.layers().begin(), huge.layers().end());
        auto& last = std::get<kan::Layer>(stages[3]);
        std::vector<double> bias(last.bias().begin(), last.bias().end()); bias[1] = 1e39;
        last.set_parameters(std::vector<double>(last.coefficients().begin(), last.coefficients().end()), bias);
        huge = kan::Network(std::move(stages));
    }
    ResidentNetwork gpu(initial, batch, Precision::Float32);
    const auto before = flat(gpu.download_parameters());
    test::throws<std::invalid_argument>([&] { gpu.upload_parameters(huge); });
    REQUIRE(flat(gpu.download_parameters()) == before);
}

TEST(upload_needs_no_allocation_and_handles_parameter_free_networks) {
    const auto initial = mixed();
    ResidentNetwork gpu(initial, batch);
    const auto allocations = gpu.workspace_allocations();
    const auto trained = cpu_trained(initial, 1);
    for (int i = 0; i < 20; ++i) gpu.upload_parameters(i % 2 ? initial : trained);
    REQUIRE(gpu.workspace_allocations() == allocations);
    REQUIRE(flat(gpu.download_parameters()) == flat(initial));
    // Fixed maps only: nothing to upload, the structure is still checked.
    const kan::Network fixed({kan::InputMap(2, kan::AffineMap{{2.0, 0.5}, {0.0, 1.0}}), kan::InputMap(2, kan::TanhMap{1.0})});
    ResidentNetwork maps(fixed, 4);
    maps.upload_parameters(fixed);
    test::throws<std::invalid_argument>([&] {
        maps.upload_parameters(kan::Network({kan::InputMap(2, kan::AffineMap{{2.0, 0.5}, {0.0, 1.5}}), kan::InputMap(2, kan::TanhMap{1.0})}));
    });
    // A moved-from executor raises like every other operation.
    ResidentNetwork moved(initial, 1);
    ResidentNetwork target(std::move(moved));
    test::throws<std::logic_error>([&] { moved.upload_parameters(initial); });
    target.upload_parameters(trained);
}
} // namespace

int main() {
    if (!kan::cuda::available()) { std::cerr << "real CUDA hardware required\n"; return 1; }
    return test::run();
}

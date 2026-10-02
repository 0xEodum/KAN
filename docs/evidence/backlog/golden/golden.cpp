// Bitwise golden dump for behaviour-preserving backlog refactors (R1-R3, R2).
// Prints every CPU basis/rational result and every resident CUDA result as
// exact hex doubles, including guard exceptions. Two builds of the library
// are equivalent for these fixtures only if their dumps are byte-identical.
//
// `--layers` (added for R2) instead dumps CPU Layer/Network execution: forward,
// backward, L2, SGD and the family operations on the resident fixtures.
// The harness compiles against the API before R2 (Layer members) and after it
// (carriers, kan/families.hpp), so both builds run the same fixtures. Since M1
// network layers are a variant (KAN layer or input map) and network gradients
// hold the matching alternative; the fixtures contain KAN layers only.
#include "kan/resident.hpp"
#if __has_include("kan/families.hpp")
#include "kan/families.hpp"
#define KAN_GOLDEN_CARRIERS 1
#endif
#if __has_include("kan/input_map.hpp")
#define KAN_GOLDEN_STAGES 1
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace kan;

namespace {
// API adapters: the dumped values never depend on which branch is compiled.
namespace api {
std::vector<double> none() { return {}; }
#ifdef KAN_GOLDEN_STAGES
std::vector<Layer> layers(const Network& n) {
    std::vector<Layer> result;
    for (const auto& stage : n.layers()) result.push_back(std::get<Layer>(stage));
    return result;
}
LayerGradients& grad(NetworkGradients& g, std::size_t j) { return std::get<LayerGradients>(g.layers[j]); }
const LayerGradients& grad(const NetworkGradients& g, std::size_t j) { return std::get<LayerGradients>(g.layers[j]); }
#else
std::vector<Layer> layers(const Network& n) { return {n.layers().begin(), n.layers().end()}; }
LayerGradients& grad(NetworkGradients& g, std::size_t j) { return g.layers[j]; }
const LayerGradients& grad(const NetworkGradients& g, std::size_t j) { return g.layers[j]; }
#endif
std::size_t outputs(const Network& n) { return layers(n).back().outputs(); }
#ifdef KAN_GOLDEN_CARRIERS
bool is_rational(const Layer& l) { return std::holds_alternative<RationalEdges>(l.carrier()); }
std::vector<double> denominators(const Layer& l) {
    const auto* r = std::get_if<RationalEdges>(&l.carrier());
    return r ? r->denominators : none();
}
const TrainableRbfConfig* trainable(const Layer& l) {
    const auto* t = std::get_if<TrainableRbfEdges>(&l.carrier());
    return t ? &t->basis : nullptr;
}
const BSplineConfig* spline(const Layer& l) {
    const auto* e = std::get_if<BasisEdges>(&l.carrier());
    return e ? std::get_if<BSplineConfig>(&e->basis) : nullptr;
}
void set_rational(Layer& l, std::span<const double> a, std::span<const double> b, std::span<const double> bias) {
    set_rational_parameters(l, a, b, bias);
}
void set_rbf(Layer& l, std::span<const double> c, std::span<const double> w) { set_rbf_parameters(l, c, w); }
void knot(Layer& l, double x) { insert_knot(l, x); }
double adapt(Layer& l, std::span<const double> x) { return adapt_grid(l, x); }
std::vector<double> centers(const LayerGradients& g) {
    const auto* t = std::get_if<TrainableRbfGradients>(&g.nonlinear);
    return t ? t->centers : none();
}
std::vector<double> log_widths(const LayerGradients& g) {
    const auto* t = std::get_if<TrainableRbfGradients>(&g.nonlinear);
    return t ? t->log_widths : none();
}
std::vector<double> denominators(const LayerGradients& g) {
    const auto* r = std::get_if<RationalGradients>(&g.nonlinear);
    return r ? r->denominators : none();
}
#else
bool is_rational(const Layer& l) { return l.is_rational(); }
std::vector<double> denominators(const Layer& l) { return {l.denominators().begin(), l.denominators().end()}; }
const TrainableRbfConfig* trainable(const Layer& l) {
    return l.is_rational() ? nullptr : std::get_if<TrainableRbfConfig>(&l.basis());
}
const BSplineConfig* spline(const Layer& l) {
    return l.is_rational() ? nullptr : std::get_if<BSplineConfig>(&l.basis());
}
void set_rational(Layer& l, std::span<const double> a, std::span<const double> b, std::span<const double> bias) {
    l.set_rational_parameters(a, b, bias);
}
void set_rbf(Layer& l, std::span<const double> c, std::span<const double> w) { l.set_rbf_parameters(c, w); }
void knot(Layer& l, double x) { l.insert_knot(x); }
double adapt(Layer& l, std::span<const double> x) { return l.adapt_grid(x); }
std::vector<double> centers(const LayerGradients& g) { return g.centers; }
std::vector<double> log_widths(const LayerGradients& g) { return g.log_widths; }
std::vector<double> denominators(const LayerGradients& g) { return g.denominators; }
#endif
} // namespace api

void hex(const char* label, std::span<const double> values) {
    std::printf("%s[%zu]", label, values.size());
    for (double v : values) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        std::printf(" %016llx", static_cast<unsigned long long>(bits));
    }
    std::printf("\n");
}

void guarded(const std::string& label, const std::function<void()>& body) {
    std::printf("== %s\n", label.c_str());
    try {
        body();
    } catch (const std::domain_error& e) {
        std::printf("domain_error: %s\n", e.what());
    } catch (const std::overflow_error& e) {
        std::printf("overflow_error: %s\n", e.what());
    } catch (const std::invalid_argument& e) {
        std::printf("invalid_argument: %s\n", e.what());
    } catch (const std::logic_error& e) {
        std::printf("logic_error: %s\n", e.what());
    } catch (const std::exception& e) {
        std::printf("exception: %s\n", e.what());
    }
}

// Fixture constructors. Typed configurations since R1; the dumped fixtures and
// their order are unchanged, so dumps stay comparable across the refactor.
BasisConfig fourier(std::size_t size, double frequency) { return FourierConfig{size, frequency}; }
BasisConfig rbf(std::vector<double> centers, double width) { return GaussianRbfConfig{std::move(centers), width}; }
BasisConfig trainable_rbf(std::vector<double> centers, std::vector<double> log_widths) {
    return TrainableRbfConfig{std::move(centers), std::move(log_widths)};
}
BasisConfig wavelet(std::vector<double> centers, std::vector<double> scales) {
    return MexicanHatConfig{std::move(centers), std::move(scales)};
}
BasisConfig spline(std::size_t degree, std::vector<double> knots) { return BSplineConfig{degree, std::move(knots)}; }

std::vector<std::pair<std::string, BasisConfig>> basis_fixtures() {
    return {
        {"chebyshev7", ChebyshevConfig{7}},
        {"chebyshev1", ChebyshevConfig{1}},
        {"legendre6", LegendreConfig{6}},
        {"hermite6", HermiteConfig{6}},
        {"jacobi_asym", JacobiConfig{6, 0.5, -0.3}},
        {"jacobi_sum_m1", JacobiConfig{5, -0.5, -0.5}},
        {"jacobi_near_bound", JacobiConfig{5, -0.999999, 3.5}},
        {"jacobi_huge", JacobiConfig{4, 1e300, 1e300}},
        {"fourier5", fourier(5, 1.3)},
        {"fourier_huge", fourier(3, 1e308)},
        {"rbf", rbf({-1, -0.3, 0.4, 1.2}, 0.7)},
        {"rbf_tiny_width", rbf({0, 1e-3}, 1e-160)},
        {"rbf_trainable", trainable_rbf({-1, 0, 0.8}, {-0.5, 0.1, -700})},
        {"mexican_hat", wavelet({0, 0.5, -1e308}, {1, 0.25, 1e-3})},
        {"bspline3", spline(3, {-1, -1, -1, -1, -0.5, 0, 0, 0.5, 1, 1, 1, 1})},
        {"bspline0", spline(0, {-1, -0.25, 0.5, 1})},
        {"bspline1", spline(1, {-1, -1, 0, 1, 1})},
        {"bspline16", spline(16, [] {
             std::vector<double> t(17, -2.0);
             t.push_back(0.0);
             t.insert(t.end(), 17, 2.0);
             return t;
         }())},
        {"bspline_huge_domain", spline(2, {-1e308, -1e308, -1e308, 0, 1e308, 1e308, 1e308})},
    };
}

const std::vector<double>& inputs() {
    static const std::vector<double> x{-3, -2, -1.5, -1, -0.999, -0.7, -0.5, -0.25, 0, 1e-310,
                                       0.25, 1.0 / 3, 0.5, 0.7, 0.999, 1, 1.5, 1.7320508075688772,
                                       2, 3, 25, 1e3, 1e150, -1e200, 1e300};
    return x;
}

void dump_basis() {
    for (const auto& [name, config] : basis_fixtures()) {
        for (double x : inputs()) {
            char label[160];
            std::snprintf(label, sizeof label, "basis %s x=%a", name.c_str(), x);
            guarded(label, [&] {
                const auto r = evaluate_basis(config, x);
                hex("v", r.values);
                hex("d", r.derivatives);
                hex("dc", r.center_derivatives);
                hex("dw", r.log_width_derivatives);
            });
        }
    }
}

void dump_rational() {
    struct Fixture {
        const char* name;
        RationalConfig config;
        std::vector<double> numerator, denominator;
    };
    std::vector<Fixture> fixtures;
    fixtures.push_back({"pade11", {1, 1, 0, 1, 1e-8}, {1, 0.5}, {-0.5}});
    fixtures.push_back({"pade32", {3, 2, 0.1, 1.3, 1e-8}, {0.2, -0.1, 0.05, 0.3}, {0.1, -0.2}});
    fixtures.push_back({"poly30", {3, 0, -0.2, 0.5, 1e-8}, {1, 2, 3, 4}, {}});
    fixtures.push_back({"pole", {1, 1, 0, 1, 1e-8}, {1, -1}, {-1}});
    fixtures.push_back({"tiny", {2, 2, 0, 1e-150, 1e-6}, {1e-300, 1e-200, 1}, {1e-100, 1}});
    fixtures.push_back({"high", {16, 16, 0, 2, 1e-8}, std::vector<double>(17, 0.01), std::vector<double>(16, 0.02)});
    for (const auto& f : fixtures) {
        for (double x : inputs()) {
            char label[160];
            std::snprintf(label, sizeof label, "rational %s x=%a", f.name, x);
            guarded(label, [&] {
                const auto r = evaluate_rational(f.config, x, f.numerator, f.denominator);
                hex("r", std::vector<double>{r.value, r.input_derivative});
                hex("da", r.numerator_derivatives);
                hex("db", r.denominator_derivatives);
            });
        }
    }
}

Layer seeded(Layer layer, double phase) {
    std::vector<double> c(layer.coefficients().size()), b(layer.bias().size());
    for (std::size_t k = 0; k < c.size(); ++k) c[k] = 0.05 * std::sin(phase + 0.37 * static_cast<double>(k));
    for (std::size_t k = 0; k < b.size(); ++k) b[k] = 0.01 * static_cast<double>(k);
    if (api::is_rational(layer)) {
        std::vector<double> d(api::denominators(layer).size());
        for (std::size_t k = 0; k < d.size(); ++k) d[k] = 0.03 * std::cos(phase + static_cast<double>(k));
        api::set_rational(layer, c, d, b);
    } else {
        layer.set_parameters(c, b);
    }
    return layer;
}

void dump_network(const char* name, const Network& network, const std::vector<double>& x,
                  std::size_t batch, double l2) {
    guarded(std::string("resident ") + name, [&] {
        cuda::ResidentNetwork gpu(network, batch);
        const auto outputs = api::outputs(network);
        std::vector<double> upstream(batch * outputs);
        for (std::size_t k = 0; k < upstream.size(); ++k) upstream[k] = std::cos(0.7 * static_cast<double>(k));
        gpu.upload_input(x, batch);
        gpu.upload_output_gradient(upstream);
        for (int step = 0; step < 3; ++step) {
            gpu.forward();
            hex("y", gpu.download_output());
            gpu.backward(l2);
            const auto g = gpu.download_gradients();
            hex("dx", g.input);
            for (std::size_t j = 0; j < g.layers.size(); ++j) {
                const auto& layer = api::grad(g, j);
                hex("dc", layer.coefficients);
                hex("db", layer.bias);
                hex("dcen", api::centers(layer));
                hex("dlw", api::log_widths(layer));
                hex("dden", api::denominators(layer));
            }
            gpu.sgd(0.05);
        }
        const auto trained = gpu.download_parameters();
        for (const auto& layer : api::layers(trained)) {
            hex("c", layer.coefficients());
            hex("b", layer.bias());
            hex("den", api::denominators(layer));
            if (const auto* trainable = api::trainable(layer)) {
                hex("cen", trainable->centers);
                hex("lw", trainable->log_widths);
            }
        }
    });
}

struct NetworkFixture {
    const char* name;
    Network network;
    std::vector<double> x;
    std::size_t batch;
    double l2;
};

std::vector<NetworkFixture> network_fixtures() {
    RationalConfig rational{3, 2, 0.1, 1.3, 1e-8};
    std::vector<NetworkFixture> fixtures;
    Network mixed({
        seeded(Layer(3, 4, ChebyshevConfig{5}), 0.1),
        seeded(Layer(4, 3, spline(3, {-1, -1, -1, -1, -0.5, 0, 0, 0.5, 1, 1, 1, 1})), 0.2),
        seeded(Layer(3, 3, trainable_rbf({-1, 0, 0.8}, {-0.5, 0.1, -0.2})), 0.3),
        seeded(Layer(3, 2, rational), 0.4),
        seeded(Layer(2, 3, wavelet({0, 0.5}, {1, 0.25})), 0.5),
        seeded(Layer(3, 2, JacobiConfig{5, 0.5, -0.3}), 0.6),
        seeded(Layer(2, 2, fourier(5, 1.3)), 0.7),
        seeded(Layer(2, 2, rbf({-1, 0, 1}, 0.7)), 0.8),
        seeded(Layer(2, 1, HermiteConfig{4}), 0.9),
        seeded(Layer(1, 2, LegendreConfig{4}), 1.0),
    });
    std::vector<double> x(6 * 3);
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = 0.9 * std::sin(0.61 * static_cast<double>(k));
    x[0] = -1;
    x[1] = 1;
    x[2] = 0;
    fixtures.push_back({"mixed", std::move(mixed), x, 6, 0.1});
    fixtures.push_back({"endpoints",
                        Network({seeded(Layer(2, 2, JacobiConfig{6, 0.5, -0.3}), 0.2),
                                 seeded(Layer(2, 1, spline(2, {-1, -1, -1, 0, 1, 1, 1})), 0.3)}),
                        {-1, 1, 1, -1, 0.5, -0.5}, 3, 0});
    fixtures.push_back({"hermite_overflow", Network({seeded(Layer(1, 1, HermiteConfig{8}), 0.1)}), {1e300}, 1, 0});
    fixtures.push_back({"rational_pole", Network({[] {
                            Layer l(1, 1, RationalConfig{1, 1, 0, 1, 1e-8});
                            api::set_rational(l, std::vector<double>{1, -1}, std::vector<double>{-1},
                                              std::vector<double>{0});
                            return l;
                        }()}),
                        {1.0}, 1, 0});
    fixtures.push_back({"rbf_tiny_width", Network({seeded(Layer(1, 1, rbf({0, 1e-3}, 1e-160)), 0.1)}),
                        {1e-158}, 1, 0});
    return fixtures;
}

void dump_resident() {
    for (const auto& f : network_fixtures()) dump_network(f.name, f.network, f.x, f.batch, f.l2);
}

void dump_parameters(const Network& network) {
    for (const auto& layer : api::layers(network)) {
        hex("c", layer.coefficients());
        hex("b", layer.bias());
        hex("den", api::denominators(layer));
        if (const auto* trainable = api::trainable(layer)) {
            hex("cen", trainable->centers);
            hex("lw", trainable->log_widths);
        }
        if (const auto* s = api::spline(layer)) hex("knots", s->knots);
    }
}

// CPU Layer/Network execution on the resident fixtures: the same three
// forward/backward(+L2)/SGD steps, then the family operations.
void dump_layers() {
    for (auto& f : network_fixtures()) {
        guarded(std::string("cpu ") + f.name, [&] {
            auto network = f.network;
            const auto outputs = api::outputs(network);
            std::vector<double> upstream(f.batch * outputs);
            for (std::size_t k = 0; k < upstream.size(); ++k) upstream[k] = std::cos(0.7 * static_cast<double>(k));
            for (int step = 0; step < 3; ++step) {
                hex("y", network.forward(f.x, f.batch));
                auto g = network.backward(f.x, f.batch, upstream);
                const auto penalty = network.regularization(f.l2);
                hex("pen", std::vector<double>{penalty.value});
                hex("dx", g.input);
                for (std::size_t j = 0; j < g.layers.size(); ++j) {
                    auto& layer = api::grad(g, j);
                    for (std::size_t k = 0; k < layer.coefficients.size(); ++k)
                        layer.coefficients[k] += api::grad(penalty.gradients, j).coefficients[k];
                    hex("dc", layer.coefficients);
                    hex("db", layer.bias);
                    hex("dcen", api::centers(layer));
                    hex("dlw", api::log_widths(layer));
                    hex("dden", api::denominators(layer));
                }
                network.sgd(g, 0.05);
            }
            dump_parameters(network);
        });
    }
    const auto fixtures = network_fixtures();
    std::vector<Layer> mixed = api::layers(fixtures.front().network);
    const std::vector<double> probe{-0.9, -0.3, 0.2, 0.7, 0.95, -0.55, 0.1, 0.45};
    guarded("cpu spline refinement", [&] {
        auto& l = mixed[1];
        api::knot(l, 0.3);
        hex("knot", std::vector<double>{api::adapt(l, std::vector<double>{-0.8, -0.7, -0.6, -0.65, 0.9, 4})});
        hex("y", l.forward(probe, 2));
        dump_parameters(Network({l}));
    });
    guarded("cpu spline outside", [&] { api::knot(mixed[1], 5); });
    guarded("cpu spline empty", [&] { api::adapt(mixed[1], std::vector<double>{7}); });
    guarded("cpu knot on chebyshev", [&] { api::knot(mixed[0], 0.1); });
    guarded("cpu rbf setter", [&] {
        auto& l = mixed[2];
        api::set_rbf(l, std::vector<double>{-0.9, 0.1, 0.7}, std::vector<double>{-0.4, 0.2, -0.1});
        hex("y", l.forward(std::vector<double>{0.1, -0.2, 0.3}, 1));
        const auto g = l.backward(std::vector<double>{0.1, -0.2, 0.3}, 1, std::vector<double>{1, -1, 0.5});
        hex("dcen", api::centers(g));
        hex("dlw", api::log_widths(g));
    });
    guarded("cpu rbf count", [&] { api::set_rbf(mixed[2], std::vector<double>{0}, std::vector<double>{0}); });
    guarded("cpu rational shape", [&] {
        api::set_rational(mixed[3], std::vector<double>{1}, std::vector<double>{}, std::vector<double>{0, 0});
    });
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--layers") == 0) {
        dump_layers();
        return 0;
    }
    const bool cpu_only = argc > 1 && std::strcmp(argv[1], "--cpu") == 0;
    dump_basis();
    dump_rational();
    if (!cpu_only) dump_resident();
    return 0;
}

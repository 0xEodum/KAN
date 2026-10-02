// Backlog M1: resident CUDA execution of input maps, parity with the CPU
// Network over forward, backward (+L2), SGD and parameter download.
#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include "kan/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <cmath>
#include <limits>

namespace {
void compare(std::span<const double> a, std::span<const double> b, double tolerance = 1e-11) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) test::near(a[i], b[i], tolerance);
}
std::vector<double> wave(std::size_t n, double scale, double phase) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = scale * std::sin(phase + 0.83 * static_cast<double>(k));
    return v;
}
kan::Layer seeded(kan::Layer layer, double phase) {
    layer.set_parameters(wave(layer.coefficients().size(), 0.2, phase), wave(layer.bias().size(), 0.05, phase + 1));
    return layer;
}
kan::LayerNormMap layer_norm(std::size_t features, bool affine) {
    kan::LayerNormMap m{1e-3, {}, {}};
    if (affine) { m.gain = wave(features, 0.3, 0.2); for (auto& g : m.gain) g += 1; m.bias = wave(features, 0.2, 0.9); }
    return m;
}
kan::Layer rational(std::size_t in, std::size_t out) {
    kan::Layer l(in, out, kan::RationalConfig{2, 1, 0.1, 1.3, 1e-8});
    kan::set_rational_parameters(l, wave(l.coefficients().size(), 0.3, 0.2), wave(in * out, 0.1, 0.7), wave(out, 0.05, 0.1));
    return l;
}
std::size_t outputs(const kan::Network& n) { return n.outputs(); }

void compare_gradients(const kan::NetworkGradients& gpu, const kan::NetworkGradients& cpu) {
    compare(gpu.input, cpu.input);
    REQUIRE(gpu.layers.size() == cpu.layers.size());
    for (std::size_t j = 0; j < gpu.layers.size(); ++j) {
        REQUIRE(gpu.layers[j].index() == cpu.layers[j].index());
        if (const auto* a = std::get_if<kan::InputMapGradients>(&gpu.layers[j])) {
            const auto& e = std::get<kan::InputMapGradients>(cpu.layers[j]);
            compare(a->input, e.input); compare(a->gain, e.gain); compare(a->bias, e.bias);
        } else {
            const auto& a2 = std::get<kan::LayerGradients>(gpu.layers[j]);
            const auto& e = std::get<kan::LayerGradients>(cpu.layers[j]);
            compare(a2.input, e.input); compare(a2.coefficients, e.coefficients); compare(a2.bias, e.bias);
        }
    }
}
void compare_parameters(const kan::Network& gpu, const kan::Network& cpu) {
    REQUIRE(gpu.layers().size() == cpu.layers().size());
    for (std::size_t j = 0; j < gpu.layers().size(); ++j) {
        REQUIRE(gpu.layers()[j].index() == cpu.layers()[j].index());
        if (const auto* m = std::get_if<kan::InputMap>(&gpu.layers()[j])) {
            const auto& e = test::input_map(cpu, j);
            REQUIRE(m->map().index() == e.map().index());
            if (const auto* ln = std::get_if<kan::LayerNormMap>(&m->map())) {
                const auto& eln = std::get<kan::LayerNormMap>(e.map());
                REQUIRE(ln->epsilon == eln.epsilon);
                compare(ln->gain, eln.gain); compare(ln->bias, eln.bias);
            } else {
                REQUIRE(m->map() == e.map()); // fixed maps are returned unchanged
            }
        } else {
            compare(test::layer(gpu, j).coefficients(), test::layer(cpu, j).coefficients());
            compare(test::layer(gpu, j).bias(), test::layer(cpu, j).bias());
        }
    }
}
// Three forward/backward(+L2)/SGD steps on both backends.
void parity(kan::Network cpu, std::size_t batch, double scale, double l2) {
    const auto x = wave(batch * cpu.inputs(), scale, 0.3);
    const auto upstream = wave(batch * outputs(cpu), 1, 0.6);
    kan::cuda::ResidentNetwork gpu(cpu, batch);
    gpu.upload_input(x, batch);
    gpu.upload_output_gradient(upstream);
    for (int step = 0; step < 3; ++step) {
        gpu.forward();
        compare(gpu.download_output(), cpu.forward(x, batch));
        gpu.backward(l2);
        auto expected = cpu.backward(x, batch, upstream);
        const auto penalty = cpu.regularization(l2).gradients;
        for (std::size_t j = 0; j < expected.layers.size(); ++j)
            if (auto* g = std::get_if<kan::LayerGradients>(&expected.layers[j]))
                for (std::size_t k = 0; k < g->coefficients.size(); ++k) g->coefficients[k] += test::grad(penalty, j).coefficients[k];
        compare_gradients(gpu.download_gradients(), expected);
        gpu.sgd(0.05);
        cpu.sgd(expected, 0.05);
        compare_parameters(gpu.download_parameters(), cpu);
    }
    REQUIRE(gpu.workspace_allocations() == 2);
}
} // namespace

TEST(resident_maps_match_cpu_in_front_of_every_carrier) {
    parity(kan::Network({kan::InputMap(3, kan::AffineMap{{0.02, -0.01, 0.03}, {0.1, 0, -0.2}}),
                         seeded(kan::Layer(3, 4, kan::ChebyshevConfig{5}), 0.1),
                         kan::InputMap(4, layer_norm(4, true)),
                         seeded(kan::Layer(4, 2, kan::BSplineConfig{2, {-3, -3, -3, -1, 0, 1, 3, 3, 3}}), 0.4)}),
           6, 40, 0.1);
    parity(kan::Network({kan::InputMap(2, kan::TanhMap{0.05}),
                         seeded(kan::Layer(2, 3, kan::TrainableRbfConfig{{-0.5, 0, 0.6}, {-0.3, 0, -0.1}}), 0.3),
                         kan::InputMap(3, layer_norm(3, false)),
                         rational(3, 1)}),
           5, 30, 0.0);
    parity(kan::Network({kan::InputMap(2, layer_norm(2, true)),
                         seeded(kan::Layer(2, 2, kan::MexicanHatConfig{{-1, 0, 1}, {0.8, 1, 0.6}}), 0.7),
                         kan::InputMap(2, kan::TanhMap{1.5})}),
           4, 20, 0.2);
}

// Wide rows (several lanes per feature, several feature blocks) and a batch
// spanning many reduction tiles; also a capacity larger than the batch.
TEST(resident_layer_norm_wide_rows_and_many_tiles) {
    parity(kan::Network({kan::InputMap(70, layer_norm(70, true)), seeded(kan::Layer(70, 3, kan::LegendreConfig{3}), 0.2)}),
           1000, 5, 0.0);
    parity(kan::Network({kan::InputMap(33, layer_norm(33, false)), seeded(kan::Layer(33, 1, kan::HermiteConfig{3}), 0.5)}),
           65, 3, 0.0);
    // 32 lanes per row (300 features) and a batch that leaves a partial warp.
    parity(kan::Network({kan::InputMap(300, layer_norm(300, true)), seeded(kan::Layer(300, 1, kan::ChebyshevConfig{2}), 0.5)}),
           37, 2, 0.0);
    // One lane per row (2 features): 32 rows per warp.
    parity(kan::Network({kan::InputMap(2, layer_norm(2, true)), seeded(kan::Layer(2, 1, kan::ChebyshevConfig{3}), 0.5)}),
           45, 1, 0.0);
}

TEST(resident_map_only_network_and_batch_zero) {
    kan::Network cpu({kan::InputMap(3, layer_norm(3, true)), kan::InputMap(3, kan::TanhMap{0.5})});
    kan::cuda::ResidentNetwork gpu(cpu, 4);
    gpu.upload_input({}, 0);
    gpu.upload_output_gradient({});
    gpu.forward();
    REQUIRE(gpu.download_output().empty());
    gpu.backward();
    const auto g = gpu.download_gradients();
    REQUIRE(g.input.empty());
    REQUIRE(test::map_grad(g, 0).gain == std::vector<double>(3, 0.0));
    REQUIRE(test::map_grad(g, 0).bias == std::vector<double>(3, 0.0));
    REQUIRE(test::map_grad(g, 1).gain.empty());
    gpu.sgd(0.1);
    compare_parameters(gpu.download_parameters(), cpu);
    // Smaller batch than capacity.
    const auto x = wave(6, 2, 0.1), u = wave(6, 1, 0.4);
    gpu.upload_input(x, 2); gpu.upload_output_gradient(u); gpu.forward(); gpu.backward();
    compare(gpu.download_output(), cpu.forward(x, 2));
    compare_gradients(gpu.download_gradients(), cpu.backward(x, 2, u));
}

TEST(resident_map_overflow_is_reported) {
    kan::Network affine({kan::InputMap(1, kan::AffineMap{{1e300}, {0}}), kan::Layer(1, 1, kan::ChebyshevConfig{2})});
    kan::cuda::ResidentNetwork gpu(affine, 1);
    gpu.upload_input(std::vector<double>{1e100}, 1);
    test::throws<std::overflow_error>([&] { gpu.forward(); });
    test::throws<std::logic_error>([&] { gpu.download_output(); });
    kan::Network norm({kan::InputMap(2, kan::LayerNormMap{})});
    kan::cuda::ResidentNetwork other(norm, 1);
    other.upload_input(std::vector<double>{1e300, -1e300}, 1);
    test::throws<std::overflow_error>([&] { other.forward(); });
    other.upload_input(std::vector<double>{1, -1}, 1);
    other.forward();
    compare(other.download_output(), norm.forward(std::vector<double>{1, -1}, 1));
}

int main() { return test::run(); }

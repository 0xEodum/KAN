// Bitwise golden dump for behaviour-preserving backlog refactors (R1-R3).
// Prints every CPU basis/rational result and every resident CUDA result as
// exact hex doubles, including guard exceptions. Two builds of the library
// are equivalent for these fixtures only if their dumps are byte-identical.
#include "kan/resident.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kan;

namespace {
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

BasisConfig polynomial(BasisKind kind, std::size_t size, double alpha = 0, double beta = 0) {
    BasisConfig c;
    c.kind = kind;
    c.size = size;
    c.alpha = alpha;
    c.beta = beta;
    return c;
}

BasisConfig fourier(std::size_t size, double frequency) {
    BasisConfig c;
    c.kind = BasisKind::Fourier;
    c.size = size;
    c.frequency = frequency;
    return c;
}

BasisConfig rbf(std::vector<double> centers, double width) {
    BasisConfig c;
    c.kind = BasisKind::GaussianRbf;
    c.size = centers.size();
    c.centers = std::move(centers);
    c.width = width;
    return c;
}

BasisConfig trainable_rbf(std::vector<double> centers, std::vector<double> log_widths) {
    BasisConfig c;
    c.kind = BasisKind::GaussianRbf;
    c.size = centers.size();
    c.centers = std::move(centers);
    c.trainable_rbf = true;
    c.log_widths = std::move(log_widths);
    return c;
}

BasisConfig wavelet(std::vector<double> centers, std::vector<double> scales) {
    BasisConfig c;
    c.kind = BasisKind::MexicanHat;
    c.size = centers.size();
    c.centers = std::move(centers);
    c.scales = std::move(scales);
    return c;
}

BasisConfig spline(std::size_t degree, std::vector<double> knots) {
    BasisConfig c;
    c.kind = BasisKind::BSpline;
    c.degree = degree;
    c.size = knots.size() - degree - 1;
    c.knots = std::move(knots);
    return c;
}

std::vector<std::pair<std::string, BasisConfig>> basis_fixtures() {
    return {
        {"chebyshev7", polynomial(BasisKind::Chebyshev, 7)},
        {"chebyshev1", polynomial(BasisKind::Chebyshev, 1)},
        {"legendre6", polynomial(BasisKind::Legendre, 6)},
        {"hermite6", polynomial(BasisKind::Hermite, 6)},
        {"jacobi_asym", polynomial(BasisKind::Jacobi, 6, 0.5, -0.3)},
        {"jacobi_sum_m1", polynomial(BasisKind::Jacobi, 5, -0.5, -0.5)},
        {"jacobi_near_bound", polynomial(BasisKind::Jacobi, 5, -0.999999, 3.5)},
        {"jacobi_huge", polynomial(BasisKind::Jacobi, 4, 1e300, 1e300)},
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
    if (layer.is_rational()) {
        std::vector<double> d(layer.denominators().size());
        for (std::size_t k = 0; k < d.size(); ++k) d[k] = 0.03 * std::cos(phase + static_cast<double>(k));
        layer.set_rational_parameters(c, d, b);
    } else {
        layer.set_parameters(c, b);
    }
    return layer;
}

void dump_network(const char* name, const Network& network, const std::vector<double>& x,
                  std::size_t batch, double l2) {
    guarded(std::string("resident ") + name, [&] {
        cuda::ResidentNetwork gpu(network, batch);
        const auto outputs = network.layers().back().outputs();
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
            for (const auto& layer : g.layers) {
                hex("dc", layer.coefficients);
                hex("db", layer.bias);
                hex("dcen", layer.centers);
                hex("dlw", layer.log_widths);
                hex("dden", layer.denominators);
            }
            gpu.sgd(0.05);
        }
        const auto trained = gpu.download_parameters();
        for (const auto& layer : trained.layers()) {
            hex("c", layer.coefficients());
            hex("b", layer.bias());
            hex("den", layer.denominators());
            if (!layer.is_rational() && layer.basis().trainable_rbf) {
                hex("cen", layer.basis().centers);
                hex("lw", layer.basis().log_widths);
            }
        }
    });
}

void dump_resident() {
    RationalConfig rational{3, 2, 0.1, 1.3, 1e-8};
    const Network mixed({
        seeded(Layer(3, 4, polynomial(BasisKind::Chebyshev, 5)), 0.1),
        seeded(Layer(4, 3, spline(3, {-1, -1, -1, -1, -0.5, 0, 0, 0.5, 1, 1, 1, 1})), 0.2),
        seeded(Layer(3, 3, trainable_rbf({-1, 0, 0.8}, {-0.5, 0.1, -0.2})), 0.3),
        seeded(Layer(3, 2, rational), 0.4),
        seeded(Layer(2, 3, wavelet({0, 0.5}, {1, 0.25})), 0.5),
        seeded(Layer(3, 2, polynomial(BasisKind::Jacobi, 5, 0.5, -0.3)), 0.6),
        seeded(Layer(2, 2, fourier(5, 1.3)), 0.7),
        seeded(Layer(2, 2, rbf({-1, 0, 1}, 0.7)), 0.8),
        seeded(Layer(2, 1, polynomial(BasisKind::Hermite, 4)), 0.9),
        seeded(Layer(1, 2, polynomial(BasisKind::Legendre, 4)), 1.0),
    });
    std::vector<double> x(6 * 3);
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = 0.9 * std::sin(0.61 * static_cast<double>(k));
    x[0] = -1;
    x[1] = 1;
    x[2] = 0;
    dump_network("mixed", mixed, x, 6, 0.1);

    const Network endpoints({seeded(Layer(2, 2, polynomial(BasisKind::Jacobi, 6, 0.5, -0.3)), 0.2),
                             seeded(Layer(2, 1, spline(2, {-1, -1, -1, 0, 1, 1, 1})), 0.3)});
    dump_network("endpoints", endpoints, {-1, 1, 1, -1, 0.5, -0.5}, 3, 0);
    dump_network("hermite_overflow", Network({seeded(Layer(1, 1, polynomial(BasisKind::Hermite, 8)), 0.1)}),
                 {1e300}, 1, 0);
    dump_network("rational_pole", Network({[] {
                     Layer l(1, 1, RationalConfig{1, 1, 0, 1, 1e-8});
                     l.set_rational_parameters(std::vector<double>{1, -1}, std::vector<double>{-1},
                                               std::vector<double>{0});
                     return l;
                 }()}),
                 {1.0}, 1, 0);
    dump_network("rbf_tiny_width", Network({seeded(Layer(1, 1, rbf({0, 1e-3}, 1e-160)), 0.1)}),
                 {1e-158}, 1, 0);
}
} // namespace

int main(int argc, char** argv) {
    const bool cpu_only = argc > 1 && std::strcmp(argv[1], "--cpu") == 0;
    dump_basis();
    dump_rational();
    if (!cpu_only) dump_resident();
    return 0;
}

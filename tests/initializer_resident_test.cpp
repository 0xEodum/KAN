// Backlog M4: initialized networks train on the resident executor through the
// existing construction and upload_parameters paths (no resident changes).
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cstdio>

namespace {
struct Data { std::vector<double> x, target; std::size_t batch = 64; };
Data product_data() {
    Data d;
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j) {
            const double a = -0.9 + 1.8 * double(i) / 7, b = -0.9 + 1.8 * double(j) / 7;
            d.x.insert(d.x.end(), {a, b});
            d.target.push_back(a * b);
        }
    return d;
}

template<class Config>
kan::Network deep(const Config& config) {
    std::vector<kan::NetworkLayer> stages;
    std::size_t in = 2;
    for (std::size_t l = 0; l < 5; ++l) {
        const std::size_t out = l == 4 ? 1 : 8;
        stages.emplace_back(kan::InputMap(in, kan::TanhMap{1.0}));
        stages.emplace_back(kan::Layer(in, out, config));
        in = out;
    }
    return kan::Network(std::move(stages));
}

std::vector<double> upstream(const std::vector<double>& y, const Data& d, double* loss) {
    std::vector<double> u(d.batch);
    *loss = 0;
    for (std::size_t b = 0; b < d.batch; ++b) {
        u[b] = 2 * (y[b] - d.target[b]) / double(d.batch);
        *loss += (y[b] - d.target[b]) * (y[b] - d.target[b]) / double(d.batch);
    }
    return u;
}

void close(std::span<const double> a, std::span<const double> e, double tolerance) {
    REQUIRE(a.size() == e.size());
    double scale = 0;
    for (double v : e) scale = std::max(scale, std::abs(v));
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - e[i]) > tolerance * (std::abs(e[i]) + scale))
            throw std::runtime_error("resident/CPU mismatch at " + std::to_string(i));
}

void compare_parameters(const kan::Network& gpu, const kan::Network& cpu, double tolerance) {
    for (std::size_t p = 0; p < cpu.layers().size(); ++p) {
        if (!std::holds_alternative<kan::Layer>(cpu.layers()[p])) continue;
        const auto& g = test::layer(gpu, p);
        const auto& c = test::layer(cpu, p);
        close(g.coefficients(), c.coefficients(), tolerance);
        close(g.bias(), c.bias(), tolerance);
        if (const auto* r = std::get_if<kan::RationalEdges>(&c.carrier()))
            close(std::get<kan::RationalEdges>(g.carrier()).denominators, r->denominators, tolerance);
    }
}

template<class Config>
void resident_matches_cpu_training(const Config& config, const char* name) {
    if (!kan::cuda::available()) { std::puts("no CUDA device: skipped"); return; }
    const auto d = product_data();
    auto cpu = deep(config);
    kan::initialize(cpu, kan::VarianceScaling{1, kan::Distribution::Normal, 4, {}});
    kan::cuda::ResidentNetwork gpu(cpu, d.batch);
    gpu.upload_input(d.x, d.batch);
    double first = 0, last = 0;
    for (int step = 0; step < 200; ++step) {
        double cpu_loss = 0, gpu_loss = 0;
        const auto u = upstream(cpu.forward(d.x, d.batch), d, &cpu_loss);
        gpu.forward();
        const auto gu = upstream(gpu.download_output(), d, &gpu_loss);
        test::near(gpu_loss, cpu_loss, 1e-8);
        gpu.upload_output_gradient(gu);
        gpu.backward();
        gpu.sgd(0.1);
        cpu.sgd(cpu.backward(d.x, d.batch, u), 0.1);
        if (step == 0) first = cpu_loss;
        last = cpu_loss;
    }
    std::printf("resident %s: loss %.3e -> %.3e\n", name, first, last);
    REQUIRE(last < 0.5 * first);
    compare_parameters(gpu.download_parameters(), cpu, 1e-7);
    // A re-initialized model is uploaded into the same executor.
    auto other = deep(config);
    kan::initialize(other, kan::NoiseInit{0.3, kan::Distribution::Uniform, 9, {}});
    const auto allocations = gpu.workspace_allocations();
    gpu.upload_parameters(other);
    REQUIRE(gpu.workspace_allocations() == allocations);
    gpu.forward();
    close(gpu.download_output(), other.forward(d.x, d.batch), 1e-11);
    compare_parameters(gpu.download_parameters(), other, 0);
}
} // namespace

TEST(resident_chebyshev_from_variance_scaling) { resident_matches_cpu_training(kan::ChebyshevConfig{5}, "chebyshev"); }
TEST(resident_bspline_from_variance_scaling) {
    resident_matches_cpu_training(kan::BSplineConfig{3, {-1, -1, -1, -1, -0.5, 0, 0.5, 1, 1, 1, 1}}, "bspline");
}
TEST(resident_rational_policies_from_variance_scaling) {
    for (auto policy : {kan::DenominatorPolicy::Guarded, kan::DenominatorPolicy::Absolute, kan::DenominatorPolicy::Smooth}) {
        kan::RationalConfig c;
        c.denominator_policy = policy;
        resident_matches_cpu_training(c, "rational");
    }
}

int main() { return test::run(); }

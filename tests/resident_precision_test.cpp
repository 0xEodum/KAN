// Backlog C1: resident precision policy. The FP64 executor stays the default
// and the parity reference; the opt-in FP32 executor stores parameters and
// activations, evaluates every basis/rational/map formula and runs the
// contractions in single precision. Its results are compared with the FP64 CPU
// reference within the FP32 tolerance defined in docs/evidence/backlog/C1.md.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

// FP32 tolerance against the FP64 reference, per entry: a relative part, a
// floor relative to the tensor's largest magnitude (cancellation in long
// reductions makes small entries carry absolute error) and the FP32 normal
// range (values below it are flushed or subnormal in single precision).
constexpr double fp32_relative = 2e-4, fp32_floor = 2e-5, fp32_underflow = 1e-37;

template<class... Parts> std::string precise(const Parts&... parts) {
    std::ostringstream out;
    out << std::setprecision(9);
    (out << ... << parts);
    return out.str();
}
void close(std::span<const double> actual, std::span<const double> expected,
           double relative = fp32_relative, double floor = fp32_floor) {
    REQUIRE(actual.size() == expected.size());
    double scale = 0;
    for (double e : expected) scale = std::max(scale, std::abs(e));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::isfinite(actual[i]));
        if (std::abs(actual[i]-expected[i]) > relative*std::abs(expected[i]) + floor*scale + fp32_underflow)
            throw std::runtime_error(precise("index ", i, " of ", actual.size(), " actual=", actual[i],
                                             " expected=", expected[i], " scale=", scale));
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
kan::Layer layer(std::size_t in, std::size_t out, kan::BasisConfig basis, double phase) {
    return seeded(kan::Layer(in, out, std::move(basis)), phase);
}
kan::Layer rational(kan::DenominatorPolicy policy, std::size_t in, std::size_t out, std::size_t m, std::size_t n) {
    kan::RationalConfig r;
    r.numerator_degree = m; r.denominator_degree = n; r.center = 0.1; r.scale = 1.3; r.denominator_policy = policy;
    kan::Layer l(in, out, r);
    const auto a = wave(l.coefficients().size(), 0.3, 1.0, 1.0);
    const auto b = wave(test::denominators(l).size(), 0.5, 2.0, 2.6);
    kan::set_rational_parameters(l, a, b, std::vector<double>(out, 0.01));
    return l;
}
// Every basis family, with ordinary parameters.
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

void gradients(const kan::NetworkGradients& actual, const kan::NetworkGradients& expected) {
    close(actual.input, expected.input);
    REQUIRE(actual.layers.size() == expected.layers.size());
    for (std::size_t j = 0; j < actual.layers.size(); ++j) {
        if (std::holds_alternative<kan::InputMapGradients>(expected.layers[j])) {
            close(test::map_grad(actual, j).input, test::map_grad(expected, j).input);
            close(test::map_grad(actual, j).gain, test::map_grad(expected, j).gain);
            close(test::map_grad(actual, j).bias, test::map_grad(expected, j).bias);
            continue;
        }
        close(test::grad(actual, j).input, test::grad(expected, j).input);
        close(test::grad(actual, j).coefficients, test::grad(expected, j).coefficients);
        close(test::grad(actual, j).bias, test::grad(expected, j).bias);
        close(test::centers(test::grad(actual, j)), test::centers(test::grad(expected, j)));
        close(test::log_widths(test::grad(actual, j)), test::log_widths(test::grad(expected, j)));
        close(test::denominators(test::grad(actual, j)), test::denominators(test::grad(expected, j)));
    }
}
void parameters(const kan::Network& actual, const kan::Network& expected) {
    for (std::size_t j = 0; j < expected.layers().size(); ++j) {
        if (std::holds_alternative<kan::InputMap>(expected.layers()[j])) {
            REQUIRE(std::holds_alternative<kan::InputMap>(actual.layers()[j]));
            const auto* a = std::get_if<kan::LayerNormMap>(&test::input_map(actual, j).map());
            const auto* e = std::get_if<kan::LayerNormMap>(&test::input_map(expected, j).map());
            REQUIRE((a == nullptr) == (e == nullptr));
            if (e) { close(a->gain, e->gain); close(a->bias, e->bias); }
            continue;
        }
        close(test::layer(actual, j).coefficients(), test::layer(expected, j).coefficients());
        close(test::layer(actual, j).bias(), test::layer(expected, j).bias());
        close(test::denominators(test::layer(actual, j)), test::denominators(test::layer(expected, j)));
        if (const auto* e = std::get_if<kan::TrainableRbfEdges>(&test::layer(expected, j).carrier())) {
            const auto& a = test::trainable(test::layer(actual, j));
            close(a.centers, e->basis.centers); close(a.log_widths, e->basis.log_widths);
        }
    }
}
// Forward, backward with L2, SGD for `steps` steps against the FP64 CPU.
void train_like_cpu(kan::Network cpu, std::size_t batch, std::size_t capacity, double lambda, int steps,
                    double rate = 0.05) {
    ResidentNetwork gpu(cpu, capacity, Precision::Float32);
    REQUIRE(gpu.precision() == Precision::Float32);
    const auto allocations = gpu.workspace_allocations();
    const auto x = wave(batch*cpu.inputs(), 0.9, 0.37), dy = wave(batch*cpu.outputs(), 0.2, 0.53, 1);
    gpu.upload_input(x, batch); gpu.upload_output_gradient(dy);
    for (int step = 0; step < steps; ++step) {
        gpu.forward(); close(gpu.download_output(), cpu.forward(x, batch));
        gpu.backward(lambda);
        auto expected = cpu.backward(x, batch, dy);
        const auto penalty = cpu.regularization(lambda).gradients;
        for (std::size_t j = 0; j < expected.layers.size(); ++j)
            if (auto* g = std::get_if<kan::LayerGradients>(&expected.layers[j]))
                for (std::size_t k = 0; k < g->coefficients.size(); ++k) g->coefficients[k] += test::grad(penalty, j).coefficients[k];
        gradients(gpu.download_gradients(), expected);
        gpu.sgd(rate); cpu.sgd(expected, rate);
    }
    parameters(gpu.download_parameters(), cpu);
    REQUIRE(gpu.workspace_allocations() == allocations);
}
} // namespace

TEST(default_precision_is_float64) {
    kan::Network net({layer(2, 1, kan::ChebyshevConfig{3}, 0.1)});
    REQUIRE(ResidentNetwork(net, 4).precision() == Precision::Float64);
    REQUIRE(ResidentNetwork(net, 4, Precision::Float64).precision() == Precision::Float64);
    REQUIRE(ResidentNetwork(net, 4, Precision::Float32).precision() == Precision::Float32);
    test::throws<std::invalid_argument>([&] { ResidentNetwork(net, 4, static_cast<Precision>(77)); });
    // The FP64 executor still accepts data beyond the FP32 range.
    ResidentNetwork fp64(net, 1);
    fp64.upload_input(std::vector<double>{1e39, 0.0}, 1);
}

// All nine basis families between input maps (affine, trainable LayerNorm,
// tanh), so every shared host/device formula runs in single precision.
TEST(float32_every_family_and_input_map_trains_like_fp64_cpu) {
    for (auto kind : families) {
        kan::Network cpu({kan::InputMap(3, kan::AffineMap{{0.9, -1.1, 0.7}, {0.05, 0.1, -0.1}}),
                          layer(3, 4, family(kind), 0.2),
                          kan::InputMap(4, kan::LayerNormMap{1e-3, {1.1, 0.9, 1.0, 1.2}, {0.1, -0.1, 0.0, 0.05}}),
                          layer(4, 2, kan::ChebyshevConfig{4}, 0.4),
                          kan::InputMap(2, kan::TanhMap{0.8})});
        train_like_cpu(std::move(cpu), 9, 12, 0.05, 3);
    }
}

// High-degree three-term recurrences (including Jacobi with alpha + beta
// close to -1 and Hermite growth) on the whole [-1, 1] domain.
TEST(float32_high_degree_recurrences_match_fp64_cpu) {
    for (kan::BasisConfig basis : {kan::BasisConfig{kan::ChebyshevConfig{12}}, kan::BasisConfig{kan::LegendreConfig{12}},
                                   kan::BasisConfig{kan::JacobiConfig{10, 2.5, -0.7}},
                                   kan::BasisConfig{kan::JacobiConfig{9, -0.4, -0.55}},
                                   kan::BasisConfig{kan::HermiteConfig{8}}}) {
        kan::Network cpu({layer(3, 2, basis, 0.3)});
        train_like_cpu(std::move(cpu), 33, 33, 0.0, 2, 0.01);
    }
}

TEST(float32_rational_policies_match_fp64_cpu) {
    for (auto policy : {kan::DenominatorPolicy::Guarded, kan::DenominatorPolicy::Absolute, kan::DenominatorPolicy::Smooth})
        for (const auto orders : {std::pair<std::size_t, std::size_t>{0, 0}, {0, 3}, {4, 1}, {3, 2}}) {
            kan::Network cpu({rational(policy, 2, 3, orders.first, orders.second), layer(3, 2, kan::ChebyshevConfig{3}, 0.5),
                              rational(policy, 2, 1, 3, 2)});
            train_like_cpu(std::move(cpu), 7, 8, 0.1, 3, 0.03);
        }
}

// The contraction engine paths in FP32: small warp kernels and cuBLAS SGEMM
// for the forward and the parameter VJP, several expansion layers sharing
// scratch, batch below capacity.
TEST(float32_contraction_paths_match_fp64_cpu) {
    kan::TrainableRbfConfig rbf{{-1, -0.6, -0.2, 0.2, 0.6, 1}, {-1, -0.9, -0.8, -0.8, -0.9, -1}};
    train_like_cpu(kan::Network({layer(37, 23, kan::ChebyshevConfig{7}, 0.1), layer(23, 41, rbf, 0.2),
                                 layer(41, 5, kan::BSplineConfig{3, {-2,-2,-2,-2,-1,0,0.5,1,2,2,2,2}}, 0.3),
                                 layer(5, 3, kan::FourierConfig{5, 1.1}, 0.4)}), 129, 200, 0.05, 3, 0.2);
    for (std::size_t batch : {1000u, 400u, 7u})
        train_like_cpu(kan::Network({layer(64, 80, kan::ChebyshevConfig{7}, 0.7),
                                     layer(80, 3, kan::JacobiConfig{5, 0.5, -0.25}, 0.8)}), batch, 1000, 0.01, 2, 0.01);
}

// Results beyond the FP32 range are nonfinite in single precision and are
// reported like any nonfinite result; the executor then recovers.
TEST(float32_reports_overflow_beyond_float_range) {
    kan::Layer l(1, 1, kan::ChebyshevConfig{2});
    l.set_parameters(std::vector<double>{0.0, 1e38}, std::vector<double>{0.0});
    kan::Network net({l});
    ResidentNetwork gpu(net, 2, Precision::Float32);
    gpu.upload_input(std::vector<double>{8.0}, 1); // y = 8e38: finite in FP64 only
    test::throws<std::overflow_error>([&] { gpu.forward(); });
    test::throws<std::logic_error>([&] { gpu.download_output(); });
    ResidentNetwork fp64(net, 2);
    fp64.upload_input(std::vector<double>{8.0}, 1); fp64.forward();
    test::near(fp64.download_output()[0], 8e38, 1e-15);
    gpu.upload_input(std::vector<double>{0.5}, 1); gpu.forward();
    close(gpu.download_output(), std::vector<double>{5e37});
    // Input VJP 1e38 * 4 overflows FP32.
    gpu.upload_output_gradient(std::vector<double>{4.0});
    test::throws<std::overflow_error>([&] { gpu.backward(); });
    // At x = 3: y = 3e38 and dx = -1e38 are finite, the coefficient VJP is -3;
    // the SGD candidate 1e38 + 1e38*3 is not: rejected, nothing committed.
    gpu.upload_input(std::vector<double>{3.0}, 1); gpu.forward();
    gpu.upload_output_gradient(std::vector<double>{-1.0}); gpu.backward();
    const auto before = gpu.download_parameters();
    test::throws<std::overflow_error>([&] { gpu.sgd(1e38); });
    REQUIRE(test::layer(gpu.download_parameters(), 0).coefficients() == test::layer(before, 0).coefficients());
    // A learning rate or L2 weight outside the FP32 range (or rounding to a
    // zero rate) cannot be applied in single precision.
    test::throws<std::invalid_argument>([&] { gpu.sgd(1e39); });
    test::throws<std::invalid_argument>([&] { gpu.sgd(1e-50); });
    test::throws<std::invalid_argument>([&] { gpu.backward(1e39); });
    gpu.sgd(1e-3);
}

// Finite FP64 data and configurations that FP32 cannot represent are rejected
// when they are uploaded (std::invalid_argument), before any execution.
TEST(float32_rejects_unrepresentable_data_and_configuration) {
    kan::Network net({layer(2, 1, kan::ChebyshevConfig{3}, 0.1)});
    ResidentNetwork gpu(net, 2, Precision::Float32);
    test::throws<std::invalid_argument>([&] { gpu.upload_input(std::vector<double>{1e39, 0.0}, 1); });
    gpu.upload_input(std::vector<double>{0.5, 1e-50}, 1); // underflow to zero is rounding, not an error
    test::throws<std::invalid_argument>([&] { gpu.upload_output_gradient(std::vector<double>{-1e40}); });
    gpu.forward(); gpu.upload_output_gradient(std::vector<double>{1.0}); gpu.backward();

    auto reject = [](kan::Network n) {
        test::throws<std::invalid_argument>([&] { ResidentNetwork(n, 2, Precision::Float32); });
        ResidentNetwork fp64(n, 2); // valid FP64 configuration
    };
    kan::Layer big(1, 1, kan::ChebyshevConfig{2});
    big.set_parameters(std::vector<double>{1e39, 0.0}, std::vector<double>{0.0});
    reject(kan::Network({big}));
    reject(kan::Network({layer(1, 1, kan::JacobiConfig{4, -1 + 1e-12, 0.5}, 0.1)}));
    reject(kan::Network({layer(1, 1, kan::FourierConfig{3, 1e-50}, 0.1)}));
    reject(kan::Network({layer(1, 1, kan::GaussianRbfConfig{{0.0, 1.0}, 1e-50}, 0.1)}));
    reject(kan::Network({layer(1, 1, kan::GaussianRbfConfig{{0.0, 1e39}, 1.0}, 0.1)}));
    reject(kan::Network({layer(1, 1, kan::TrainableRbfConfig{{0.0, 1.0}, {100.0, 0.0}}, 0.1)}));
    reject(kan::Network({layer(1, 1, kan::MexicanHatConfig{{0.0}, {1e-50}}, 0.1)}));
    // Distinct knots that round to one FP32 value would change the spline.
    reject(kan::Network({layer(1, 1, kan::BSplineConfig{3, {-1, -1, -1, -1, 0.5, 0.5 + 1e-12, 1, 1, 1, 1}}, 0.1)}));
    kan::RationalConfig tiny_scale; tiny_scale.scale = 1e-50;
    reject(kan::Network({kan::Layer(1, 1, tiny_scale)}));
    reject(kan::Network({kan::InputMap(2, kan::AffineMap{{1.0, 1e-50}, {0.0, 0.0}}), layer(2, 1, kan::ChebyshevConfig{3}, 0.1)}));
    reject(kan::Network({kan::InputMap(2, kan::LayerNormMap{1e-50, {}, {}}), layer(2, 1, kan::ChebyshevConfig{3}, 0.1)}));
    reject(kan::Network({kan::InputMap(2, kan::TanhMap{1e-50}), layer(2, 1, kan::ChebyshevConfig{3}, 0.1)}));
}

// Log-space paths in single precision: a Gaussian whose value underflows FP32
// keeps a representable input derivative, and a Mexican hat whose
// normalization/scale intermediate overflows FP32 keeps a representable one.
TEST(float32_log_space_tails_keep_representable_derivatives) {
    for (kan::BasisConfig basis : {kan::BasisConfig{kan::GaussianRbfConfig{{0.0}, 1e-30}},
                                   kan::BasisConfig{kan::TrainableRbfConfig{{0.0}, {std::log(1e-30)}}},
                                   kan::BasisConfig{kan::MexicanHatConfig{{0.0}, {1e-30}}}}) {
        kan::Layer l(1, 1, basis);
        l.set_parameters(std::vector<double>{1.0}, std::vector<double>{0.0});
        kan::Network cpu({l});
        ResidentNetwork gpu(cpu, 1, Precision::Float32);
        const std::vector<double> x{1.1e-29}, dy{1.0}; // q = 11
        gpu.upload_input(x, 1); gpu.forward();
        close(gpu.download_output(), cpu.forward(x, 1));
        gpu.upload_output_gradient(dy); gpu.backward();
        const auto actual = gpu.download_gradients(), expected = cpu.backward(x, 1, dy);
        REQUIRE(expected.input[0] != 0);
        REQUIRE(std::abs(expected.input[0]) > 1e-30 && std::abs(expected.input[0]) < 1e30);
        gradients(actual, expected);
    }
}

// The guarded pole test resolves the FP32 rounding of Q: its relative
// threshold is max(epsilon, n * 2^-23), so a denominator that FP64 accepts at
// epsilon = 1e-8 but FP32 cannot distinguish from a pole is reported.
TEST(float32_guarded_pole_threshold_follows_float_resolution) {
    kan::RationalConfig c; c.numerator_degree = 0; c.denominator_degree = 1;
    kan::Layer l(1, 1, c);
    kan::set_rational_parameters(l, std::vector<double>{1.0}, std::vector<double>{-1.0}, std::vector<double>{0.0});
    kan::Network net({l});
    const std::vector<double> near_pole{1 - 1e-7}; // |Q|/bound = 5e-8
    ResidentNetwork fp64(net, 1), fp32(net, 1, Precision::Float32);
    fp64.upload_input(near_pole, 1); fp64.forward();
    fp32.upload_input(near_pole, 1);
    test::throws<std::domain_error>([&] { fp32.forward(); });
    fp32.upload_input(std::vector<double>{0.5}, 1); fp32.forward();
    close(fp32.download_output(), std::vector<double>{2.0});
}

TEST(float32_zero_batch_and_parameter_rounding) {
    kan::Network cpu({layer(3, 2, kan::LegendreConfig{4}, 0.3), rational(kan::DenominatorPolicy::Smooth, 2, 1, 2, 2)});
    ResidentNetwork empty(cpu, 0, Precision::Float32);
    empty.upload_input({}, 0); empty.upload_output_gradient({}); empty.forward(); empty.backward(0.25);
    REQUIRE(empty.download_output().empty());
    const auto g = empty.download_gradients(), penalty = cpu.regularization(0.25).gradients;
    REQUIRE(g.input.empty());
    close(test::grad(g, 0).coefficients, test::grad(penalty, 0).coefficients);
    for (double b : test::grad(g, 0).bias) REQUIRE(b == 0);
    // Parameters are stored in FP32: downloads are the rounded values.
    const auto stored = empty.download_parameters();
    const auto& c = test::layer(cpu, 0).coefficients();
    for (std::size_t k = 0; k < c.size(); ++k)
        REQUIRE(test::layer(stored, 0).coefficients()[k] == static_cast<double>(static_cast<float>(c[k])));
    REQUIRE(empty.workspace_allocations() == ResidentNetwork(cpu, 0).workspace_allocations());
    ResidentNetwork moved(std::move(empty));
    REQUIRE(moved.precision() == Precision::Float32);
    test::throws<std::logic_error>([&] { empty.precision(); });
}

int main() { return test::run(); }

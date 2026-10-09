// Backlog M3 phase 2: the SiLU residual branch on the resident executor.
//
// Contract under test (docs/CONTRACT.md, "Residual branch (backlog M3)"):
// - every carrier (BasisEdges, TrainableRbfEdges, RationalEdges) in every
//   precision computes y += silu(X)*W^T after the carrier's output, the
//   VJPs dW = U^T*silu(X) + lambda*W and dX += silu'(X) (.) (U*W), and SGD
//   updates W with the other parameters; results agree with the FP64 CPU at
//   the executor's parameters within the precision's tolerance;
// - train_step (both overloads, both losses) is bitwise the eager sequence,
//   a failing step (including one the branch itself makes fail) leaves W at
//   the last good step;
// - the branch is structure for upload_parameters (presence must match) and
//   its weights are values (FP32 representability); download_parameters and
//   download_gradients carry it;
// - extreme inputs give finite results and no status.
#include "kan/families.hpp"
#include "kan/initializers.hpp"
#include "kan/resident.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

namespace {
using kan::cuda::Loss;
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

constexpr Precision precisions[] = {Precision::Float64, Precision::Float32, Precision::TensorFloat32};
const char* name_of(Precision p) {
    return p == Precision::Float64 ? "f64" : p == Precision::Float32 ? "f32" : "tf32";
}

// Per entry |a - e| <= relative*|e| + floor*max|e| (+ the FP32 normal range).
struct Tolerance { double relative, floor; };
Tolerance tolerance(Precision p) {
    if (p == Precision::Float64) return {1e-11, 1e-12};
    if (p == Precision::Float32) return {2e-4, 5e-5};
    return {1e-2, 1e-2};
}
template<class... Parts> std::string text(const Parts&... parts) {
    std::ostringstream out;
    out << std::setprecision(17);
    (out << ... << parts);
    return out.str();
}
void close(std::span<const double> actual, std::span<const double> expected, Tolerance t, const std::string& what) {
    if (actual.size() != expected.size())
        throw std::runtime_error(text(what, ": size ", actual.size(), " != ", expected.size()));
    double scale = 0;
    for (double e : expected) scale = std::max(scale, std::abs(e));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i]-expected[i]) > t.relative*std::abs(expected[i]) + t.floor*scale + 1e-37)
            throw std::runtime_error(text(what, ": index ", i, " of ", actual.size(), " actual=", actual[i],
                                          " expected=", expected[i], " scale=", scale));
    }
}

std::vector<double> wave(std::size_t count, double scale, double frequency, double phase = 0) {
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = scale*std::sin(frequency*static_cast<double>(i)+phase);
    return v;
}
kan::Layer seeded(kan::Layer l, double phase, double amplitude = 1.0) {
    const auto scale = amplitude/std::sqrt(static_cast<double>(l.inputs()*l.terms()));
    l.set_parameters(wave(l.coefficients().size(), scale, 0.731, phase), wave(l.outputs(), 0.05, 1.3, phase));
    return l;
}
kan::Layer branched(kan::Layer l, double phase, double amplitude = 0.8) {
    const auto scale = amplitude/std::sqrt(static_cast<double>(l.inputs()));
    l.set_residual(kan::SiluResidual{wave(l.inputs()*l.outputs(), scale, 0.917, phase)});
    return l;
}
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
kan::BasisConfig trainable_rbf() {
    return kan::TrainableRbfConfig{{-1, -0.5, 0, 0.5, 1}, {-0.3, -0.2, -0.1, -0.2, -0.3}};
}
const std::vector<double> spline_knots{-1.5, -1.5, -1.5, -1.5, -0.5, 0.25, 1.5, 1.5, 1.5, 1.5};
kan::BasisConfig spline() { return kan::BSplineConfig{3, spline_knots}; }

// Every carrier and map kind; the branch on every carrier, one layer without it.
kan::Network mixed() {
    std::vector<kan::NetworkLayer> layers;
    layers.emplace_back(kan::InputMap(3, kan::AffineMap{{0.5, 0.4, 0.3}, {0.1, -0.1, 0.0}}));
    layers.emplace_back(branched(seeded(kan::Layer(3, 4, kan::ChebyshevConfig{5}), 0.1), 0.2));
    layers.emplace_back(kan::InputMap(4, kan::LayerNormMap{1e-3, {1.0, 0.9, 1.1, 1.0}, {0.0, 0.05, -0.05, 0.1}}));
    layers.emplace_back(branched(seeded(kan::Layer(4, 4, trainable_rbf()), 0.7), 0.4));
    layers.emplace_back(kan::InputMap(4, kan::TanhMap{0.8}));
    layers.emplace_back(branched(rational(4, 3, kan::DenominatorPolicy::Absolute), 0.6));
    layers.emplace_back(seeded(kan::Layer(3, 3, spline()), 1.9));
    layers.emplace_back(branched(seeded(kan::Layer(3, 2, kan::LegendreConfig{4}), 1.1), 0.8));
    return kan::Network(std::move(layers));
}
kan::Network single(kan::Layer l) { return kan::Network({std::move(l)}); }
// Crosses the small-kernel thresholds of the branch in both precisions
// (batch*outputs*inputs and outputs*inputs) at the fixture's batches.
kan::Network wide() {
    return kan::Network({branched(seeded(kan::Layer(192, 192, kan::ChebyshevConfig{3}), 0.3, 0.3), 0.5, 0.3),
                         branched(seeded(kan::Layer(192, 3, kan::ChebyshevConfig{3}), 0.9, 0.3), 0.7, 0.3)});
}
struct Fixture {
    std::string name;
    kan::Network network;
    std::size_t capacity, batch;
    double rate;
};
std::vector<Fixture> fixtures() {
    std::vector<Fixture> f;
    f.push_back({"chebyshev", single(branched(seeded(kan::Layer(3, 4, kan::ChebyshevConfig{5}), 0.3), 0.1)), 9, 7, 0.05});
    f.push_back({"bspline", single(branched(seeded(kan::Layer(3, 4, spline()), 0.5), 0.3)), 9, 7, 0.05});
    f.push_back({"trainable_rbf", single(branched(seeded(kan::Layer(3, 4, trainable_rbf()), 0.9), 0.5)), 9, 7, 0.05});
    f.push_back({"rational_guarded", single(branched(rational(3, 4, kan::DenominatorPolicy::Guarded), 0.7)), 9, 7, 0.05});
    f.push_back({"rational_smooth", single(branched(rational(3, 4, kan::DenominatorPolicy::Smooth), 0.9)), 9, 7, 0.05});
    f.push_back({"mixed", mixed(), 11, 9, 0.05});
    f.push_back({"wide", wide(), 640, 600, 0.005});
    return f;
}
std::vector<double> input_for(const kan::Network& n, std::size_t rows, double phase = 0.2) {
    return wave(rows*n.inputs(), 0.9, 0.37, phase);
}
std::vector<double> target_for(const kan::Network& n, std::size_t rows, double phase = 0.4) {
    return wave(rows*n.outputs(), 0.5, 0.53, phase);
}
std::vector<double> upstream_for(const kan::Network& n, std::size_t rows) {
    return wave(rows*n.outputs(), 1.0/static_cast<double>(std::max<std::size_t>(rows, 1)), 0.53, 0.4);
}

// Every trainable parameter in a fixed order, the residual weights included.
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
        if (l.residual()) add(l.residual()->weights);
    }
    return v;
}
std::vector<double> parameters(ResidentNetwork& gpu) { return flat(gpu.download_parameters()); }

// CPU gradients of backward(l2): the VJPs plus the L2 gradient.
kan::NetworkGradients cpu_gradients(const kan::Network& n, std::span<const double> x, std::size_t rows,
                                    std::span<const double> u, double l2) {
    auto g = n.backward(x, rows, u);
    const auto r = n.regularization(l2);
    for (std::size_t j = 0; j < g.layers.size(); ++j) {
        auto* layer = std::get_if<kan::LayerGradients>(&g.layers[j]);
        if (!layer) continue;
        const auto& penalty = test::grad(r.gradients, j);
        for (std::size_t k = 0; k < layer->coefficients.size(); ++k) layer->coefficients[k] += penalty.coefficients[k];
        REQUIRE(layer->residual.size() == penalty.residual.size());
        for (std::size_t k = 0; k < layer->residual.size(); ++k) layer->residual[k] += penalty.residual[k];
    }
    return g;
}
// TF32 rounds the operands of the cuBLAS dW = U^T*S to 10 mantissa bits
// (2^-11 relative each); for sums with heavy cancellation the error scales
// with sum_b |U[b,o]|*|S[b,i]|, not with the result. Per KAN layer j this
// bound is formed from the CPU's layer input (a prefix network's forward)
// and upstream (the next stage's input gradient, or the network upstream).
std::vector<std::vector<double>> residual_bounds(const kan::Network& n, std::span<const double> x, std::size_t rows,
                                                 std::span<const double> u, const kan::NetworkGradients& e) {
    std::vector<std::vector<double>> bounds(n.layers().size());
    const auto stages = n.layers();
    for (std::size_t j = 0; j < stages.size(); ++j) {
        const auto* l = std::get_if<kan::Layer>(&stages[j]);
        if (!l || !l->residual()) continue;
        const auto input = j ? kan::Network(std::vector<kan::NetworkLayer>(stages.begin(), stages.begin()+j)).forward(x, rows)
                             : std::vector<double>(x.begin(), x.end());
        std::vector<double> upstream(u.begin(), u.end());
        if (j+1 < stages.size()) {
            if (const auto* m = std::get_if<kan::InputMapGradients>(&e.layers[j+1])) upstream = m->input;
            else upstream = test::grad(e, j+1).input;
        }
        const auto in = l->inputs(), out = l->outputs();
        auto& b = bounds[j];
        b.assign(in*out, 0.0);
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t o = 0; o < out; ++o)
                for (std::size_t i = 0; i < in; ++i) {
                    const double v = input[r*in+i];
                    b[o*in+i] += std::abs(upstream[r*out+o])*std::abs(v/(1+std::exp(-v)));
                }
    }
    return bounds;
}
void compare_gradients(const kan::NetworkGradients& a, const kan::NetworkGradients& e, Tolerance t, const std::string& what,
                       bool with_input = true, const std::vector<std::vector<double>>* bounds = nullptr) {
    if (with_input) close(a.input, e.input, t, what + " network input");
    REQUIRE(a.layers.size() == e.layers.size());
    for (std::size_t j = 0; j < a.layers.size(); ++j) {
        const auto at = what + " layer " + std::to_string(j);
        if (std::holds_alternative<kan::InputMapGradients>(e.layers[j])) {
            if (with_input || j) close(test::map_grad(a, j).input, test::map_grad(e, j).input, t, at + " input");
            close(test::map_grad(a, j).gain, test::map_grad(e, j).gain, t, at + " gain");
            close(test::map_grad(a, j).bias, test::map_grad(e, j).bias, t, at + " bias");
            continue;
        }
        const auto& ag = test::grad(a, j);
        const auto& eg = test::grad(e, j);
        if (with_input || j) close(ag.input, eg.input, t, at + " input");
        close(ag.coefficients, eg.coefficients, t, at + " coefficients");
        close(ag.bias, eg.bias, t, at + " bias");
        close(test::centers(ag), test::centers(eg), t, at + " centers");
        close(test::log_widths(ag), test::log_widths(eg), t, at + " log widths");
        close(test::denominators(ag), test::denominators(eg), t, at + " denominators");
        if (bounds && !(*bounds)[j].empty()) {
            const auto& bound = (*bounds)[j];
            double scale = 0;
            for (double v : eg.residual) scale = std::max(scale, std::abs(v));
            for (std::size_t k = 0; k < eg.residual.size(); ++k)
                if (!std::isfinite(ag.residual[k]) || std::abs(ag.residual[k]-eg.residual[k]) >
                        t.relative*std::abs(eg.residual[k]) + t.floor*scale + 2e-3*bound[k])
                    throw std::runtime_error(text(at, " residual (TF32 bound): index ", k, " actual=", ag.residual[k],
                                                  " expected=", eg.residual[k], " bound=", bound[k]));
        } else {
            close(ag.residual, eg.residual, t, at + " residual");
        }
    }
}
std::vector<double> flat(const kan::NetworkGradients& gradients) {
    std::vector<double> v;
    auto add = [&](std::span<const double> s) { v.insert(v.end(), s.begin(), s.end()); };
    for (const auto& stage : gradients.layers) {
        if (const auto* m = std::get_if<kan::InputMapGradients>(&stage)) { add(m->gain); add(m->bias); continue; }
        const auto& g = std::get<kan::LayerGradients>(stage);
        add(g.coefficients); add(g.bias); add(test::centers(g)); add(test::log_widths(g)); add(test::denominators(g));
        add(g.residual);
    }
    return v;
}

// The MSE upstream in the executor's arithmetic: (y - t) * (2/N) in T.
std::vector<double> mse_upstream(std::span<const double> y, std::span<const double> t, Precision p) {
    std::vector<double> g(y.size());
    const double scale = 2.0/static_cast<double>(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        if (p == Precision::Float64) g[i] = (y[i]-t[i])*scale;
        else g[i] = static_cast<double>((static_cast<float>(y[i])-static_cast<float>(t[i]))*static_cast<float>(scale));
    }
    return g;
}
double eager_mse_step(ResidentNetwork& gpu, std::span<const double> x, std::span<const double> t, std::size_t rows,
                      double rate, double l2, Precision p) {
    gpu.upload_input(x, rows);
    gpu.forward();
    const auto y = gpu.download_output();
    gpu.upload_output_gradient(mse_upstream(y, t, p));
    gpu.backward(l2);
    gpu.sgd(rate);
    double sum = 0;
    for (std::size_t i = 0; i < y.size(); ++i) sum += (y[i]-t[i])*(y[i]-t[i]);
    return sum/static_cast<double>(y.size());
}
template<class Exception, class Fn> std::string message_of(Fn fn) {
    try { fn(); } catch (const Exception& error) { return error.what(); }
    throw std::runtime_error("expected exception was not thrown");
}
bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
void require_branch(const kan::Network& n, const kan::Network& reference) {
    REQUIRE(n.layers().size() == reference.layers().size());
    for (std::size_t j = 0; j < n.layers().size(); ++j) {
        if (std::holds_alternative<kan::InputMap>(reference.layers()[j])) continue;
        REQUIRE(test::layer(n, j).residual().has_value() == test::layer(reference, j).residual().has_value());
    }
}
} // namespace

// Forward, every gradient (input, coefficients, bias, nonlinear, residual) and
// SGD against the FP64 CPU at the executor's parameters, at a batch below
// capacity, with and without L2; the gradients carry the branch's presence.
TEST(eager_forward_backward_sgd_match_the_cpu) {
    for (const auto p : precisions) {
        const auto t = tolerance(p);
        for (auto& f : fixtures()) {
            for (const std::size_t rows : {f.batch, f.batch/2+1}) {
                for (const double l2 : {0.0, 1e-3}) {
                    const auto what = text(f.name, " ", name_of(p), " batch ", rows, " l2 ", l2);
                    ResidentNetwork gpu(f.network, f.capacity, p);
                    const auto cpu = gpu.download_parameters();
                    require_branch(cpu, f.network);
                    const auto x = input_for(f.network, rows), u = upstream_for(f.network, rows);
                    gpu.upload_input(x, rows); gpu.upload_output_gradient(u);
                    gpu.forward();
                    close(gpu.download_output(), cpu.forward(x, rows), t, what + " output");
                    gpu.backward(l2);
                    const auto g = gpu.download_gradients();
                    const auto reference = cpu_gradients(cpu, x, rows, u, l2);
                    const auto bounds = p == Precision::TensorFloat32 ? residual_bounds(cpu, x, rows, u, reference)
                                                                      : std::vector<std::vector<double>>{};
                    compare_gradients(g, reference, t, what, true, bounds.empty() ? nullptr : &bounds);
                    // SGD: p - rate*g of the downloaded values, in T.
                    const auto before = flat(cpu), gradient = flat(g);
                    gpu.sgd(f.rate);
                    std::vector<double> expected(before.size());
                    for (std::size_t i = 0; i < before.size(); ++i) {
                        if (p == Precision::Float64) expected[i] = before[i]-f.rate*gradient[i];
                        else expected[i] = static_cast<float>(before[i]) - static_cast<float>(f.rate)*static_cast<float>(gradient[i]);
                    }
                    close(parameters(gpu), expected, p == Precision::Float64 ? Tolerance{1e-15, 1e-16} : Tolerance{2e-7, 1e-7},
                          what + " sgd");
                }
            }
        }
    }
}

// Batch zero: empty output and input gradient; the parameter gradients are the
// L2 gradients alone (lambda*W for the branch).
TEST(empty_batch_gives_the_l2_gradients) {
    for (const auto p : precisions) {
        for (auto& f : fixtures()) {
            ResidentNetwork gpu(f.network, f.capacity, p);
            const auto cpu = gpu.download_parameters();
            gpu.upload_input({}, 0); gpu.upload_output_gradient({});
            gpu.forward();
            REQUIRE(gpu.download_output().empty());
            gpu.backward(0.25);
            const auto g = gpu.download_gradients();
            REQUIRE(g.input.empty());
            compare_gradients(g, cpu_gradients(cpu, {}, 0, {}, 0.25), tolerance(p), f.name + " empty " + name_of(p));
            gpu.sgd(f.rate);
        }
    }
}

// train_step skips the network input gradient (the first layer here has the
// branch, so only its dX is skipped, not dW): bitwise the eager sequence for
// OutputGradient, MSE against a resident target and staged batches.
TEST(train_steps_are_bitwise_the_eager_steps) {
    for (const auto p : precisions) {
        for (auto& f : fixtures()) {
            for (const double l2 : {0.0, 1e-3}) {
                ResidentNetwork graph(f.network, f.capacity, p), eager(f.network, f.capacity, p);
                const auto x = input_for(f.network, f.batch), u = upstream_for(f.network, f.batch);
                for (auto* r : {&graph, &eager}) { r->upload_input(x, f.batch); r->upload_output_gradient(u); }
                for (int s = 0; s < 3; ++s) {
                    graph.train_step(f.rate, l2);
                    eager.forward(); eager.backward(l2); eager.sgd(f.rate);
                }
                if (parameters(graph) != parameters(eager)) throw std::runtime_error("output gradient mismatch: " + f.name);
                // MSE against a resident target.
                const auto target = target_for(f.network, f.batch);
                graph.upload_input(x, f.batch); graph.upload_target(target);
                for (int s = 0; s < 3; ++s) {
                    graph.train_step(f.rate, l2, Loss::MeanSquaredError);
                    const double loss = eager_mse_step(eager, x, target, f.batch, f.rate, l2, p);
                    test::near(graph.download_loss(), loss, p == Precision::Float64 ? 1e-13 : 1e-5);
                }
                if (parameters(graph) != parameters(eager)) throw std::runtime_error("mse mismatch: " + f.name);
                // Staged batches of varying size.
                graph.set_status_interval(2);
                const std::size_t rows[] = {f.batch, f.batch/2+1, f.capacity};
                for (std::size_t s = 0; s < std::size(rows); ++s) {
                    const auto xs = input_for(f.network, rows[s], 0.3*static_cast<double>(s));
                    const auto ts = target_for(f.network, rows[s], 0.5*static_cast<double>(s));
                    graph.train_step(xs, ts, rows[s], f.rate, l2);
                    eager_mse_step(eager, xs, ts, rows[s], f.rate, l2, p);
                }
                graph.check_status();
                if (parameters(graph) != parameters(eager)) throw std::runtime_error("staged mismatch: " + f.name);
                REQUIRE(graph.trained_steps() == 9);
            }
        }
    }
}

// A step that fails in the branch's own forward contraction (S*W^T overflows)
// or in its weight VJP (U^T*S overflows) commits nothing: the weights stay at
// the last good step and training continues.
TEST(failing_steps_roll_back_the_residual_weights) {
    for (const auto p : precisions) {
        const bool f64 = p == Precision::Float64;
        // Forward: two weights near the top of the range; silu(10) ~ 10 overflows the sum.
        {
            auto l = seeded(kan::Layer(2, 1, kan::ChebyshevConfig{3}), 0.2, 0.1);
            const double big = f64 ? 1e308 : 1e38;
            l.set_residual(kan::SiluResidual{{big, big}});
            const auto n = single(std::move(l));
            ResidentNetwork gpu(n, 2, p), reference(n, 2, p);
            const std::vector<double> good{0.1, 0.1, -0.2, 0.05}, bad{10.0, 10.0, 0.1, 0.1}, u{1e-3, -1e-3};
            for (auto* r : {&gpu, &reference}) { r->upload_input(good, 2); r->upload_output_gradient(u); }
            reference.forward(); reference.backward(); reference.sgd(1e-3);
            gpu.train_step(1e-3);
            gpu.upload_input(bad, 2); gpu.upload_output_gradient(u);
            const auto message = message_of<std::overflow_error>([&] { gpu.train_step(1e-3); });
            REQUIRE(contains(message, "training step 1"));
            REQUIRE(gpu.trained_steps() == 1);
            if (parameters(gpu) != parameters(reference)) throw std::runtime_error(text("forward rollback ", name_of(p)));
            // Eager forward reports the same overflow.
            test::throws<std::overflow_error>([&] { gpu.forward(); });
            gpu.upload_input(good, 2); gpu.upload_output_gradient(u);
            gpu.train_step(1e-3);
            REQUIRE(gpu.trained_steps() == 2);
        }
        // Backward only: B-spline edges are zero far outside their support, so
        // only dW = sum u*silu(x) overflows (the network input gradient is skipped).
        {
            const auto n = single(branched(seeded(kan::Layer(1, 1, spline()), 0.4), 0.2));
            ResidentNetwork gpu(n, 1, p), reference(n, 1, p);
            const std::vector<double> good{0.3}, far{1e10}, u{0.1}, huge{f64 ? 1e300 : 1e30};
            for (auto* r : {&gpu, &reference}) { r->upload_input(good, 1); r->upload_output_gradient(u); }
            reference.forward(); reference.backward(); reference.sgd(1e-3);
            gpu.train_step(1e-3);
            gpu.upload_input(far, 1); gpu.upload_output_gradient(huge);
            gpu.set_status_interval(4);
            gpu.train_step(1e-3);
            gpu.train_step(1e-3);
            const auto message = message_of<std::overflow_error>([&] { gpu.check_status(); });
            REQUIRE(contains(message, "training step 1"));
            REQUIRE(gpu.trained_steps() == 1);
            if (parameters(gpu) != parameters(reference)) throw std::runtime_error(text("backward rollback ", name_of(p)));
            gpu.forward();
            test::throws<std::overflow_error>([&] { gpu.backward(); });
        }
    }
}

// Branch presence is structure, weights are values.
TEST(upload_and_download_parameters_carry_the_branch) {
    for (const auto p : precisions) {
        const auto t = tolerance(p);
        for (auto& f : fixtures()) {
            ResidentNetwork gpu(f.network, f.capacity, p);
            const auto downloaded = gpu.download_parameters();
            require_branch(downloaded, f.network);
            if (p == Precision::Float64) REQUIRE(flat(downloaded) == flat(f.network));
            // Round trip.
            gpu.upload_parameters(downloaded);
            REQUIRE(flat(gpu.download_parameters()) == flat(downloaded));
            // Other weights are used by the next forward.
            std::vector<kan::NetworkLayer> layers(f.network.layers().begin(), f.network.layers().end());
            for (auto& stage : layers)
                if (auto* l = std::get_if<kan::Layer>(&stage); l && l->residual()) {
                    auto w = l->residual()->weights;
                    for (auto& v : w) v = -0.5*v + 0.01;
                    l->set_residual(kan::SiluResidual{w});
                }
            const kan::Network other(layers);
            gpu.upload_parameters(other);
            const auto x = input_for(f.network, f.batch);
            gpu.upload_input(x, f.batch);
            gpu.forward();
            const auto output = gpu.download_output();
            close(output, gpu.download_parameters().forward(x, f.batch), t, f.name + " uploaded weights");
            // Mismatched presence in either direction is rejected; nothing changes.
            std::vector<kan::NetworkLayer> stripped(layers);
            for (auto& stage : stripped)
                if (auto* l = std::get_if<kan::Layer>(&stage); l && l->residual()) { l->set_residual(std::nullopt); break; }
            const auto before = parameters(gpu);
            const auto message = message_of<std::invalid_argument>([&] { gpu.upload_parameters(kan::Network(stripped)); });
            REQUIRE(contains(message, "residual branch"));
            REQUIRE(parameters(gpu) == before);
            gpu.forward();
            REQUIRE(gpu.download_output() == output);
            ResidentNetwork plain(kan::Network(stripped), f.capacity, p);
            test::throws<std::invalid_argument>([&] { plain.upload_parameters(other); });
            // FP32 representability of the weights.
            if (p != Precision::Float64) {
                std::vector<kan::NetworkLayer> huge(layers);
                for (auto& stage : huge)
                    if (auto* l = std::get_if<kan::Layer>(&stage); l && l->residual()) {
                        auto w = l->residual()->weights; w.back() = 1e300;
                        l->set_residual(kan::SiluResidual{w});
                        break;
                    }
                test::throws<std::invalid_argument>([&] { gpu.upload_parameters(kan::Network(huge)); });
                test::throws<std::invalid_argument>([&] { ResidentNetwork(kan::Network(huge), 2, p); });
                REQUIRE(parameters(gpu) == before);
            }
        }
    }
}

// Inputs far outside every carrier's useful range: silu and silu' stay finite
// (no exp overflow), nothing is reported, and the results match the CPU.
TEST(extreme_inputs_are_finite) {
    for (const auto p : precisions) {
        const double r = p == Precision::Float64 ? 1e3 : 100.0;
        for (const auto& n : {single(branched(seeded(kan::Layer(2, 3, spline()), 0.3), 0.4)),
                              single(branched(seeded(kan::Layer(2, 3, trainable_rbf()), 0.6), 0.2))}) {
            const std::vector<double> x{r, -r, -r, r, 0.5*r, -0.5*r};
            ResidentNetwork gpu(n, 3, p);
            const auto cpu = gpu.download_parameters();
            const auto u = upstream_for(n, 3);
            gpu.upload_input(x, 3); gpu.upload_output_gradient(u);
            gpu.forward(); gpu.backward(1e-3);
            close(gpu.download_output(), cpu.forward(x, 3), tolerance(p), "extreme output");
            compare_gradients(gpu.download_gradients(), cpu_gradients(cpu, x, 3, u, 1e-3), tolerance(p), "extreme");
            gpu.sgd(1e-3);
            gpu.set_status_interval(1);
            gpu.train_step(1e-3);
            gpu.check_status();
        }
    }
}

// Phase 1's demonstration network (NoiseInit + branch, Chebyshev) trained with
// resident MSE steps tracks the CPU trajectory while it sits on the saddle and
// leaves it like the CPU. The escape (epochs 200-300 at rate 0.03) is chaotic:
// one ulp in one CPU weight changes the epoch-300 loss by 97% (M3 evidence,
// phase 2), so only the pre-escape trajectory is compared value by value.
TEST(noise_initialized_training_tracks_the_cpu) {
    std::vector<double> x, target;
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j) {
            const double a = -0.9 + 1.8*double(i)/7, b = -0.9 + 1.8*double(j)/7;
            x.insert(x.end(), {a, b});
            target.push_back(a*b);
        }
    // [Tanh, KAN 2->8, (Tanh, KAN 8->8) x 3, Tanh, KAN 8->1], NoiseInit with the branch.
    const auto initial = [] {
        std::vector<kan::NetworkLayer> stages;
        std::size_t in = 2;
        for (std::size_t l = 0; l <= 4; ++l) {
            const std::size_t out = l == 4 ? 1 : 8;
            kan::Layer layer(in, out, kan::ChebyshevConfig{5});
            layer.set_residual(kan::SiluResidual{std::vector<double>(in*out, 0.0)});
            stages.emplace_back(kan::InputMap(in, kan::TanhMap{1.0}));
            stages.emplace_back(std::move(layer));
            in = out;
        }
        kan::Network n(std::move(stages));
        kan::initialize(n, kan::NoiseInit{0.3, kan::Distribution::Uniform, 1, {}});
        return n;
    }();
    auto cpu = initial;
    constexpr std::size_t epochs = 1000;
    constexpr double rate = 0.03;
    const std::size_t checkpoints[] = {0, 50, 100, 150, 200, 999};
    std::vector<double> expected;
    for (std::size_t e = 0; e < epochs; ++e) {
        const auto y = cpu.forward(x, 64);
        std::vector<double> u(64);
        double s = 0;
        for (std::size_t b = 0; b < 64; ++b) { u[b] = 2*(y[b]-target[b])/64.0; s += (y[b]-target[b])*(y[b]-target[b]); }
        if (std::find(std::begin(checkpoints), std::end(checkpoints), e) != std::end(checkpoints)) expected.push_back(s/64.0);
        cpu.sgd(cpu.backward(x, 64, u), rate);
    }
    std::printf("cpu loss:");
    for (double v : expected) std::printf(" %.6e", v);
    std::printf("\n");
    for (const auto p : precisions) {
        ResidentNetwork gpu(initial, 64, p);
        gpu.upload_input(x, 64); gpu.upload_target(target);
        gpu.set_status_interval(50);
        std::vector<double> actual;
        for (std::size_t e = 0; e < epochs; ++e) {
            gpu.train_step(rate, 0.0, Loss::MeanSquaredError);
            if (std::find(std::begin(checkpoints), std::end(checkpoints), e) != std::end(checkpoints)) actual.push_back(gpu.download_loss());
        }
        std::printf("%s loss:", name_of(p));
        for (double v : actual) std::printf(" %.6e", v);
        std::printf("\n");
        const auto tracked = std::size(checkpoints)-1; // up to epoch 200
        close(std::span(actual).first(tracked), std::span(expected).first(tracked),
              p == Precision::Float64 ? Tolerance{1e-12, 0} : Tolerance{1e-3, 0}, text("training ", name_of(p)));
        REQUIRE(expected.back() < 0.5*0.1205);
        REQUIRE(actual.back() < 0.5*0.1205);
    }
}

int main() {
    if (!kan::cuda::available()) { std::puts("real CUDA hardware required"); return 1; }
    return test::run();
}

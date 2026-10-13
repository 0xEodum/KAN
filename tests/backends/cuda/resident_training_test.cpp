// Backlog C9: on-device training steps of the resident executor.
//
// Contract under test:
// - train_step(rate, l2, loss) runs forward + loss gradient + backward + SGD
//   as one CUDA-graph replay. With Loss::OutputGradient it is bitwise the
//   eager forward(); backward(l2); sgd(rate) sequence on the uploaded upstream;
//   with Loss::MeanSquaredError the upstream is 2(y - t)/(batch*outputs)
//   against the uploaded target, computed on the device (bitwise the same
//   operations in the executor precision), and download_loss() returns
//   sum (y - t)^2/(batch*outputs) of that step's forward.
// - train_step(input, target, batch, rate, l2) stages a new host batch
//   (validated and converted before returning, copied asynchronously) and
//   trains on it with the MSE loss.
// - Status is checked once every status_interval() steps (default 1) and by
//   every synchronous call. A nonfinite result or rational pole raised by a
//   step is reported as before (std::overflow_error / std::domain_error) by
//   the check that observes it; the first failing step of the interval is the
//   one reported, trained_steps() then equals its index, its update and those
//   of the remaining steps of the interval are not committed, and the
//   executor stays usable with the parameters of the last good step.
// - Graphs are re-captured when the batch, the active parameter buffer, the
//   loss, the learning rate or the L2 weight change; parameter uploads are
//   seen by the next replay. Nothing allocates device storage.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "support/families.hpp"
#include "support/network.hpp"
#include "support/test.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {
using kan::cuda::Loss;
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

constexpr Precision precisions[] = {Precision::Float64, Precision::Float32, Precision::TensorFloat32};

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
kan::RationalConfig rational_config(kan::DenominatorPolicy policy, std::size_t m = 3, std::size_t n = 2) {
    kan::RationalConfig r;
    r.numerator_degree = m; r.denominator_degree = n; r.center = 0.1; r.scale = 1.3; r.denominator_policy = policy;
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

// Every carrier and map kind: affine -> Chebyshev -> LayerNorm(gain, bias) ->
// trainable RBF -> tanh -> rational (safe policy) -> B-spline.
kan::Network mixed() {
    std::vector<kan::NetworkLayer> layers;
    layers.emplace_back(kan::InputMap(3, kan::AffineMap{{0.5, 0.4, 0.3}, {0.1, -0.1, 0.0}}));
    layers.emplace_back(seeded(kan::Layer(3, 4, kan::ChebyshevConfig{5}), 0.1));
    layers.emplace_back(kan::InputMap(4, kan::LayerNormMap{1e-3, {1.0, 0.9, 1.1, 1.0}, {0.0, 0.05, -0.05, 0.1}}));
    layers.emplace_back(seeded(kan::Layer(4, 4, trainable_rbf()), 0.7));
    layers.emplace_back(kan::InputMap(4, kan::TanhMap{0.8}));
    layers.emplace_back(rational(4, 3, kan::DenominatorPolicy::Absolute));
    layers.emplace_back(seeded(kan::Layer(3, 2, kan::BSplineConfig{3, spline_knots}), 1.9));
    return kan::Network(std::move(layers));
}
// Crosses the cuBLAS thresholds of both precisions (C1/C2 contraction paths);
// small coefficients keep the hidden activations inside [-1, 1].
kan::Network wide() {
    return kan::Network({seeded(kan::Layer(64, 80, kan::ChebyshevConfig{7}), 0.3, 0.3),
                         seeded(kan::Layer(80, 3, kan::ChebyshevConfig{7}), 0.9, 0.3)});
}
kan::Network small() {
    return kan::Network({seeded(kan::Layer(3, 5, kan::ChebyshevConfig{6}), 0.2),
                         seeded(kan::Layer(5, 2, kan::LegendreConfig{4}), 0.6)});
}
struct Fixture {
    const char* name;
    kan::Network network;
    std::size_t batch;
    double rate; // stable SGD rate for the fixture's data (the wide network diverges at 0.05)
};
std::vector<Fixture> fixtures() {
    return {{"small", small(), 7, 0.05}, {"mixed", mixed(), 9, 0.05}, {"wide", wide(), 400, 0.005}};
}
std::vector<double> input_for(const kan::Network& n, std::size_t rows, double phase = 0.2) {
    return wave(rows*n.inputs(), 0.9, 0.37, phase);
}
std::vector<double> target_for(const kan::Network& n, std::size_t rows, double phase = 0.4) {
    return wave(rows*n.outputs(), 0.5, 0.53, phase);
}
// The gradient of a batch-mean loss (scaled by 1/rows, as torch_reference.py).
std::vector<double> upstream_for(const kan::Network& n, std::size_t rows) {
    return wave(rows*n.outputs(), 1.0/static_cast<double>(std::max<std::size_t>(rows, 1)), 0.53, 0.4);
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
std::vector<double> parameters(ResidentNetwork& gpu) { return flat(gpu.download_parameters()); }

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
double mse(std::span<const double> y, std::span<const double> t) {
    double sum = 0;
    for (std::size_t i = 0; i < y.size(); ++i) sum += (y[i]-t[i])*(y[i]-t[i]);
    return sum/static_cast<double>(y.size());
}
// One eager MSE step on the reference executor (host-computed upstream).
double eager_mse_step(ResidentNetwork& gpu, std::span<const double> x, std::span<const double> t, std::size_t rows,
                      double rate, double l2, Precision p) {
    gpu.upload_input(x, rows);
    gpu.forward();
    const auto y = gpu.download_output();
    gpu.upload_output_gradient(mse_upstream(y, t, p));
    gpu.backward(l2);
    gpu.sgd(rate);
    return mse(y, t);
}
template<class Exception, class Fn> std::string message_of(Fn fn) {
    try { fn(); } catch (const Exception& error) { return error.what(); }
    throw std::runtime_error("expected exception was not thrown");
}
bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

TEST(train_step_is_bitwise_the_eager_step) {
    for (const auto p : precisions) {
        for (auto& f : fixtures()) {
            for (const double l2 : {0.0, 1e-3}) {
                ResidentNetwork graph(f.network, f.batch, p), eager(f.network, f.batch, p);
                const auto x = input_for(f.network, f.batch), u = upstream_for(f.network, f.batch);
                for (auto* r : {&graph, &eager}) { r->upload_input(x, f.batch); r->upload_output_gradient(u); }
                for (int s = 0; s < 4; ++s) {
                    graph.train_step(f.rate, l2);
                    eager.forward(); eager.backward(l2); eager.sgd(f.rate);
                }
                if (parameters(graph) != parameters(eager)) throw std::runtime_error(std::string("mismatch: ") + f.name);
                REQUIRE(graph.trained_steps() == 4);
                // The resident upstream survives the steps: the eager path continues.
                graph.forward(); graph.backward(l2);
                eager.forward(); eager.backward(l2);
                REQUIRE(graph.download_gradients().input == eager.download_gradients().input);
            }
        }
    }
}

TEST(train_step_lifecycle_and_arguments) {
    for (const auto p : precisions) {
        const auto n = small();
        ResidentNetwork gpu(n, 4, p);
        test::throws<std::logic_error>([&] { gpu.train_step(0.1); });                          // no input
        gpu.upload_input(input_for(n, 4), 4);
        test::throws<std::logic_error>([&] { gpu.train_step(0.1); });                          // no upstream
        test::throws<std::logic_error>([&] { gpu.train_step(0.1, 0.0, Loss::MeanSquaredError); }); // no target
        test::throws<std::logic_error>([&] { gpu.download_loss(); });
        gpu.upload_output_gradient(upstream_for(n, 4));
        test::throws<std::invalid_argument>([&] { gpu.train_step(0.0); });
        test::throws<std::invalid_argument>([&] { gpu.train_step(-1.0); });
        test::throws<std::invalid_argument>([&] { gpu.train_step(0.1, -1.0); });
        test::throws<std::invalid_argument>([&] { gpu.train_step(std::nan("")); });
        test::throws<std::invalid_argument>([&] { gpu.train_step(0.1, 0.0, static_cast<Loss>(7)); });
        test::throws<std::invalid_argument>([&] { gpu.upload_target(std::vector<double>(3)); });
        test::throws<std::invalid_argument>([&] { gpu.upload_target(std::vector<double>(8, std::nan(""))); });
        REQUIRE(gpu.trained_steps() == 0);
        gpu.train_step(0.1);
        // Like sgd(): outputs and gradients are stale, input and upstream stay.
        test::throws<std::logic_error>([&] { gpu.download_output(); });
        test::throws<std::logic_error>([&] { gpu.download_gradients(); });
        test::throws<std::logic_error>([&] { gpu.download_loss(); }); // not an MSE step
        gpu.train_step(0.1);
        REQUIRE(gpu.trained_steps() == 2);
        // Input upload invalidates the upstream and the target.
        gpu.upload_input(input_for(n, 4), 4);
        test::throws<std::logic_error>([&] { gpu.train_step(0.1); });
        gpu.upload_target(target_for(n, 4));
        gpu.upload_input(input_for(n, 4), 4);
        test::throws<std::logic_error>([&] { gpu.train_step(0.1, 0.0, Loss::MeanSquaredError); });
        // Empty batches have no mean squared error.
        gpu.upload_input({}, 0);
        test::throws<std::invalid_argument>([&] { gpu.upload_target({}); gpu.train_step(0.1, 0.0, Loss::MeanSquaredError); });
        test::throws<std::invalid_argument>([&] { gpu.train_step({}, {}, 0, 0.1); });
        // Empty batch with an uploaded (empty) upstream: L2 only, as sgd().
        gpu.upload_output_gradient({});
        gpu.train_step(0.1, 1e-2);
        REQUIRE(gpu.trained_steps() == 3);
    }
}

TEST(device_mse_matches_host_mse_bitwise_and_cpu) {
    for (const auto p : precisions) {
        for (auto& f : fixtures()) {
            ResidentNetwork graph(f.network, f.batch, p), eager(f.network, f.batch, p);
            const auto x = input_for(f.network, f.batch), t = target_for(f.network, f.batch);
            graph.upload_input(x, f.batch); graph.upload_target(t);
            for (int s = 0; s < 3; ++s) {
                const auto cpu = eager.download_parameters();
                const auto y_cpu = cpu.forward(x, f.batch);
                const double expected = eager_mse_step(eager, x, t, f.batch, f.rate, 1e-3, p);
                graph.train_step(f.rate, 1e-3, Loss::MeanSquaredError);
                const double loss = graph.download_loss();
                // Same value as the host reduction of the same outputs (order differs) ...
                const double relative = p == Precision::Float64 ? 1e-13 : 1e-5;
                test::near(loss, expected, relative);
                // ... and as the FP64 CPU at the executor's parameters.
                test::near(loss, mse(y_cpu, t), p == Precision::Float64 ? 1e-12 : p == Precision::Float32 ? 2e-3 : 5e-2);
                if (parameters(graph) != parameters(eager)) throw std::runtime_error(std::string("mismatch: ") + f.name);
            }
            // The device-computed upstream is not an uploaded one.
            test::throws<std::logic_error>([&] { graph.train_step(f.rate); });
        }
    }
}

TEST(staged_batches_match_resident_steps) {
    for (const auto p : precisions) {
        for (auto& f : fixtures()) {
            const auto capacity = f.batch;
            ResidentNetwork staged(f.network, capacity, p), eager(f.network, capacity, p);
            staged.set_status_interval(3);
            // Varying batch sizes force re-capture; the sources are clobbered
            // after each call (the call must have consumed them).
            const std::size_t rows[] = {capacity, capacity, capacity/2+1, capacity, 1, capacity};
            double last = 0;
            for (std::size_t s = 0; s < std::size(rows); ++s) {
                auto x = input_for(f.network, rows[s], 0.1*static_cast<double>(s));
                auto t = target_for(f.network, rows[s], 0.3*static_cast<double>(s));
                const auto x_copy = x, t_copy = t;
                staged.train_step(x, t, rows[s], f.rate, 1e-3);
                std::fill(x.begin(), x.end(), 1e300); std::fill(t.begin(), t.end(), std::nan(""));
                last = eager_mse_step(eager, x_copy, t_copy, rows[s], f.rate, 1e-3, p);
            }
            test::near(staged.download_loss(), last, p == Precision::Float64 ? 1e-13 : 1e-5);
            if (parameters(staged) != parameters(eager)) throw std::runtime_error(std::string("mismatch: ") + f.name);
            REQUIRE(staged.trained_steps() == std::size(rows));
            // The last staged batch is the executor's input: evaluate it eagerly.
            REQUIRE(staged.batch() == capacity);
            staged.forward(); eager.forward();
            REQUIRE(staged.download_output() == eager.download_output());
            // Resident MSE steps on the last staged batch continue the same trajectory.
            staged.train_step(f.rate, 1e-3, Loss::MeanSquaredError);
            eager_mse_step(eager, input_for(f.network, capacity, 0.5), target_for(f.network, capacity, 1.5), capacity, f.rate, 1e-3, p);
            if (parameters(staged) != parameters(eager)) throw std::runtime_error(std::string("resident mismatch: ") + f.name);
        }
    }
}

TEST(staged_batch_validation_changes_nothing) {
    for (const auto p : precisions) {
        const auto n = small();
        ResidentNetwork gpu(n, 6, p), reference(n, 6, p);
        const auto x = input_for(n, 6), t = target_for(n, 6);
        gpu.train_step(x, t, 6, 0.05);
        const double first = eager_mse_step(reference, x, t, 6, 0.05, 0.0, p);
        auto bad = x; bad[4] = std::nan("");
        test::throws<std::invalid_argument>([&] { gpu.train_step(bad, t, 6, 0.05); });
        auto bad_target = t; bad_target[1] = INFINITY;
        test::throws<std::invalid_argument>([&] { gpu.train_step(x, bad_target, 6, 0.05); });
        test::throws<std::invalid_argument>([&] { gpu.train_step(x, t, 7, 0.05); });              // above capacity
        test::throws<std::invalid_argument>([&] { gpu.train_step(x, std::vector<double>(3), 6, 0.05); });
        test::throws<std::invalid_argument>([&] { gpu.train_step(x, t, 6, 0.0); });
        if (p != Precision::Float64) {
            auto huge = x; huge[0] = 1e300;
            test::throws<std::invalid_argument>([&] { gpu.train_step(huge, t, 6, 0.05); });
        }
        REQUIRE(gpu.trained_steps() == 1);
        REQUIRE(parameters(gpu) == parameters(reference));
        // Rejections change nothing: the loss of the last step stays available
        // and the staged input/target are the last accepted batch.
        test::near(gpu.download_loss(), first, p == Precision::Float64 ? 1e-13 : 1e-5);
        gpu.train_step(0.05, 0.0, Loss::MeanSquaredError);
        eager_mse_step(reference, x, t, 6, 0.05, 0.0, p);
        REQUIRE(parameters(gpu) == parameters(reference));
    }
}

// Input that overflows the Chebyshev recurrence (T_5 ~ 16 x^5) in the executor precision.
double overflowing(Precision p) { return p == Precision::Float64 ? 1e70 : 1e9; }

TEST(deferred_overflow_is_attributed_to_its_step) {
    for (const auto p : precisions) {
        const auto n = small();
        ResidentNetwork gpu(n, 5, p), reference(n, 5, p);
        gpu.set_status_interval(4);
        REQUIRE(gpu.status_interval() == 4);
        std::vector<std::vector<double>> xs;
        for (int s = 0; s < 8; ++s) xs.push_back(input_for(n, 5, 0.3*s));
        xs[2][3] = overflowing(p);
        const auto t = target_for(n, 5);
        for (int s = 0; s < 2; ++s) eager_mse_step(reference, xs[s], t, 5, 0.05, 0.0, p);
        // Steps 0-2 return: the interval is not complete. Step 3 completes it
        // and reports the failure of step 2.
        for (int s = 0; s < 3; ++s) gpu.train_step(xs[s], t, 5, 0.05);
        const auto text = message_of<std::overflow_error>([&] { gpu.train_step(xs[3], t, 5, 0.05); });
        REQUIRE(contains(text, "nonfinite resident numerical result"));
        REQUIRE(contains(text, "training step 2"));
        REQUIRE(gpu.trained_steps() == 2);
        // Parameters of the last good step; outputs, gradients and loss are stale.
        REQUIRE(parameters(gpu) == parameters(reference));
        test::throws<std::logic_error>([&] { gpu.download_loss(); });
        gpu.check_status(); // nothing pending any more
        // The executor recovers: training continues from the last good step.
        for (int s = 4; s < 8; ++s) {
            gpu.train_step(xs[s], t, 5, 0.05);
            eager_mse_step(reference, xs[s], t, 5, 0.05, 0.0, p);
        }
        REQUIRE(gpu.trained_steps() == 6);
        REQUIRE(parameters(gpu) == parameters(reference));
    }
}

// Guarded rational edge with Q = 1 - z at the default center/scale: x = 1 is a pole.
kan::Network pole_network() {
    kan::RationalConfig config;
    config.numerator_degree = 1; config.denominator_degree = 1;
    kan::Layer l(1, 1, config);
    kan::set_rational_parameters(l, std::vector<double>{0.2, 0.5}, std::vector<double>{-1.0}, std::vector<double>{0.0});
    return kan::Network({std::move(l)});
}

TEST(first_failure_of_an_interval_wins) {
    for (const auto p : precisions) {
        const auto n = pole_network();
        ResidentNetwork gpu(n, 3, p), reference(n, 3, p);
        gpu.set_status_interval(5);
        const std::vector<double> good{0.1, -0.4, 0.3}, pole{0.2, 1.0, -0.1}, t{0.1, 0.2, 0.3};
        // A target whose squared error overflows the loss (forward phase).
        const double far = p == Precision::Float64 ? 1e200 : 1e30;
        const std::vector<double> t_far{0.1, far, 0.3};
        constexpr double rate = 1e-12; // keeps the denominator at the pole
        eager_mse_step(reference, good, t, 3, rate, 0.0, p);
        gpu.train_step(good, t, 3, rate);  // step 0: fine
        gpu.train_step(pole, t, 3, rate);  // step 1: pole
        // Step 2: a loss overflow, which would be reported only if it were first.
        gpu.train_step(good, t_far, 3, rate);
        const auto text = message_of<std::domain_error>([&] { gpu.check_status(); });
        REQUIRE(contains(text, "unsafe resident rational denominator"));
        REQUIRE(contains(text, "training step 1"));
        REQUIRE(gpu.trained_steps() == 1);
        REQUIRE(parameters(gpu) == parameters(reference));
        // Interval one (the default) reports the failing call itself.
        gpu.set_status_interval(1);
        test::throws<std::domain_error>([&] { gpu.train_step(pole, t, 3, rate); });
        REQUIRE(gpu.trained_steps() == 1);
        // The loss overflow alone commits nothing either (resident target).
        gpu.upload_input(good, 3); gpu.upload_target(t_far);
        const auto overflow = message_of<std::overflow_error>([&] { gpu.train_step(rate, 0.0, Loss::MeanSquaredError); });
        REQUIRE(contains(overflow, "training step 1"));
        REQUIRE(parameters(gpu) == parameters(reference));
    }
}

TEST(synchronous_calls_report_pending_failures) {
    for (const auto p : precisions) {
        const auto n = small();
        auto x_bad = input_for(n, 4);
        x_bad[0] = overflowing(p);
        const auto x = input_for(n, 4), t = target_for(n, 4);
        const std::function<void(ResidentNetwork&)> calls[] = {
            [&](ResidentNetwork& g) { g.download_parameters(); },
            [&](ResidentNetwork& g) { g.synchronize(); },
            [&](ResidentNetwork& g) { g.forward(); },
            [&](ResidentNetwork& g) { g.upload_input(x, 4); },
            [&](ResidentNetwork& g) { g.upload_parameters(n); },
            [&](ResidentNetwork& g) { g.set_status_interval(2); },
            // Reports the failure; the loss is stale afterwards (std::logic_error).
            [&](ResidentNetwork& g) { try { g.download_loss(); } catch (const std::logic_error&) {} },
        };
        for (const auto& call : calls) {
            ResidentNetwork gpu(n, 4, p);
            gpu.set_status_interval(100);
            gpu.train_step(x, t, 4, 0.05);
            gpu.train_step(x_bad, t, 4, 0.05);
            gpu.train_step(x, t, 4, 0.05);
            test::throws<std::overflow_error>([&] { call(gpu); });
            REQUIRE(gpu.trained_steps() == 1);
            call(gpu); // reported once; the call itself then works
        }
    }
}

TEST(status_interval_arguments) {
    ResidentNetwork gpu(small(), 2);
    REQUIRE(gpu.status_interval() == 1);
    test::throws<std::invalid_argument>([&] { gpu.set_status_interval(0); });
    REQUIRE(gpu.status_interval() == 1);
    gpu.check_status(); // nothing pending
}

TEST(graph_replay_sees_uploaded_parameters_and_changes) {
    for (const auto p : precisions) {
        for (auto& f : fixtures()) {
            ResidentNetwork gpu(f.network, f.batch, p), eager(f.network, f.batch, p);
            const auto x = input_for(f.network, f.batch), u = upstream_for(f.network, f.batch);
            for (auto* r : {&gpu, &eager}) { r->upload_input(x, f.batch); r->upload_output_gradient(u); }
            auto step = [&](double rate, double l2) {
                gpu.train_step(rate, l2);
                eager.forward(); eager.backward(l2); eager.sgd(rate);
            };
            const double r = f.rate;
            step(r, 0.0); step(r, 0.0);
            // Parameters replaced between replays.
            const auto other = ResidentNetwork(f.network, f.batch, p).download_parameters();
            gpu.upload_parameters(other); eager.upload_parameters(other);
            step(r, 0.0);
            // Learning-rate schedule and L2 changes between replays.
            step(0.4*r, 0.0); step(0.2*r, 1e-3); step(0.2*r, 1e-3); step(0.6*r, 0.0);
            // A smaller batch, then the full batch again.
            const auto rows = f.batch/2+1;
            const auto x2 = input_for(f.network, rows, 0.7), u2 = upstream_for(f.network, rows);
            for (auto* e : {&gpu, &eager}) { e->upload_input(x2, rows); e->upload_output_gradient(u2); }
            step(r, 0.0); step(r, 0.0);
            for (auto* e : {&gpu, &eager}) { e->upload_input(x, f.batch); e->upload_output_gradient(u); }
            step(r, 0.0);
            if (parameters(gpu) != parameters(eager)) throw std::runtime_error(std::string("mismatch: ") + f.name);
            REQUIRE(gpu.trained_steps() == 10);
        }
    }
}

TEST(training_allocates_no_device_storage) {
    for (const auto p : precisions) {
        const auto n = wide();
        ResidentNetwork gpu(n, 64, p);
        const auto allocations = gpu.workspace_allocations();
        gpu.set_status_interval(8);
        for (int s = 0; s < 20; ++s) gpu.train_step(input_for(n, 64 - s % 3), target_for(n, 64 - s % 3), 64 - s % 3, 0.01);
        gpu.upload_input(input_for(n, 64), 64); gpu.upload_output_gradient(upstream_for(n, 64));
        for (int s = 0; s < 5; ++s) gpu.train_step(0.01, 1e-4);
        gpu.check_status();
        REQUIRE(gpu.workspace_allocations() == allocations);
        REQUIRE(gpu.trained_steps() == 25);
    }
}

TEST(trainable_rbf_width_failure_is_deferred) {
    for (const auto p : precisions) {
        const kan::Network n({seeded(kan::Layer(2, 2, trainable_rbf()), 0.4)});
        ResidentNetwork gpu(n, 4, p), reference(n, 4, p);
        gpu.set_status_interval(3);
        const auto x = input_for(n, 4), t = target_for(n, 4);
        eager_mse_step(reference, x, t, 4, 0.05, 0.0, p);
        gpu.train_step(x, t, 4, 0.05);
        // A huge rate drives a log width beyond the exponent range: exp() is
        // not finite, which the candidate validation rejects.
        gpu.train_step(x, t, 4, p == Precision::Float64 ? 1e300 : 1e37);
        test::throws<std::overflow_error>([&] { gpu.train_step(x, t, 4, 0.05); });
        REQUIRE(gpu.trained_steps() == 1);
        REQUIRE(parameters(gpu) == parameters(reference));
    }
}

TEST(moved_from_training_calls_fail) {
    ResidentNetwork gpu(small(), 2);
    ResidentNetwork moved(std::move(gpu));
    test::throws<std::logic_error>([&] { gpu.train_step(0.1); });
    test::throws<std::logic_error>([&] { gpu.check_status(); });
    test::throws<std::logic_error>([&] { gpu.status_interval(); });
    REQUIRE(moved.status_interval() == 1);
}
} // namespace

int main() {
    if (!kan::cuda::available()) { std::cerr << "real CUDA hardware required\n"; return 1; }
    return test::run();
}

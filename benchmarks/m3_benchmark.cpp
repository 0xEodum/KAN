#include "kan/cuda.hpp"
#include "kan/network.hpp"
#ifdef KAN_BENCH_RESIDENT
#include "kan/resident.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>

namespace {
// Benchmark networks hold KAN layers only (no input maps).
const kan::Layer& kan_layer(const kan::Network& n, std::size_t j) { return std::get<kan::Layer>(n.layers()[j]); }
std::vector<std::reference_wrapper<const kan::Layer>> kan_layers(const kan::Network& n) {
    std::vector<std::reference_wrapper<const kan::Layer>> result;
    for (const auto& stage : n.layers()) result.push_back(std::cref(std::get<kan::Layer>(stage)));
    return result;
}
const kan::LayerGradients& kan_grad(const kan::NetworkGradients& g, std::size_t j) { return std::get<kan::LayerGradients>(g.layers[j]); }
using Clock = std::chrono::steady_clock;
constexpr double learning_rate = 0.001;
const char* names[] = {"bspline", "mexican_hat", "trainable_rbf"};
struct Case { int id, family; std::vector<std::size_t> widths; std::size_t batch; };
kan::BasisConfig family_basis(int family) {
    if (family == 0) return kan::BSplineConfig{3, {-1,-1,-1,-1,-0.5,0,0.5,1,1,1,1}};
    const std::vector<double> centers{-1.0, -2.0/3.0, -1.0/3.0, 0.0, 1.0/3.0, 2.0/3.0, 1.0};
    if (family == 1) return kan::MexicanHatConfig{centers, {0.35,0.45,0.55,0.65,0.75,0.85,0.95}};
    return kan::TrainableRbfConfig{centers, {-0.8,-0.6,-0.4,-0.2,0,0.2,0.4}};
}
// Configured centers / trainable log widths (empty where the family has none).
const std::vector<double>& centers(const kan::Layer& l) {
    static const std::vector<double> none;
    if (const auto* e = std::get_if<kan::BasisEdges>(&l.carrier()))
        if (const auto* m = std::get_if<kan::MexicanHatConfig>(&e->basis)) return m->centers;
    if (const auto* t = std::get_if<kan::TrainableRbfEdges>(&l.carrier())) return t->basis.centers;
    return none;
}
const std::vector<double>& log_widths(const kan::Layer& l) {
    static const std::vector<double> none;
    if (const auto* t = std::get_if<kan::TrainableRbfEdges>(&l.carrier())) return t->basis.log_widths;
    return none;
}
// Trainable RBF gradient vectors (empty for the other carriers).
std::span<const double> centers(const kan::LayerGradients& g) {
    const auto* t = std::get_if<kan::TrainableRbfGradients>(&g.nonlinear);
    return t ? std::span<const double>(t->centers) : std::span<const double>();
}
std::span<const double> log_widths(const kan::LayerGradients& g) {
    const auto* t = std::get_if<kan::TrainableRbfGradients>(&g.nonlinear);
    return t ? std::span<const double>(t->log_widths) : std::span<const double>();
}
kan::Network network(const Case& c) {
    const auto basis = family_basis(c.family);
    std::vector<kan::Layer> layers;
    for (std::size_t l = 1; l < c.widths.size(); ++l) {
        kan::Layer layer(c.widths[l-1], c.widths[l], basis);
        std::vector<double> coefficients(layer.coefficients().size()), bias(layer.outputs());
        for (std::size_t j = 0; j < coefficients.size(); ++j)
            coefficients[j] = 0.02 * std::sin(static_cast<double>((j + 1) * (l + 1))) /
                              (static_cast<double>(layer.inputs()) * static_cast<double>(1 + j % kan::basis_size(basis)));
        for (std::size_t j = 0; j < bias.size(); ++j) bias[j] = 0.01 * std::cos(static_cast<double>(j + l));
        layer.set_parameters(coefficients, bias); layers.push_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
std::vector<double> data(std::size_t count, double scale, std::size_t offset) {
    std::vector<double> result(count);
    for (std::size_t j = 0; j < count; ++j) result[j] = scale * std::sin(static_cast<double>((j + offset) % 997) * 0.071);
    return result;
}
double milliseconds(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
double quantile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const double index = p * static_cast<double>(values.size()-1);
    const auto lo = static_cast<std::size_t>(index), hi = std::min(lo+1, values.size()-1);
    return values[lo] + (values[hi]-values[lo]) * (index-static_cast<double>(lo));
}
double checksum(std::span<const double> values) {
    double result = 0;
    for (std::size_t i = 0; i < values.size(); ++i) result += values[i] * static_cast<double>(1 + i % 17);
    return result;
}
double checksum(const kan::NetworkGradients& g) {
    double result = checksum(g.input);
    for (std::size_t j = 0; j < g.layers.size(); ++j) { const auto& layer = kan_grad(g, j); result += checksum(layer.coefficients) + checksum(layer.bias) + checksum(centers(layer)) + checksum(log_widths(layer)); }
    return result;
}
double checksum(const kan::Network& n) {
    double result = 0;
    for (const kan::Layer& l : kan_layers(n)) result += checksum(l.coefficients()) + checksum(l.bias()) + checksum(centers(l)) + checksum(log_widths(l));
    return result;
}
struct Result {
    kan::Network parameters;
    std::vector<double> output, times;
    kan::NetworkGradients gradients;
    double setup_ms = 0;
    std::size_t allocations = 0;
};
Result host(const Case& c, const std::vector<double>& input, const std::vector<double>& upstream,
            int warmups, int repeats, bool legacy) {
    auto start = Clock::now();
    Result result{network(c)}; result.setup_ms = milliseconds(start);
    for (int step = 0; step < warmups + repeats; ++step) {
        start = Clock::now();
        if (!legacy) {
            result.output = result.parameters.forward(input, c.batch);
            result.gradients = result.parameters.backward(input, c.batch, upstream);
        } else {
            const auto layers = kan_layers(result.parameters);
            std::vector<std::vector<double>> activations;
            auto current = std::span<const double>(input);
            for (const auto& layer : layers) {
                activations.push_back(kan::cuda::forward(layer, current, c.batch)); current = activations.back();
            }
            result.output = activations.back();
            // Match Network::backward's public full-call behavior: recompute activations.
            activations.clear(); current = input;
            for (const auto& layer : layers) {
                activations.push_back(kan::cuda::forward(layer, current, c.batch)); current = activations.back();
            }
            result.gradients.layers.resize(layers.size()); auto dy = std::span<const double>(upstream);
            for (std::size_t l = layers.size(); l-- > 0;) {
                auto x = l == 0 ? std::span<const double>(input) : std::span<const double>(activations[l-1]);
                result.gradients.layers[l] = kan::cuda::backward(layers[l], x, c.batch, dy);
                dy = kan_grad(result.gradients,l).input;
            }
            result.gradients.input = kan_grad(result.gradients, 0).input;
        }
        result.parameters.sgd(result.gradients, learning_rate);
        const auto elapsed = milliseconds(start);
        if (step >= warmups) result.times.push_back(elapsed);
    }
    return result;
}
#ifdef KAN_BENCH_RESIDENT
Result resident(const Case& c, const std::vector<double>& input, const std::vector<double>& upstream,
                int warmups, int repeats, bool transfers) {
    auto start = Clock::now();
    Result result{network(c)};
    kan::cuda::ResidentNetwork gpu(result.parameters, c.batch);
    gpu.upload_input(input, c.batch); gpu.upload_output_gradient(upstream); gpu.synchronize();
    result.setup_ms = milliseconds(start); result.allocations = gpu.workspace_allocations();
    for (int step = 0; step < warmups + repeats; ++step) {
        start = Clock::now();
        if (transfers) { gpu.upload_input(input, c.batch); gpu.upload_output_gradient(upstream); }
        gpu.forward(); gpu.backward();
        if (transfers) { result.output = gpu.download_output(); result.gradients = gpu.download_gradients(); }
        gpu.sgd(learning_rate); gpu.synchronize();
        const auto elapsed = milliseconds(start);
        if (step >= warmups) result.times.push_back(elapsed);
    }
    result.parameters = gpu.download_parameters();
    if (result.allocations != gpu.workspace_allocations()) throw std::runtime_error("workspace allocation count changed");
    // SGD invalidates the downloadable output/gradient state. Replay the identical
    // trajectory outside timing for steady-mode pre-update verification; compare
    // its final parameters independently with the actually timed network.
    if (!transfers) {
        kan::cuda::ResidentNetwork replay(network(c), c.batch);
        replay.upload_input(input, c.batch); replay.upload_output_gradient(upstream);
        for (int step = 0; step < warmups + repeats; ++step) {
            replay.forward(); replay.backward();
            if (step == warmups + repeats - 1) {
                result.output = replay.download_output(); result.gradients = replay.download_gradients();
            }
            replay.sgd(learning_rate);
        }
        replay.synchronize();
        const auto replay_parameters = replay.download_parameters();
        for (std::size_t l = 0; l < result.parameters.layers().size(); ++l) {
            const auto expected = kan_layer(result.parameters,l);
            const auto actual = kan_layer(replay_parameters,l);
            if (!std::equal(expected.coefficients().begin(), expected.coefficients().end(), actual.coefficients().begin()) ||
                !std::equal(expected.bias().begin(), expected.bias().end(), actual.bias().begin()) ||
                centers(expected) != centers(actual) || log_widths(expected) != log_widths(actual))
                throw std::runtime_error("resident verification replay changed parameters");
        }
    }
    return result;
}
#endif
double compare(std::span<const double> expected, std::span<const double> actual) {
    if (expected.size() != actual.size()) throw std::runtime_error("verification shape mismatch");
    double max_error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto error = std::abs(expected[i]-actual[i]);
        if (!std::isfinite(actual[i]) || error > 2e-10 * (1 + std::abs(expected[i])))
            throw std::runtime_error("verification numerical mismatch");
        max_error = std::max(max_error, error);
    }
    return max_error;
}
double verify(const Result& expected, const Result& actual) {
    double error = compare(expected.output, actual.output);
    error = std::max(error, compare(expected.gradients.input, actual.gradients.input));
    for (std::size_t l = 0; l < expected.parameters.layers().size(); ++l) {
        error = std::max(error, compare(kan_grad(expected.gradients,l).coefficients, kan_grad(actual.gradients,l).coefficients));
        error = std::max(error, compare(kan_grad(expected.gradients,l).bias, kan_grad(actual.gradients,l).bias));
        error = std::max(error, compare(centers(kan_grad(expected.gradients,l)), centers(kan_grad(actual.gradients,l))));
        error = std::max(error, compare(log_widths(kan_grad(expected.gradients,l)), log_widths(kan_grad(actual.gradients,l))));
        error = std::max(error, compare(centers(kan_layer(expected.parameters,l)), centers(kan_layer(actual.parameters,l))));
        error = std::max(error, compare(log_widths(kan_layer(expected.parameters,l)), log_widths(kan_layer(actual.parameters,l))));
        error = std::max(error, compare(kan_layer(expected.parameters,l).coefficients(), kan_layer(actual.parameters,l).coefficients()));
        error = std::max(error, compare(kan_layer(expected.parameters,l).bias(), kan_layer(actual.parameters,l).bias()));
    }
    return error;
}
void print(const Case& c, const char* backend, const Result& result, double error, int warmups, int repeats) {
    std::cout << c.id << ',' << names[c.family] << ',';
    for (std::size_t i = 0; i < c.widths.size(); ++i) std::cout << (i ? "x" : "") << c.widths[i];
    std::cout << ',' << c.batch << ",7,double," << backend << ',' << warmups << ',' << repeats << ','
              << result.setup_ms << ',' << quantile(result.times, 0.5) << ','
              << quantile(result.times, 0.75)-quantile(result.times, 0.25) << ','
              << checksum(result.output) << ',' << checksum(result.gradients) << ',' << checksum(result.parameters) << ','
              << error << ',' << result.allocations << ',';
    for (std::size_t i = 0; i < result.times.size(); ++i) std::cout << (i ? ";" : "") << result.times[i];
    std::cout << '\n' << std::flush;
}
}
int main(int argc, char** argv) {
    try {
        int warmups = 2, repeats = 7, selected = -1; std::string backend = "all";
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (i+1 >= argc) throw std::invalid_argument("option requires value");
            const std::string value = argv[++i];
            if (option == "--warmups") warmups = std::stoi(value);
            else if (option == "--repeats") repeats = std::stoi(value);
            else if (option == "--case") selected = std::stoi(value);
            else if (option == "--backend") backend = value;
            else throw std::invalid_argument("unknown option");
        }
        if (warmups < 0 || repeats < 1 || selected < -1 || selected >= 12) throw std::invalid_argument("invalid benchmark options");
        if (backend != "all" && backend != "cpu" && backend != "resident") throw std::invalid_argument("unknown backend");
#ifndef KAN_BENCH_RESIDENT
        if (backend == "resident") throw std::runtime_error("resident benchmark not built");
#endif
        if (backend != "cpu" && !kan::cuda::available()) throw std::runtime_error("real CUDA device required");
        std::cout << std::setprecision(17)
          << "case,family,topology,batch,terms,precision,backend,warmups,repeats,setup_ms,median_ms,iqr_ms,output_checksum,gradient_checksum,parameter_checksum,max_abs_error,workspace_allocations,samples_ms\n";
        int id = 0;
        for (int family = 0; family < 3; ++family)
            for (int shape = 0; shape < 2; ++shape)
                for (const std::size_t batch : {32u, 1024u}) {
                    Case c{id++, family, shape == 0 ? std::vector<std::size_t>{16,24,8} : std::vector<std::size_t>{64,64,32,16}, batch};
                    if (selected != -1 && selected != c.id) continue;
                    const auto input = data(batch*c.widths.front(), 0.75, 1);
                    const auto upstream = data(batch*c.widths.back(), 0.02/static_cast<double>(batch), 23);
                    const auto reference = host(c, input, upstream, warmups, repeats, false);
                    if (backend == "all" || backend == "cpu") print(c, "cpu_full", reference, 0, warmups, repeats);
                    if (false) {
                        const auto result = host(c, input, upstream, warmups, repeats, true);
                        print(c, "m1_host_full", result, verify(reference, result), warmups, repeats);
                    }
#ifdef KAN_BENCH_RESIDENT
                    if (backend == "all" || backend == "resident") {
                        for (const bool transfers : {false, true}) {
                            const auto result = resident(c, input, upstream, warmups, repeats, transfers);
                            print(c, transfers ? "m3_transfer_full" : "m3_resident_full", result, verify(reference, result), warmups, repeats);
                        }
                    }
#endif
                }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

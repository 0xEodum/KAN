// M2: denominator policies on the resident CUDA executor (parity with the CPU).
#include "kan/cuda.hpp"
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "support/families.hpp"
#include "support/test.hpp"
#include <cmath>
#include <utility>

namespace {
using kan::DenominatorPolicy;
constexpr DenominatorPolicy safe_policies[] = {DenominatorPolicy::Absolute, DenominatorPolicy::Smooth};

void compare(std::span<const double> a, std::span<const double> b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) test::near(a[i], b[i], 5e-10);
}
kan::Layer rational(DenominatorPolicy policy, std::size_t in, std::size_t out, std::size_t m, std::size_t n) {
    kan::RationalConfig r;
    r.numerator_degree = m; r.denominator_degree = n; r.center = 0.1; r.scale = 1.3; r.denominator_policy = policy;
    kan::Layer l(in, out, r);
    std::vector<double> a(l.coefficients().size()), b(test::denominators(l).size());
    for (std::size_t k = 0; k < a.size(); ++k) a[k] = 0.3 * std::sin(static_cast<double>(k + 1));
    // Mixed signs so that S changes sign across samples (both Absolute branches).
    for (std::size_t k = 0; k < b.size(); ++k) b[k] = 0.5 * std::cos(static_cast<double>(2 * k + 1));
    kan::set_rational_parameters(l, a, b, std::vector<double>(out, 0.01));
    return l;
}
} // namespace

TEST(policy_mixed_network_all_vjps_and_trajectory_match_cpu) {
    for (auto policy : {DenominatorPolicy::Guarded, DenominatorPolicy::Absolute, DenominatorPolicy::Smooth})
        for (const auto orders : {std::pair<std::size_t, std::size_t>{0, 0}, {0, 3}, {4, 1}, {3, 2}, {16, 16}}) {
            // The guarded policy needs moderate denominators to stay clear of poles.
            if (policy == DenominatorPolicy::Guarded && orders.second == 16) continue;
            kan::Layer basis(3, 2, kan::ChebyshevConfig{3});
            basis.set_parameters(std::vector<double>(18, 0.02), std::vector<double>(2, 0));
            kan::Network cpu({rational(policy, 2, 3, orders.first, orders.second), basis,
                              rational(policy, 2, 1, 3, 2)});
            kan::cuda::ResidentNetwork gpu(cpu, 8);
            const auto count = gpu.workspace_allocations();
            const std::vector<double> x{-0.5, 0.2, 0.8, -0.2, 0.1, 0.4}, dy{0.2, -0.1, 0.3};
            gpu.upload_input(x, 3); gpu.upload_output_gradient(dy);
            for (int step = 0; step < 3; ++step) {
                gpu.forward(); compare(gpu.download_output(), cpu.forward(x, 3));
                gpu.backward(0.1);
                auto expected = cpu.backward(x, 3, dy);
                const auto reg = cpu.regularization(0.1).gradients;
                for (std::size_t j = 0; j < expected.layers.size(); ++j)
                    for (std::size_t k = 0; k < expected.layers[j].coefficients.size(); ++k)
                        expected.layers[j].coefficients[k] += reg.layers[j].coefficients[k];
                const auto actual = gpu.download_gradients();
                compare(actual.input, expected.input);
                for (std::size_t j = 0; j < expected.layers.size(); ++j) {
                    compare(actual.layers[j].coefficients, expected.layers[j].coefficients);
                    compare(test::denominators(actual.layers[j]), test::denominators(expected.layers[j]));
                    compare(actual.layers[j].bias, expected.layers[j].bias);
                }
                gpu.sgd(0.03); cpu.sgd(expected, 0.03);
            }
            const auto trained = gpu.download_parameters();
            for (std::size_t j = 0; j < cpu.layers().size(); ++j) {
                compare(trained.layers()[j].coefficients(), cpu.layers()[j].coefficients());
                compare(test::denominators(trained.layers()[j]), test::denominators(cpu.layers()[j]));
                REQUIRE(trained.layers()[j].carrier().index() == cpu.layers()[j].carrier().index());
            }
            REQUIRE(std::get<kan::RationalEdges>(trained.layers()[0].carrier()).config.denominator_policy == policy);
            REQUIRE(count == gpu.workspace_allocations());
        }
}

namespace {
kan::Network pole_network(DenominatorPolicy policy) {
    kan::RationalConfig r;
    r.numerator_degree = 0; r.denominator_degree = 1; r.denominator_policy = policy;
    kan::Layer layer(1, 1, r);
    kan::set_rational_parameters(layer, std::vector<double>{1}, std::vector<double>{-0.5}, std::vector<double>{0});
    return kan::Network({layer});
}
double resident_step(kan::cuda::ResidentNetwork& gpu) {
    gpu.forward();
    const double u = gpu.download_output()[0] - 4;
    gpu.upload_output_gradient(std::vector<double>{u});
    gpu.backward(); gpu.sgd(0.0625);
    return 0.5 * u * u;
}
} // namespace

TEST(resident_sgd_into_a_pole_stops_guarded_but_continues_under_safe_policies) {
    kan::cuda::ResidentNetwork guarded(pole_network(DenominatorPolicy::Guarded), 1);
    guarded.upload_input(std::vector<double>{1}, 1);
    resident_step(guarded);
    test::throws<std::domain_error>([&] { guarded.forward(); });
    for (auto policy : safe_policies) {
        auto cpu = pole_network(policy);
        kan::cuda::ResidentNetwork gpu(cpu, 1);
        gpu.upload_input(std::vector<double>{1}, 1);
        const double first = resident_step(gpu);
        double last = first;
        for (int step = 0; step < 200; ++step) {
            last = resident_step(gpu);
            REQUIRE(std::isfinite(last));
        }
        REQUIRE(last < 1e-6 * first);
        // Same trajectory on the CPU.
        const std::vector<double> x{1};
        for (int step = 0; step < 201; ++step) {
            const double u = cpu.forward(x, 1)[0] - 4;
            cpu.sgd(cpu.backward(x, 1, std::vector<double>{u}), 0.0625);
        }
        const auto trained = gpu.download_parameters();
        compare(trained.layers()[0].coefficients(), cpu.layers()[0].coefficients());
        compare(test::denominators(trained.layers()[0]), test::denominators(cpu.layers()[0]));
    }
}

TEST(resident_safe_policies_report_overflow_not_poles) {
    for (auto policy : safe_policies) {
        kan::RationalConfig r;
        r.numerator_degree = 1; r.denominator_degree = 1; r.denominator_policy = policy;
        kan::Layer l(1, 1, r);
        kan::set_rational_parameters(l, std::vector<double>{1, -1}, std::vector<double>{-1}, std::vector<double>{0});
        kan::cuda::ResidentNetwork gpu(kan::Network({l}), 1);
        gpu.upload_input(std::vector<double>{1}, 1);
        gpu.upload_output_gradient(std::vector<double>{1});
        gpu.forward(); gpu.backward();
        test::near(gpu.download_output()[0], 0);
        r.numerator_degree = 0;
        kan::Layer huge(1, 1, r);
        kan::set_rational_parameters(huge, std::vector<double>{1}, std::vector<double>{1e200}, std::vector<double>{0});
        kan::cuda::ResidentNetwork over(kan::Network({huge}), 1);
        over.upload_input(std::vector<double>{1e110}, 1);
        // S = b z = 1e310 overflows in Horner for both policies (for a finite
        // S, 1 + |S| cannot overflow; Smooth also fails when only S^2 does).
        test::throws<std::overflow_error>([&] { over.forward(); });
        // S = 1e210 is finite: Absolute evaluates, Smooth overflows in S^2.
        over.upload_input(std::vector<double>{1e10}, 1);
        if (policy == DenominatorPolicy::Smooth) {
            test::throws<std::overflow_error>([&] { over.forward(); });
        } else {
            over.forward();
            test::near(over.download_output()[0] / 1e-210, 1);
        }
    }
}

int main() {
    if (!kan::cuda::available()) { std::cout << "CUDA device unavailable\n"; return 0; }
    return test::run();
}

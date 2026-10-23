// M5 phase 1, RED test 9: a complete PQC layer call (forward, backward, SGD)
// performs no per-edge or per-sample heap allocation: the count is the same
// for every width and batch (the allocation probe of backlog M4).
#include "kan/families.hpp"
#include "kan/layer.hpp"
#include "support/test.hpp"
#include <cstdlib>
#include <iostream>
#include <new>

namespace {
bool counting = false;
std::size_t allocation_count = 0;
} // namespace
void* operator new(std::size_t size) {
    if (counting) ++allocation_count;
    if (auto* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

TEST(complete_pqc_layer_call_has_no_per_edge_heap_allocations) {
    kan::PqcConfig config;
    config.qubits = 3;
    config.repetitions = 4;
    config.entangler = kan::Entangler::CnotRing;
    std::size_t reference = 0;
    bool first = true, constant = true;
    for (std::size_t width : {1u, 8u, 32u})
        for (std::size_t batch : {1u, 64u}) {
            kan::Layer layer(width, width, config);
            const auto& e = std::get<kan::PqcEdges>(layer.carrier());
            std::vector<double> c(e.coefficients.size(), 0.01), a(e.angles.size(), 0.3),
                w(e.encoding_weights.size(), 0.9), b(e.encoding_biases.size(), 0.1), bias(width);
            kan::set_pqc_parameters(layer, c, a, w, b, bias);
            std::vector<double> x(width * batch, 0.2), up(width * batch, 0.1);
            allocation_count = 0;
            counting = true;
            auto y = layer.forward(x, batch);
            auto gradient = layer.backward(x, batch, up);
            layer.sgd(gradient, 0.001);
            counting = false;
            std::cout << "width=" << width << " batch=" << batch << " full_call_allocations=" << allocation_count
                      << '\n';
            if (first) reference = allocation_count;
            first = false;
            constant = constant && allocation_count == reference;
            REQUIRE(y.size() == width * batch);
        }
    REQUIRE(constant);
    REQUIRE(reference < 64);
}
int main() { return test::run(); }

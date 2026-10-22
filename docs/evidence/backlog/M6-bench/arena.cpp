// Backlog M6: arena bytes of B-spline networks (test hook ResidentLayout),
// base (c06b2ca, -DM6_BASE: shared grids only) against M6 (shared and per-input grids).
// Usage: arena <f64|f32> <widths> <capacity> <intervals> <degree> <shared|input>
// Build: scripts/link_harness.sh <build> docs/evidence/backlog/M6-bench/arena.cpp <exe> -Isrc [-DM6_BASE]
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 7) return 2;
    std::vector<std::size_t> w; std::stringstream s(argv[2]); std::string item;
    while (std::getline(s, item, 'x')) w.push_back(std::stoul(item));
    const std::size_t capacity = std::stoul(argv[3]), G = std::stoul(argv[4]), p = std::stoul(argv[5]);
    const bool input = std::string(argv[6]) == "input";
    std::vector<double> t(p, -1);
    for (std::size_t j = 0; j <= G; ++j) t.push_back(-1 + 2.0 * double(j) / double(G));
    t.insert(t.end(), p, 1.0);
    std::vector<kan::NetworkLayer> layers;
    for (std::size_t j = 0; j + 1 < w.size(); ++j) {
        kan::BSplineConfig c{p, t};
#ifndef M6_BASE
        if (input) c = kan::per_input_grids(c, w[j]);
#endif
        layers.emplace_back(kan::Layer(w[j], w[j + 1], c));
    }
    kan::cuda::testing::ResidentLayout layout;
    kan::cuda::testing::ResidentTestScope scope(kan::cuda::testing::SplinePath::Automatic, &layout);
    kan::cuda::ResidentNetwork r(kan::Network(std::move(layers)), capacity,
                                 std::string(argv[1]) == "f64" ? kan::cuda::Precision::Float64 : kan::cuda::Precision::Float32);
    std::printf("%s,%s,%zu,%zu,%zu,%s,%zu,%d\n", argv[1], argv[2], capacity, G, p, argv[6], layout.arena_bytes, int(layout.sparse.at(0)));
}

// Per-kernel timing probe for the wavelet launch cost model (M7 review).
// Reads lines: <f32|f64> <in> <out> <K> <batch> <fs> <bs> <slots> <splits> <isplits>
// Network: Chebyshev in->in (so the wavelet layer computes dx), then the
// wavelet layer in->out with K wavelets. Each line runs forward()+backward()
// `reps` times inside an NVTX range named by the line.
#include "kan/families.hpp"
#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include <nvtx3/nvToolsExt.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

std::vector<double> wave(std::size_t n, double s, double f, double p = 0) {
    std::vector<double> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = s * std::sin(f * double(k) + p);
    return v;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 10;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string p;
        std::size_t inputs, outputs, K, batch;
        unsigned fs, bs, slots, splits, isplits;
        in >> p >> inputs >> outputs >> K >> batch >> fs >> bs >> slots >> splits >> isplits;
        kan::Layer first(inputs, inputs, kan::ChebyshevConfig{2});
        first.set_parameters(wave(inputs * inputs * 2, 0.3 / std::sqrt(double(inputs)), 0.7), std::vector<double>(inputs, 0.0));
        kan::Layer layer(inputs, outputs, kan::WaveletConfig{K});
        std::vector<double> c(layer.coefficients().size());
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = 0.1 / std::sqrt(double(inputs * K)) * std::sin(0.37 * double(k));
        layer.set_parameters(c, std::vector<double>(outputs, 0.0));
        kan::Network net({first, layer});
        const auto precision = p == "f64" ? kan::cuda::Precision::Float64 : kan::cuda::Precision::Float32;
        std::unique_ptr<kan::cuda::ResidentNetwork> r;
        try {
            kan::cuda::testing::ResidentTestScope scope(kan::cuda::testing::WaveletLaunch{fs, bs, slots, splits, isplits});
            r = std::make_unique<kan::cuda::ResidentNetwork>(net, batch, precision);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "skip %s: %s\n", line.c_str(), e.what());
            continue;
        }
        r->upload_input(wave(batch * inputs, 0.95, 0.113), batch);
        r->upload_output_gradient(wave(batch * outputs, 1.0 / double(batch), 0.07, 0.1));
        r->forward(); r->backward();
        nvtxRangePushA(line.c_str());
        for (int i = 0; i < reps; ++i) { r->forward(); r->backward(); }
        nvtxRangePop();
    }
}

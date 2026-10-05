// R9: cost of ResidentNetwork::upload_parameters relative to constructing the
// executor, for a small and a large Chebyshev network in FP64 and FP32.
//   upload_timing timing                  CSV: case,precision,operation,median_ms,repeats
//   upload_timing profile <case> <prec>   constructs, then 10 uploads inside
//                                         cudaProfilerStart/Stop (for nsys
//                                         --capture-range=cudaProfilerApi)
#include "kan/resident.hpp"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {
using kan::cuda::Precision;
using kan::cuda::ResidentNetwork;

struct Case { const char* name; std::vector<std::size_t> widths; std::size_t capacity; int repeats; };
const Case cases[] = {
    {"small 64x64x32x16 K7 cap1024", {64, 64, 32, 16}, 1024, 21},
    {"large 1024x1024x1024 K7 cap4096", {1024, 1024, 1024}, 4096, 7},
};

kan::Network network(const Case& c, double phase) {
    std::vector<kan::Layer> layers;
    for (std::size_t j = 0; j + 1 < c.widths.size(); ++j) {
        kan::Layer l(c.widths[j], c.widths[j+1], kan::ChebyshevConfig{7});
        std::vector<double> coefficients(l.coefficients().size()), bias(l.outputs());
        const double scale = 1.0/std::sqrt(static_cast<double>(l.inputs()*l.terms()));
        for (std::size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = scale*std::sin(0.37*static_cast<double>(i)+phase);
        for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = 0.01*std::cos(static_cast<double>(i)+phase);
        l.set_parameters(coefficients, bias);
        layers.push_back(std::move(l));
    }
    return kan::Network(std::move(layers));
}
double median_ms(int repeats, const std::function<void()>& f) {
    std::vector<double> t;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        f();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size()/2];
}
Precision precision(const std::string& s) { return s == "fp32" ? Precision::Float32 : Precision::Float64; }
}

int main(int argc, char** argv) {
    if (!kan::cuda::available()) { std::cerr << "CUDA device required\n"; return 1; }
    const std::string mode = argc > 1 ? argv[1] : "timing";
    if (mode == "profile" && argc == 4) {
        const auto& c = cases[std::atoi(argv[2])];
        const auto a = network(c, 0.0), b = network(c, 1.0);
        ResidentNetwork gpu(a, c.capacity, precision(argv[3]));
        gpu.upload_parameters(b); // warm-up outside the capture
        cudaProfilerStart();
        for (int r = 0; r < 10; ++r) gpu.upload_parameters(r % 2 ? b : a);
        cudaProfilerStop();
        return 0;
    }
    std::cout << "case,precision,operation,median_ms,repeats\n";
    for (const auto& c : cases) {
        const auto a = network(c, 0.0), b = network(c, 1.0);
        for (const char* p : {"fp64", "fp32"}) {
            const auto prec = precision(p);
            { ResidentNetwork warm(a, c.capacity, prec); warm.upload_parameters(b); }
            const auto construct = median_ms(c.repeats, [&] { ResidentNetwork gpu(a, c.capacity, prec); gpu.synchronize(); });
            ResidentNetwork gpu(a, c.capacity, prec);
            int r = 0;
            const auto upload = median_ms(c.repeats, [&] { gpu.upload_parameters(++r % 2 ? b : a); });
            const auto download = median_ms(c.repeats, [&] { (void)gpu.download_parameters(); });
            std::cout << c.name << ',' << p << ",construct," << construct << ',' << c.repeats << '\n'
                      << c.name << ',' << p << ",upload_parameters," << upload << ',' << c.repeats << '\n'
                      << c.name << ',' << p << ",download_parameters," << download << ',' << c.repeats << '\n';
        }
    }
    return 0;
}

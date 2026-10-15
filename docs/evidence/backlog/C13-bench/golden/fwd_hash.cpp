#include "kan/resident.hpp"
#include "backends/cuda/resident_testing.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
int main() {
    for (std::size_t terms : {60u, 130u, 300u}) for (int prec = 0; prec < 2; ++prec) for (std::size_t degree : {3u, 5u}) {
        kan::BSplineConfig b{degree, std::vector<double>(degree + 1, -1.0)};
        for (std::size_t k = 1; k < terms - degree; ++k) b.knots.push_back(-1.0 + 2.0 * k / double(terms - degree));
        b.knots.insert(b.knots.end(), degree + 1, 1.0);
        kan::Layer l(16, 8, b); std::vector<double> c(l.coefficients().size());
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = 0.2 * std::sin(0.731 * k);
        l.set_parameters(c, std::vector<double>(8, 0.01));
        kan::cuda::testing::ResidentTestScope scope(kan::cuda::testing::SplinePath::Dense);
        kan::cuda::ResidentNetwork r(kan::Network({l}), 100, prec ? kan::cuda::Precision::TensorFloat32 : kan::cuda::Precision::Float32);
        std::vector<double> x(1600); for (std::size_t k = 0; k < x.size(); ++k) x[k] = 1.02 * std::sin(0.377 * k);
        r.upload_input(x, 100); r.forward(); auto y = r.download_output();
        std::uint64_t h = 1469598103934665603ULL; for (double v : y) { std::uint64_t u; std::memcpy(&u, &v, 8); h = (h ^ u) * 1099511628211ULL; }
        std::printf("terms %zu prec %d degree %zu hash %016llx\n", terms, prec, degree, (unsigned long long)h);
    }
}

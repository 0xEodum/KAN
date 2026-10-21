// Old quadrature (6856b4f moments.cpp) vs new closed form of init::wavelet_moments.
#include "init/moments.hpp"
#include "init/portable_math.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <numbers>
#include <vector>
using namespace kan;
namespace old {
constexpr std::size_t panel_nodes = 8, panels = 36; constexpr double hat_window = 12;
template<class F> double uniform_mean(F f, double lo, double hi, double a, double b, std::size_t count) {
    static const auto rule = init::gauss_legendre(panel_nodes);
    const double width = (b - a) / double(count); double sum = 0;
    for (std::size_t p = 0; p < count; ++p) { const double left = a + width * double(p), half = width / 2, mid = left + half; double panel = 0;
        for (std::size_t i = 0; i < panel_nodes; ++i) panel += rule.weights[i] * f(mid + half * rule.nodes[i]); sum += panel * half; }
    return sum / (hi - lo);
}
double mexican_hat_moment(double center, double s, double lo, double hi) {
    const double amplitude2 = 4 / (3 * std::sqrt(std::numbers::pi) * s);
    const double a = std::max(lo, center - hat_window * s), b = std::min(hi, center + hat_window * s);
    return uniform_mean([&](double x) { const double q = (x - center) / s, r = 1 - q * q; return amplitude2 * r * r * init::portable_exp(-q * q); }, lo, hi, a, b, panels);
}
std::vector<double> moments(const std::vector<double>& t, const std::vector<double>& l) {
    std::vector<double> s(l.size()); for (size_t j=0;j<l.size();++j) s[j]=init::portable_exp(l[j]);
    double lo=t[0]-s[0], hi=t[0]+s[0]; for (size_t j=1;j<s.size();++j){lo=std::min(lo,t[j]-s[j]); hi=std::max(hi,t[j]+s[j]);}
    std::vector<double> m(s.size()); std::map<std::pair<double,double>,double> known;
    for (size_t j=0;j<s.size();++j){ auto [e,ins]=known.try_emplace({t[j],l[j]},0.0); if(ins) e->second=mexican_hat_moment(t[j],s[j],lo,hi); m[j]=e->second; }
    return m;
}
}
long double ref_mass(long double q) { return erfl(q) / 2 + (q - 2 * q * q * q) * expl(-q * q) / (3 * sqrtl(3.14159265358979323846264338327950288L)); }
std::vector<long double> reference(const std::vector<double>& t, const std::vector<double>& l) {
    std::vector<double> s(l.size()); for (size_t j=0;j<l.size();++j) s[j]=init::portable_exp(l[j]);
    double lo=t[0]-s[0], hi=t[0]+s[0]; for (size_t j=1;j<s.size();++j){lo=std::min(lo,t[j]-s[j]); hi=std::max(hi,t[j]+s[j]);}
    std::vector<long double> m(s.size());
    for (size_t j=0;j<s.size();++j) m[j]=(ref_mass(((long double)hi-t[j])/s[j])-ref_mass(((long double)lo-t[j])/s[j]))/((long double)hi-lo);
    return m;
}
int main() {
    double worst = 0, worst_old = 0, worst_new = 0; std::uint64_t st = 3;
    auto u = [&]{ st = st*6364136223846793005ull+1442695040888963407ull; return (double(st>>11)+0.5)*0x1p-53; };
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + trial % 50;
        std::vector<double> t(n), l(n);
        const double spread = std::pow(10.0, -3 + 6 * u()), lscale = -8 + 12 * u();
        for (std::size_t j = 0; j < n; ++j) { t[j] = spread * (2 * u() - 1); l[j] = lscale + 3 * (u() - 0.5); }
        const auto a = old::moments(t, l); const auto b = init::wavelet_moments(t, l).second_moments;
        const auto r = reference(t, l);
        for (std::size_t j = 0; j < n; ++j) {
            worst = std::max(worst, std::abs(a[j] - b[j]) / a[j]);
            worst_old = std::max(worst_old, (double)(std::abs(a[j] - r[j]) / r[j]));
            worst_new = std::max(worst_new, (double)(std::abs(b[j] - r[j]) / r[j]));
        }
    }
    std::printf("200 random sets (1-50 wavelets, translations spread 1e-3..1e3, log scales -9.5..5.5): max relative difference quadrature vs closed form %.3g; against a long-double erfl reference: quadrature %.3g, closed form %.3g\n", worst, worst_old, worst_new);
    // Timing at 1024 x 1024 distinct wavelets.
    const std::size_t count = 1024 * 1024; std::vector<double> t(count), l(count);
    for (std::size_t j = 0; j < count; ++j) { t[j] = std::sin(0.37 * double(j)); l[j] = -1.2 + 0.8 * std::cos(0.91 * double(j)); }
    auto t0 = std::chrono::steady_clock::now(); auto b = init::wavelet_moments(t, l);
    const double tn = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now(); auto a = old::moments(t, l);
    const double to = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double w2 = 0; for (std::size_t j = 0; j < count; ++j) w2 = std::max(w2, std::abs(a[j] - b.second_moments[j]) / a[j]);
    std::printf("1024x1024 distinct wavelets: quadrature %.3f s (%.2f us/wavelet), closed form %.3f s (%.3f us/wavelet), max rel diff %.3g\n", to, to/count*1e6, tn, tn/count*1e6, w2);
}

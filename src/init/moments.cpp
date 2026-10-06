// Reference measures and second moments of the basis families (backlog M4).
// Closed forms where they exist; otherwise deterministic Gauss-Legendre
// quadrature evaluated with portable arithmetic (see portable_math.hpp).
#include "moments.hpp"
#include "portable_math.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <variant>

namespace kan {
namespace {
using init::gauss_legendre;
using init::portable_exp;

constexpr std::size_t panel_nodes = 8;
constexpr std::size_t panels = 36;
constexpr double rbf_window = 9;   // exp(-2*81) is far below double resolution
constexpr double hat_window = 12;  // (1-q^2)^2 exp(-q^2) at q = 12: about 1e-58

BasisMoments constant_moments(std::size_t terms, double variance, double first, double rest) {
    BasisMoments m{variance, std::vector<double>(terms, rest)};
    m.second_moments[0] = first;
    return m;
}

BasisMoments moments(const ChebyshevConfig& c) { return constant_moments(c.size, 0.5, 1, 0.5); } // arcsine
BasisMoments moments(const LegendreConfig& c) {
    BasisMoments m{1.0 / 3, std::vector<double>(c.size)};
    for (std::size_t k = 0; k < c.size; ++k) m.second_moments[k] = 1.0 / double(2 * k + 1);
    return m;
}
// Normalized weight (1-x)^a (1+x)^b: m_n = h_n / mu_0, computed by its ratio
// recurrence (the gamma-function form has a removable 0/0 at a+b = -1).
BasisMoments moments(const JacobiConfig& c) {
    const double a = c.alpha, b = c.beta, ab = a + b;
    BasisMoments m{4 * (a + 1) * (b + 1) / ((ab + 2) * (ab + 2) * (ab + 3)), std::vector<double>(c.size)};
    m.second_moments[0] = 1;
    if (c.size > 1) m.second_moments[1] = (a + 1) * (b + 1) / (ab + 3);
    for (std::size_t k = 2; k < c.size; ++k) {
        const double n = double(k);
        m.second_moments[k] = m.second_moments[k - 1] * ((a + n) * (b + n) / (n * (ab + n))) *
                              ((2 * n + ab - 1) / (2 * n + ab + 1));
    }
    return m;
}
BasisMoments moments(const HermiteConfig& c) { // x ~ N(0, 1/2): E[H_n^2] = 2^n n!
    BasisMoments m{0.5, std::vector<double>(c.size)};
    m.second_moments[0] = 1;
    for (std::size_t k = 1; k < c.size; ++k) m.second_moments[k] = 2 * double(k) * m.second_moments[k - 1];
    return m;
}
BasisMoments moments(const FourierConfig& c) { // uniform on [-pi/w, pi/w]
    const double half = std::numbers::pi / c.frequency;
    return constant_moments(c.size, half * half / 3, 1, 0.5);
}

// Uniform on [lo, hi]: E[f] = (1/(hi-lo)) * integral over [a, b] of f by
// `count` panels of the 8-point rule.
template<class F>
double uniform_mean(F f, double lo, double hi, double a, double b, std::size_t count) {
    static const auto rule = gauss_legendre(panel_nodes);
    const double width = (b - a) / double(count);
    double sum = 0;
    for (std::size_t p = 0; p < count; ++p) {
        const double left = a + width * double(p), half = width / 2, mid = left + half;
        double panel = 0;
        for (std::size_t i = 0; i < panel_nodes; ++i) panel += rule.weights[i] * f(mid + half * rule.nodes[i]);
        sum += panel * half;
    }
    return sum / (hi - lo);
}

std::pair<double, double> center_range(const std::vector<double>& centers, double largest) {
    const auto [lo, hi] = std::minmax_element(centers.begin(), centers.end());
    if (*lo < *hi) return {*lo, *hi};
    return {*lo - largest, *lo + largest};
}

double uniform_variance(double lo, double hi) {
    const double half = hi / 2 - lo / 2;
    return half * half / 3;
}

// Gaussian terms exp(-((x-c)/w)^2), uniform on the center range.
BasisMoments gaussian_moments(const std::vector<double>& centers, const std::vector<double>& widths) {
    const auto [lo, hi] = center_range(centers, *std::max_element(widths.begin(), widths.end()));
    BasisMoments m{uniform_variance(lo, hi), std::vector<double>(centers.size())};
    for (std::size_t k = 0; k < centers.size(); ++k) {
        const double c = centers[k], w = widths[k];
        const double a = std::max(lo, c - rbf_window * w), b = std::min(hi, c + rbf_window * w);
        m.second_moments[k] = uniform_mean([&](double x) {
            const double q = (x - c) / w;
            return portable_exp(-2 * q * q);
        }, lo, hi, a, b, panels);
    }
    return m;
}
BasisMoments moments(const GaussianRbfConfig& c) {
    return gaussian_moments(c.centers, std::vector<double>(c.centers.size(), c.width));
}
BasisMoments moments(const TrainableRbfConfig& c) {
    std::vector<double> widths(c.log_widths.size());
    std::transform(c.log_widths.begin(), c.log_widths.end(), widths.begin(), portable_exp);
    return gaussian_moments(c.centers, widths);
}
BasisMoments moments(const MexicanHatConfig& c) {
    const auto [lo, hi] = center_range(c.centers, *std::max_element(c.scales.begin(), c.scales.end()));
    BasisMoments m{uniform_variance(lo, hi), std::vector<double>(c.centers.size())};
    const double sqrt_pi = std::sqrt(std::numbers::pi);
    for (std::size_t k = 0; k < c.centers.size(); ++k) {
        const double center = c.centers[k], s = c.scales[k], amplitude2 = 4 / (3 * sqrt_pi * s);
        const double a = std::max(lo, center - hat_window * s), b = std::min(hi, center + hat_window * s);
        m.second_moments[k] = uniform_mean([&](double x) {
            const double q = (x - center) / s, r = 1 - q * q;
            return amplitude2 * r * r * portable_exp(-q * q);
        }, lo, hi, a, b, panels);
    }
    return m;
}
// Uniform on the domain [t_p, t_K]: the (p+1)-point rule is exact on each span.
BasisMoments moments(const BSplineConfig& c) {
    const auto terms = basis_size(c);
    const auto& t = c.knots;
    const double lo = t[c.degree], hi = t[terms];
    const auto rule = gauss_legendre(c.degree + 1);
    BasisMoments m{uniform_variance(lo, hi), std::vector<double>(terms, 0.0)};
    const BasisConfig config = c;
    for (std::size_t j = c.degree; j < terms; ++j) {
        if (!(t[j] < t[j + 1])) continue;
        const double half = t[j + 1] / 2 - t[j] / 2, mid = t[j] + half;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            const auto values = evaluate_basis(config, mid + half * rule.nodes[i]).values;
            for (std::size_t k = 0; k < terms; ++k) m.second_moments[k] += rule.weights[i] * half * values[k] * values[k];
        }
    }
    const double length = 2 * (hi / 2 - lo / 2);
    for (auto& value : m.second_moments) value /= length;
    return m;
}
} // namespace

BasisMoments reference_moments(const BasisConfig& config) {
    validate_basis(config);
    auto m = std::visit([](const auto& c) { return moments(c); }, config);
    bool finite = std::isfinite(m.variance) && m.variance > 0;
    for (double v : m.second_moments) finite = finite && std::isfinite(v);
    if (!finite) throw std::overflow_error("reference moments are not finite");
    return m;
}

namespace init {
std::vector<double> rational_moments(DenominatorPolicy policy, std::size_t numerator_degree,
                                     std::span<const double> beta) {
    std::vector<double> mu(numerator_degree + 1, 0.0);
    constexpr std::size_t rational_panels = 16;
    static const auto rule = gauss_legendre(panel_nodes);
    const double width = 2.0 / double(rational_panels);
    for (std::size_t p = 0; p < rational_panels; ++p) {
        const double half = width / 2, mid = -1 + width * double(p) + half;
        for (std::size_t i = 0; i < panel_nodes; ++i) {
            const double u = mid + half * rule.nodes[i];
            double s = 0;
            for (std::size_t k = beta.size(); k > 0; --k) s = (s + beta[k - 1]) * u;
            const double q = policy == DenominatorPolicy::Guarded ? 1 + s
                           : policy == DenominatorPolicy::Absolute ? 1 + std::abs(s) : 1 + s * s;
            const double weight = rule.weights[i] * half / (2 * q * q);
            double power = 1; // u^(2k)
            for (std::size_t k = 0; k <= numerator_degree; ++k) {
                mu[k] += weight * power;
                power *= u * u;
            }
        }
    }
    return mu;
}
} // namespace init

} // namespace kan

// Reference measures and second moments of the basis families (backlog M4).
// Closed forms where they exist; otherwise deterministic Gauss-Legendre
// quadrature evaluated with portable arithmetic (see portable_math.hpp).
#include "init/moments.hpp"
#include "init/portable_math.hpp"
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
constexpr double rbf_window = 9;  // exp(-2*81) is far below double resolution
constexpr double hat_window = 12; // (1-q^2)^2 exp(-q^2) at q = 12: about 1e-58

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
        m.second_moments[k] =
            m.second_moments[k - 1] * ((a + n) * (b + n) / (n * (ab + n))) * ((2 * n + ab - 1) / (2 * n + ab + 1));
    }
    return m;
}
// x ~ N(0, 1/2) for both normalizations, so switching the normalization keeps
// the assumed input distribution. Physicists: E[H_n^2] = 2^n n!. Functions
// (backlog M5): E[psi_n^2] = integral H_n^2 exp(-2x^2) / (2^n n! pi)
// = Gamma(n+1/2) / (sqrt(2) pi n!), so m_0 = 1/sqrt(2 pi), m_n = m_(n-1) (2n-1)/(2n).
BasisMoments moments(const HermiteConfig& c) {
    BasisMoments m{0.5, std::vector<double>(c.size)};
    if (c.normalization == HermiteNormalization::Function) {
        m.second_moments[0] = 1 / std::sqrt(2 * std::numbers::pi);
        for (std::size_t k = 1; k < c.size; ++k)
            m.second_moments[k] = m.second_moments[k - 1] * (double(2 * k - 1) / double(2 * k));
        return m;
    }
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
template<class F> double uniform_mean(F f, double lo, double hi, double a, double b, std::size_t count) {
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
        m.second_moments[k] = uniform_mean(
            [&](double x) {
                const double q = (x - c) / w;
                return portable_exp(-2 * q * q);
            },
            lo, hi, a, b, panels);
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
// E[psi^2] of the L2-normalized Mexican hat with this center and scale, x
// uniform on [lo, hi] (the MexicanHatConfig dictionary; the per-edge wavelets
// of backlog M7 use the closed form hat_mass).
double mexican_hat_moment(double center, double s, double lo, double hi) {
    const double sqrt_pi = std::sqrt(std::numbers::pi);
    const double amplitude2 = 4 / (3 * sqrt_pi * s);
    const double a = std::max(lo, center - hat_window * s), b = std::min(hi, center + hat_window * s);
    return uniform_mean(
        [&](double x) {
            const double q = (x - center) / s, r = 1 - q * q;
            return amplitude2 * r * r * portable_exp(-q * q);
        },
        lo, hi, a, b, panels);
}
BasisMoments moments(const MexicanHatConfig& c) {
    const auto [lo, hi] = center_range(c.centers, *std::max_element(c.scales.begin(), c.scales.end()));
    BasisMoments m{uniform_variance(lo, hi), std::vector<double>(c.centers.size())};
    for (std::size_t k = 0; k < c.centers.size(); ++k)
        m.second_moments[k] = mexican_hat_moment(c.centers[k], c.scales[k], lo, hi);
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
            for (std::size_t k = 0; k < terms; ++k)
                m.second_moments[k] += rule.weights[i] * half * values[k] * values[k];
        }
    }
    const double length = 2 * (hi / 2 - lo / 2);
    for (auto& value : m.second_moments) value /= length;
    return m;
}
} // namespace

namespace {
// Mass of the L2-normalized Mexican hat's square below q (in units of its
// scale), minus 1/2: G(q) = integral_{-inf}^{q} (4/(3 sqrt(pi))) (1 - t^2)^2
// exp(-t^2) dt - 1/2 = erf(q)/2 + (q - 2 q^3) exp(-q^2)/(3 sqrt(pi)), odd in q,
// with erf(q) = (2/sqrt(pi)) q exp(-q^2) sum_n (2 q^2)^n/(2n+1)!! (positive
// terms: no cancellation in the series). Beyond |q| = 7 G is +-1/2 within
// 1e-19. Basic operations and portable_exp only, so the value is bitwise
// the same on every platform (backlog M7 review: replaces a 288-node
// quadrature per wavelet, about 3.6 us, by about 0.1 us).
double hat_mass(double q) {
    if (!(q > -7)) return -0.5;
    if (!(q < 7)) return 0.5;
    const double q2 = q * q, ratio = 2 * q2;
    double term = 1, sum = 1;
    for (int n = 1; term > 0x1p-60 * sum; ++n) {
        term *= ratio / double(2 * n + 1);
        sum += term;
    }
    return portable_exp(-q2) * (q * sum + (q - 2 * q * q2) / 3) / std::sqrt(std::numbers::pi);
}
} // namespace

BasisMoments init::wavelet_moments(std::span<const double> translations, std::span<const double> log_scales) {
    std::vector<double> scales(log_scales.size());
    std::transform(log_scales.begin(), log_scales.end(), scales.begin(), portable_exp);
    double lo = translations[0] - scales[0], hi = translations[0] + scales[0];
    for (std::size_t j = 1; j < scales.size(); ++j) {
        lo = std::min(lo, translations[j] - scales[j]);
        hi = std::max(hi, translations[j] + scales[j]);
    }
    BasisMoments m{uniform_variance(lo, hi), std::vector<double>(scales.size())};
    // E[psi_j^2] = (G(q_hi) - G(q_lo)) / (hi - lo), q = (x - tau_j)/s_j; the
    // span holds the wavelet's lobe, so q_lo <= -1 < 1 <= q_hi (no cancellation).
    const double length = 2 * (hi / 2 - lo / 2);
    for (std::size_t j = 0; j < scales.size(); ++j) {
        const double tau = translations[j], s = scales[j];
        m.second_moments[j] = (hat_mass((hi - tau) / s) - hat_mass((lo - tau) / s)) / length;
    }
    bool finite = std::isfinite(m.variance) && m.variance > 0;
    for (double v : m.second_moments) finite = finite && std::isfinite(v) && v > 0;
    if (!finite) throw std::overflow_error("wavelet reference moments are not finite");
    return m;
}

BasisMoments reference_moments(const BasisConfig& config) {
    validate_basis(config);
    if (const auto* spline = std::get_if<BSplineConfig>(&config); spline && spline->grids != 1)
        throw std::invalid_argument("reference moments of per-input spline grids: use spline_grid for one input");
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
            const double q = policy == DenominatorPolicy::Guarded    ? 1 + s
                             : policy == DenominatorPolicy::Absolute ? 1 + std::abs(s)
                                                                     : 1 + s * s;
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

double pqc_second_moment(const PqcConfig& config) {
    const double states = double(std::size_t{1} << config.qubits); // 2^n, exact
    if (config.entangler != Entangler::None) return 1 / (states + 1);
    if (config.readout == PqcReadout::PauliZ) return 1.0 / 3;
    double power = 1; // 3^n, exact
    for (std::size_t q = 0; q < config.qubits; ++q) power *= 3;
    return 1 / power;
}
} // namespace init

} // namespace kan

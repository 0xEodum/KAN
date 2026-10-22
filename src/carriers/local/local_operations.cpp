// Operations specific to the local families (B-spline knot insertion and grid
// adaptation, trainable RBF centers and widths, per-edge wavelet parameters).
// Each one reads the carrier it
// requires, builds the replacement carrier and commits it through
// Layer::set_carrier, which revalidates the whole layer; a failure leaves the
// layer unchanged. Rational operations are in carriers/rational.
#include "kan/families.hpp"
#include "carriers/local/grid_refit.hpp"
#include "core/checks.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>

namespace kan {
namespace {
using detail::checked_size;
using detail::require_finite;
using detail::result_finite;

// The spline carrier of a layer, validated before its knots are indexed.
const BasisEdges* spline_edges(const Layer& layer) {
    const auto* edges = std::get_if<BasisEdges>(&layer.carrier());
    const auto* spline = edges ? std::get_if<BSplineConfig>(&edges->basis) : nullptr;
    if (!spline) return nullptr;
    validate_basis(edges->basis);
    if (edges->coefficients.size() != checked_size(checked_size(layer.inputs(), layer.outputs()), basis_size(*spline)))
        throw std::invalid_argument("layer is uninitialized or moved from");
    return edges;
}
} // namespace

namespace {
// Knot vector of grid g of a validated spline (backlog M6: `grids` of equal length).
std::span<const double> grid_knots(const BSplineConfig& spline, std::size_t g) {
    const auto n = spline.knots.size() / spline.grids;
    return std::span<const double>(spline.knots).subspan(g * n, n);
}

// Boehm insertion of knots[g] into every grid g; edge (o, i) uses grid i (per
// input) or the shared grid. The single-knot form is insert_knot's arithmetic.
void insert_into_grids(Layer& layer, const BasisEdges& edges, std::span<const double> knots) {
    const auto& spline = std::get<BSplineConfig>(edges.basis);
    const auto size = basis_size(spline);
    const auto p = spline.degree;
    if (knots.size() != spline.grids) throw std::invalid_argument("insert_knots needs one knot per spline grid");
    std::vector<std::size_t> spans(spline.grids), multiplicities(spline.grids);
    std::vector<double> next_knots;
    next_knots.reserve(spline.knots.size() + spline.grids);
    for (std::size_t g = 0; g < spline.grids; ++g) {
        const auto t = grid_knots(spline, g);
        const double x = knots[g];
        if (!std::isfinite(x) || x <= t[p] || x >= t[size]) throw std::invalid_argument("knot must be strictly inside a spline domain");
        multiplicities[g] = static_cast<std::size_t>(std::count(t.begin(), t.end(), x));
        if (multiplicities[g] >= p + 1) throw std::invalid_argument("knot multiplicity exceeds degree+1");
        spans[g] = static_cast<std::size_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin() - 1);
        next_knots.insert(next_knots.end(), t.begin(), t.begin() + static_cast<std::ptrdiff_t>(spans[g] + 1));
        next_knots.push_back(x);
        next_knots.insert(next_knots.end(), t.begin() + static_cast<std::ptrdiff_t>(spans[g] + 1), t.end());
    }
    BSplineConfig next_basis{p, std::move(next_knots), spline.grids};
    validate_basis(next_basis);
    const auto next_size = basis_size(next_basis);
    const auto inputs = layer.inputs(), count = inputs * layer.outputs();
    std::vector<double> next(checked_size(count, next_size));
    // Boehm insertion of x on every edge.
    for (std::size_t edge = 0; edge < count; ++edge) {
        const auto g = spline.grids == 1 ? 0 : edge % inputs;
        const auto t = grid_knots(spline, g);
        const double x = knots[g];
        const auto span = spans[g], multiplicity = multiplicities[g];
        const auto* c = edges.coefficients.data() + edge * size;
        auto* q = next.data() + edge * next_size;
        for (std::size_t j = 0; j < next_size; ++j) {
            if (j <= span - p) q[j] = c[j];
            else if (j >= span - multiplicity + 1) q[j] = c[j - 1];
            else {
                const double denominator = t[j + p] - t[j], numerator = x - t[j];
                const double alpha = std::isfinite(denominator) ? numerator / denominator
                                                                : (x / 2 - t[j] / 2) / (t[j + p] / 2 - t[j] / 2);
                q[j] = (1 - alpha) * c[j - 1] + alpha * c[j];
            }
        }
    }
    result_finite(next);
    layer.set_carrier(BasisEdges{std::move(next_basis), std::move(next)});
}

// adapt_grid's rule on one grid: the median of the samples in the most
// populated span (ties: the lowest), its midpoint if the median is not
// strictly inside. `each(f)` calls f(x) for every sample.
template<class Each>
double adapted_knot(std::span<const double> t, std::size_t degree, Each&& each) {
    const auto size = t.size() - degree - 1;
    std::vector<std::vector<double>> spans(size);
    each([&](double x) {
        if (x < t[degree] || x > t[size]) return;
        auto index = x == t[size] ? size - 1
                                  : static_cast<std::size_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin() - 1);
        spans[index].push_back(x);
    });
    std::size_t best = degree;
    for (std::size_t k = degree; k < size; ++k)
        if (spans[k].size() > spans[best].size()) best = k;
    auto values = std::move(spans[best]);
    if (values.empty()) throw std::invalid_argument("no in-domain samples to adapt");
    std::sort(values.begin(), values.end());
    double x = values[values.size() / 2];
    if (values.size() % 2 == 0) x = std::midpoint(values[values.size() / 2 - 1], x);
    if (x <= t[best] || x >= t[best + 1]) x = std::midpoint(t[best], t[best + 1]);
    if (x <= t[best] || x >= t[best + 1]) throw std::invalid_argument("span has no representable interior knot");
    return x;
}
} // namespace

void insert_knot(Layer& layer, double x) {
    const auto* edges = spline_edges(layer);
    if (!edges || !std::isfinite(x)) throw std::invalid_argument("knot must be strictly inside a spline domain");
    if (std::get<BSplineConfig>(edges->basis).grids != 1)
        throw std::invalid_argument("a layer with per-input grids inserts one knot per grid (insert_knots)");
    insert_into_grids(layer, *edges, std::span<const double>(&x, 1));
}

void insert_knots(Layer& layer, std::span<const double> knots) {
    const auto* edges = spline_edges(layer);
    if (!edges) throw std::invalid_argument("knot insertion requires splines");
    insert_into_grids(layer, *edges, knots);
}

double adapt_grid(Layer& layer, std::span<const double> samples) {
    const auto* edges = spline_edges(layer);
    require_finite(samples);
    if (!edges) throw std::invalid_argument("adaptation requires splines");
    const auto& spline = std::get<BSplineConfig>(edges->basis);
    if (spline.grids != 1)
        throw std::invalid_argument("a layer with per-input grids adapts from layer inputs (batch x inputs)");
    const double x = adapted_knot(spline.knots, spline.degree, [&](auto&& f) { for (double v : samples) f(v); });
    insert_knot(layer, x);
    return x;
}

std::vector<double> adapt_grid(Layer& layer, std::span<const double> input, std::size_t batch) {
    const auto* edges = spline_edges(layer);
    if (!edges) throw std::invalid_argument("adaptation requires splines");
    const auto inputs = layer.inputs();
    if (input.size() != checked_size(batch, inputs)) throw std::invalid_argument("adaptation input shape mismatch");
    require_finite(input);
    const auto& spline = std::get<BSplineConfig>(edges->basis);
    std::vector<double> knots(spline.grids);
    for (std::size_t g = 0; g < spline.grids; ++g)
        knots[g] = adapted_knot(grid_knots(spline, g), spline.degree, [&](auto&& f) {
            if (spline.grids == 1) for (double v : input) f(v);
            else for (std::size_t b = 0; b < batch; ++b) f(input[b * inputs + g]);
        });
    insert_into_grids(layer, *edges, knots);
    return knots;
}

void refit_grid(Layer& layer, std::span<const double> input, std::size_t batch, const GridRefit& options) {
    const auto* edges = spline_edges(layer);
    if (!edges) throw std::invalid_argument("grid refit requires splines");
    if (!(options.grid_eps >= 0 && options.grid_eps <= 1)) throw std::invalid_argument("grid_eps must be in [0, 1]");
    if (!(options.ridge >= 0) || !std::isfinite(options.ridge)) throw std::invalid_argument("refit ridge must be finite and nonnegative");
    if (!(options.margin >= 0) || !std::isfinite(options.margin)) throw std::invalid_argument("refit margin must be finite and nonnegative");
    const auto inputs = layer.inputs(), outputs = layer.outputs();
    if (input.size() != checked_size(batch, inputs)) throw std::invalid_argument("refit input shape mismatch");
    require_finite(input);
    if (batch == 0) throw std::invalid_argument("grid refit needs samples");
    const auto& spline = std::get<BSplineConfig>(edges->basis);
    const auto p = spline.degree, terms = basis_size(spline);
    const auto intervals = options.intervals ? options.intervals : terms - p;
    const auto next_terms = checked_size(intervals, 1) + p;
    if (next_terms < intervals) throw std::overflow_error("refit term count overflows");
    const auto grids = options.per_input ? inputs : 1;
    // Sorted samples per input (and pooled for a shared grid).
    std::vector<std::vector<double>> sorted(inputs, std::vector<double>(batch));
    for (std::size_t i = 0; i < inputs; ++i) {
        for (std::size_t b = 0; b < batch; ++b) sorted[i][b] = input[b * inputs + i];
        std::sort(sorted[i].begin(), sorted[i].end());
    }
    std::vector<double> pooled;
    if (grids == 1 && inputs > 1) {
        pooled.assign(input.begin(), input.end());
        std::sort(pooled.begin(), pooled.end());
    }
    BSplineConfig next_basis{p, {}, grids};
    next_basis.knots.reserve(checked_size(grids, next_terms + p + 1));
    for (std::size_t g = 0; g < grids; ++g) {
        const auto& x = grids == 1 && inputs > 1 ? pooled : sorted[g];
        const auto n = x.size();
        const auto load = [&](std::size_t k) { return x[k]; };
        if (!(x.front() < x.back()))
            throw std::invalid_argument("grid refit: the samples of input " + std::to_string(g) + " span no interval");
        const double lo = detail::refit_knot(load, n, intervals, 0, options.grid_eps, options.margin);
        const double hi = detail::refit_knot(load, n, intervals, intervals, options.grid_eps, options.margin);
        next_basis.knots.insert(next_basis.knots.end(), p, lo);
        for (std::size_t j = 0; j <= intervals; ++j)
            next_basis.knots.push_back(detail::refit_knot(load, n, intervals, j, options.grid_eps, options.margin));
        next_basis.knots.insert(next_basis.knots.end(), p, hi);
        const auto t = std::span<const double>(next_basis.knots).subspan(g * (next_terms + p + 1));
        for (std::size_t j = p; j < p + intervals; ++j)
            if (!(t[j] < t[j + 1]) || !std::isfinite(t[j + 1]))
                throw std::invalid_argument("grid refit: knots of input " + std::to_string(g) +
                                            " coincide (repeated samples; raise grid_eps)");
    }
    validate_basis(next_basis);
    // Least squares per input over its sorted samples (grid_refit.hpp).
    const auto P = p + 1;
    std::vector<double> next(checked_size(checked_size(inputs, outputs), next_terms));
    std::vector<double> band(checked_size(next_terms, P)), nn(P * P), no(P * P), rhs(next_terms);
    struct Run { std::size_t g, current; std::vector<double> no; };
    std::vector<Run> runs;
    for (std::size_t i = 0; i < inputs; ++i) {
        const auto fresh = grid_knots(next_basis, grids == 1 ? 0 : i);
        const auto current = grid_knots(spline, spline.grids == 1 ? 0 : i);
        const auto& x = sorted[i];
        const auto load = [&](std::size_t k) { return x[k]; };
        std::fill(band.begin(), band.end(), 0.0);
        runs.clear();
        for (std::size_t begin = 0; begin < batch;) {
            const auto g = detail::spline_first_term(fresh.data(), next_terms, p, x[begin]);
            const auto gc = detail::spline_first_term(current.data(), terms, p, x[begin]);
            auto end = begin + 1;
            while (end < batch && detail::spline_first_term(fresh.data(), next_terms, p, x[end]) == g &&
                   detail::spline_first_term(current.data(), terms, p, x[end]) == gc) ++end;
            if (!detail::refit_run_sums<0>(fresh.data(), next_terms, current.data(), terms, p, load, begin, end, g, gc, nn.data(), no.data()))
                throw std::logic_error("grid refit: inconsistent spline spans");
            for (std::size_t r = 0; r < P; ++r)
                for (std::size_t q = r; q < P; ++q) band[(g + q) * P + (q - r)] += nn[r * P + q];
            if (gc != terms) runs.push_back({g, gc, no});
            begin = end;
        }
        if (!detail::band_cholesky(band.data(), next_terms, p, options.ridge))
            throw std::invalid_argument("grid refit: the least-squares system of input " + std::to_string(i) +
                                        " is singular (knot spans without samples; use ridge > 0)");
        for (std::size_t o = 0; o < outputs; ++o) {
            const double* c = edges->coefficients.data() + (o * inputs + i) * terms;
            std::fill(rhs.begin(), rhs.end(), 0.0);
            for (const auto& run : runs)
                for (std::size_t r = 0; r < P; ++r) {
                    double sum = rhs[run.g + r];
                    for (std::size_t q = 0; q < P; ++q) sum += run.no[r * P + q] * c[run.current + q];
                    rhs[run.g + r] = sum;
                }
            detail::band_solve(band.data(), next_terms, p, rhs.data(), 1);
            std::copy(rhs.begin(), rhs.end(), next.begin() + static_cast<std::ptrdiff_t>((o * inputs + i) * next_terms));
        }
    }
    result_finite(next);
    layer.set_carrier(BasisEdges{std::move(next_basis), std::move(next)});
}

void set_rbf_parameters(Layer& layer, std::span<const double> centers, std::span<const double> log_widths) {
    const auto* edges = std::get_if<TrainableRbfEdges>(&layer.carrier());
    if (!edges) throw std::invalid_argument("RBF parameters require a trainable Gaussian basis");
    // The term count is derived from the centers; it must not change here.
    const auto size = edges->basis.centers.size();
    if (centers.size() != size || log_widths.size() != size)
        throw std::invalid_argument("RBF parameter count must match the layer's basis size");
    layer.set_carrier(TrainableRbfEdges{{{centers.begin(), centers.end()}, {log_widths.begin(), log_widths.end()}},
                                        edges->coefficients});
}

void set_wavelet_parameters(Layer& layer, std::span<const double> coefficients, std::span<const double> translations,
                            std::span<const double> log_scales, std::span<const double> bias) {
    const auto* edges = std::get_if<WaveletEdges>(&layer.carrier());
    if (!edges) throw std::invalid_argument("wavelet parameters require a wavelet layer");
    // set_carrier validates the shapes for the layer, finiteness and the log scales.
    layer.set_carrier(WaveletEdges{edges->config, {coefficients.begin(), coefficients.end()},
                                   {translations.begin(), translations.end()}, {log_scales.begin(), log_scales.end()}},
                      bias);
}

} // namespace kan

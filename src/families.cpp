// Family-specific operations. Each one reads the carrier it requires, builds
// the replacement carrier and commits it through Layer::set_carrier, which
// revalidates the whole layer; a failure leaves the layer unchanged.
#include "kan/families.hpp"
#include "detail/checks.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
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

void insert_knot(Layer& layer, double x) {
    const auto* edges = spline_edges(layer);
    if (!edges || !std::isfinite(x)) throw std::invalid_argument("knot must be strictly inside a spline domain");
    const auto& spline = std::get<BSplineConfig>(edges->basis);
    const auto size = basis_size(spline);
    const auto& t = spline.knots;
    const auto p = spline.degree;
    if (x <= t[p] || x >= t[size]) throw std::invalid_argument("knot must be strictly inside a spline domain");
    const auto multiplicity = static_cast<std::size_t>(std::count(t.begin(), t.end(), x));
    if (multiplicity >= p + 1) throw std::invalid_argument("knot multiplicity exceeds degree+1");
    const auto span = static_cast<std::size_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin() - 1);
    std::vector<double> knots;
    knots.reserve(t.size() + 1);
    knots.insert(knots.end(), t.begin(), t.begin() + static_cast<std::ptrdiff_t>(span + 1));
    knots.push_back(x);
    knots.insert(knots.end(), t.begin() + static_cast<std::ptrdiff_t>(span + 1), t.end());
    BSplineConfig next_basis{p, std::move(knots)};
    validate_basis(next_basis);
    const auto next_size = basis_size(next_basis);
    const auto count = layer.inputs() * layer.outputs();
    std::vector<double> next(checked_size(count, next_size));
    // Boehm insertion of x on every edge.
    for (std::size_t edge = 0; edge < count; ++edge) {
        const auto* c = edges->coefficients.data() + edge * size;
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

double adapt_grid(Layer& layer, std::span<const double> samples) {
    const auto* edges = spline_edges(layer);
    require_finite(samples);
    if (!edges) throw std::invalid_argument("adaptation requires splines");
    const auto& spline = std::get<BSplineConfig>(edges->basis);
    const auto& t = spline.knots;
    const auto size = basis_size(spline);
    std::vector<std::vector<double>> spans(size);
    for (double x : samples) {
        if (x < t[spline.degree] || x > t[size]) continue;
        auto index = x == t[size] ? size - 1
                                  : static_cast<std::size_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin() - 1);
        spans[index].push_back(x);
    }
    std::size_t best = spline.degree;
    for (std::size_t k = spline.degree; k < size; ++k)
        if (spans[k].size() > spans[best].size()) best = k;
    auto values = std::move(spans[best]);
    if (values.empty()) throw std::invalid_argument("no in-domain samples to adapt");
    std::sort(values.begin(), values.end());
    double x = values[values.size() / 2];
    if (values.size() % 2 == 0) x = std::midpoint(values[values.size() / 2 - 1], x);
    if (x <= t[best] || x >= t[best + 1]) x = std::midpoint(t[best], t[best + 1]);
    if (x <= t[best] || x >= t[best + 1]) throw std::invalid_argument("span has no representable interior knot");
    insert_knot(layer, x);
    return x;
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

void set_rational_parameters(Layer& layer, std::span<const double> numerator,
                             std::span<const double> denominators, std::span<const double> bias) {
    const auto* edges = std::get_if<RationalEdges>(&layer.carrier());
    if (!edges || numerator.size() != edges->coefficients.size() ||
        denominators.size() != edges->denominators.size() || bias.size() != layer.bias().size())
        throw std::invalid_argument("rational parameter shape or layer type mismatch");
    require_finite(numerator);
    require_finite(denominators);
    require_finite(bias);
    layer.set_carrier(RationalEdges{edges->config, {numerator.begin(), numerator.end()},
                                    {denominators.begin(), denominators.end()}},
                      bias);
}

} // namespace kan

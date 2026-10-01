#pragma once

// Host-only bridge from the typed public configuration to the flat view used by
// the shared formulas. The view borrows the configuration's vectors.

#include "basis_formulas.hpp"
#include "kan/basis.hpp"
#include <type_traits>
#include <variant>

namespace kan::detail {

inline BasisView basis_view(const BasisConfig& config) {
    return std::visit([](const auto& c) -> BasisView {
        using T = std::decay_t<decltype(c)>;
        BasisView view{BasisKind::Chebyshev, basis_size(c), 0, 0, 1, 1,
                       nullptr, nullptr, nullptr, nullptr, 0, false};
        if constexpr (std::is_same_v<T, ChebyshevConfig>) {
            view.kind = BasisKind::Chebyshev;
        } else if constexpr (std::is_same_v<T, LegendreConfig>) {
            view.kind = BasisKind::Legendre;
        } else if constexpr (std::is_same_v<T, HermiteConfig>) {
            view.kind = BasisKind::Hermite;
        } else if constexpr (std::is_same_v<T, JacobiConfig>) {
            view.kind = BasisKind::Jacobi;
            view.alpha = c.alpha;
            view.beta = c.beta;
        } else if constexpr (std::is_same_v<T, FourierConfig>) {
            view.kind = BasisKind::Fourier;
            view.frequency = c.frequency;
        } else if constexpr (std::is_same_v<T, GaussianRbfConfig>) {
            view.kind = BasisKind::GaussianRbf;
            view.centers = c.centers.data();
            view.width = c.width;
        } else if constexpr (std::is_same_v<T, TrainableRbfConfig>) {
            view.kind = BasisKind::GaussianRbf;
            view.centers = c.centers.data();
            view.log_widths = c.log_widths.data();
            view.trainable = true;
        } else if constexpr (std::is_same_v<T, BSplineConfig>) {
            view.kind = BasisKind::BSpline;
            view.knots = c.knots.data();
            view.degree = c.degree;
        } else {
            static_assert(std::is_same_v<T, MexicanHatConfig>, "unhandled basis configuration");
            view.kind = BasisKind::MexicanHat;
            view.centers = c.centers.data();
            view.scales = c.scales.data();
        }
        return view;
    }, config);
}

} // namespace kan::detail

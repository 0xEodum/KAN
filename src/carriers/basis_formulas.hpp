#pragma once

// Family dispatch over the single-source linear-carrier basis formulas shared
// by the CPU backend and the resident CUDA kernels. Callers validate
// configuration and finite input.
//
// Scalar (backlog C1): every formula is a template over the floating-point
// type of its data. The CPU and the FP64 resident executor instantiate double,
// the FP32 resident executor float. Literals are written as Scalar(...) so a
// float instantiation never promotes to double; the double instantiation
// performs exactly the operations it performed before C1.
//
// Guard: callable `Scalar(Scalar)` returning its argument. It is invoked on
// every value the contract requires to be finite; the CPU guard throws, the
// device guard records a status bit and execution continues.

// The per-family evaluators live with their execution group:
// carriers/polynomial (global recurrences), carriers/trigonometric (Fourier) and
// carriers/local (B-spline, Gaussian RBF, Mexican hat). This header holds the
// compile-time and runtime family dispatch over them.

#include "carriers/basis_types.hpp"
#include "carriers/local/local_formulas.hpp"
#include "carriers/polynomial/polynomial_formulas.hpp"
#include "carriers/trigonometric/fourier_formulas.hpp"
#include "core/host_device.hpp"
#include "kan/basis.hpp"
#include <cstddef>
#include <stdexcept>

namespace kan::detail {

// Family fixed at compile time: device kernels instantiate one family each, so
// no kernel carries another family's code, registers or spline scratch.
template<BasisKind Kind, class Scalar, class Guard>
KAN_HOST_DEVICE void basis_terms_for(const BasisViewOf<Scalar>& basis, Scalar x, const BasisRowOf<Scalar>& row,
                                     const Guard& guard) {
    if constexpr (Kind == BasisKind::BSpline)
        spline_terms(basis.knots, basis.terms, basis.degree, x, row.values, row.derivatives, guard);
    else if constexpr (Kind == BasisKind::MexicanHat)
        mexican_hat_terms(basis.centers, basis.scales, basis.terms, x, row.values, row.derivatives, guard);
    else if constexpr (Kind == BasisKind::GaussianRbf)
        gaussian_terms(basis, x, row, guard);
    else if constexpr (Kind == BasisKind::Fourier)
        fourier_terms(basis.frequency, basis.terms, x, row.values, row.derivatives, guard);
    else {
        static_assert(Kind == BasisKind::Chebyshev || Kind == BasisKind::Legendre ||
                      Kind == BasisKind::Jacobi || Kind == BasisKind::Hermite,
                      "basis family without an evaluator");
        polynomial_terms(Kind, basis.alpha, basis.beta, basis.terms, x, row.values, row.derivatives, guard);
    }
}

// Host-side runtime family dispatch over the compile-time evaluators: CPU
// evaluation and kernel launch selection. `visit` receives a tag whose `value`
// is the family.
template<BasisKind Kind>
struct BasisFamilyTag {
    static constexpr BasisKind value = Kind;
};

template<class Visitor>
void visit_basis_family(BasisKind kind, Visitor&& visit) {
    switch (kind) {
    case BasisKind::Chebyshev: visit(BasisFamilyTag<BasisKind::Chebyshev>{}); return;
    case BasisKind::Legendre: visit(BasisFamilyTag<BasisKind::Legendre>{}); return;
    case BasisKind::Jacobi: visit(BasisFamilyTag<BasisKind::Jacobi>{}); return;
    case BasisKind::Hermite: visit(BasisFamilyTag<BasisKind::Hermite>{}); return;
    case BasisKind::Fourier: visit(BasisFamilyTag<BasisKind::Fourier>{}); return;
    case BasisKind::GaussianRbf: visit(BasisFamilyTag<BasisKind::GaussianRbf>{}); return;
    case BasisKind::BSpline: visit(BasisFamilyTag<BasisKind::BSpline>{}); return;
    case BasisKind::MexicanHat: visit(BasisFamilyTag<BasisKind::MexicanHat>{}); return;
    }
    // Validation rejects unknown kinds before any dispatch.
    throw std::logic_error("unvalidated basis kind");
}

template<class Scalar, class Guard>
void basis_terms(const BasisViewOf<Scalar>& basis, Scalar x, const BasisRowOf<Scalar>& row, const Guard& guard) {
    visit_basis_family(basis.kind, [&](auto family) {
        basis_terms_for<decltype(family)::value>(basis, x, row, guard);
    });
}

} // namespace kan::detail

#pragma once

// Family tag and the flat basis view/row types shared by every basis family's
// formulas (carriers/{polynomial,trigonometric,local}), the family dispatch in
// basis_formulas.hpp, the CPU backend and the resident CUDA kernels.

#include <cstddef>

namespace kan::detail {

// Family tag for kernel dispatch. The public API selects a family by its
// configuration type; trainable RBFs are GaussianRbf with BasisView::trainable,
// and HermiteFunction is HermiteConfig with the Function normalization (backlog
// M5; appended so the existing tags and kernel instantiations keep their values).
enum class BasisKind {
    Chebyshev,
    Legendre,
    Jacobi,
    Hermite,
    Fourier,
    GaussianRbf,
    BSpline,
    MexicanHat,
    HermiteFunction
};

// Non-owning view of one validated basis. Pointer fields are read only for the
// families that use them; `log_widths` is read only when `trainable` is set.
template<class Scalar> struct BasisViewOf {
    BasisKind kind;
    std::size_t terms;
    Scalar alpha, beta, frequency, width;
    const Scalar* centers;
    const Scalar* log_widths;
    const Scalar* scales;
    const Scalar* knots;
    std::size_t degree;
    bool trainable;
};
using BasisView = BasisViewOf<double>;

// Output rows of length `terms`. The nonlinear RBF rows may be null.
template<class Scalar> struct BasisRowOf {
    Scalar* values;
    Scalar* derivatives;
    Scalar* center_derivatives;
    Scalar* log_width_derivatives;
};
using BasisRow = BasisRowOf<double>;

} // namespace kan::detail

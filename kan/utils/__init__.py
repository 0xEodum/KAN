"""
Utility functions for Kolmogorov-Arnold Networks.

This module provides various utilities for working with KAN models,
including visualization, symbolic manipulation, and weight initialization.
"""

from .visualization import (
    plot_basis_functions,
    plot_layer_functions,
    plot_network_functions,
    plot_function_approximation
)

from .initializers import (
    init_chebyshev_normal,
    init_chebyshev_uniform,
    init_chebyshev_orthogonal,
    init_chebyshev_zeros,
    init_chebyshev_identity,
    init_jacobi_normal,
    init_jacobi_uniform,
    init_jacobi_orthogonal,
    init_jacobi_identity,
    get_initializer
)

# Import symbolic utilities if sympy is available
try:
    import sympy
    from .symbolic import (
        get_layer_symbolic_expr,
        get_network_symbolic_expr,
        simplify_expressions,
        export_to_latex,
        export_to_function
    )
    _sympy_available = True
except ImportError:
    _sympy_available = False


def has_sympy():
    """Check if sympy is available for symbolic operations."""
    return _sympy_available
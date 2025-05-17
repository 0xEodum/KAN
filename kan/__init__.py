"""
KAN - Kolmogorov-Arnold Networks

A PyTorch-based library for neural networks using explicit mathematical forms
based on the Kolmogorov-Arnold representation theorem.

These networks provide greater interpretability and mathematical rigor
compared to traditional neural networks.
"""

__version__ = "0.3.0"

from .layers.chebykan import ChebyKANLayer
from .layers.jacobikan import JacobiKANLayer
from .layers.hermitekan import HermiteKANLayer
from .basis.chebyshev import ChebyshevBasis
from .basis.jacobi import JacobiBasis
from .basis.hermite import HermiteBasis
from .layers.base import KANLayer, KANSequential

# Import utilities
from .utils.visualization import (
    plot_basis_functions,
    plot_layer_functions,
    plot_network_functions,
    plot_function_approximation
)

from .utils.initializers import (
    init_chebyshev_normal,
    init_chebyshev_uniform,
    init_chebyshev_orthogonal,
    init_chebyshev_zeros,
    init_chebyshev_identity,
    init_jacobi_normal,
    init_jacobi_uniform,
    init_jacobi_orthogonal,
    init_jacobi_identity,
    init_hermite_normal,
    init_hermite_uniform,
    init_hermite_orthogonal,
    init_hermite_identity,
    get_initializer
)

# Import symbolic utilities if sympy is available
try:
    import sympy
    from .utils.symbolic import (
        get_layer_symbolic_expr,
        get_network_symbolic_expr,
        simplify_expressions,
        export_to_latex,
        export_to_function
    )
    _sympy_available = True
except ImportError:
    _sympy_available = False
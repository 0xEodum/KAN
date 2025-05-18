"""
KAN - Сети Колмогорова-Арнольда

Библиотека на основе PyTorch для нейронных сетей, использующих явные математические формы,
основанные на теореме представления Колмогорова-Арнольда.

Эти сети обеспечивают большую интерпретируемость и математическую строгость
по сравнению с традиционными нейронными сетями.
"""

__version__ = "0.4.0"

# Слои KAN
from .layers.chebykan import ChebyKANLayer
from .layers.jacobikan import JacobiKANLayer
from .layers.hermitekan import HermiteKANLayer
from .layers.splinekan import BSplineKANLayer, CubicSplineKANLayer, AdaptiveSplineKANLayer
from .layers.recursive_splinekan import RecursiveSplineKANLayer
from .layers.base import KANLayer, KANSequential
from .layers.relukan import ReLUKANLayer

# Базисные функции
from .basis.chebyshev import ChebyshevBasis
from .basis.jacobi import JacobiBasis
from .basis.hermite import HermiteBasis
from .basis.bspline import BSplineBasis
from .basis.cubic_spline import CubicSplineBasis
from .basis.adaptive_spline import AdaptiveSplineBasis
from .basis.recursive_bspline import RecursiveBSplineBasis
from .basis.relu import ReLUBasis

# Импорт утилит
from .utils.visualization import (
    plot_basis_functions,
    plot_layer_functions,
    plot_network_functions,
    plot_function_approximation
)

# Импорт всех инициализаторов
from .utils.initializers import (
    # Инициализаторы для полиномов Чебышева
    init_chebyshev_normal,
    init_chebyshev_uniform,
    init_chebyshev_orthogonal,
    init_chebyshev_zeros,
    init_chebyshev_identity,
    
    # Инициализаторы для полиномов Якоби
    init_jacobi_normal,
    init_jacobi_uniform,
    init_jacobi_orthogonal,
    init_jacobi_identity,
    
    # Инициализаторы для полиномов Эрмита
    init_hermite_normal,
    init_hermite_uniform,
    init_hermite_orthogonal,
    init_hermite_identity,
    
    # Инициализаторы для сплайнов
    init_bspline_normal,
    init_bspline_uniform,
    init_bspline_identity,
    init_cubic_spline_normal,
    init_cubic_spline_uniform,
    init_cubic_spline_identity,
    init_adaptive_spline_normal,
    init_adaptive_spline_uniform,
    init_adaptive_spline_identity,
    
    # Единая функция для получения инициализатора
    get_initializer
)

# Импорт рекурсивных инициализаторов - они будут доступны через get_initializer
from .utils.recursive_initializers import (
    init_recursive_bspline_normal,
    init_recursive_bspline_uniform,
    init_recursive_bspline_orthogonal,
    init_recursive_bspline_identity,
    init_recursive_bspline_grid_based
)

# Импорт символьных утилит, если доступен sympy
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
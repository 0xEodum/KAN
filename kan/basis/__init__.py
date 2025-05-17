"""
Базисные функции для сетей Колмогорова-Арнольда (KAN).

Этот модуль предоставляет различные базисные функции, которые могут использоваться в слоях KAN
для представления одномерных функций в соответствии с теоремой Колмогорова-Арнольда.
"""

from .base import BasisFunction
from .chebyshev import ChebyshevBasis
from .jacobi import JacobiBasis
from .hermite import HermiteBasis
from .bspline import BSplineBasis
from .cubic_spline import CubicSplineBasis
from .adaptive_spline import AdaptiveSplineBasis

# Словарь, сопоставляющий имена базисных функций с их классами
BASIS_REGISTRY = {
    'chebyshev': ChebyshevBasis,
    'jacobi': JacobiBasis,
    'hermite': HermiteBasis,
    'bspline': BSplineBasis,
    'cubic_spline': CubicSplineBasis,
    'adaptive_spline': AdaptiveSplineBasis,
}


def get_basis(name: str, **kwargs):
    """
    Получает базисную функцию по имени.
    
    Args:
        name: Имя базисной функции
        **kwargs: Дополнительные аргументы для передачи конструктору базиса
        
    Returns:
        Экземпляр запрошенной базисной функции
    """
    if name not in BASIS_REGISTRY:
        raise ValueError(f"Неизвестная базисная функция: {name}. Доступные варианты: "
                       f"{', '.join(BASIS_REGISTRY.keys())}")
    
    return BASIS_REGISTRY[name](**kwargs)
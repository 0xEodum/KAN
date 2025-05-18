"""
Слои KAN для построения сетей Колмогорова-Арнольда.

Этот модуль предоставляет различные реализации слоев для архитектур KAN,
каждая из которых использует разные базисные функции для представления
одномерных компонентов разложения Колмогорова-Арнольда.
"""

from .base import KANLayer, KANSequential
from .chebykan import ChebyKANLayer
from .jacobikan import JacobiKANLayer
from .hermitekan import HermiteKANLayer
from .splinekan import BSplineKANLayer, CubicSplineKANLayer, AdaptiveSplineKANLayer
from .recursive_splinekan import RecursiveSplineKANLayer

# Словарь, сопоставляющий имена слоев с их классами
LAYER_REGISTRY = {
    'chebykan': ChebyKANLayer,
    'jacobikan': JacobiKANLayer,
    'hermitekan': HermiteKANLayer,
    'bsplinekan': BSplineKANLayer,
    'cubic_splinekan': CubicSplineKANLayer,
    'adaptive_splinekan': AdaptiveSplineKANLayer,
    'recursive_splinekan': RecursiveSplineKANLayer,
}


def get_layer(name: str, **kwargs):
    """
    Получает слой KAN по имени.
    
    Args:
        name: Имя слоя
        **kwargs: Дополнительные аргументы для передачи конструктору слоя
        
    Returns:
        Экземпляр запрошенного слоя KAN
    """
    if name not in LAYER_REGISTRY:
        raise ValueError(f"Неизвестный слой: {name}. Доступные варианты: "
                       f"{', '.join(LAYER_REGISTRY.keys())}")
    
    return LAYER_REGISTRY[name](**kwargs)
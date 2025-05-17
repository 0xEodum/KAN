import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union

from ..basis.bspline import BSplineBasis
from ..basis.cubic_spline import CubicSplineBasis
from ..basis.adaptive_spline import AdaptiveSplineBasis
from .base import KANLayer


class BSplineKANLayer(KANLayer):
    """
    KAN слой, использующий B-сплайновые базисные функции.
    
    B-сплайны обеспечивают локальную поддержку и контролируемую гладкость,
    что делает их эффективными для аппроксимации функций с локальными особенностями
    и разрывами.
    """
    
    def __init__(self, input_dim: int, output_dim: int, degree: int = 3,
                 num_knots: int = None, domain: Tuple[float, float] = (-1, 1),
                 uniform: bool = True, init_scale: float = None):
        """
        Инициализирует BSplineKAN слой.
        
        Args:
            input_dim: Количество входных измерений
            output_dim: Количество выходных измерений
            degree: Степень B-сплайнов (0-постоянные, 1-линейные, 2-квадратичные, 3-кубические)
            num_knots: Количество внутренних узлов (если None, устанавливается как degree+2)
            domain: Область определения [a, b]
            uniform: Использовать равномерное распределение узлов
            init_scale: Масштаб для инициализации коэффициентов (если None, используется 1/(input_dim * num_basis))
        """
        # Создание базиса B-сплайнов
        basis_function = BSplineBasis(
            degree=degree,
            num_knots=num_knots,
            domain=domain,
            uniform=uniform
        )
        
        # Инициализация базового класса
        super(BSplineKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Сохранение параметров для последующего использования
        self.degree = degree
        self.num_knots = basis_function.num_knots
        self.uniform = uniform
        self.domain = domain
        
        # Количество базисных функций
        self.num_basis = basis_function.num_basis
        
        # Инициализация коэффициентов
        init_scale = init_scale or 1.0 / (input_dim * self.num_basis)
        self.bspline_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, self.num_basis)
        )
        nn.init.normal_(self.bspline_coeffs, mean=0.0, std=init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход слоя.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Выходной тензор формы (batch_size, output_dim)
        """
        # Проверка размерности x
        x = x.view(-1, self.input_dim)
        
        # Делегирование вычисления базисной функции
        return self.basis_function.forward(x, self.bspline_coeffs)
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Возвращает аналитическую форму функции слоя.
        
        Returns:
            Словарь, содержащий информацию об аналитической форме
        """
        return {
            'type': 'BSplineKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'degree': self.degree,
            'num_knots': self.num_knots,
            'domain': self.domain,
            'uniform': self.uniform,
            'basis': self.basis_function.name,
            'coefficients': self.bspline_coeffs.detach().cpu().numpy(),
            'knots': self.basis_function.knots.detach().cpu().numpy()
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Получает коэффициенты, используемые этим слоем.
        
        Returns:
            Тензор формы (input_dim, output_dim, num_basis), содержащий коэффициенты
        """
        return self.bspline_coeffs
    
    def extra_repr(self) -> str:
        """
        Возвращает строковое представление слоя.
        
        Returns:
            Строковое представление
        """
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'degree={self.degree}, num_knots={self.num_knots}, uniform={self.uniform}')


class CubicSplineKANLayer(KANLayer):
    """
    KAN слой, использующий кубические сплайны.
    
    Кубические сплайны обеспечивают непрерывность второй производной и
    широко используются в компьютерной графике и численном анализе
    благодаря их оптимальным свойствам гладкости.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_knots: int,
                 domain: Tuple[float, float] = (-1, 1), 
                 knot_method: str = 'uniform',
                 boundary_condition: str = 'natural',
                 init_scale: float = None):
        """
        Инициализирует CubicSplineKAN слой.
        
        Args:
            input_dim: Количество входных измерений
            output_dim: Количество выходных измерений
            num_knots: Количество узлов
            domain: Область определения [a, b]
            knot_method: Метод распределения узлов ('uniform', 'chebyshev', 'adaptive')
            boundary_condition: Граничные условия ('natural', 'clamped', 'not-a-knot')
            init_scale: Масштаб для инициализации коэффициентов (если None, используется 1/(input_dim * num_knots))
        """
        # Создание базиса кубических сплайнов
        basis_function = CubicSplineBasis(
            num_knots=num_knots,
            domain=domain,
            knot_method=knot_method,
            boundary_condition=boundary_condition
        )
        
        # Инициализация базового класса
        super(CubicSplineKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Сохранение параметров для последующего использования
        self.num_knots = num_knots
        self.domain = domain
        self.knot_method = knot_method
        self.boundary_condition = boundary_condition
        
        # Инициализация коэффициентов
        init_scale = init_scale or 1.0 / (input_dim * num_knots)
        self.spline_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, num_knots)
        )
        nn.init.normal_(self.spline_coeffs, mean=0.0, std=init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход слоя.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Выходной тензор формы (batch_size, output_dim)
        """
        # Проверка размерности x
        x = x.view(-1, self.input_dim)
        
        # Делегирование вычисления базисной функции
        return self.basis_function.forward(x, self.spline_coeffs)
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Возвращает аналитическую форму функции слоя.
        
        Returns:
            Словарь, содержащий информацию об аналитической форме
        """
        return {
            'type': 'CubicSplineKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_knots': self.num_knots,
            'domain': self.domain,
            'knot_method': self.knot_method,
            'boundary_condition': self.boundary_condition,
            'basis': self.basis_function.name,
            'coefficients': self.spline_coeffs.detach().cpu().numpy(),
            'knots': self.basis_function.knots.detach().cpu().numpy()
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Получает коэффициенты, используемые этим слоем.
        
        Returns:
            Тензор формы (input_dim, output_dim, num_knots), содержащий коэффициенты
        """
        return self.spline_coeffs
    
    def extra_repr(self) -> str:
        """
        Возвращает строковое представление слоя.
        
        Returns:
            Строковое представление
        """
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'num_knots={self.num_knots}, knot_method={self.knot_method}, '
                f'boundary_condition={self.boundary_condition}')


class AdaptiveSplineKANLayer(KANLayer):
    """
    KAN слой, использующий адаптивные сплайны с оптимизируемыми узлами.
    
    Адаптивные сплайны автоматически размещают узлы в областях, где они наиболее
    необходимы, что позволяет достичь лучшей аппроксимации с меньшим количеством параметров.
    """
    
    def __init__(self, input_dim: int, output_dim: int, degree: int = 3,
                 num_knots: int = 10, domain: Tuple[float, float] = (-1, 1),
                 init_strategy: str = 'uniform', regularization: float = 0.01,
                 min_distance: float = 1e-3, init_scale: float = None):
        """
        Инициализирует AdaptiveSplineKAN слой.
        
        Args:
            input_dim: Количество входных измерений
            output_dim: Количество выходных измерений
            degree: Степень сплайнов
            num_knots: Количество узлов
            domain: Область определения [a, b]
            init_strategy: Стратегия инициализации узлов ('uniform', 'chebyshev', 'random')
            regularization: Коэффициент регуляризации для предотвращения скопления узлов
            min_distance: Минимальное расстояние между узлами
            init_scale: Масштаб для инициализации коэффициентов
        """
        # Создание базиса адаптивных сплайнов
        basis_function = AdaptiveSplineBasis(
            degree=degree,
            num_knots=num_knots,
            domain=domain,
            init_strategy=init_strategy,
            regularization=regularization,
            min_distance=min_distance
        )
        
        # Инициализация базового класса
        super(AdaptiveSplineKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Сохранение параметров для последующего использования
        self.degree = degree
        self.num_knots = num_knots
        self.domain = domain
        self.init_strategy = init_strategy
        self.regularization = regularization
        self.min_distance = min_distance
        
        # Количество базисных функций
        self.num_basis = num_knots + degree
        
        # Инициализация коэффициентов
        init_scale = init_scale or 1.0 / (input_dim * self.num_basis)
        self.adaptive_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, self.num_basis)
        )
        nn.init.normal_(self.adaptive_coeffs, mean=0.0, std=init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход слоя.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Выходной тензор формы (batch_size, output_dim)
        """
        # Проверка размерности x
        x = x.view(-1, self.input_dim)
        
        # Делегирование вычисления базисной функции
        return self.basis_function.forward(x, self.adaptive_coeffs)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Получает регуляризационную потерю для предотвращения скопления узлов.
        
        Returns:
            Значение регуляризационной потери
        """
        return self.basis_function.get_regularization_loss()
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Возвращает аналитическую форму функции слоя.
        
        Returns:
            Словарь, содержащий информацию об аналитической форме
        """
        # Получаем текущие позиции узлов
        knots = self.basis_function._get_sorted_knots().detach().cpu().numpy()
        
        return {
            'type': 'AdaptiveSplineKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'degree': self.degree,
            'num_knots': self.num_knots,
            'domain': self.domain,
            'init_strategy': self.init_strategy,
            'regularization': self.regularization,
            'min_distance': self.min_distance,
            'basis': self.basis_function.name,
            'coefficients': self.adaptive_coeffs.detach().cpu().numpy(),
            'knots': knots
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Получает коэффициенты, используемые этим слоем.
        
        Returns:
            Тензор формы (input_dim, output_dim, num_basis), содержащий коэффициенты
        """
        return self.adaptive_coeffs
    
    def get_knots(self) -> np.ndarray:
        """
        Получает текущие позиции узлов.
        
        Returns:
            Массив позиций узлов
        """
        return self.basis_function._get_sorted_knots().detach().cpu().numpy()
    
    def extra_repr(self) -> str:
        """
        Возвращает строковое представление слоя.
        
        Returns:
            Строковое представление
        """
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'degree={self.degree}, num_knots={self.num_knots}, '
                f'regularization={self.regularization}')
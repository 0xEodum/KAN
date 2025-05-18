import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union

from ..basis.recursive_bspline import RecursiveBSplineBasis
from .base import KANLayer


class RecursiveSplineKANLayer(KANLayer):
    """
    KAN слой, использующий рекурсивную реализацию B-сплайновых базисных функций.
    
    Этот слой реализует алгоритм обработки, аналогичный представленному в original KANLayer.py,
    но адаптированный к архитектуре KAN фреймворка. Он предоставляет дополнительные
    возможности для работы с сетками, такие как адаптация сетки к распределению данных
    и вычисление коэффициентов методом наименьших квадратов.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_knots: int = 5, 
                 degree: int = 3,
                 domain: Tuple[float, float] = (-1, 1),
                 uniform: bool = True,
                 grid_eps: float = 0.02,
                 noise_scale: float = 0.1,
                 scale_base_mu: float = 0.0,
                 scale_base_sigma: float = 0.0,
                 scale_sp: float = 1.0,
                 sp_trainable: bool = True,
                 sb_trainable: bool = True,
                 sparse_init: bool = False,
                 init_scale: float = None):
        """
        Инициализирует RecursiveSplineKANLayer.
        
        Args:
            input_dim: Количество входных измерений
            output_dim: Количество выходных измерений
            num_knots: Количество внутренних узлов сетки
            degree: Степень сплайнов (0 - постоянный, 1 - линейный, 2 - квадратичный, 3 - кубический и т.д.)
            domain: Область определения сплайнов [a, b]
            uniform: Использовать равномерное распределение узлов
            grid_eps: Параметр для интерполяции между равномерной и адаптивной сеткой
                     (0 = полностью адаптивная, 1 = полностью равномерная)
            noise_scale: Масштаб шума при инициализации
            scale_base_mu: Среднее значение для базовой функции
            scale_base_sigma: Стандартное отклонение для базовой функции
            scale_sp: Масштаб для сплайновой составляющей
            sp_trainable: Делать ли scale_sp обучаемым параметром
            sb_trainable: Делать ли scale_base обучаемым параметром
            sparse_init: Использовать ли разреженную инициализацию
            init_scale: Масштаб для инициализации коэффициентов (если None, вычисляется автоматически)
        """
        # Создание B-сплайновой базисной функции
        basis_function = RecursiveBSplineBasis(
            degree=degree,
            num_knots=num_knots,
            domain=domain,
            uniform=uniform,
            grid_eps=grid_eps
        )
        
        # Инициализация базового класса
        super(RecursiveSplineKANLayer, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_function=basis_function
        )
        
        # Сохранение параметров
        self.degree = degree
        self.num_knots = num_knots
        self.domain = domain
        self.uniform = uniform
        self.grid_eps = grid_eps
        
        # Вычисление init_scale, если не задан
        if init_scale is None:
            init_scale = 1.0 / (input_dim * basis_function.num_basis)
        
        # Инициализация коэффициентов B-сплайнов
        self.spline_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, basis_function.num_basis)
        )
        
        # Инициализация с шумом
        nn.init.normal_(self.spline_coeffs, mean=0.0, std=init_scale * noise_scale)
        
        # Создание маски для разреженной инициализации
        if sparse_init:
            # Создаем маску с половиной нулевых значений
            mask = torch.rand(input_dim, output_dim) > 0.5
        else:
            mask = torch.ones(input_dim, output_dim)
        
        self.register_buffer('mask', mask.float())
        
        # Параметры для базовой функции и масштабирования
        self.scale_base = nn.Parameter(
            scale_base_mu * torch.ones(input_dim, output_dim) / np.sqrt(input_dim) +
            scale_base_sigma * (torch.rand(input_dim, output_dim) * 2 - 1) / np.sqrt(input_dim)
        )
        self.scale_base.requires_grad_(sb_trainable)
        
        self.scale_sp = nn.Parameter(
            torch.ones(input_dim, output_dim) * scale_sp / np.sqrt(input_dim) * self.mask
        )
        self.scale_sp.requires_grad_(sp_trainable)
        
        # Базовая функция (по умолчанию SiLU/Swish)
        self.base_fun = torch.nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход слоя.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Выходной тензор формы (batch_size, output_dim)
        """
        # Обеспечиваем правильную форму x
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_dim)
        
        # Вычисляем базовую составляющую
        base_component = self.base_fun(x)  # (batch_size, input_dim)
        
        # Вычисляем сплайновую составляющую используя базисную функцию
        spline_values = self.basis_function.forward(x, self.spline_coeffs)  # (batch_size, output_dim)
        
        # Объединяем компоненты и применяем маску
        y = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for i in range(self.input_dim):
            for o in range(self.output_dim):
                y[:, o] += (
                    self.scale_base[i, o] * base_component[:, i] +
                    self.scale_sp[i, o] * spline_values[:, o, i]
                ) * self.mask[i, o]
        
        return y
    
    def get_analytical_form(self) -> Dict[str, Any]:
        """
        Возвращает аналитическую форму функции слоя.
        
        Returns:
            Словарь, содержащий информацию об аналитической форме
        """
        return {
            'type': 'RecursiveSplineKANLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'degree': self.degree,
            'num_knots': self.num_knots,
            'domain': self.domain,
            'uniform': self.uniform,
            'grid_eps': self.grid_eps,
            'basis': self.basis_function.name,
            'coefficients': self.spline_coeffs.detach().cpu().numpy(),
            'scale_base': self.scale_base.detach().cpu().numpy(),
            'scale_sp': self.scale_sp.detach().cpu().numpy(),
            'mask': self.mask.detach().cpu().numpy(),
            'grid': self.basis_function.grid.detach().cpu().numpy()
        }
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Получает коэффициенты, используемые этим слоем.
        
        Returns:
            Тензор формы (input_dim, output_dim, num_basis), содержащий коэффициенты
        """
        return self.spline_coeffs
    
    def update_grid_from_samples(self, x: torch.Tensor, mode: str = 'sample'):
        """
        Обновляет сетку на основе распределения входных образцов.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            mode: Режим обновления ('sample' или 'grid')
            
        Returns:
            None
        """
        # Делегируем обновление сетки базисной функции
        x_sorted, y_eval = self.basis_function.update_grid_from_samples(
            x, self.spline_coeffs, mode
        )
        
        # Пересчитываем коэффициенты на основе обновленной сетки
        new_coeffs = self.basis_function.compute_coefficients(x_sorted, y_eval)
        
        # Обновляем коэффициенты
        self.spline_coeffs.data = new_coeffs
    
    def initialize_grid_from_parent(self, parent_layer, x: torch.Tensor, mode: str = 'sample'):
        """
        Инициализирует сетку на основе родительского слоя и образцов.
        
        Args:
            parent_layer: Родительский слой KAN (обычно с более грубой сеткой)
            x: Входной тензор формы (batch_size, input_dim)
            mode: Режим инициализации ('sample' или 'grid')
            
        Returns:
            None
        """
        # Проверяем, что parent_layer - это совместимый слой
        if not isinstance(parent_layer, (RecursiveSplineKANLayer, KANLayer)):
            raise ValueError("Родительский слой должен быть экземпляром KANLayer или RecursiveSplineKANLayer")
        
        # Получаем сетку родительского слоя
        if hasattr(parent_layer, 'basis_function') and hasattr(parent_layer.basis_function, 'grid'):
            parent_grid = parent_layer.basis_function.grid
        elif hasattr(parent_layer, 'grid'):
            parent_grid = parent_layer.grid
        else:
            raise ValueError("Не удалось получить сетку родительского слоя")
        
        # Получаем coeffs родительского слоя
        if hasattr(parent_layer, 'spline_coeffs'):
            parent_coeffs = parent_layer.spline_coeffs
        elif hasattr(parent_layer, 'coef'):
            parent_coeffs = parent_layer.coef
        elif hasattr(parent_layer, 'get_coefficients'):
            parent_coeffs = parent_layer.get_coefficients()
        else:
            raise ValueError("Не удалось получить коэффициенты родительского слоя")
        
        # Создаем промежуточный слой для интерполяции сетки
        interp_layer = RecursiveSplineKANLayer(
            input_dim=1,
            output_dim=self.input_dim,
            degree=1,  # Линейная интерполяция
            num_knots=parent_grid.shape[0] - 1,
            domain=(-1, 1),
            uniform=False
        )
        
        # Устанавливаем сетку промежуточного слоя
        interp_layer.basis_function.grid = parent_grid
        
        # Создаем параметры для линейной интерполяции
        t = torch.linspace(-1, 1, self.num_knots + 1, device=x.device)
        interp_inputs = t.unsqueeze(1)
        
        # Вычисляем новую сетку путем интерполяции
        new_grid = interp_layer(interp_inputs).reshape(self.num_knots + 1, self.input_dim).t()
        
        # Расширяем сетку
        extended_grid = self.basis_function._extend_grid(new_grid.unsqueeze(0), self.degree).squeeze(0)
        
        # Обновляем сетку базисной функции
        self.basis_function.grid.data = extended_grid
        
        # Вычисляем значения в новых точках
        if mode == 'grid':
            # Используем более плотную сетку для более точной интерполяции
            dense_grid = torch.linspace(self.domain[0], self.domain[1], 2 * self.num_knots + 1, device=x.device)
            sample_points = dense_grid.repeat(self.input_dim, 1).t()
        else:
            # Используем исходные образцы
            sample_points = x
        
        # Вычисляем значения на родительской сетке
        parent_values = parent_layer(sample_points)
        
        # Преобразуем в формат для compute_coefficients
        shaped_values = parent_values.unsqueeze(1).expand(-1, self.input_dim, -1)
        
        # Вычисляем коэффициенты для новой сетки
        new_coeffs = self.basis_function.compute_coefficients(sample_points, shaped_values)
        
        # Обновляем коэффициенты
        self.spline_coeffs.data = new_coeffs
    
    def get_subset(self, in_indices: List[int], out_indices: List[int]):
        """
        Получает подмножество слоя для указанных входных и выходных индексов.
        
        Args:
            in_indices: Список индексов входных нейронов
            out_indices: Список индексов выходных нейронов
            
        Returns:
            Новый слой RecursiveSplineKANLayer с указанным подмножеством
        """
        # Создаем новый слой с уменьшенными размерностями
        subset_layer = RecursiveSplineKANLayer(
            input_dim=len(in_indices),
            output_dim=len(out_indices),
            num_knots=self.num_knots,
            degree=self.degree,
            domain=self.domain,
            uniform=self.uniform,
            grid_eps=self.grid_eps
        )
        
        # Копируем сетку для выбранных входных измерений
        subset_layer.basis_function.grid.data = self.basis_function.grid[in_indices]
        
        # Копируем коэффициенты для выбранных входных и выходных измерений
        subset_layer.spline_coeffs.data = self.spline_coeffs[in_indices][:, out_indices]
        
        # Копируем масштабные параметры и маску
        subset_layer.scale_base.data = self.scale_base[in_indices][:, out_indices]
        subset_layer.scale_sp.data = self.scale_sp[in_indices][:, out_indices]
        subset_layer.mask.data = self.mask[in_indices][:, out_indices]
        
        return subset_layer
    
    def swap(self, i1: int, i2: int, mode: str = 'in'):
        """
        Меняет местами i1-й и i2-й нейроны на входе или выходе.
        
        Args:
            i1: Индекс первого нейрона
            i2: Индекс второго нейрона
            mode: 'in' для входных нейронов, 'out' для выходных
            
        Returns:
            None
        """
        with torch.no_grad():
            if mode == 'in':
                # Обмен сеткой
                grid_temp = self.basis_function.grid[i1].clone()
                self.basis_function.grid[i1] = self.basis_function.grid[i2].clone()
                self.basis_function.grid[i2] = grid_temp
                
                # Обмен коэффициентами
                coef_temp = self.spline_coeffs[i1].clone()
                self.spline_coeffs[i1] = self.spline_coeffs[i2].clone()
                self.spline_coeffs[i2] = coef_temp
                
                # Обмен scale_base
                scale_base_temp = self.scale_base[i1].clone()
                self.scale_base[i1] = self.scale_base[i2].clone()
                self.scale_base[i2] = scale_base_temp
                
                # Обмен scale_sp
                scale_sp_temp = self.scale_sp[i1].clone()
                self.scale_sp[i1] = self.scale_sp[i2].clone()
                self.scale_sp[i2] = scale_sp_temp
                
                # Обмен маской
                mask_temp = self.mask[i1].clone()
                self.mask[i1] = self.mask[i2].clone()
                self.mask[i2] = mask_temp
                
            elif mode == 'out':
                # Обмен коэффициентами
                coef_temp = self.spline_coeffs[:, i1].clone()
                self.spline_coeffs[:, i1] = self.spline_coeffs[:, i2].clone()
                self.spline_coeffs[:, i2] = coef_temp
                
                # Обмен scale_base
                scale_base_temp = self.scale_base[:, i1].clone()
                self.scale_base[:, i1] = self.scale_base[:, i2].clone()
                self.scale_base[:, i2] = scale_base_temp
                
                # Обмен scale_sp
                scale_sp_temp = self.scale_sp[:, i1].clone()
                self.scale_sp[:, i1] = self.scale_sp[:, i2].clone()
                self.scale_sp[:, i2] = scale_sp_temp
                
                # Обмен маской
                mask_temp = self.mask[:, i1].clone()
                self.mask[:, i1] = self.mask[:, i2].clone()
                self.mask[:, i2] = mask_temp
            
            else:
                raise ValueError("mode должен быть 'in' или 'out'")
    
    def extra_repr(self) -> str:
        """
        Возвращает строковое представление слоя.
        
        Returns:
            Строковое представление
        """
        return (
            f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
            f'degree={self.degree}, num_knots={self.num_knots}, grid_eps={self.grid_eps}, '
            f'domain={self.domain}, uniform={self.uniform}'
        )
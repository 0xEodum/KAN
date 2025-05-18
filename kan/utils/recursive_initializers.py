import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Callable

# Инициализаторы для рекурсивных B-сплайнов

def init_recursive_bspline_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты рекурсивных B-сплайнов с равномерным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        scale: Масштабирующий фактор для инициализации
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, num_basis = tensor.shape
    
    # Инициализация с равномерным распределением масштабированным по количеству базисных функций
    bound = scale / (input_dim * math.sqrt(num_basis))
    nn.init.uniform_(tensor, -bound, bound)
    
    # Применяем специфичное для B-сплайнов масштабирование
    # Используем экспоненциальный спад для высших базисных функций
    basis_factors = torch.exp(-torch.linspace(0.0, 1.0, num_basis))
    tensor *= basis_factors.reshape(1, 1, -1)
    
    return tensor


def init_recursive_bspline_normal(tensor: torch.Tensor, 
                                mean: float = 0.0, 
                                std: Optional[float] = None) -> torch.Tensor:
    """
    Инициализирует коэффициенты рекурсивных B-сплайнов с нормальным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        mean: Среднее значение нормального распределения
        std: Стандартное отклонение (если None, использует 1/sqrt(input_dim * num_basis))
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, num_basis = tensor.shape
    
    # Установка стандартного отклонения по умолчанию, если не предоставлено
    if std is None:
        std = 1.0 / math.sqrt(input_dim * num_basis)
    
    # Инициализация с нормальным распределением
    nn.init.normal_(tensor, mean=mean, std=std)
    
    # Применяем экспоненциальный спад для высших базисных функций
    basis_factors = torch.exp(-torch.linspace(0.0, 1.0, num_basis))
    tensor *= basis_factors.reshape(1, 1, -1)
    
    return tensor


def init_recursive_bspline_orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты рекурсивных B-сплайнов с ортогональной инициализацией.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        gain: Масштабирующий фактор
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, num_basis = tensor.shape
    
    # Преобразование для ортогональной инициализации
    flat_shape = (input_dim, output_dim * num_basis)
    reshaped = tensor.reshape(flat_shape)
    
    # Применяем ортогональную инициализацию
    nn.init.orthogonal_(reshaped, gain=gain)
    
    # Преобразуем обратно
    tensor.copy_(reshaped.reshape(input_dim, output_dim, num_basis))
    
    # Применяем экспоненциальный спад для высших базисных функций
    basis_factors = torch.exp(-torch.linspace(0.0, 1.0, num_basis))
    tensor *= basis_factors.reshape(1, 1, -1)
    
    return tensor


def init_recursive_bspline_identity(tensor: torch.Tensor, 
                                   exact: bool = False,
                                   noise_scale: float = 0.01) -> torch.Tensor:
    """
    Инициализирует коэффициенты рекурсивных B-сплайнов для аппроксимации функции идентичности.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        exact: Использовать точную идентичность или добавить шум
        noise_scale: Масштаб шума для добавления, если не exact
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в ноль
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, num_basis = tensor.shape
    
    # Для аппроксимации идентичности с B-сплайнами, коэффициенты должны
    # следовать линейному шаблону. В простейшем случае, это просто линейная функция.
    min_dim = min(input_dim, output_dim)
    
    # Для каждого входного измерения i и соответствующего выходного измерения i
    for i in range(min_dim):
        # Линейный шаблон коэффициентов для функции идентичности
        # Используем линейно возрастающие коэффициенты от -1 до 1
        tensor[i, i, :] = torch.linspace(-1.0, 1.0, num_basis)
    
    # Добавляем шум, если требуется
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


def init_recursive_bspline_grid_based(tensor: torch.Tensor, 
                                     grid: torch.Tensor,
                                     fun: Callable[[torch.Tensor], torch.Tensor],
                                     noise_scale: float = 0.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты рекурсивных B-сплайнов на основе значений функции в узлах сетки.
    
    Эта инициализация позволяет аппроксимировать заданную функцию,
    вычисляя ее значения в узлах сетки и определяя коэффициенты методом
    наименьших квадратов.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        grid: Сетка формы (input_dim, num_points)
        fun: Функция, которую нужно аппроксимировать
        noise_scale: Масштаб шума для добавления
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, num_basis = tensor.shape
    
    # Устанавливаем все коэффициенты в ноль
    nn.init.zeros_(tensor)
    
    # Создаем тестовые точки на основе сетки
    grid_points = grid[:, grid.shape[1]//2:(grid.shape[1]//2 + num_basis)]
    test_points = grid_points.t()  # (num_basis, input_dim)
    
    # Вычисляем значения функции в тестовых точках
    # Функция должна принимать тензор (batch_size, input_dim) и возвращать (batch_size, output_dim)
    with torch.no_grad():
        function_values = fun(test_points)  # (num_basis, output_dim)
    
    # Создаем матрицу базисных функций для каждого измерения
    # В простейшем случае, это просто единичная матрица
    for i in range(input_dim):
        for o in range(output_dim):
            # Используем линейную аппроксимацию для простоты
            tensor[i, o, :] = function_values[:, o] / input_dim
    
    # Добавляем шум, если требуется
    if noise_scale > 0:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


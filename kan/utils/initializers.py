import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Callable


# ====== Инициализаторы для полиномов Чебышева ======

def init_chebyshev_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты Чебышева с равномерным распределением.
    
    Эта инициализация основана на идее, что величина коэффициентов Чебышева
    обычно уменьшается с возрастанием степени для гладких функций.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        scale: Масштабирующий фактор для инициализации
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Инициализация с равномерным распределением масштабированным по степени
    bound = scale / (input_dim * math.sqrt(degree + 1))
    nn.init.uniform_(tensor, -bound, bound)
    
    # Применяем масштабирование по степени для преобладания низких степеней
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_chebyshev_normal(tensor: torch.Tensor, 
                         mean: float = 0.0, 
                         std: Optional[float] = None) -> torch.Tensor:
    """
    Инициализирует коэффициенты Чебышева с нормальным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        mean: Среднее значение нормального распределения
        std: Стандартное отклонение (если None, использует 1/sqrt(input_dim * (degree+1)))
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Установка стандартного отклонения по умолчанию, если не предоставлено
    if std is None:
        std = 1.0 / math.sqrt(input_dim * (degree + 1))
    
    # Инициализация с нормальным распределением
    nn.init.normal_(tensor, mean=mean, std=std)
    
    # Применяем масштабирование по степени для преобладания низких степеней
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_chebyshev_orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты Чебышева с ортогональной инициализацией.
    
    Это адаптированная версия nn.init.orthogonal_, модифицированная для работы с
    3D тензорной структурой коэффициентов KAN.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        gain: Масштабирующий фактор
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Преобразование для ортогональной инициализации
    flat_shape = (input_dim, output_dim * degree_plus_one)
    reshaped = tensor.reshape(flat_shape)
    
    # Применяем ортогональную инициализацию
    nn.init.orthogonal_(reshaped, gain=gain)
    
    # Преобразуем обратно
    tensor.copy_(reshaped.reshape(input_dim, output_dim, degree_plus_one))
    
    # Применяем масштабирование по степени
    degree_factors = torch.linspace(1.0, 0.1, degree + 1)
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_chebyshev_zeros(tensor: torch.Tensor) -> torch.Tensor:
    """
    Инициализирует коэффициенты Чебышева нулями, кроме постоянного члена.
    
    Эта инициализация устанавливает все коэффициенты в ноль, кроме постоянного
    члена (степень 0), который инициализируется малыми случайными значениями.
    Это полезно для начала с почти идентичной функции.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в ноль
    nn.init.zeros_(tensor)
    
    # Инициализируем только постоянный член (степень 0)
    input_dim, output_dim, _ = tensor.shape
    const_term = tensor[:, :, 0]
    nn.init.normal_(const_term, mean=0.0, std=0.01)
    
    # Для входного измерения i, устанавливаем линейный член (степень 1) для выхода i близким к 1
    # Это помогает создать приблизительно идентичную начальную функцию
    for i in range(min(input_dim, output_dim)):
        tensor[i, i, 1] = 1.0 + torch.randn(1).item() * 0.01
    
    return tensor


def init_chebyshev_identity(tensor: torch.Tensor, 
                           exact: bool = False,
                           noise_scale: float = 0.01) -> torch.Tensor:
    """
    Инициализирует коэффициенты Чебышева для аппроксимации функции идентичности.
    
    Для функции идентичности f(x) = x разложение Чебышева имеет
    T_1(x) = x как единственный ненулевой член. Эта инициализация устанавливает
    коэффициенты для создания отображения, близкого к идентичному,
    с возможным добавлением шума.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        exact: Использовать точную идентичность или добавить шум
        noise_scale: Масштаб шума для добавления, если не exact
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в ноль
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, degree_plus_one = tensor.shape
    
    # Проверка, что степень не менее 1 (нам нужен T_1(x) = x для идентичности)
    if degree_plus_one < 2:
        raise ValueError("Степень должна быть не менее 1 для инициализации идентичности")
    
    # Отображение идентичности: установка коэффициента для T_1(x) = x
    min_dim = min(input_dim, output_dim)
    for i in range(min_dim):
        tensor[i, i, 1] = 1.0
    
    # Добавляем шум, если требуется
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


# ====== Инициализаторы для полиномов Якоби ======

def init_jacobi_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты Якоби с равномерным распределением.
    
    Эта инициализация аналогична случаю Чебышева, но адаптирована для полиномов Якоби.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        scale: Масштабирующий фактор для инициализации
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Инициализация с равномерным распределением масштабированным по степени
    bound = scale / (input_dim * math.sqrt(degree + 1))
    nn.init.uniform_(tensor, -bound, bound)
    
    # Применяем масштабирование по степени для преобладания низких степеней
    # Для полиномов Якоби мы уменьшаем веса быстрее с степенью,
    # поскольку полиномы высоких степеней могут иметь большие значения
    degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_normal(tensor: torch.Tensor, 
                      mean: float = 0.0, 
                      std: Optional[float] = None) -> torch.Tensor:
    """
    Инициализирует коэффициенты Якоби с нормальным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        mean: Среднее значение нормального распределения
        std: Стандартное отклонение (если None, использует 1/sqrt(input_dim * (degree+1)))
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Установка стандартного отклонения по умолчанию, если не предоставлено
    if std is None:
        std = 1.0 / math.sqrt(input_dim * (degree + 1))
    
    # Инициализация с нормальным распределением
    nn.init.normal_(tensor, mean=mean, std=std)
    
    # Применяем масштабирование по степени
    degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты Якоби с ортогональной инициализацией.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        gain: Масштабирующий фактор
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Преобразование для ортогональной инициализации
    flat_shape = (input_dim, output_dim * degree_plus_one)
    reshaped = tensor.reshape(flat_shape)
    
    # Применяем ортогональную инициализацию
    nn.init.orthogonal_(reshaped, gain=gain)
    
    # Преобразуем обратно
    tensor.copy_(reshaped.reshape(input_dim, output_dim, degree_plus_one))
    
    # Применяем масштабирование по степени
    degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_jacobi_identity(tensor: torch.Tensor, 
                        alpha: float = 0.0, 
                        beta: float = 0.0,
                        exact: bool = False,
                        noise_scale: float = 0.01) -> torch.Tensor:
    """
    Инициализирует коэффициенты Якоби для аппроксимации функции идентичности.
    
    Для функции идентичности f(x) = x, нам нужно определить коэффициенты
    в разложении Якоби. Для α = β = 0 (Лежандр), коэффициент P_1 равен 1,
    а для других значений α, β нужны специфические значения.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        alpha: Параметр α полиномов Якоби
        beta: Параметр β полиномов Якоби
        exact: Использовать точную идентичность или добавить шум
        noise_scale: Масштаб шума для добавления, если не exact
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в ноль
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, degree_plus_one = tensor.shape
    
    # Проверка, что степень не менее 1 (нам нужен как минимум P_1 для идентичности)
    if degree_plus_one < 2:
        raise ValueError("Степень должна быть не менее 1 для инициализации идентичности")
    
    # Для полиномов Якоби P_1^(α,β)(x) = ((α + β + 2)x + (α - β))/2
    # Для представления идентичности f(x) = x, нам нужно скорректировать коэффициент
    identity_coef = 2.0 / (alpha + beta + 2)  # Чтобы сделать коэффициент x = 1
    
    # Отображение идентичности: установка коэффициента для P_1
    min_dim = min(input_dim, output_dim)
    for i in range(min_dim):
        tensor[i, i, 1] = identity_coef
    
    # Если α ≠ β, нам нужен постоянный член для компенсации (α - β) в P_1
    if alpha != beta and degree_plus_one > 0:
        const_coef = -(alpha - beta) * identity_coef / 2
        for i in range(min_dim):
            tensor[i, i, 0] = const_coef
    
    # Добавляем шум, если требуется
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


# ====== Инициализаторы для полиномов Эрмита ======

def init_hermite_uniform(tensor: torch.Tensor, 
                        scaling: str = 'physicist',
                        scale: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты Эрмита с равномерным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        scaling: Тип полиномов Эрмита ('physicist' или 'probabilist')
        scale: Масштабирующий фактор для инициализации
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Инициализация с равномерным распределением масштабированным по степени
    bound = scale / (input_dim * math.sqrt(degree + 1))
    nn.init.uniform_(tensor, -bound, bound)
    
    # Применяем масштабирование по степени для преобладания низких степеней
    # Полиномы Эрмита быстро растут с степенью, поэтому используем более сильное масштабирование
    if scaling == 'physicist':
        # Для физических полиномов Эрмита рост быстрее
        # H_n(x) ~ 2^(n/2) для больших n
        degree_factors = torch.exp(-torch.linspace(0.0, 3.0, degree + 1))
    else:  # scaling == 'probabilist'
        # Для вероятностных полиномов Эрмита рост немного медленнее
        # He_n(x) ~ 1 для больших n
        degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_hermite_normal(tensor: torch.Tensor, 
                       scaling: str = 'physicist',
                       mean: float = 0.0, 
                       std: Optional[float] = None) -> torch.Tensor:
    """
    Инициализирует коэффициенты Эрмита с нормальным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        scaling: Тип полиномов Эрмита ('physicist' или 'probabilist')
        mean: Среднее значение нормального распределения
        std: Стандартное отклонение (если None, использует 1/sqrt(input_dim * (degree+1)))
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Установка стандартного отклонения по умолчанию, если не предоставлено
    if std is None:
        std = 1.0 / math.sqrt(input_dim * (degree + 1))
    
    # Инициализация с нормальным распределением
    nn.init.normal_(tensor, mean=mean, std=std)
    
    # Применяем масштабирование по степени
    if scaling == 'physicist':
        degree_factors = torch.exp(-torch.linspace(0.0, 3.0, degree + 1))
    else:  # scaling == 'probabilist'
        degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_hermite_orthogonal(tensor: torch.Tensor, 
                           scaling: str = 'physicist',
                           gain: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты Эрмита с ортогональной инициализацией.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        scaling: Тип полиномов Эрмита ('physicist' или 'probabilist')
        gain: Масштабирующий фактор
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, degree_plus_one = tensor.shape
    degree = degree_plus_one - 1
    
    # Преобразование для ортогональной инициализации
    flat_shape = (input_dim, output_dim * degree_plus_one)
    reshaped = tensor.reshape(flat_shape)
    
    # Применяем ортогональную инициализацию
    nn.init.orthogonal_(reshaped, gain=gain)
    
    # Преобразуем обратно
    tensor.copy_(reshaped.reshape(input_dim, output_dim, degree_plus_one))
    
    # Применяем масштабирование по степени
    if scaling == 'physicist':
        degree_factors = torch.exp(-torch.linspace(0.0, 3.0, degree + 1))
    else:  # scaling == 'probabilist'
        degree_factors = torch.exp(-torch.linspace(0.0, 2.0, degree + 1))
    
    tensor *= degree_factors.reshape(1, 1, -1)
    
    return tensor


def init_hermite_identity(tensor: torch.Tensor, 
                         scaling: str = 'physicist',
                         exact: bool = False,
                         noise_scale: float = 0.01) -> torch.Tensor:
    """
    Инициализирует коэффициенты Эрмита для аппроксимации функции идентичности.
    
    Для функции идентичности f(x) = x, нам нужно определить коэффициенты
    в разложении Эрмита.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, degree+1)
        scaling: Тип полиномов Эрмита ('physicist' или 'probabilist')
        exact: Использовать точную идентичность или добавить шум
        noise_scale: Масштаб шума для добавления, если не exact
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в ноль
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, degree_plus_one = tensor.shape
    
    # Проверка, что степень не менее 1 (нам нужен как минимум H_1 для идентичности)
    if degree_plus_one < 2:
        raise ValueError("Степень должна быть не менее 1 для инициализации идентичности")
    
    # Для идентичности f(x) = x:
    # В физическом масштабировании: H_1(x) = 2x, поэтому коэффициент должен быть 0.5
    # В вероятностном масштабировании: He_1(x) = x, поэтому коэффициент должен быть 1.0
    identity_coef = 0.5 if scaling == 'physicist' else 1.0
    
    # Отображение идентичности: установка коэффициента для H_1 или He_1
    min_dim = min(input_dim, output_dim)
    for i in range(min_dim):
        tensor[i, i, 1] = identity_coef
    
    # Добавляем шум, если требуется
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


# ====== Инициализаторы для B-сплайнов ======

def init_bspline_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты B-сплайнов с равномерным распределением.
    
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
    # B-сплайны формируют разбиение единицы (их сумма = 1), поэтому нам нужны малые коэффициенты
    tensor *= 0.5
    
    return tensor


def init_bspline_normal(tensor: torch.Tensor, 
                        mean: float = 0.0, 
                        std: Optional[float] = None) -> torch.Tensor:
    """
    Инициализирует коэффициенты B-сплайнов с нормальным распределением.
    
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
    
    return tensor


def init_bspline_identity(tensor: torch.Tensor, 
                         exact: bool = False,
                         noise_scale: float = 0.01) -> torch.Tensor:
    """
    Инициализирует коэффициенты B-сплайнов для аппроксимации функции идентичности.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        exact: Использовать точную идентичность или добавить шум
        noise_scale: Масштаб шума для добавления, если не exact
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в нуль
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, num_basis = tensor.shape
    
    # Для аппроксимации идентичности с B-сплайнами, коэффициенты должны
    # следовать линейному шаблону. В простейшем случае, это просто линейная функция.
    min_dim = min(input_dim, output_dim)
    
    # Для каждого входного измерения i и соответствующего выходного измерения i
    for i in range(min_dim):
        # Линейный шаблон коэффициентов для функции идентичности
        # Используем линейно возрастающие коэффициенты от -1 до 1
        tensor[i, i, :] = torch.linspace(-0.5, 0.5, num_basis)
    
    # Добавляем шум, если требуется
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


# ====== Инициализаторы для кубических сплайнов ======

def init_cubic_spline_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты кубических сплайнов с равномерным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_knots)
        scale: Масштабирующий фактор для инициализации
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, num_knots = tensor.shape
    
    # Инициализация с равномерным распределением
    bound = scale / (input_dim * math.sqrt(num_knots))
    nn.init.uniform_(tensor, -bound, bound)
    
    return tensor


def init_cubic_spline_normal(tensor: torch.Tensor, 
                             mean: float = 0.0, 
                             std: Optional[float] = None) -> torch.Tensor:
    """
    Инициализирует коэффициенты кубических сплайнов с нормальным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_knots)
        mean: Среднее значение нормального распределения
        std: Стандартное отклонение (если None, использует 1/sqrt(input_dim * num_knots))
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, num_knots = tensor.shape
    
    # Установка стандартного отклонения по умолчанию, если не предоставлено
    if std is None:
        std = 1.0 / math.sqrt(input_dim * num_knots)
    
    # Инициализация с нормальным распределением
    nn.init.normal_(tensor, mean=mean, std=std)
    
    return tensor


def init_cubic_spline_identity(tensor: torch.Tensor, 
                               exact: bool = False,
                               noise_scale: float = 0.01) -> torch.Tensor:
    """
    Инициализирует коэффициенты кубических сплайнов для аппроксимации функции идентичности.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_knots)
        exact: Использовать точную идентичность или добавить шум
        noise_scale: Масштаб шума для добавления, если не exact
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в нуль
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, num_knots = tensor.shape
    
    # Для кубических сплайнов идентичность можно аппроксимировать линейно возрастающими значениями
    min_dim = min(input_dim, output_dim)
    
    # Для каждого входного измерения i и соответствующего выходного измерения i
    for i in range(min_dim):
        # Линейный шаблон коэффициентов для функции идентичности
        # Для кубических сплайнов мы используем линейно возрастающие значения в узлах
        tensor[i, i, :] = torch.linspace(-1.0, 1.0, num_knots)
    
    # Добавляем шум, если требуется
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale
        tensor += noise
    
    return tensor


# ====== Инициализаторы для адаптивных сплайнов ======

def init_adaptive_spline_uniform(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Инициализирует коэффициенты адаптивных сплайнов с равномерным распределением.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        scale: Масштабирующий фактор для инициализации
        
    Returns:
        Инициализированный тензор
    """
    input_dim, output_dim, num_basis = tensor.shape
    
    # Инициализация с равномерным распределением, уменьшая масштаб
    # для компенсации адаптивности узлов
    bound = scale / (input_dim * math.sqrt(num_basis)) * 0.5
    nn.init.uniform_(tensor, -bound, bound)
    
    return tensor


def init_adaptive_spline_normal(tensor: torch.Tensor, 
                              mean: float = 0.0, 
                              std: Optional[float] = None) -> torch.Tensor:
    """
    Инициализирует коэффициенты адаптивных сплайнов с нормальным распределением.
    
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
        std = 1.0 / math.sqrt(input_dim * num_basis) * 0.5
    
    # Инициализация с нормальным распределением
    nn.init.normal_(tensor, mean=mean, std=std)
    
    return tensor


def init_adaptive_spline_identity(tensor: torch.Tensor, 
                                 exact: bool = False,
                                 noise_scale: float = 0.01) -> torch.Tensor:
    """
    Инициализирует коэффициенты адаптивных сплайнов для аппроксимации функции идентичности.
    
    Args:
        tensor: Тензор для инициализации формы (input_dim, output_dim, num_basis)
        exact: Использовать точную идентичность или добавить шум
        noise_scale: Масштаб шума для добавления, если не exact
        
    Returns:
        Инициализированный тензор
    """
    # Устанавливаем все коэффициенты в нуль
    nn.init.zeros_(tensor)
    
    input_dim, output_dim, num_basis = tensor.shape
    
    # Для адаптивных сплайнов идентичность создается через шаблон коэффициентов
    # аналогичный B-сплайнам, но с большей осторожностью из-за адаптивности узлов
    min_dim = min(input_dim, output_dim)
    
    # Для каждого входного измерения i и соответствующего выходного измерения i
    for i in range(min_dim):
        # Создание шаблона с меньшей амплитудой
        tensor[i, i, :] = torch.linspace(-0.25, 0.25, num_basis)
    
    # Добавляем шум, если требуется
    if not exact:
        noise = torch.randn_like(tensor) * noise_scale * 0.5
        tensor += noise
    
    return tensor


def get_initializer(name: str) -> Callable:
    """
    Получает функцию инициализации по имени.
    
    Args:
        name: Имя инициализатора
        
    Returns:
        Функция инициализации
    """
    initializers = {
        # Инициализаторы для полиномов Чебышева
        'chebyshev_normal': init_chebyshev_normal,
        'chebyshev_uniform': init_chebyshev_uniform,
        'chebyshev_orthogonal': init_chebyshev_orthogonal,
        'chebyshev_zeros': init_chebyshev_zeros,
        'chebyshev_identity': init_chebyshev_identity,
        
        # Инициализаторы для полиномов Якоби
        'jacobi_normal': init_jacobi_normal,
        'jacobi_uniform': init_jacobi_uniform,
        'jacobi_orthogonal': init_jacobi_orthogonal,
        'jacobi_identity': init_jacobi_identity,
        
        # Инициализаторы для полиномов Эрмита
        'hermite_normal': init_hermite_normal,
        'hermite_uniform': init_hermite_uniform,
        'hermite_orthogonal': init_hermite_orthogonal,
        'hermite_identity': init_hermite_identity,
        
        # Инициализаторы для B-сплайнов
        'bspline_normal': init_bspline_normal,
        'bspline_uniform': init_bspline_uniform,
        'bspline_identity': init_bspline_identity,
        
        # Инициализаторы для кубических сплайнов
        'cubic_spline_normal': init_cubic_spline_normal,
        'cubic_spline_uniform': init_cubic_spline_uniform,
        'cubic_spline_identity': init_cubic_spline_identity,
        
        # Инициализаторы для адаптивных сплайнов
        'adaptive_spline_normal': init_adaptive_spline_normal,
        'adaptive_spline_uniform': init_adaptive_spline_uniform,
        'adaptive_spline_identity': init_adaptive_spline_identity,
        
        # Для обратной совместимости сохраняем оригинальные имена
        'normal': init_chebyshev_normal,
        'uniform': init_chebyshev_uniform,
        'orthogonal': init_chebyshev_orthogonal,
        'zeros': init_chebyshev_zeros,
        'identity': init_chebyshev_identity
    }
    
    if name not in initializers:
        raise ValueError(f"Неизвестный инициализатор: {name}. Доступные инициализаторы: "
                       f"{', '.join(initializers.keys())}")
    
    return initializers[name]
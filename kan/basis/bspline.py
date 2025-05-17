import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union, Callable
from .base import BasisFunction


class BSplineBasis(BasisFunction):
    """
    B-сплайн базисные функции для KAN.
    
    B-сплайны (базисные сплайны) - это кусочно-полиномиальные функции, которые
    образуют базис для пространства сплайнов. Они обладают свойствами
    локальной поддержки и управляемой гладкости, что делает их полезными для
    аппроксимации функций с локальными особенностями.
    
    B-сплайн степени k определяется рекурсивно через узловой вектор:
    B_{i,0}(x) = 1, если t_i <= x < t_{i+1}, иначе 0
    B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x) + 
                 (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
    """
    
    def __init__(self, degree: int, num_knots: int = None, 
                domain: Tuple[float, float] = (-1, 1), uniform: bool = True):
        """
        Инициализирует B-сплайн базис.
        
        Args:
            degree: Степень B-сплайнов (0 - постоянный, 1 - линейный, 
                   2 - квадратичный, 3 - кубический и т.д.)
            num_knots: Количество внутренних узлов. Если None, устанавливается как degree + 2.
                      Общее количество базисных функций будет num_knots + degree.
            domain: Область определения сплайнов [a, b]
            uniform: Использовать равномерное распределение узлов
        """
        super().__init__(degree)
        
        # Количество внутренних узлов
        self.num_knots = num_knots if num_knots is not None else degree + 2
        self.uniform = uniform
        self._domain = domain
        
        # Генерация узлового вектора
        # Для B-сплайнов степени k нужно повторить крайние узлы (k+1) раз
        a, b = domain
        
        if uniform:
            # Равномерное распределение узлов
            inner_knots = np.linspace(a, b, self.num_knots)
        else:
            # Неравномерное распределение с скоплением узлов в центре
            # (можно настроить под конкретную задачу)
            t = np.linspace(0, 1, self.num_knots)
            inner_knots = a + (b - a) * (0.5 + 0.5 * np.sin(np.pi * (t - 0.5)))
        
        # Создание полного узлового вектора с повторением границ
        knots = np.concatenate([
            np.ones(degree + 1) * a,
            inner_knots,
            np.ones(degree + 1) * b
        ])
        
        # Общее количество базисных функций
        self.num_basis = len(knots) - degree - 1
        
        # Сохранение узлового вектора как параметр (без градиента)
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32))
        
        # Для нормализации входов
        # Полезно для вычисления индексов узлов
        self.register_buffer("domain_tensor", torch.tensor(domain, dtype=torch.float32))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """
        Регистрирует буфер для базисной функции.
        
        Args:
            name: Имя буфера
            tensor: Тензор для регистрации
        """
        if not hasattr(self, name):
            setattr(self, name, tensor)
        else:
            # Обновляем существующий буфер
            existing_buffer = getattr(self, name)
            existing_buffer.data = tensor.data
    
    @property
    def name(self) -> str:
        """Возвращает имя базисной функции."""
        return f"B-spline-{self.degree}-{self.num_knots}"
    
    @property
    def domain(self) -> Tuple[float, float]:
        """Возвращает область определения B-сплайнов."""
        return self._domain
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Нормализует входные данные в область определения сплайнов.
        
        Args:
            x: Входной тензор
            
        Returns:
            Нормализованный входной тензор
        """
        # Сигмоидная нормализация в область domain
        a, b = self._domain
        normalized = a + (b - a) * torch.sigmoid(x)
        return normalized
    
    def forward(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет B-сплайновое разложение в точках x с заданными коэффициентами.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            coefficients: Тензор коэффициентов формы (input_dim, output_dim, num_basis)
            
        Returns:
            Выходной тензор формы (batch_size, output_dim)
        """
        # Проверка размерности x
        x = x.view(-1, coefficients.shape[0])
        
        # Нормализация входа
        x = self.normalize_input(x)
        
        # Вычисление базисных функций
        basis_values = self.basis_functions(x)
        
        # Применение коэффициентов с использованием einsum
        return torch.einsum('bid,iod->bo', basis_values, coefficients)
    
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет значения всех B-сплайн базисных функций в точках x.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Тензор формы (batch_size, input_dim, num_basis), содержащий
            значения B-сплайн функций для каждой точки.
        """
        return self._de_boor_values(x)
    
    def _de_boor_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет значения B-сплайнов используя алгоритм де Бура.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Тензор формы (batch_size, input_dim, num_basis)
        """
        batch_size, input_dim = x.shape
        
        # Создаем выходной тензор
        result = torch.zeros(batch_size, input_dim, self.num_basis, 
                           device=x.device, dtype=x.dtype)
        
        # Для каждой точки вычисляем значения базисных функций
        for batch_idx in range(batch_size):
            for dim_idx in range(input_dim):
                # Текущее значение x
                t = x[batch_idx, dim_idx].item()
                
                # Корректировка для граничных значений
                t = min(max(t, self._domain[0]), self._domain[1] - 1e-7)
                
                # Поиск интервала, в котором находится t
                k = self.degree
                
                # Находим индекс i такой, что u_i <= t < u_{i+1}
                i = self._find_span(t)
                
                # B-сплайны степени 0 (константные)
                # B_i,0(t) = 1 если u_i <= t < u_{i+1}, иначе 0
                d = [0.0] * (k + 1)
                for j in range(k + 1):
                    if self.knots[i - k + j].item() <= t < self.knots[i - k + j + 1].item():
                        d[j] = 1.0
                
                # Вычисление B-сплайнов более высоких степеней по рекурсивной формуле
                for r in range(1, k + 1):
                    for j in range(k - r + 1):
                        # Веса для рекурсивной формулы
                        u_left = self.knots[i - k + j + r].item()
                        u_right = self.knots[i + j + 1].item()
                        
                        # Если знаменатель равен нулю, устанавливаем коэффициент равным нулю
                        c1 = 0.0 if u_left - self.knots[i - k + j].item() == 0.0 else \
                             (t - self.knots[i - k + j].item()) / (u_left - self.knots[i - k + j].item())
                        
                        c2 = 0.0 if self.knots[i + j + 1].item() - self.knots[i + j + 1 - r].item() == 0.0 else \
                             (self.knots[i + j + 1].item() - t) / (self.knots[i + j + 1].item() - self.knots[i + j + 1 - r].item())
                        
                        # Новое значение
                        d[j] = c1 * d[j] + c2 * d[j + 1]
                
                # Заполняем тензор результатов
                # Значения нулевые везде, кроме k+1 базисных функций начиная с i-k
                for j in range(k + 1):
                    basis_idx = i - k + j
                    if 0 <= basis_idx < self.num_basis:
                        result[batch_idx, dim_idx, basis_idx] = d[j]
        
        return result
    
    def _find_span(self, t: float) -> int:
        """
        Находит индекс узла i такой, что u_i <= t < u_{i+1}.
        
        Args:
            t: Значение параметра
            
        Returns:
            Индекс узла
        """
        # Корректировка для граничных значений
        if t >= self.knots[-1].item():
            return len(self.knots) - self.degree - 2
        
        # Бинарный поиск
        low = self.degree
        high = len(self.knots) - 1
        
        while low <= high:
            mid = (low + high) // 2
            if t < self.knots[mid].item():
                high = mid - 1
            elif t >= self.knots[mid + 1].item():
                low = mid + 1
            else:
                return mid
        
        return high
    
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                 order: int = 1) -> torch.Tensor:
        """
        Вычисляет производную B-сплайнового разложения.
        
        Args:
            x: Входной тензор
            coefficients: Тензор коэффициентов
            order: Порядок производной
            
        Returns:
            Тензор производной формы (batch_size, output_dim)
        """
        # Для порядка 0 просто возвращаем оригинальные значения
        if order == 0:
            return self.forward(x, coefficients)
        
        # Если порядок превышает степень сплайна, результат - нули
        if order > self.degree:
            return torch.zeros_like(x[:, :coefficients.shape[1]], device=x.device)
        
        # Преобразуем коэффициенты для вычисления производной
        derivative_coeffs = self._derivative_coefficients(coefficients)
        
        # Создаем B-сплайн меньшей степени для производной
        derivative_basis = BSplineBasis(
            degree=self.degree - 1,
            num_knots=self.num_knots,
            domain=self.domain,
            uniform=self.uniform
        )
        
        # Передаем узлы из текущего базиса с учетом уменьшения степени
        derivative_knots = self.knots[1:-1].clone()
        derivative_basis.knots = derivative_knots
        
        # Рекурсивное вычисление для производных высших порядков
        if order > 1:
            return derivative_basis.derivative(x, derivative_coeffs, order - 1)
        else:
            return derivative_basis.forward(x, derivative_coeffs)
    
    def _derivative_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет коэффициенты для производной B-сплайнового разложения.
        
        Производная B-сплайна степени k выражается через B-сплайны степени k-1:
        d/dt B_{i,k}(t) = k/(u_{i+k}-u_i) * B_{i,k-1}(t) - k/(u_{i+k+1}-u_{i+1}) * B_{i+1,k-1}(t)
        
        Args:
            coefficients: Тензор коэффициентов формы (input_dim, output_dim, num_basis)
            
        Returns:
            Коэффициенты производной формы (input_dim, output_dim, num_basis-1)
        """
        input_dim, output_dim, num_basis = coefficients.shape
        device = coefficients.device
        
        # Для B-сплайна степени k, мы имеем num_basis-1 B-сплайнов степени k-1
        derivative_coeffs = torch.zeros(input_dim, output_dim, num_basis - 1, 
                                     device=device, dtype=coefficients.dtype)
        
        # Вычисление коэффициентов производной
        for i in range(num_basis - 1):
            # Для каждого i от 0 до num_basis-2
            # Разница узлов для нормализации
            u_diff1 = self.knots[i + self.degree + 1] - self.knots[i + 1]
            u_diff0 = self.knots[i + self.degree] - self.knots[i]
            
            # Предотвращение деления на ноль
            scale0 = 0.0 if u_diff0 == 0.0 else self.degree / u_diff0
            scale1 = 0.0 if u_diff1 == 0.0 else self.degree / u_diff1
            
            derivative_coeffs[:, :, i] = scale0 * coefficients[:, :, i] - scale1 * coefficients[:, :, i + 1]
        
        return derivative_coeffs
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union, Callable
from .base import BasisFunction


class RecursiveBSplineBasis(BasisFunction):
    """
    Рекурсивная реализация B-сплайновых базисных функций для KAN.
    
    Эта реализация использует рекурсивное определение B-сплайнов:
    B_{i,0}(x) = 1, если t_i <= x < t_{i+1}, иначе 0
    B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x) + 
                 (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
    
    В отличие от итеративного алгоритма де Бура, используемого в BSplineBasis,
    эта реализация применяет векторизованные операции для эффективного вычисления
    на больших батчах данных. Также она включает дополнительные функции для 
    динамического обновления сеток.
    """
    
    def __init__(self, degree: int, num_knots: int = None, 
                domain: Tuple[float, float] = (-1, 1), 
                uniform: bool = True,
                grid_eps: float = 0.02):
        """
        Инициализирует рекурсивную B-сплайн базу.
        
        Args:
            degree: Степень B-сплайнов (0 - константный, 1 - линейный, 2 - квадратичный, и т.д.)
            num_knots: Количество внутренних узлов (если None, устанавливается как degree + 2)
            domain: Область определения сплайнов [a, b]
            uniform: Использовать равномерное распределение узлов
            grid_eps: Параметр для интерполяции между равномерной и адаптивной сеткой
                      (0 = полностью адаптивная, 1 = полностью равномерная)
        """
        super().__init__(degree)
        
        # Сохраняем параметры
        self.num_knots = num_knots if num_knots is not None else degree + 2
        self._domain = domain
        self.uniform = uniform
        self.grid_eps = grid_eps
        
        # Генерация узлового вектора
        a, b = domain
        
        if uniform:
            # Равномерное распределение узлов
            inner_knots = torch.linspace(a, b, self.num_knots)
        else:
            # Неравномерное распределение с скоплением узлов в центре
            t = torch.linspace(0, 1, self.num_knots)
            inner_knots = torch.tensor(a + (b - a) * (0.5 + 0.5 * torch.sin(np.pi * (t - 0.5))))
        
        # Расширяем сетку
        grid = inner_knots[None, :].expand(1, self.num_knots)
        grid = self._extend_grid(grid, k_extend=degree)
        
        # Сохраняем сетку как параметр (без градиента)
        self.register_buffer("grid", grid.squeeze(0))
        
        # Количество базисных функций
        self.num_basis = len(inner_knots) + degree
    
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
        return f"RecursiveB-spline-{self.degree}-{self.num_knots}"
    
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
        
        # Вычисление базисных функций и применение коэффициентов
        basis_values = self.basis_functions(x)
        
        # Применение коэффициентов с использованием einsum
        return torch.einsum('bid,iod->bo', basis_values, coefficients)
    
    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет значения всех B-сплайновых базисных функций в точках x.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Тензор формы (batch_size, input_dim, num_basis), содержащий
            значения B-сплайновых функций для каждой точки.
        """
        return self._batch_bspline_values(x)
    
    def _batch_bspline_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет значения B-сплайнов рекурсивно с использованием векторизованных операций.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Тензор формы (batch_size, input_dim, num_basis)
        """
        batch_size, input_dim = x.shape
        device = x.device
        
        # Подготавливаем x для вычислений
        x_unsqueezed = x.unsqueeze(dim=2)
        grid_unsqueezed = self.grid.unsqueeze(dim=0)
        
        # Рекурсивное вычисление B-сплайнов
        if self.degree == 0:
            # Базовый случай: B-сплайны степени 0 (индикаторные функции)
            value = (x_unsqueezed >= grid_unsqueezed[:, :, :-1]) & (x_unsqueezed < grid_unsqueezed[:, :, 1:])
            value = value.float()
        else:
            # Рекурсивный случай
            # Создаем временный базис меньшей степени
            temp_basis = RecursiveBSplineBasis(
                degree=self.degree - 1,
                num_knots=self.num_knots,
                domain=self.domain,
                uniform=self.uniform
            )
            temp_basis.grid = self.grid
            
            # Вычисляем B-сплайны меньшей степени
            B_km1 = temp_basis._batch_bspline_values(x)
            
            # Вычисляем B-сплайны степени k по рекурсивной формуле
            # B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x) + 
            #               (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
            
            # Пустой тензор для результата
            value = torch.zeros(batch_size, input_dim, self.num_basis, device=device)
            
            for i in range(self.num_basis - 1):  # -1 потому что мы комбинируем (i) и (i+1)
                # Коэффициенты для первого слагаемого
                denom1 = self.grid[i + self.degree] - self.grid[i]
                
                # Коэффициенты для второго слагаемого
                denom2 = self.grid[i + self.degree + 1] - self.grid[i + 1]
                
                # Применяем формулу с проверкой деления на ноль
                mask1 = (denom1 != 0)
                mask2 = (denom2 != 0)
                
                # Первое слагаемое: (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x)
                if i < B_km1.shape[2]:  # Проверка индекса
                    term1 = torch.zeros_like(x_unsqueezed)
                    if mask1:
                        term1 = (x_unsqueezed - self.grid[i]) / denom1 * B_km1[:, :, i:i+1]
                
                # Второе слагаемое: (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
                if i + 1 < B_km1.shape[2]:  # Проверка индекса
                    term2 = torch.zeros_like(x_unsqueezed)
                    if mask2:
                        term2 = (self.grid[i + self.degree + 1] - x_unsqueezed) / denom2 * B_km1[:, :, i+1:i+2]
                
                # Суммируем слагаемые
                value[:, :, i] = (term1 + term2).squeeze(2)
        
        # Обработка значений на границе области
        # На правой границе должно быть B(b) = 1 для последнего базиса
        right_boundary = (x_unsqueezed == self.grid[-1])
        if right_boundary.any():
            value[:, :, -1] = value[:, :, -1] + right_boundary.float().squeeze(2)
        
        # Проверка на NaN и замена NaN на 0
        value = torch.nan_to_num(value)
        
        return value
    
    def _extend_grid(self, grid: torch.Tensor, k_extend: int) -> torch.Tensor:
        """
        Расширяет сетку, добавляя k точек слева и справа.
        
        Args:
            grid: Сетка формы (in_dim, num_points)
            k_extend: Количество точек для добавления с каждой стороны
            
        Returns:
            Расширенная сетка
        """
        # Вычисляем шаг сетки
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        
        # Расширяем сетку с обеих сторон
        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        
        return grid
    
    def update_grid_from_samples(self, x: torch.Tensor, coefficients: torch.Tensor, mode: str = 'sample'):
        """
        Обновляет сетку на основе распределения входных образцов.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            coefficients: Текущие коэффициенты
            mode: Режим обновления ('sample' или 'grid')
        """
        batch_size = x.shape[0]
        in_dim = x.shape[1]
        
        # Нормализуем входные данные
        x_norm = self.normalize_input(x)
        
        # Сортируем x для каждого измерения
        x_sorted, _ = torch.sort(x_norm, dim=0)
        
        # Вычисляем адаптивную сетку
        def get_grid(num_interval):
            # Выбираем индексы для равномерного разделения отсортированных данных
            indices = [int(batch_size / num_interval * i) for i in range(num_interval)] + [-1]
            
            # Получаем grid_adaptive из процентилей выборки
            grid_adaptive = x_sorted[indices, :].permute(1, 0)
            
            # Добавляем небольшое поле по краям
            margin = 0.01 * (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]])
            
            # Вычисляем grid_uniform (равномерную сетку)
            h = (grid_adaptive[:, [-1]] + margin - grid_adaptive[:, [0]] + margin) / num_interval
            grid_uniform = grid_adaptive[:, [0]] - margin + h * torch.arange(num_interval + 1, 
                                                                            device=x.device)[None, :]
            
            # Интерполируем между grid_uniform и grid_adaptive
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            
            return grid
        
        # Вычисляем новую сетку
        num_interval = self.num_knots - 1
        new_grid = get_grid(num_interval)
        
        # Если режим 'grid', используем более плотную сетку для оценки
        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_dense = sample_grid.permute(1, 0)
            y_dense = self.forward(x_dense, coefficients)
        else:
            # Иначе используем исходные образцы
            y_dense = self.forward(x_sorted, coefficients)
        
        # Расширяем сетку
        extended_grid = self._extend_grid(new_grid, k_extend=self.degree)
        
        # Обновляем сетку
        self.grid.data = extended_grid.squeeze(0)
        
        # Возвращаем данные для пересчета коэффициентов
        return x_sorted, y_dense
    
    def compute_coefficients(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет коэффициенты B-сплайнов методом наименьших квадратов.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            y: Целевой тензор формы (batch_size, in_dim, out_dim)
            
        Returns:
            Коэффициенты формы (in_dim, out_dim, num_basis)
        """
        batch_size = x.shape[0]
        in_dim = x.shape[1]
        out_dim = y.shape[2]
        
        # Вычисляем базисные функции для всех точек
        basis_values = self.basis_functions(x)  # (batch_size, in_dim, num_basis)
        
        # Преобразуем для метода наименьших квадратов
        # Для каждого (in_dim, out_dim) решаем задачу наименьших квадратов
        coefficients = torch.zeros(in_dim, out_dim, self.num_basis, device=x.device)
        
        for i in range(in_dim):
            for o in range(out_dim):
                # Матрица A: (batch_size, num_basis)
                A = basis_values[:, i, :]
                
                # Вектор b: (batch_size)
                b = y[:, i, o]
                
                # Решаем задачу наименьших квадратов: A * x = b
                try:
                    # Используем pytorch solver
                    solution = torch.linalg.lstsq(A, b.unsqueeze(1)).solution
                    coefficients[i, o, :] = solution.squeeze()
                except:
                    # Альтернативное решение через псевдообратную матрицу
                    ATA = A.t() @ A
                    ATb = A.t() @ b
                    # Регуляризация для численной стабильности
                    reg = 1e-8 * torch.eye(ATA.shape[0], device=ATA.device)
                    solution = torch.linalg.solve(ATA + reg, ATb)
                    coefficients[i, o, :] = solution
        
        return coefficients
    
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
        derivative_basis = RecursiveBSplineBasis(
            degree=self.degree - 1,
            num_knots=self.num_knots,
            domain=self.domain,
            uniform=self.uniform,
            grid_eps=self.grid_eps
        )
        
        # Передаем узлы из текущего базиса с учетом уменьшения степени
        derivative_basis.grid = self.grid
        
        # Рекурсивное вычисление для производных высших порядков
        if order > 1:
            return derivative_basis.derivative(x, derivative_coeffs, order - 1)
        else:
            return derivative_basis.forward(x, derivative_coeffs)
    
    def _derivative_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет коэффициенты для производной B-сплайнового разложения.
        
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
        
        # Вычисление коэффициентов производной по формуле:
        # d/dx(B_{i,k}(x)) = k * (B_{i,k-1}(x) / (t_{i+k} - t_i) - B_{i+1,k-1}(x) / (t_{i+k+1} - t_{i+1}))
        for i in range(num_basis - 1):
            # Разница узлов для нормализации
            u_diff1 = self.grid[i + self.degree + 1] - self.grid[i + 1]
            u_diff0 = self.grid[i + self.degree] - self.grid[i]
            
            # Предотвращение деления на ноль
            scale0 = 0.0 if u_diff0 == 0.0 else self.degree / u_diff0
            scale1 = 0.0 if u_diff1 == 0.0 else self.degree / u_diff1
            
            derivative_coeffs[:, :, i] = scale0 * coefficients[:, :, i] - scale1 * coefficients[:, :, i + 1]
        
        return derivative_coeffs
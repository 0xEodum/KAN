import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union, Callable
from .base import BasisFunction


class CubicSplineBasis(BasisFunction):
    """
    Кубические сплайны для KAN.
    
    Кубические сплайны - это кусочно-полиномиальные функции третьей степени,
    которые имеют непрерывную первую и вторую производные на всей области определения.
    Они обеспечивают гладкую интерполяцию между точками и широко используются в
    компьютерной графике, численном анализе и обработке сигналов.
    
    Эта реализация основана на естественных кубических сплайнах, которые имеют
    нулевую вторую производную на концах.
    """
    
    def __init__(self, num_knots: int, domain: Tuple[float, float] = (-1, 1), 
                knot_method: str = 'uniform', boundary_condition: str = 'natural'):
        """
        Инициализирует базис кубических сплайнов.
        
        Args:
            num_knots: Количество узлов (включая граничные)
            domain: Область определения сплайнов [a, b]
            knot_method: Метод размещения узлов ('uniform', 'chebyshev', 'adaptive')
            boundary_condition: Граничные условия ('natural', 'clamped', 'not-a-knot')
                               natural: вторая производная равна нулю на концах
                               clamped: первая производная задана на концах (0 по умолчанию)
                               not-a-knot: третья производная непрерывна на первом и последнем внутреннем узле
        """
        # Кубические сплайны имеют степень 3
        super().__init__(degree=3)
        
        self.num_knots = num_knots
        self._domain = domain
        self.knot_method = knot_method
        self.boundary_condition = boundary_condition
        
        # Проверка количества узлов
        if num_knots < 2:
            raise ValueError("Количество узлов должно быть не менее 2")
        
        # Генерация узлов в зависимости от выбранного метода
        knots = self._generate_knots()
        
        # Сохранение узлов как параметр модели
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32))
        
        # Вычисление базисных функций для каждого интервала
        self._precompute_basis_coefficients()
    
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
    
    def _generate_knots(self) -> np.ndarray:
        """
        Генерирует узловые точки в соответствии с выбранным методом.
        
        Returns:
            Массив узлов
        """
        a, b = self._domain
        
        if self.knot_method == 'uniform':
            # Равномерное распределение узлов
            knots = np.linspace(a, b, self.num_knots)
        elif self.knot_method == 'chebyshev':
            # Узлы Чебышева концентрируются у краев, помогая избежать эффекта Рунге
            t = np.linspace(0, np.pi, self.num_knots)
            knots = a + (b - a) * (1 - np.cos(t)) / 2
        elif self.knot_method == 'adaptive':
            # Неравномерное распределение с концентрацией в центре
            # Полезно для функций с большими изменениями в середине
            t = np.linspace(0, 1, self.num_knots)
            knots = a + (b - a) * (0.5 + 0.5 * np.sin(np.pi * (t - 0.5)))
        else:
            raise ValueError(f"Неизвестный метод расстановки узлов: {self.knot_method}")
        
        return knots
    
    def _precompute_basis_coefficients(self):
        """
        Предвычисляет коэффициенты для базисных функций кубического сплайна.
        
        Для каждого интервала [t_i, t_{i+1}] кубический сплайн представляется как:
        S_i(x) = a_i + b_i(x-t_i) + c_i(x-t_i)^2 + d_i(x-t_i)^3
        
        Эта функция вычисляет матрицы коэффициентов для каждого интервала.
        """
        knots = self.knots.detach().cpu().numpy()
        n = len(knots)
        
        # Для естественных кубических сплайнов с n узлами у нас есть n-1 интервалов
        # и 4*(n-1) коэффициентов (a, b, c, d для каждого интервала)
        
        # Создаем систему уравнений для вычисления коэффициентов
        # Для каждого внутреннего узла у нас есть 4 условия:
        # 1. Значение сплайна слева = значение в узле
        # 2. Значение сплайна справа = значение в узле
        # 3. Первая производная слева = первая производная справа
        # 4. Вторая производная слева = вторая производная справа
        
        # Дополнительные условия на границах зависят от выбранного граничного условия
        
        # Здесь мы вычисляем только коэффициенты "формы", которые не зависят
        # от конкретных значений функции, а только от положения узлов.
        # Конкретные значения будут применяться во время прямого прохода.
        
        # Для хранения предвычисленных коэффициентов базисных функций
        # для каждого узла и каждого интервала
        basis_coeffs = np.zeros((n, n-1, 4))
        
        # В данной реализации мы используем метод "разделенных разностей"
        # для вычисления коэффициентов, что позволяет напрямую получить
        # значения базисных функций в любой точке
        
        # Для каждого узла i мы строим базисную функцию, которая равна 1 в узле i
        # и 0 в других узлах, с соответствующими условиями на производные
        for i in range(n):
            # Значения в узлах: 1 в узле i, 0 в остальных
            y = np.zeros(n)
            y[i] = 1.0
            
            # Вычисляем коэффициенты для этой базисной функции
            coeffs = self._compute_spline_coefficients(knots, y)
            
            # Сохраняем коэффициенты
            for j in range(n-1):
                basis_coeffs[i, j] = coeffs[j]
        
        # Сохраняем предвычисленные коэффициенты
        self.register_buffer("basis_coeffs", torch.tensor(basis_coeffs, dtype=torch.float32))
    
    def _compute_spline_coefficients(self, knots: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Вычисляет коэффициенты кубического сплайна для заданных узлов и значений.
        
        Args:
            knots: Массив узловых точек
            values: Значения функции в узлах
            
        Returns:
            Массив коэффициентов формы (n-1) x 4, где n - количество узлов.
            Для каждого интервала [t_i, t_{i+1}]: [a_i, b_i, c_i, d_i]
        """
        n = len(knots)
        h = np.diff(knots)  # Шаги между узлами
        
        # Создаем систему для вычисления вторых производных в узлах
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Заполняем матрицу системы
        # Для внутренних узлов
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            
            b[i] = 6 * ((values[i+1] - values[i]) / h[i] - (values[i] - values[i-1]) / h[i-1])
        
        # Граничные условия
        if self.boundary_condition == 'natural':
            # Естественные граничные условия: f''(a) = f''(b) = 0
            A[0, 0] = 1.0
            A[n-1, n-1] = 1.0
        elif self.boundary_condition == 'clamped':
            # Зажатые граничные условия: f'(a) = f'(b) = 0
            A[0, 0] = 2 * h[0]
            A[0, 1] = h[0]
            b[0] = 6 * ((values[1] - values[0]) / h[0] - 0)  # f'(a) = 0
            
            A[n-1, n-2] = h[n-2]
            A[n-1, n-1] = 2 * h[n-2]
            b[n-1] = 6 * (0 - (values[n-1] - values[n-2]) / h[n-2])  # f'(b) = 0
        elif self.boundary_condition == 'not-a-knot':
            # Условие "not-a-knot": f'''(t_1-) = f'''(t_1+) и f'''(t_{n-2}-) = f'''(t_{n-2}+)
            if n > 3:
                A[0, 0] = h[1]
                A[0, 1] = -(h[0] + h[1])
                A[0, 2] = h[0]
                
                A[n-1, n-3] = h[n-2]
                A[n-1, n-2] = -(h[n-3] + h[n-2])
                A[n-1, n-1] = h[n-3]
            else:
                # Если узлов мало, используем естественные граничные условия
                A[0, 0] = 1.0
                A[n-1, n-1] = 1.0
        else:
            raise ValueError(f"Неизвестное граничное условие: {self.boundary_condition}")
        
        # Решаем систему для получения вторых производных
        try:
            m = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # В случае вырожденной матрицы используем псевдообратную
            m = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Вычисляем коэффициенты сплайна для каждого интервала
        coeffs = np.zeros((n-1, 4))
        for i in range(n-1):
            # a_i = f(t_i)
            coeffs[i, 0] = values[i]
            
            # b_i = f'(t_i) = (f(t_{i+1}) - f(t_i))/h_i - h_i*(2*m_i + m_{i+1})/6
            coeffs[i, 1] = (values[i+1] - values[i]) / h[i] - h[i] * (2 * m[i] + m[i+1]) / 6
            
            # c_i = m_i/2
            coeffs[i, 2] = m[i] / 2
            
            # d_i = (m_{i+1} - m_i)/(6*h_i)
            coeffs[i, 3] = (m[i+1] - m[i]) / (6 * h[i])
        
        return coeffs
    
    @property
    def name(self) -> str:
        """Возвращает имя базисной функции."""
        return f"CubicSpline-{self.num_knots}-{self.knot_method}-{self.boundary_condition}"
    
    @property
    def domain(self) -> Tuple[float, float]:
        """Возвращает область определения сплайнов."""
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
        Вычисляет разложение по кубическим сплайнам в точках x с заданными коэффициентами.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            coefficients: Тензор коэффициентов формы (input_dim, output_dim, num_knots)
            
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
        Вычисляет значения всех базисных кубических сплайнов в точках x.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Тензор формы (batch_size, input_dim, num_knots) с значениями
            базисных сплайнов для каждой точки.
        """
        batch_size, input_dim = x.shape
        
        # Создаем выходной тензор
        result = torch.zeros(batch_size, input_dim, self.num_knots, 
                           device=x.device, dtype=x.dtype)
        
        # Для каждой точки вычисляем значения всех базисных функций
        for batch_idx in range(batch_size):
            for dim_idx in range(input_dim):
                t = x[batch_idx, dim_idx]
                
                # Ограничиваем t в пределах домена
                t = torch.clamp(t, self.knots[0], self.knots[-1] - 1e-6)
                
                # Находим интервал, в котором находится t
                interval_idx = self._find_interval(t)
                
                # Вычисляем локальную координату в интервале
                local_t = t - self.knots[interval_idx]
                
                # Для каждой базисной функции (соответствующей узлу)
                for basis_idx in range(self.num_knots):
                    # Получаем коэффициенты для данной базисной функции в данном интервале
                    a, b, c, d = self.basis_coeffs[basis_idx, interval_idx]
                    
                    # Вычисляем значение базисной функции в точке t
                    # S(t) = a + b*(t-t_i) + c*(t-t_i)^2 + d*(t-t_i)^3
                    value = a + local_t * (b + local_t * (c + local_t * d))
                    
                    result[batch_idx, dim_idx, basis_idx] = value
        
        return result
    
    def _find_interval(self, t: torch.Tensor) -> int:
        """
        Находит индекс интервала, в котором находится t.
        
        Args:
            t: Значение параметра
            
        Returns:
            Индекс интервала i такой, что knots[i] <= t < knots[i+1]
        """
        # Бинарный поиск для нахождения интервала
        knots = self.knots
        
        # Обрабатываем крайние случаи
        if t >= knots[-2]:
            return len(knots) - 2
        if t <= knots[0]:
            return 0
        
        # Бинарный поиск
        left, right = 0, len(knots) - 2
        while left <= right:
            mid = (left + right) // 2
            if t < knots[mid]:
                right = mid - 1
            elif t >= knots[mid + 1]:
                left = mid + 1
            else:
                return mid
        
        # Если мы оказались здесь, что-то пошло не так
        # Возвращаем ближайший допустимый интервал
        return max(0, min(left, len(knots) - 2))
    
    def derivative(self, x: torch.Tensor, coefficients: torch.Tensor, 
                 order: int = 1) -> torch.Tensor:
        """
        Вычисляет производную сплайнового разложения.
        
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
        
        # Если порядок больше 3, результат нулевой (кубические сплайны имеют степень 3)
        if order > 3:
            return torch.zeros_like(x[:, :coefficients.shape[1]], device=x.device)
        
        # Нормализация входа
        x = self.normalize_input(x)
        
        batch_size, input_dim = x.shape
        output_dim = coefficients.shape[1]
        
        # Создаем выходной тензор
        result = torch.zeros(batch_size, output_dim, device=x.device, dtype=x.dtype)
        
        # Для каждой точки вычисляем значение производной
        for batch_idx in range(batch_size):
            for dim_idx in range(input_dim):
                t = x[batch_idx, dim_idx]
                
                # Ограничиваем t в пределах домена
                t = torch.clamp(t, self.knots[0], self.knots[-1] - 1e-6)
                
                # Находим интервал, в котором находится t
                interval_idx = self._find_interval(t)
                
                # Вычисляем локальную координату в интервале
                local_t = t - self.knots[interval_idx]
                
                # Для каждого выходного измерения
                for out_idx in range(output_dim):
                    deriv_value = 0.0
                    
                    # Суммируем вклады от всех базисных функций
                    for basis_idx in range(self.num_knots):
                        # Получаем коэффициенты для данной базисной функции в данном интервале
                        a, b, c, d = self.basis_coeffs[basis_idx, interval_idx]
                        
                        # Вычисляем значение производной соответствующего порядка
                        if order == 1:
                            # S'(t) = b + 2*c*(t-t_i) + 3*d*(t-t_i)^2
                            deriv_value += coefficients[dim_idx, out_idx, basis_idx] * (b + local_t * (2 * c + local_t * 3 * d))
                        elif order == 2:
                            # S''(t) = 2*c + 6*d*(t-t_i)
                            deriv_value += coefficients[dim_idx, out_idx, basis_idx] * (2 * c + local_t * 6 * d)
                        elif order == 3:
                            # S'''(t) = 6*d
                            deriv_value += coefficients[dim_idx, out_idx, basis_idx] * 6 * d
                    
                    result[batch_idx, out_idx] += deriv_value
        
        return result
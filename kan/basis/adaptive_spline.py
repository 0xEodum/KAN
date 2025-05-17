
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union, Callable
from .base import BasisFunction


class AdaptiveSplineBasis(BasisFunction):
    """
    Адаптивные сплайны с оптимизируемыми узлами для KAN.
    
    Этот класс реализует сплайны, у которых положение узлов не фиксировано,
    а оптимизируется в процессе обучения. Это позволяет сплайнам адаптироваться
    к характеру аппроксимируемой функции, размещая больше узлов в областях
    с большими изменениями и меньше узлов в областях с меньшими изменениями.
    
    В отличие от обычных сплайнов, которые обычно требуют большего числа узлов
    для аппроксимации сложных функций, адаптивные сплайны могут достичь
    сопоставимой точности с меньшим количеством параметров.
    """
    
    def __init__(self, degree: int, num_knots: int, 
                domain: Tuple[float, float] = (-1, 1),
                init_strategy: str = 'uniform',
                regularization: float = 0.0,
                min_distance: float = 1e-3):
        """
        Инициализирует адаптивный сплайновый базис.
        
        Args:
            degree: Степень сплайнов (1 - линейный, 2 - квадратичный, 3 - кубический и т.д.)
            num_knots: Количество внутренних узлов
            domain: Область определения сплайнов [a, b]
            init_strategy: Стратегия инициализации узлов ('uniform', 'chebyshev', 'random')
            regularization: Коэффициент регуляризации для предотвращения скопления узлов
            min_distance: Минимальное расстояние между соседними узлами
        """
        super().__init__(degree)
        
        self.num_knots = num_knots
        self._domain = domain
        self.init_strategy = init_strategy
        self.regularization = regularization
        self.min_distance = min_distance
        
        # Инициализация узлов
        knots = self._initialize_knots()
        
        # Параметры для хранения позиций узлов (обучаемые)
        self.knot_params = nn.Parameter(knots)
        
        # Буфер для кэширования отсортированных узлов
        self.register_buffer("sorted_knots", torch.zeros_like(knots))
        self.register_buffer("arange", torch.arange(0, degree + 1, dtype=torch.float32))
    
    def _initialize_knots(self) -> torch.Tensor:
        """
        Инициализирует позиции узлов в соответствии с выбранной стратегией.
        
        Returns:
            Тензор с позициями узлов
        """
        a, b = self._domain
        n = self.num_knots
        
        if self.init_strategy == 'uniform':
            # Равномерное распределение узлов
            knots = torch.linspace(a, b, n)
        elif self.init_strategy == 'chebyshev':
            # Узлы Чебышева - больше узлов у краев
            theta = torch.linspace(np.pi, 0, n)
            knots = torch.tensor(a + (b - a) * (1 - torch.cos(theta)) / 2)
        elif self.init_strategy == 'random':
            # Случайное распределение
            torch.manual_seed(42)  # Для воспроизводимости
            knots = a + (b - a) * torch.rand(n)
            knots, _ = torch.sort(knots)
        else:
            raise ValueError(f"Неизвестная стратегия инициализации: {self.init_strategy}")
        
        # Преобразуем в параметры с помощью инверсии сигмоиды
        # Это позволяет нам использовать сигмоиду для обеспечения ограничений на узлы
        knots_norm = (knots - a) / (b - a)  # Нормализация в [0, 1]
        params = torch.log(knots_norm / (1 - knots_norm))  # Инверсия сигмоиды
        
        return params
    
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
    
    def _get_sorted_knots(self) -> torch.Tensor:
        """
        Получает отсортированные узлы с ограничениями на домен и минимальное расстояние.
        
        Returns:
            Отсортированные узлы
        """
        a, b = self._domain
        
        # Применяем сигмоиду для получения значений в интервале (0, 1)
        knots_norm = torch.sigmoid(self.knot_params)
        
        # Масштабируем в домен
        knots = a + (b - a) * knots_norm
        
        # Сортируем узлы
        knots_sorted, _ = torch.sort(knots)
        
        # Применяем минимальное расстояние между узлами
        if self.min_distance > 0:
            # Применяем корректировки для соблюдения минимального расстояния
            for i in range(1, len(knots_sorted)):
                # Если узлы слишком близко, двигаем текущий узел
                if knots_sorted[i] - knots_sorted[i-1] < self.min_distance:
                    knots_sorted[i] = knots_sorted[i-1] + self.min_distance
            
            # Проверяем верхнюю границу после корректировок
            if knots_sorted[-1] > b:
                # Если последний узел вышел за границу, нормализуем все узлы
                scale = (b - a) / (knots_sorted[-1] - knots_sorted[0])
                knots_sorted = a + (knots_sorted - knots_sorted[0]) * scale
        
        # Обновляем кэшированные узлы
        self.sorted_knots.copy_(knots_sorted)
        
        return knots_sorted
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Вычисляет регуляризационную потерю для предотвращения скопления узлов.
        
        Returns:
            Значение регуляризационной потери
        """
        if self.regularization <= 0:
            return torch.tensor(0.0, device=self.knot_params.device)
        
        knots = self._get_sorted_knots()
        
        # Вычисляем расстояния между соседними узлами
        distances = knots[1:] - knots[:-1]
        
        # Регуляризация для равномерного распределения узлов
        # Используем обратные квадраты расстояний
        inverse_squared_distances = 1.0 / (distances.pow(2) + 1e-8)
        reg_loss = self.regularization * inverse_squared_distances.mean()
        
        return reg_loss
    
    @property
    def name(self) -> str:
        """Возвращает имя базисной функции."""
        return f"AdaptiveSpline-{self.degree}-{self.num_knots}"
    
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
        Вычисляет сплайновое разложение в точках x с заданными коэффициентами.
        
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
        Вычисляет значения всех базисных функций адаптивного сплайна в точках x.
        
        Args:
            x: Входной тензор формы (batch_size, input_dim)
            
        Returns:
            Тензор формы (batch_size, input_dim, num_basis) содержащий
            значения базисных функций для каждой точки.
        """
        batch_size, input_dim = x.shape
        
        # Получаем отсортированные узлы
        knots = self._get_sorted_knots()
        
        # Количество базисных функций
        num_basis = self.num_knots + self.degree
        
        # Создаем расширенный узловой вектор для B-сплайнов
        # (повторяем концевые узлы degree раз)
        extended_knots = torch.cat([
            torch.ones(self.degree) * self._domain[0],
            knots,
            torch.ones(self.degree) * self._domain[1]
        ])
        
        # Создаем выходной тензор
        result = torch.zeros(batch_size, input_dim, num_basis, 
                           device=x.device, dtype=x.dtype)
        
        # Функция для вычисления значения B-сплайна степени k в точке t
        def b_spline_value(t, i, k):
            if k == 0:
                # Базовый случай: B-сплайн степени 0
                if extended_knots[i] <= t < extended_knots[i+1] or \
                   (t == extended_knots[i+1] and extended_knots[i+1] == extended_knots[-1]):
                    return 1.0
                else:
                    return 0.0
            
            # Рекурсивное вычисление B-сплайна
            denom1 = extended_knots[i+k] - extended_knots[i]
            term1 = 0.0 if denom1 == 0.0 else \
                   (t - extended_knots[i]) / denom1 * b_spline_value(t, i, k-1)
            
            denom2 = extended_knots[i+k+1] - extended_knots[i+1]
            term2 = 0.0 if denom2 == 0.0 else \
                   (extended_knots[i+k+1] - t) / denom2 * b_spline_value(t, i+1, k-1)
            
            return term1 + term2
        
        # Для каждой точки вычисляем значения всех базисных функций
        for batch_idx in range(batch_size):
            for dim_idx in range(input_dim):
                # Текущее значение x
                t = x[batch_idx, dim_idx].item()
                
                # Ограничиваем t в пределах домена
                t = min(max(t, self._domain[0]), self._domain[1])
                
                # Для каждой базисной функции
                for basis_idx in range(num_basis):
                    # Вычисляем значение B-сплайна
                    value = b_spline_value(t, basis_idx, self.degree)
                    result[batch_idx, dim_idx, basis_idx] = value
        
        return result
    
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
        
        # Если порядок больше степени сплайна, результат нулевой
        if order > self.degree:
            return torch.zeros_like(x[:, :coefficients.shape[1]], device=x.device)
        
        # Для адаптивных сплайнов вычисление производных более сложное,
        # так как узлы меняются. Используем численное дифференцирование.
        epsilon = 1e-6
        
        # Нормализованные входы
        x_norm = self.normalize_input(x)
        
        # Вычисляем функцию в точке x и соседних точках
        if order == 1:
            # Первая производная (центральная разность)
            x_plus = x_norm + epsilon
            x_minus = x_norm - epsilon
            
            y_plus = self.forward(x_plus, coefficients)
            y_minus = self.forward(x_minus, coefficients)
            
            return (y_plus - y_minus) / (2 * epsilon)
        elif order == 2:
            # Вторая производная
            x_plus = x_norm + epsilon
            x_minus = x_norm - epsilon
            
            y = self.forward(x_norm, coefficients)
            y_plus = self.forward(x_plus, coefficients)
            y_minus = self.forward(x_minus, coefficients)
            
            return (y_plus - 2 * y + y_minus) / (epsilon ** 2)
        else:
            # Высшие производные через рекурсию
            first_deriv = self.derivative(x, coefficients, 1)
            return self.derivative(x, first_deriv, order - 1)
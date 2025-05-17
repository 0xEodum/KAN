import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from ..layers.base import KANLayer
from ..basis.chebyshev import ChebyshevBasis
from ..basis.jacobi import JacobiBasis
from ..basis.hermite import HermiteBasis
from ..basis.bspline import BSplineBasis
from ..basis.cubic_spline import CubicSplineBasis
from ..basis.adaptive_spline import AdaptiveSplineBasis


def require_sympy():
    """Проверяет, доступен ли sympy; вызывает ошибку, если нет."""
    if not SYMPY_AVAILABLE:
        raise ImportError(
            "Символьные операции требуют sympy. "
            "Пожалуйста, установите его с помощью 'pip install sympy'."
        )


# Функции для получения символьных выражений для различных базисов

def get_chebyshev_symbolic(degree: int) -> List[sp.Expr]:
    """
    Получает символьные выражения для полиномов Чебышева до заданной степени.
    
    Args:
        degree: Максимальная степень полиномов
        
    Returns:
        Список символьных выражений для T_0(x) до T_degree(x)
    """
    require_sympy()
    
    x = sp.Symbol('x')
    cheby_polys = [None] * (degree + 1)
    
    # Начальные значения
    cheby_polys[0] = 1
    if degree > 0:
        cheby_polys[1] = x
    
    # Рекуррентное соотношение
    for n in range(2, degree + 1):
        cheby_polys[n] = 2 * x * cheby_polys[n-1] - cheby_polys[n-2]
    
    return cheby_polys


def get_jacobi_symbolic(degree: int, alpha: float, beta: float) -> List[sp.Expr]:
    """
    Получает символьные выражения для полиномов Якоби до заданной степени.
    
    Args:
        degree: Максимальная степень полиномов
        alpha: Первый параметр для полиномов Якоби (α > -1)
        beta: Второй параметр для полиномов Якоби (β > -1)
        
    Returns:
        Список символьных выражений для P_0^(α,β)(x) до P_degree^(α,β)(x)
    """
    require_sympy()
    
    x = sp.Symbol('x')
    jacobi_polys = [None] * (degree + 1)
    
    # Начальные значения
    jacobi_polys[0] = 1
    if degree > 0:
        jacobi_polys[1] = ((alpha + beta + 2) * x + (alpha - beta)) / 2
    
    # Рекуррентное соотношение для n ≥ 2
    for n in range(2, degree + 1):
        n_float = float(n)
        ab_sum = alpha + beta
        ab_diff = alpha**2 - beta**2
        
        # Общий член в числителе: (2n + α + β - 1)
        common_term = 2 * n_float + ab_sum - 1
        
        # Коэффициент для P_{n-1}
        coef1 = common_term * ((2 * n_float + ab_sum) * (2 * n_float + ab_sum - 2) * x + ab_diff)
        
        # Коэффициент для P_{n-2}
        coef2 = -2 * (n_float + alpha - 1) * (n_float + beta - 1) * (2 * n_float + ab_sum)
        
        # Знаменатель
        denom = 2 * n_float * (n_float + ab_sum) * (2 * n_float + ab_sum - 2)
        
        # Вычисляем P_n
        jacobi_polys[n] = (coef1 * jacobi_polys[n-1] + coef2 * jacobi_polys[n-2]) / denom
    
    return jacobi_polys


def get_hermite_symbolic(degree: int, scaling: str = 'physicist') -> List[sp.Expr]:
    """
    Получает символьные выражения для полиномов Эрмита до заданной степени.
    
    Args:
        degree: Максимальная степень полиномов
        scaling: Тип полиномов Эрмита:
                 'physicist' (по умолчанию) - H_n(x) с рекурсией H_{n+1} = 2x·H_n - 2n·H_{n-1}
                 'probabilist' - He_n(x) с рекурсией He_{n+1} = x·He_n - n·He_{n-1}
        
    Returns:
        Список символьных выражений для H_0(x) до H_degree(x) или He_0(x) до He_degree(x)
    """
    require_sympy()
    
    x = sp.Symbol('x')
    hermite_polys = [None] * (degree + 1)
    
    # Начальные значения
    hermite_polys[0] = 1
    
    if degree > 0:
        if scaling == 'physicist':
            hermite_polys[1] = 2 * x
        else:  # scaling == 'probabilist'
            hermite_polys[1] = x
    
    # Рекуррентное соотношение
    for n in range(2, degree + 1):
        if scaling == 'physicist':
            # H_{n}(x) = 2x·H_{n-1}(x) - 2(n-1)·H_{n-2}(x)
            hermite_polys[n] = 2 * x * hermite_polys[n-1] - 2 * (n-1) * hermite_polys[n-2]
        else:  # scaling == 'probabilist'
            # He_{n}(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)
            hermite_polys[n] = x * hermite_polys[n-1] - (n-1) * hermite_polys[n-2]
    
    return hermite_polys


def get_bspline_symbolic(knots: List[float], degree: int) -> List[sp.Expr]:
    """
    Получает символьные выражения для B-сплайнов с заданными узлами и степенью.
    
    Обратите внимание, что точное символьное представление B-сплайнов сложно,
    поэтому мы используем приближенное представление на основе кусочно-полиномиальных функций.
    
    Args:
        knots: Список узловых точек
        degree: Степень B-сплайнов
        
    Returns:
        Список символьных выражений для B-сплайнов
    """
    require_sympy()
    
    x = sp.Symbol('x')
    
    # Расширяем узловой вектор для B-сплайнов
    extended_knots = ([knots[0]] * degree + knots + [knots[-1]] * degree)
    
    # Количество базисных функций
    num_basis = len(knots) + degree - 1
    
    # Для каждого B-сплайна создаем кусочно-полиномиальную функцию
    bspline_exprs = []
    
    for i in range(num_basis):
        # Для каждого B-сплайна вычисляем его кусочно-полиномиальное представление
        # Обратите внимание, что это приближение; точная формула рекурсивна
        # и сложна для символьного представления
        expr = sp.Piecewise((0, x < extended_knots[i]), (0, x >= extended_knots[i+degree+1]))
        
        # Добавляем приближенное выражение для B-сплайна между его узлами поддержки
        for j in range(i, i+degree):
            if extended_knots[j+1] > extended_knots[j]:
                # Линейное приближение в каждом интервале
                t = (x - extended_knots[j]) / (extended_knots[j+1] - extended_knots[j])
                piece = t if j == i else (1 - t) if j == i+degree-1 else 1
                expr = expr + sp.Piecewise(
                    (piece, sp.And(x >= extended_knots[j], x < extended_knots[j+1])),
                    (0, True)
                )
        
        bspline_exprs.append(expr)
    
    return bspline_exprs


def get_cubic_spline_symbolic(knots: List[float]) -> List[sp.Expr]:
    """
    Получает символьные выражения для кубических сплайнов с заданными узлами.
    
    Для кубических сплайнов мы используем метод естественных сплайнов,
    а символьное представление даем в виде кусочно-полиномиальных функций.
    
    Args:
        knots: Список узловых точек
        
    Returns:
        Список символьных выражений для кубических сплайнов
    """
    require_sympy()
    
    x = sp.Symbol('x')
    n = len(knots)
    
    # Для каждого узла создаем кубический сплайн, который равен 1 в этом узле
    # и 0 в других узлах
    spline_exprs = []
    
    for i in range(n):
        # Создаем выражение для сплайна, который равен 1 в i-м узле
        expr = sp.Piecewise((0, x < knots[0]), (0, x > knots[-1]))
        
        for j in range(n-1):
            if j == 0 and i == 0:
                # Для первого узла в первом интервале
                t = (x - knots[j]) / (knots[j+1] - knots[j])
                piece = (1 - t)**3
                expr = expr + sp.Piecewise(
                    (piece, sp.And(x >= knots[j], x < knots[j+1])),
                    (0, True)
                )
            elif j == n-2 and i == n-1:
                # Для последнего узла в последнем интервале
                t = (x - knots[j]) / (knots[j+1] - knots[j])
                piece = t**3
                expr = expr + sp.Piecewise(
                    (piece, sp.And(x >= knots[j], x <= knots[j+1])),
                    (0, True)
                )
            elif j == i-1:
                # Для узла в предыдущем интервале
                t = (x - knots[j]) / (knots[j+1] - knots[j])
                piece = t**3
                expr = expr + sp.Piecewise(
                    (piece, sp.And(x >= knots[j], x < knots[j+1])),
                    (0, True)
                )
            elif j == i:
                # Для узла в текущем интервале
                t = (x - knots[j]) / (knots[j+1] - knots[j])
                piece = 1 - 3*(1-t)**2 + 2*(1-t)**3
                expr = expr + sp.Piecewise(
                    (piece, sp.And(x >= knots[j], x < knots[j+1])),
                    (0, True)
                )
            elif j+1 == i:
                # Для узла в следующем интервале
                t = (x - knots[j]) / (knots[j+1] - knots[j])
                piece = 1 - 3*t**2 + 2*t**3
                expr = expr + sp.Piecewise(
                    (piece, sp.And(x >= knots[j], x < knots[j+1])),
                    (0, True)
                )
        
        spline_exprs.append(expr)
    
    return spline_exprs


def get_adaptive_spline_symbolic(knots: List[float], degree: int) -> List[sp.Expr]:
    """
    Получает символьные выражения для адаптивных сплайнов.
    
    Для адаптивных сплайнов мы используем то же самое представление, что и для B-сплайнов,
    поскольку их математическая форма идентична, но с оптимизируемыми узлами.
    
    Args:
        knots: Список узловых точек (оптимизированных)
        degree: Степень сплайнов
        
    Returns:
        Список символьных выражений для адаптивных сплайнов
    """
    # Используем то же представление, что и для B-сплайнов
    return get_bspline_symbolic(knots, degree)


def get_layer_symbolic_expr(layer: KANLayer, input_var_names: Optional[List[str]] = None) -> Dict[int, Union[sp.Expr, str]]:
    """
    Получает символьные выражения для каждого выхода слоя KAN.
    
    Args:
        layer: Слой KAN
        input_var_names: Имена для входных переменных (по умолчанию 'x_0', 'x_1', ...)
        
    Returns:
        Словарь, отображающий индексы выходов на символьные выражения
    """
    require_sympy()
    
    # Проверяем, что это тип слоя, который мы можем обработать
    if not hasattr(layer, 'basis_function'):
        return {0: "Не удается сгенерировать символьное выражение для этого типа слоя"}
    
    # Получаем параметры слоя
    basis = layer.basis_function
    coeffs = layer.get_coefficients().detach().cpu().numpy()
    input_dim = coeffs.shape[0]
    output_dim = coeffs.shape[1]
    
    # Устанавливаем имена входных переменных по умолчанию, если не предоставлены
    if input_var_names is None:
        input_var_names = [f'x_{i}' for i in range(input_dim)]
    
    # Создаем символьные переменные для входов
    input_vars = [sp.Symbol(name) for name in input_var_names]
    
    # Различная обработка в зависимости от типа базиса
    result = {}
    
    if isinstance(basis, ChebyshevBasis):
        # Получаем символьные полиномы Чебышева
        cheby_polys = get_chebyshev_symbolic(basis.degree)
        
        # Для каждого выходного измерения
        for o in range(output_dim):
            # Инициализируем выражение для этого выхода
            expr = 0
            
            # Для каждого входного измерения
            for i in range(input_dim):
                # Получаем преобразованный вход (применяем tanh для нормализации)
                x_transformed = sp.tanh(input_vars[i])
                
                # Суммируем вклад от каждой базисной функции
                input_expr = 0
                for d in range(basis.degree + 1):
                    # Заменяем преобразованный вход в полином Чебышева
                    poly_expr = cheby_polys[d].subs(sp.Symbol('x'), x_transformed)
                    # Умножаем на коэффициент и добавляем к сумме
                    input_expr += coeffs[i, o, d] * poly_expr
                
                # Добавляем вклад этого входа к выходу
                expr += input_expr
            
            # Сохраняем выражение для этого выхода
            result[o] = expr
    elif isinstance(basis, JacobiBasis):
        # Получаем символьные полиномы Якоби
        jacobi_polys = get_jacobi_symbolic(basis.degree, basis.alpha, basis.beta)
        
        # Для каждого выходного измерения
        for o in range(output_dim):
            # Инициализируем выражение для этого выхода
            expr = 0
            
            # Для каждого входного измерения
            for i in range(input_dim):
                # Получаем преобразованный вход (применяем tanh для нормализации)
                x_transformed = sp.tanh(input_vars[i])
                
                # Суммируем вклад от каждой базисной функции
                input_expr = 0
                for d in range(basis.degree + 1):
                    # Заменяем преобразованный вход в полином Якоби
                    poly_expr = jacobi_polys[d].subs(sp.Symbol('x'), x_transformed)
                    # Умножаем на коэффициент и добавляем к сумме
                    input_expr += coeffs[i, o, d] * poly_expr
                
                # Добавляем вклад этого входа к выходу
                expr += input_expr
            
            # Сохраняем выражение для этого выхода
            result[o] = expr
    elif isinstance(basis, HermiteBasis):
        # Получаем символьные полиномы Эрмита
        hermite_polys = get_hermite_symbolic(basis.degree, basis.scaling)
        
        # Для каждого выходного измерения
        for o in range(output_dim):
            # Инициализируем выражение для этого выхода
            expr = 0
            
            # Для каждого входного измерения
            for i in range(input_dim):
                # Получаем преобразованный вход, используя нормализацию базиса
                # Для Эрмита мы используем масштабированный tanh для обработки неограниченной области
                if hasattr(basis, '_normalize_domain') and basis._normalize_domain:
                    x_transformed = sp.tanh(input_vars[i]) * 3.0
                else:
                    x_transformed = input_vars[i]
                
                # Суммируем вклад от каждой базисной функции
                input_expr = 0
                for d in range(basis.degree + 1):
                    # Заменяем преобразованный вход в полином Эрмита
                    poly_expr = hermite_polys[d].subs(sp.Symbol('x'), x_transformed)
                    # Умножаем на коэффициент и добавляем к сумме
                    input_expr += coeffs[i, o, d] * poly_expr
                
                # Добавляем вклад этого входа к выходу
                expr += input_expr
            
            # Сохраняем выражение для этого выхода
            result[o] = expr
    elif isinstance(basis, BSplineBasis):
        # Получаем узлы B-сплайна
        knots = basis.knots.detach().cpu().numpy()
        
        # Получаем символьные выражения для B-сплайнов
        bspline_exprs = get_bspline_symbolic(knots, basis.degree)
        
        # Для каждого выходного измерения
        for o in range(output_dim):
            # Инициализируем выражение для этого выхода
            expr = 0
            
            # Для каждого входного измерения
            for i in range(input_dim):
                # Получаем преобразованный вход 
                a, b = basis.domain
                x_transformed = a + (b - a) * sp.sigmoid(input_vars[i])
                
                # Суммируем вклад от каждой базисной функции
                input_expr = 0
                for d in range(len(bspline_exprs)):
                    # Заменяем преобразованный вход в B-сплайн
                    if d < coeffs.shape[2]:
                        spline_expr = bspline_exprs[d].subs(sp.Symbol('x'), x_transformed)
                        # Умножаем на коэффициент и добавляем к сумме
                        input_expr += coeffs[i, o, d] * spline_expr
                
                # Добавляем вклад этого входа к выходу
                expr += input_expr
            
            # Сохраняем выражение для этого выхода
            result[o] = expr
    elif isinstance(basis, CubicSplineBasis):
        # Получаем узлы кубического сплайна
        knots = basis.knots.detach().cpu().numpy()
        
        # Получаем символьные выражения для кубических сплайнов
        spline_exprs = get_cubic_spline_symbolic(knots)
        
        # Для каждого выходного измерения
        for o in range(output_dim):
            # Инициализируем выражение для этого выхода
            expr = 0
            
            # Для каждого входного измерения
            for i in range(input_dim):
                # Получаем преобразованный вход
                a, b = basis.domain
                x_transformed = a + (b - a) * sp.sigmoid(input_vars[i])
                
                # Суммируем вклад от каждой базисной функции
                input_expr = 0
                for d in range(len(spline_exprs)):
                    # Заменяем преобразованный вход в сплайн
                    if d < coeffs.shape[2]:
                        spline_expr = spline_exprs[d].subs(sp.Symbol('x'), x_transformed)
                        # Умножаем на коэффициент и добавляем к сумме
                        input_expr += coeffs[i, o, d] * spline_expr
                
                # Добавляем вклад этого входа к выходу
                expr += input_expr
            
            # Сохраняем выражение для этого выхода
            result[o] = expr
    elif isinstance(basis, AdaptiveSplineBasis):
        # Получаем текущие оптимизированные узлы
        knots = basis._get_sorted_knots().detach().cpu().numpy()
        
        # Получаем символьные выражения для адаптивных сплайнов
        adaptive_exprs = get_adaptive_spline_symbolic(knots, basis.degree)
        
        # Для каждого выходного измерения
        for o in range(output_dim):
            # Инициализируем выражение для этого выхода
            expr = 0
            
            # Для каждого входного измерения
            for i in range(input_dim):
                # Получаем преобразованный вход
                a, b = basis.domain
                x_transformed = a + (b - a) * sp.sigmoid(input_vars[i])
                
                # Суммируем вклад от каждой базисной функции
                input_expr = 0
                for d in range(len(adaptive_exprs)):
                    # Заменяем преобразованный вход в сплайн
                    if d < coeffs.shape[2]:
                        spline_expr = adaptive_exprs[d].subs(sp.Symbol('x'), x_transformed)
                        # Умножаем на коэффициент и добавляем к сумме
                        input_expr += coeffs[i, o, d] * spline_expr
                
                # Добавляем вклад этого входа к выходу
                expr += input_expr
            
            # Сохраняем выражение для этого выхода
            result[o] = expr
    else:
        # Для неподдерживаемых типов базисов
        for o in range(output_dim):
            result[o] = f"Символьное выражение не реализовано для {type(basis).__name__}"
    
    return result


def get_network_symbolic_expr(model, input_var_names: Optional[List[str]] = None) -> Dict[int, Union[sp.Expr, str]]:
    """
    Получает символьные выражения для выходов сети KAN.
    
    Args:
        model: Модель KAN (последовательная или одиночный слой)
        input_var_names: Имена для входных переменных (по умолчанию 'x_0', 'x_1', ...)
        
    Returns:
        Словарь, отображающий индексы выходов на символьные выражения
    """
    require_sympy()
    
    # Проверяем, является ли модель последовательной
    if hasattr(model, 'children'):
        layers = [m for m in model.children() if isinstance(m, KANLayer)]
    else:
        layers = [model] if isinstance(model, KANLayer) else []
    
    if not layers:
        return {0: "В модели не найдены слои KAN"}
    
    # Если это одиночный слой, просто получаем его выражения
    if len(layers) == 1:
        return get_layer_symbolic_expr(layers[0], input_var_names)
    
    # Для многослойных сетей нам нужно составить выражения
    current_exprs = get_layer_symbolic_expr(layers[0], input_var_names)
    
    # Обрабатываем каждый последующий слой
    for layer_idx in range(1, len(layers)):
        layer = layers[layer_idx]
        
        # Получаем символьные выражения для этого слоя
        # Нам нужно создать временные имена переменных для выходов предыдущего слоя
        temp_var_names = [f'temp_{i}' for i in range(len(current_exprs))]
        layer_exprs = get_layer_symbolic_expr(layer, temp_var_names)
        
        # Создаем карту подстановок
        subs_map = {sp.Symbol(temp_var_names[i]): expr 
                   for i, expr in current_exprs.items()
                   if isinstance(expr, sp.Expr)}
        
        # Применяем подстановки для составления выражений
        new_exprs = {}
        for out_idx, expr in layer_exprs.items():
            if isinstance(expr, sp.Expr):
                new_exprs[out_idx] = expr.subs(subs_map)
            else:
                new_exprs[out_idx] = expr
        
        current_exprs = new_exprs
    
    return current_exprs


def simplify_expressions(exprs: Dict[int, sp.Expr], 
                        simplify_method: str = 'basic') -> Dict[int, sp.Expr]:
    """
    Упрощает символьные выражения.
    
    Args:
        exprs: Словарь, отображающий индексы на выражения
        simplify_method: Метод упрощения ('basic', 'full', или 'rational')
        
    Returns:
        Словарь, отображающий индексы на упрощенные выражения
    """
    require_sympy()
    
    result = {}
    for idx, expr in exprs.items():
        if isinstance(expr, sp.Expr):
            if simplify_method == 'basic':
                result[idx] = sp.expand(expr)
            elif simplify_method == 'full':
                result[idx] = sp.simplify(expr)
            elif simplify_method == 'rational':
                result[idx] = sp.cancel(expr)
            else:
                result[idx] = expr
        else:
            result[idx] = expr
    
    return result


def export_to_latex(exprs: Dict[int, Union[sp.Expr, str]]) -> Dict[int, str]:
    """
    Преобразует символьные выражения в строки LaTeX.
    
    Args:
        exprs: Словарь, отображающий индексы на выражения
        
    Returns:
        Словарь, отображающий индексы на строки LaTeX
    """
    require_sympy()
    
    result = {}
    for idx, expr in exprs.items():
        if isinstance(expr, sp.Expr):
            result[idx] = sp.latex(expr)
        else:
            result[idx] = str(expr)
    
    return result


def export_to_function(exprs: Dict[int, Union[sp.Expr, str]], 
                     function_name: str = 'kan_function',
                     library: str = 'numpy') -> str:
    """
    Преобразует символьные выражения в функцию Python.
    
    Args:
        exprs: Словарь, отображающий индексы на выражения
        function_name: Имя для сгенерированной функции
        library: Численная библиотека для использования ('numpy', 'torch', или 'jax')
        
    Returns:
        Строка, содержащая определение функции Python
    """
    require_sympy()
    
    # Получаем все переменные, используемые в выражениях
    all_vars = set()
    for expr in exprs.values():
        if isinstance(expr, sp.Expr):
            all_vars.update(expr.free_symbols)
    
    # Сортируем переменные по имени
    sorted_vars = sorted(all_vars, key=lambda s: s.name)
    var_names = [var.name for var in sorted_vars]
    
    # Преобразуем выражения в соответствующую библиотеку
    if library == 'numpy':
        import_line = "import numpy as np"
        module = "np"
    elif library == 'torch':
        import_line = "import torch"
        module = "torch"
    elif library == 'jax':
        import_line = "import jax.numpy as jnp"
        module = "jnp"
    else:
        raise ValueError(f"Неподдерживаемая библиотека: {library}")
    
    # Создаем определение функции
    function_lines = [
        import_line,
        "",
        f"def {function_name}({', '.join(var_names)}):",
        "    # Сгенерированная функция KAN",
    ]
    
    # Добавляем вычисление для каждого выхода
    for idx, expr in exprs.items():
        if isinstance(expr, sp.Expr):
            # Преобразуем в строку с соответствующей библиотекой
            expr_str = str(expr)
            # Заменяем математические функции эквивалентами библиотеки
            replacements = {
                'sin': f'{module}.sin',
                'cos': f'{module}.cos',
                'tan': f'{module}.tan',
                'exp': f'{module}.exp',
                'log': f'{module}.log',
                'sqrt': f'{module}.sqrt',
                'tanh': f'{module}.tanh',
                'acos': f'{module}.arccos',
                'asin': f'{module}.arcsin',
                'atan': f'{module}.arctan',
                'sigmoid': f'{module}.sigmoid'
            }
            for old, new in replacements.items():
                expr_str = expr_str.replace(old + '(', new + '(')
            
            function_lines.append(f"    y_{idx} = {expr_str}")
        else:
            function_lines.append(f"    # Выход {idx}: {expr}")
            function_lines.append(f"    y_{idx} = None")
    
    # Возвращаем все выходы
    output_vars = [f"y_{idx}" for idx in sorted(exprs.keys())]
    return_line = f"    return {', '.join(output_vars)}"
    if len(output_vars) > 1:
        return_line = f"    return ({', '.join(output_vars)})"
    function_lines.append(return_line)
    
    return "\n".join(function_lines)
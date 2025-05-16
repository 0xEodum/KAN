# KAN - Kolmogorov-Arnold Networks

Реализация нейронных сетей на основе теоремы Колмогорова-Арнольда о представлении непрерывных многомерных функций через композиции непрерывных функций меньшей размерности.

## Обзор

KAN (Kolmogorov-Arnold Networks) - это нейронные сети, которые используют явные математические формы для представления функций. Они основаны на теореме Колмогорова-Арнольда, которая утверждает, что любая непрерывная функция нескольких переменных может быть представлена в виде композиции непрерывных функций одной переменной и операции сложения.

Этот подход обеспечивает несколько преимуществ:
- **Интерпретируемость**: Можно получить явную математическую формулу для аппроксимируемой функции
- **Математическая строгость**: Основан на фундаментальной теореме из теории функций
- **Эффективность**: При правильном выборе базисных функций может требовать меньше параметров для аппроксимации

## Особенности

- Различные базисные функции для представления одномерных компонентов:
  - Полиномы Чебышева (реализовано)
  - Полиномы Якоби (планируется)
  - B-сплайны (планируется)
  - Вейвлеты (планируется)
- Модульная и расширяемая архитектура
- Символьные вычисления для извлечения математических формул
- Инструменты визуализации для анализа выученных функций

## Установка

```bash
# Базовая установка
pip install kolmogorov-arnold-networks

# С поддержкой символьных вычислений
pip install kolmogorov-arnold-networks[symbolic]

# Полная установка со всеми зависимостями
pip install kolmogorov-arnold-networks[all]
```

## Использование

### Базовый пример функциональной аппроксимации

```python
import torch
import numpy as np
from kan.layers.chebykan import ChebyKANLayer
from kan.utils.visualization import plot_function_approximation

# Создание данных для аппроксимации
x = np.linspace(-1, 1, 100)
y = np.sin(2 * np.pi * x)
x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Создание модели KAN
model = ChebyKANLayer(input_dim=1, output_dim=1, degree=8)

# Обучение модели
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')

# Визуализация результатов
plot_function_approximation(model, x, y)
```

### Получение символьного представления

```python
from kan.utils.symbolic import get_layer_symbolic_expr, export_to_latex

# Получение символьного выражения
expr_dict = get_layer_symbolic_expr(model)
latex_form = export_to_latex(expr_dict)

print("Символьное представление выученной функции:")
print(latex_form[0])  # Для первого выхода
```

## Примеры

Репозиторий содержит примеры использования KAN для различных задач:

- Аппроксимация функций (`examples/function_approximation.py`)
- Классификация MNIST (`examples/mnist_classification.py`)

## Структура проекта

```
kan/
├── basis/           # Реализации базисных функций
├── layers/          # Слои нейронной сети
├── models/          # Полные модели
├── utils/           # Вспомогательные инструменты
└── __init__.py
```

## Требования

- Python 3.7+
- PyTorch 1.8+
- NumPy 1.19+
- Matplotlib 3.3+
- SymPy 1.8+ (опционально, для символьных вычислений)

## Цитирование

Если вы используете эту библиотеку в своих исследованиях, пожалуйста, цитируйте:

```
@misc{kolmogorov-arnold-networks,
  author = {Author},
  title = {Kolmogorov-Arnold Networks},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/kolmogorov-arnold-networks}}
}
```

## Лицензия

MIT
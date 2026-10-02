# Бэклог по итогам ревью M1–M4

Дата ревью: 2026-10-01, ветка `cpp-foundation`, HEAD `7b58b61` (M4 закрыта, M5 — NEXT).
Источник истины по стадиям — [ROADMAP.md](ROADMAP.md); этот файл хранит находки ревью,
их обоснование и предлагаемый порядок работ. Перенос пунктов в ROADMAP (новая стадия,
изменение области M5) фиксируется отдельной записью в `docs/evidence`, как требует `AGENTS.md`.

## Быстрое восстановление контекста

- Код небольшой: ~1.6k строк в `src/` + `include/`. Вся GPU-логика — `src/resident.cu` (641 строка),
  устаревший M1-API — `src/cuda.cu`. Контракт — [CONTRACT.md](CONTRACT.md).
- Вердикт ревью: математика верна, это настоящий KAN (`y_o = b_o + Σ_i φ_{o,i}(x_i)`).
  Главные долги — устройство типов носителей, практическая обучаемость глубоких сетей
  и эффективность CUDA (FP64 + наивные свёртки вместо GEMM).
- Рекомендованный порядок: **R (рефакторинг носителей) → C1/C2 (точность + GEMM) → M5**.
  Пункты M (математика) можно вести параллельно, они меняют контракт.
- Замеры и скрипты: [evidence/review-2026-10-01](evidence/review-2026-10-01/)
  (`torch_reference.py`, `resident_bench.cpp`, `build_resident_bench.cmd`).

Обозначения приоритета: **P0** — блокирует следующую веху или даёт кратный выигрыш,
**P1** — важно, **P2** — желательно.

---

## R. Организация кода

Ключевая мысль: директории `carriers/…` — правильная идея, но проблема в типах, а не в папках.
Группировать стоит по свойству, важному для исполнения, а не по «orthogonal/harmonic»
(Фурье тоже ортогонален, B-сплайн — нет).

| ID | P | Задача | Где / почему |
|---|---|---|---|
| R1 | P0 | (ГОТОВО, [evidence](evidence/backlog/R1.md)) Заменить плоский `BasisConfig` на `std::variant<ChebyshevConfig, …, BSplineConfig>` (pybind11 поддерживает variant) | `include/kan/basis.hpp:10-23` — все поля всех семейств в одной структуре |
| R2 | P0 | (ГОТОВО, [evidence](evidence/backlog/R2.md)) Убрать флаг `rational_` и ветвления из `Layer`: разделить носители, **линейные по параметрам** (общий движок «раскладка + GEMM»), и **нелинейные** (rational, обучаемые RBF, будущий PQC) со своими VJP. Семейно-специфичные методы (`insert_knot`, `adapt_grid`, `set_rbf_parameters`) вынести с общего класса | `src/layer.cpp:88`, `:127`, sgd/validate; M5 иначе добавит третью ветку |
| R3 | P0 | (ГОТОВО, [evidence](evidence/backlog/R3.md)) Единственный источник формулы семейства: `__host__ __device__` header-функции, общие для CPU и CUDA | Сейчас дублируются: `src/basis.cpp` ↔ `src/resident.cu:48-172`, `src/rational.cpp` ↔ `src/resident.cu:258-327` (Якоби, log-space Гаусс/Mexican hat, Cox–de Boor, Горнер) |
| R4 | P1 | Раскладка каталогов: `carriers/{polynomial,trigonometric,local,rational,quantum}`, `backends/{cpu,cuda}`, `core/` (Layer, Network, ошибки, формы) | Предложение ревью; `local/` = B-сплайн, RBF, Mexican hat |
| R5 | P1 | Переименовать тесты и бенчмарки по фичам, зеркально `carriers/` | Сейчас `m3_layer_test`, `m4_resident_test` и т.п. — названы по вехам |
| R6 | P2 | Добавить `.clang-format` и привести код M3/M4 к нему | Плотные строки с несколькими операторами, напр. `src/resident.cu:520-523`, `src/rational.cpp` |
| R7 | P2 | Объявить deprecated или пустить через resident-исполнитель M1-API `src/cuda.cu` | Только Chebyshev, malloc на каждый вызов, `coefficient_gradient_kernel` за O(K²·B) (`src/cuda.cu:140`) |

---

## M. Математика и обучаемость

Проверено и верно: рекуррентная формула Якоби (против `scipy.special.eval_jacobi`,
макс. отн. ошибка 6e-15, включая |x|>1 и α+β=−1), производная Cox–de Boor, вставка
узла Boehm, производная и нормировка Mexican hat (Ricker), d/dlog-width RBF = 2q²e^{−q²},
VJP rational (dr/da = z^k/Q, dr/db = −r·z^k/Q, Q = 1+Σb·z^k).

Для линейных по коэффициентам семейств слой KAN ≡ раскладка Φ: ℝ^I→ℝ^{I·K} + плотный
линейный слой. Это определение, а не ошибка, и на этом строится C2.

| ID | P | Задача | Где / почему |
|---|---|---|---|
| M1 | P0 | (ГОТОВО, [evidence](evidence/backlog/M1.md)) Явная типизированная карта входа (affine / tanh / LayerNorm) как отдельный слой | Контракт запрещает неявную нормировку, но инструмента нет. Полиномы при \|x\|≫1 растут как (2\|x\|)^n → взрыв градиентов; локальные базисы вне [t_p, t_K] дают ноль и нулевой градиент («мёртвое» ребро) |
| M2 | P0 | (ГОТОВО, [evidence](evidence/backlog/M2.md)) Rational: безопасный знаменатель без полюсов — PAU (Molina et al. 2019) `Q = 1+\|Σ b_k z^k\|` или гладкий `Q = 1+(Σ…)²` — как опция политики сингулярностей | Сейчас guard бросает `domain_error` (`src/rational.cpp:41`, GPU статус 2): один шаг SGD в полюс обрывает обучение без восстановления |
| M3 | P1 | Residual-ветка `w_b·silu(x)` (как в оригинальном KAN), опционально | Единственный путь градиента вне сетки для B-сплайна/RBF |
| M4 | P1 | Инициализаторы (variance-preserving по семейству, шумовая как в pykan) | Контракт признаёт: нулевая инициализация не обучает многослойные сети |
| M5 | P1 | Нормированные функции Эрмита `H_n(x)e^{−x²/2}/√(2^n n! √π)` как вариант | Физические H_n растут ~2^n·n!, плохая обусловленность |
| M6 | P2 | Сетка на вход (как в pykan), а не одна на весь слой; refit сетки по квантилям; `adapt_grid` с прогоном сэмплов через предыдущие слои | Сейчас узлы/центры/масштабы общие для слоя; `adapt_grid` вставляет по одному узлу |
| M7 | P2 | Обучаемые scale/translation на ребро для Mexican hat (Wav-KAN) | Сейчас это KAN со словарём фиксированных вейвлетов |

---

## C. CUDA

Замер 2026-10-01, RTX 3090 (FP64:FP32 = 1:64), Chebyshev K=7, полный шаг
forward+backward+SGD. Одиночные прогоны под WDDM, вне замороженного протокола проекта —
диагностика, а не принятое evidence.

| Топология, batch | resident FP64 | PyTorch FP64 | PyTorch FP32 |
|---|---:|---:|---:|
| 64→64→32→16, 1024 | **2.46 мс** | 3.11 | 3.60 |
| 256→256→256→10, 8192 | 310 мс | 87 | 9.5 |
| 1024→1024→1024, 4096 | 7039 мс | 613 | 22.9 |

Интерпретация: на малой сети resident обгоняет eager PyTorch (там всё упирается в запуск
ядер). На крупных ~360 GFLOP/шаг дают ≈51 GFLOPS — около 9% пика FP64 (~0.56 TFLOPS,
PyTorch FP64 упирается в пик). Сверх этого FP32 даёт ещё ~27× на GeForce.

| ID | P | Задача | Где / почему |
|---|---|---|---|
| C1 | P0 | Политика точности: шаблон по `Scalar`; FP32 (опц. TF32/BF16) для обучения, FP64 — эталон паритета | Самый крупный множитель на GeForce |
| C2 | P0 | Свести свёртку к GEMM (cuBLAS/cuBLASLt, bias через epilogue): `Y = Φ·Cᵀ + b`, `dC = Uᵀ·Φ`, `dX = Σ_k (U·C)⊙Φ'` | `src/resident.cu:174-212` — наивные GEMM без тайлинга, некоалесцированный доступ к коэффициентам |
| C3 | P1 | Fused-ядро: вычислять базис в shared memory при загрузке тайла X; в backward пересчитывать Φ', а не хранить | Сейчас пишутся тензоры V и D размером B·I·K в глобальную память (для 1024-wide, B=4096 — ~235 МБ каждый на слой) |
| C4 | P1 | (ГОТОВО вместе с R3, [evidence](evidence/backlog/R3.md)) Шаблонизировать `basis_kernel` по семейству | `src/resident.cu:48`, скретч `double lower[18], next[18]` (`:67`) задаёт регистры/local memory для всех семейств |
| C5 | P1 | Разреженный путь B-сплайна: хранить `(span, p+1 значений)` | Ненулевых p+1, а хранится и умножается все K; после `adapt_grid` K растёт |
| C6 | P1 | Rational forward: sample — быстрый индекс | `rational_forward_kernel`, `index%outputs` (`:283-286`) → запись кэшей с шагом I·capacity |
| C7 | P1 | Rational forward: убрать вычисление всех VJP ради проверки конечности | `:317-322`, лишние FP64-деления (очень дороги на GA102) |
| C8 | P1 | Rational parameter VJP: один warp на ребро, собирающий все m+n+1 сумм сразу | `rational_parameter_kernel` (`:337`): warp на параметр, каждый заново читает z, P, Q и пересчитывает степени |
| C9 | P1 | Проверять статус раз за шаг / раз в N шагов; захват шага в CUDA Graph | `result()` (`:482`) — memcpy статуса + sync после forward, backward и sgd (3 раза за шаг) |
| C10 | P2 | `--fmad=false` только в сборке паритета, в сборке производительности включить FMA | Сейчас выключено везде |
| C11 | P0 (процесс) | (ГОТОВО) Включить счётчики Nsight Compute: NVIDIA Control Panel → Developer → Manage GPU Performance Counters → «Allow access to all users» (или ncu от администратора) | `ERR_NVGPUCTRPERM` во всех трёх вехах — оптимизация шла без occupancy/bandwidth |
| C12 | P2 | Nonlinear RBF reduction: коалесцированный доступ | `nonlinear_partial_kernel`: чтения `dx`/`dw`/`c` с шагом K |

---

## Воспроизведение замеров

```powershell
# PyTorch-эталон (нужен torch с CUDA)
python docs/evidence/review-2026-10-01/torch_reference.py

# resident-исполнитель: нужна Release CUDA-сборка (scripts/build.ps1 -Cuda ... -BuildDirectory build-m4-cuda)
cmd /c docs\evidence\review-2026-10-01\build_resident_bench.cmd build-m4-cuda
& "$env:TEMP\resident_bench.exe"
```

## Журнал статуса

| Дата | Изменение |
|---|---|
| 2026-10-01 | Бэклог создан по итогам ревью M1–M4; все пункты открыты |
| 2026-10-01 | C11 закрыт владельцем (счётчики Nsight Compute доступны) |
| 2026-10-01 | Бэклог стал стадией B в ROADMAP; M5 стартует только после закрытия всех пунктов ([решение](evidence/backlog-gate.md)). Первый проход: R3, R1 |
| 2026-10-01 | R3 закрыт: формулы базисов и rational — общие `KAN_HOST_DEVICE`-шаблоны в `src/detail/`; golden-дамп CPU+CUDA побитово идентичен, 19/19 CTest, GCC 11/11. Профилирование выявило и устранило три регрессии ([evidence](evidence/backlog/R3.md)) |
| 2026-10-01 | C4 закрыт вместе с R3: `basis_kernel` инстанцируется по семейству (66 → 36–62 регистров, скретч сплайна только у B-сплайна), время ядра −0.2…−12.6% ([evidence](evidence/backlog/R3.md)). Проход 1: R3, C4, R1 |
| 2026-10-01 | R1 закрыт: `BasisConfig` = `std::variant` типизированных конфигов по семействам, размер локальных семейств выводится, `TrainableRbfConfig` — отдельный тип, `BasisKind` ушёл в `detail`; golden побитово идентичен, 20/20 CTest, GCC 12/12 ([evidence](evidence/backlog/R1.md)). Проход 1 завершён: R3, C4, R1 |
| 2026-10-02 | R2 закрыт: `Layer` хранит `kan::Carrier = std::variant<BasisEdges, TrainableRbfEdges, RationalEdges>`, общий линейный движок «раскладка Φ + свёртка» (`src/carriers/linear_engine.hpp`), нелинейные VJP по носителям, семейные операции — свободные функции `kan/families.hpp`, resident — план на носитель; golden побитово идентичен (+ новый `--layers` дамп), 21/21 CTest, GCC 13/13, покрытие 98.5%, CPU быстрее в 1.1–2.2× ([evidence](evidence/backlog/R2.md)). Проход 2: R2, затем M1 и M2 |
| 2026-10-02 | M2 закрыт: `RationalConfig::denominator_policy` = `Guarded` (по умолчанию, прежнее поведение) / `Absolute` (PAU, 1+\|S\|) / `Smooth` (1+S²); общие host/device формулы, ядра инстанцируются по политике; шаг SGD в полюс больше не обрывает обучение при безопасных политиках; golden (оба режима) побитово идентичен, 23/23 CTest, GCC 14/14, покрытие 98.5% ([evidence](evidence/backlog/M2.md)). Остаток стоимости параметрического VJP → C8, ненулевая инициализация знаменателя → M4 |
| 2026-10-02 | M1 закрыт: отдельный вид слоя `kan::InputMap` с `AffineMap` (фиксированная, хелперы `affine_from_range`/`affine_from_moments`), `TanhMap`, `LayerNormMap` (обучаемые gain/bias); `Network` = последовательность `std::variant<Layer, InputMap>`; общие host/device формулы, CUDA-ядра карт ≈1% шага; демонстрация: входы в [100, 500] без карты — переполнение (Chebyshev) или нулевой градиент (B-сплайн), с картой — loss 8e-20 ([evidence](evidence/backlog/M1.md)) |
| 2026-10-02 | Проход 2 завершён: R2, M2, M1. M1 и M2 велись параллельно в worktree и слиты в `39e115c` (тесты M2 переведены на гетерогенный `Network::layers()`); на объединённом дереве MSVC+CUDA+Python 26/26, GCC 15/15, golden (оба режима) побитово идентичен `5dc6819` |

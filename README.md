# Региональные эмбеддинги для скоринга

Компактная PyTorch-модель использует `score_dubai` как основной скор и
добавляет к его логиту региональную поправку:

```text
final_logit = logit(score_dubai) + alpha * region_delta_logit
```

Таргет обучения — `flag_6m_30p`. Модель учитывает регион регистрации,
фактический регион и переданный `equal_flag`. Этот флаг не вычисляется из
кодов: он может быть равен нулю даже при совпадающих регионах.

## Архитектура

- 70 модельных кодов регионов `0..69`;
- общий embedding `70 × 4` для обеих ролей;
- два исходных эмбеддинга, абсолютная разность и произведение;
- отдельный входной `equal_flag`;
- MLP `17 → 24 → 8 → 1` с LayerNorm, SiLU и Dropout;
- обучаемый коэффициент `alpha`;
- выходная голова инициализируется нулями, поэтому до обучения результат
  в точности равен `score_dubai`.

В конфигурации по умолчанию у модели **986 обучаемых параметров**:

| Компонент | Параметры |
|---|---:|
| Shared embedding | 280 |
| Linear 17 → 24 | 432 |
| LayerNorm 24 | 48 |
| Linear 24 → 8 | 200 |
| LayerNorm 8 | 16 |
| Linear 8 → 1 | 9 |
| alpha | 1 |
| **Итого** | **986** |

## Запуск

Проект зафиксирован на Python `3.12.11` и PyTorch `2.13.0` с CUDA 13.0.
На Windows окружение воспроизводится через `uv`:

```powershell
winget install --id astral-sh.uv --exact --source winget
uv sync --extra dev
uv run region-model-info
uv run pytest
```

`uv` автоматически установит нужную версию Python из `.python-version` и
создаст `.venv`.

Альтернативная ручная установка:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
region-model-info
pytest
```

Минимальный шаг обучения:

```python
import torch
from regional_score import RegionalResidualScorer

model = RegionalResidualScorer()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

score_dubai = torch.tensor([0.20, 0.75])
region_ids = torch.tensor([[0, 0], [12, 7]], dtype=torch.long)
equal_flag = torch.tensor([1.0, 0.0])
targets = torch.tensor([[0.0], [1.0]])

optimizer.zero_grad()
logits = model(score_dubai, region_ids, equal_flag)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

Полный цикл обучения ожидает, что `train_loader` и `val_loader` возвращают
батчи `(base_scores, region_ids, equal_flags, targets)`.

Обучение на `synthetic_data/applications.csv`:

```bash
.venv/bin/train-region-model
```

По умолчанию train заканчивается `2025-12-31`, validation — `2026-04-30`,
последующие заявки образуют test. Checkpoint сохраняется в
`checkpoints/regional_residual.pt`.

Низкоуровневый вызов `fit`:

```python
from regional_score import EarlyStopping, ModelCheckpoint, WeightMonitor, fit

early_stopping = EarlyStopping(
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True,
)
weight_monitor = WeightMonitor(["embedding.weight", "alpha"])
checkpoint = ModelCheckpoint("checkpoints", every_n_epochs=5)

history = fit(
    model=model,
    train_batches=train_loader,
    val_batches=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=100,
    callbacks=[early_stopping, weight_monitor, checkpoint],
    device="cuda",
    max_grad_norm=1.0,
)

embedding_stats = weight_monitor.history["embedding.weight"]
print(early_stopping.best_epoch, early_stopping.best_value)
print(embedding_stats[-1])
```

`WeightMonitor` отдельно хранит для каждого параметра среднее, стандартное
отклонение, L2-норму и максимальное абсолютное значение после каждой эпохи.
`ModelCheckpoint` сохраняет переносимые CPU-снимки `state_dict` в файлы
`checkpoints/epoch_0005.pt`, `checkpoints/epoch_0010.pt` и так далее.

## Перенос single-target модели на другой таргет

Можно предобучить модель на одном таргете, а затем использовать её embedding
и скрытый backbone для другого таргета:

```python
import torch

from regional_score import (
    RegionalResidualScorer,
    initialize_single_target_from_pretrained,
    set_single_target_backbone_trainable,
)

model = RegionalResidualScorer()
report = initialize_single_target_from_pretrained(
    model,
    "checkpoints/epoch_0020.pt",
    freeze_backbone=True,
)

# Optimizer создаётся после заморозки: сначала обучаются новая голова и alpha.
optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=1e-3,
)

# После нескольких эпох можно разморозить backbone и пересоздать optimizer
# с меньшим learning rate.
set_single_target_backbone_trainable(model, True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

По умолчанию последний `Linear 8 → 1` и `alpha` не переносятся, поскольку они
специфичны для таргета. Для близких таргетов их можно включить параметрами
`transfer_head=True` и `transfer_alpha=True`. Размеры embedding и скрытых слоёв
исходной и новой моделей должны совпадать.

## Подбор гиперпараметров single-target модели

Отдельный скрипт перебирает размер embedding и скрытых слоёв, dropout, batch
size, learning rate, weight decay, балансировку классов и число эпох:

```powershell
uv run python sweep_single_target.py `
  --data path/to/applications.csv `
  --trials 30 `
  --epoch-options 5 10 20 40
```

Первые trial сравнивают текущую конфигурацию на каждом указанном числе эпох,
остальные выбираются случайно из полного пространства параметров. Early
stopping не используется: каждый trial проходит ровно заданное число эпох.

Результаты сохраняются в `checkpoints/hyperparameter_sweep`:

- `trial_*.pt` — отдельная итоговая модель каждого запуска;
- `results.csv` — результаты в порядке выполнения;
- `leaderboard.csv` — модели по убыванию validation ROC AUC;
- `best_model.pt` — копия лучшей модели;
- `search_plan.json` — полный воспроизводимый план эксперимента.

Для ранжирования используется `roc_auc_score` вероятности
`sigmoid(final_logit)` на validation. Test-выборка во время подбора не
используется.

## Несколько таргетов

Для совместного обучения горизонтов используется общий региональный backbone
и независимые головы с отдельными `alpha`:

```python
from regional_score import (
    MaskedMultiTargetBCELoss,
    MultiTargetRegionalResidualScorer,
)

target_names = ("30@3", "30@6", "30@9")
model = MultiTargetRegionalResidualScorer(target_names)
criterion = MaskedMultiTargetBCELoss(
    target_names,
    target_weights={"30@3": 1.0, "30@6": 1.0, "30@9": 1.5},
)
```

Для этой модели `base_scores` и `targets` имеют форму `[batch, 3]`, а
`region_ids` — `[batch, 2]`. В `forward` также передаётся `equal_flag`.
Таргеты, которые ещё не созрели на дату среза, должны быть отмечены `NaN`:
loss их игнорирует, а не считает отрицательным классом. Веса и поправки можно
мониторить отдельно по именам `target_heads.30@3.weight`, `alpha.30@3` и т. д.

Для полностью независимого переобучения нужно создать по одному
`RegionalResidualScorer` на каждый таргет и вызвать `fit` на соответствующей
выборке. Это полезный baseline для проверки, даёт ли shared backbone выигрыш.

Для обучения основной скор должен быть рассчитан out-of-fold либо зафиксирован
до обучения региональной ветки. Иначе региональная модель может получить
завышенную offline-оценку из-за утечки.

## ОКВЭД-эмбеддинги поверх готового скора

Отдельный пакет `okved_score` сравнивает две компактные residual-модели,
которые не переобучают исходный бустинг:

```text
flat:
primary_okved -> full-code embedding -> MLP -> delta_logit

hierarchical:
primary_okved -> [L1, L2, L3] -> level embeddings
              -> masked mean -> MLP -> delta_logit

final_logit = logit(boost_score) + alpha * delta_logit
```

Иерархия строится по точкам и использует накопительные пути:

```text
69.10   -> 69 -> 69.10
46.74.2 -> 46 -> 46.74 -> 46.74.2
```

Неизвестный полный код попадает в `UNK`. В иерархической модели известные
родительские уровни сохраняются, поэтому она может переносить сигнал на
редкие и новые leaf-коды. Отсутствующие глубокие уровни кодируются `PAD` и
не участвуют в pooling.

Готовый `boost_score` должен быть OOF на train и рассчитан frozen-моделью на
validation/test. В текущем эксперименте оцениваются только:

1. исходный `boost_score`;
2. flat residual NN;
3. hierarchical residual NN.

Синтетические данные:

```bash
uv run python synthetic_data/generate_okved.py \
  --rows 20000 \
  --regime mixed \
  --oov-rate 0.04 \
  --oov-cutoff 2026-01-01
```

После генерации единый benchmark запускается командой:

```bash
uv run python benchmark_okved.py \
  --data synthetic_data/okved_applications.csv \
  --output-dir checkpoints/okved_benchmark
```

Validation используется для early stopping обеих сетей. Test применяется
один раз для итогового сравнения ROC AUC/Gini, PR AUC, log loss и Brier score;
результаты также разбиваются на frequent, rare и OOV сегменты.
Флаг `--match-parameter-budget` уменьшает ширину level embeddings так, чтобы
число параметров hierarchical-модели было близко к flat-модели; фактические
числа параметров сохраняются в `results.json`.

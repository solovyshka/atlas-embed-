# Региональные эмбеддинги для скоринга

Компактная PyTorch-модель добавляет региональную поправку к логиту основной
скоринговой модели:

```text
final_logit = base_logit + alpha * region_delta_logit
```

Модель учитывает регион прописки, рождения и подачи заявления. Одна общая
таблица эмбеддингов позволяет регионам делить статистическую силу между тремя
ролями.

## Архитектура

- 70 регионов и два служебных ID: `MISSING=70`, `UNKNOWN=71`;
- общий embedding `72 × 4`;
- три исходных эмбеддинга;
- абсолютные разности и произведения для каждой из трёх пар;
- три индикатора совпадения и число уникальных регионов;
- MLP `40 → 24 → 8 → 1` с LayerNorm, SiLU и Dropout;
- обучаемый коэффициент `alpha`, начальное значение `0.1`.

В конфигурации по умолчанию у модели **1 546 обучаемых параметров**:

| Компонент | Параметры |
|---|---:|
| Shared embedding | 288 |
| Linear 40 → 24 | 984 |
| LayerNorm 24 | 48 |
| Linear 24 → 8 | 200 |
| LayerNorm 8 | 16 |
| Linear 8 → 1 | 9 |
| alpha | 1 |
| **Итого** | **1 546** |

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

base_logits = torch.tensor([0.2, -0.4])
region_ids = torch.tensor([[0, 0, 0], [12, 7, 12]], dtype=torch.long)
targets = torch.tensor([[0.0], [1.0]])

optimizer.zero_grad()
logits = model(base_logits, region_ids)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

Полный цикл обучения ожидает, что `train_loader` и `val_loader` возвращают
батчи `(base_logits, region_ids, targets)`:

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

Для этой модели `base_logits` и `targets` имеют форму `[batch, 3]`. Таргеты,
которые ещё не созрели на дату среза, должны быть отмечены `NaN`: loss их
игнорирует, а не считает отрицательным классом. Веса и поправки можно
мониторить отдельно по именам `target_heads.30@3.weight`, `alpha.30@3` и т. д.

Для полностью независимого переобучения нужно создать по одному
`RegionalResidualScorer` на каждый таргет и вызвать `fit` на соответствующей
выборке. Это полезный baseline для проверки, даёт ли shared backbone выигрыш.

Для обучения основной скор должен быть рассчитан out-of-fold либо зафиксирован
до обучения региональной ветки. Иначе региональная модель может получить
завышенную offline-оценку из-за утечки.

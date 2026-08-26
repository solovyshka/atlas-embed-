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

Для обучения основной скор должен быть рассчитан out-of-fold либо зафиксирован
до обучения региональной ветки. Иначе региональная модель может получить
завышенную offline-оценку из-за утечки.

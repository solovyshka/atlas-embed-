from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import RegionalResidualScorer


class MultiTargetRegionalResidualScorer(RegionalResidualScorer):
    """Модель для нескольких таргетов с общим региональным backbone.

    Порядок ``target_names`` задаёт порядок колонок во всех тензорах. Например,
    для ``("30@3", "30@6", "30@9")``:

    - ``base_logits``: float-тензор ``[batch, 3]`` с отдельным базовым логитом
      для каждого таргета. Нужно передавать логиты, а не вероятности;
    - ``region_ids``: long-тензор ``[batch, 3]`` в порядке
      registration, birth, application;
    - результат: float-тензор логитов ``[batch, 3]`` в порядке target_names.

    Общие embedding и скрытые слои учатся на всех таргетах, а последний
    Linear-слой и коэффициент alpha у каждого таргета свои.
    """

    def __init__(
        self,
        target_names: Sequence[str],
        num_regions: int = 72,
        embedding_dim: int = 4,
        hidden_dims: tuple[int, int] = (24, 8),
        dropout: float = 0.15,
        initial_alpha: float = 0.1,
    ) -> None:
        # names определяет не только имена голов, но и порядок колонок входа
        # и выхода. После обучения этот порядок должен сохраняться.
        names = tuple(target_names)
        if not names:
            raise ValueError("target_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("target_names must be unique")
        if any(not name or "." in name for name in names):
            raise ValueError("target names must be non-empty and cannot contain '.'")

        super().__init__(
            num_regions=num_regions,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            initial_alpha=initial_alpha,
        )
        self.target_names = names

        # Keep the shared feature extractor and replace the single output layer
        # with one independently monitorable head and gate per target.
        self.region_tower = nn.Sequential(*list(self.region_tower.children())[:-1])
        del self.alpha
        self.target_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dims[1], 1) for name in names}
        )
        self.alpha = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(initial_alpha)))
                for name in names
            }
        )

    @property
    def num_targets(self) -> int:
        return len(self.target_names)

    def regional_delta_logits(self, region_ids: Tensor) -> Tensor:
        """Вернуть региональные поправки до умножения на target-specific alpha."""
        shared_features = self.region_tower(self.region_features(region_ids))
        return torch.cat(
            [self.target_heads[name](shared_features) for name in self.target_names],
            dim=1,
        )

    def forward(self, base_logits: Tensor, region_ids: Tensor) -> Tensor:
        """Добавить отдельную региональную поправку к каждому базовому логиту."""
        if base_logits.ndim != 2 or base_logits.shape[1] != self.num_targets:
            raise ValueError(
                f"base_logits must have shape [batch, {self.num_targets}]"
            )
        if base_logits.shape[0] != region_ids.shape[0]:
            raise ValueError("base_logits and region_ids batch sizes must match")

        delta_logits = self.regional_delta_logits(region_ids)
        # ParameterDict позволяет независимо отслеживать alpha.30@3,
        # alpha.30@6 и остальные коэффициенты через WeightMonitor.
        alpha = torch.stack([self.alpha[name] for name in self.target_names])
        return base_logits + alpha.unsqueeze(0) * delta_logits


class MaskedMultiTargetBCELoss(nn.Module):
    """BCE для нескольких таргетов с весами и незрелыми наблюдениями.

    ``logits`` и ``targets`` должны иметь форму ``[batch, num_targets]`` и тот
    же порядок колонок, что ``target_names`` в модели. Значения targets:

    - ``0.0`` — событие не произошло;
    - ``1.0`` — событие произошло;
    - ``NaN`` — горизонт ещё не созрел и это наблюдение нельзя использовать.

    Loss сначала усредняется отдельно по доступным наблюдениям каждого таргета,
    затем target losses объединяются с помощью ``target_weights``. Поэтому
    длинный горизонт с меньшим числом зрелых договоров не теряется среди строк.
    """

    def __init__(
        self,
        target_names: Sequence[str],
        target_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        names = tuple(target_names)
        if not names:
            raise ValueError("target_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("target_names must be unique")

        weights = target_weights or {}
        unknown_names = set(weights) - set(names)
        if unknown_names:
            unknown = ", ".join(sorted(unknown_names))
            raise ValueError(f"weights provided for unknown targets: {unknown}")

        weight_values = torch.tensor(
            [float(weights.get(name, 1.0)) for name in names],
            dtype=torch.float32,
        )
        if torch.any(weight_values < 0) or not torch.any(weight_values > 0):
            raise ValueError("target weights must be non-negative with one positive")

        self.target_names = names
        self.register_buffer("target_weights", weight_values)

    def per_target(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Вернуть loss каждого таргета; для полностью незрелого таргета — NaN."""
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have the same shape")
        if logits.ndim != 2 or logits.shape[1] != len(self.target_names):
            raise ValueError(
                f"logits must have shape [batch, {len(self.target_names)}]"
            )

        # NaN — именно маска недоступного таргета, а не отрицательный класс.
        observed = ~torch.isnan(targets)
        safe_targets = torch.nan_to_num(targets)
        element_losses = F.binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            reduction="none",
        )
        observed_count = observed.sum(dim=0)
        summed_losses = (element_losses * observed).sum(dim=0)
        losses = summed_losses / observed_count.clamp_min(1)
        return losses.masked_fill(observed_count == 0, torch.nan)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        losses = self.per_target(logits, targets)
        active = ~torch.isnan(losses)
        active_weights = self.target_weights.to(logits) * active
        weight_sum = active_weights.sum()
        if weight_sum.item() == 0:
            raise ValueError("batch contains no weighted target observations")
        return (torch.nan_to_num(losses) * active_weights).sum() / weight_sum

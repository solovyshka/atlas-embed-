from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .model import RegionalResidualScorer

PretrainedSource = (
    RegionalResidualScorer
    | Mapping[str, Tensor]
    | str
    | Path
)


@dataclass(frozen=True)
class TransferReport:
    """Результат переноса параметров между single-target моделями."""

    loaded_keys: tuple[str, ...]
    frozen_keys: tuple[str, ...]


def _is_backbone_key(name: str) -> bool:
    # Последний Linear (region_tower.8) является target-specific головой.
    return name.startswith("embedding.") or (
        name.startswith("region_tower.") and not name.startswith("region_tower.8.")
    )


def _load_source_state(
    source: PretrainedSource,
    map_location: torch.device | str,
) -> Mapping[str, Tensor]:
    if isinstance(source, RegionalResidualScorer):
        return source.state_dict()
    if isinstance(source, (str, Path)):
        payload = torch.load(source, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint must contain a state dict")

        # ModelCheckpoint сохраняет веса внутри model_state_dict. Также
        # поддерживается обычный torch.save(model.state_dict(), path).
        state = payload.get("model_state_dict", payload)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint model_state_dict must be a mapping")
        return state
    return source


def initialize_single_target_from_pretrained(
    model: RegionalResidualScorer,
    source: PretrainedSource,
    *,
    transfer_head: bool = False,
    transfer_alpha: bool = False,
    freeze_backbone: bool = False,
    map_location: torch.device | str = "cpu",
) -> TransferReport:
    """Инициализировать single-target модель весами другого таргета.

    По умолчанию переносятся:

    - общая таблица региональных embedding;
    - Linear и LayerNorm скрытого backbone до последнего слоя.

    Выходной ``region_tower.8`` и ``alpha`` остаются новой случайной
    инициализацией, потому что они наиболее специфичны для таргета.

    Args:
        model: Новая модель для следующего таргета. Архитектура должна
            совпадать с предобученной моделью.
        source: Предобученная модель, её ``state_dict`` или путь к checkpoint.
        transfer_head: Перенести последний Linear 8 → 1.
        transfer_alpha: Перенести residual-коэффициент alpha.
        freeze_backbone: Запретить обновление перенесённого backbone. Optimizer
            следует создавать после вызова этой функции.
        map_location: Устройство для чтения checkpoint.
    """
    source_state = _load_source_state(source, map_location)
    target_state = model.state_dict()

    keys_to_load = [
        name
        for name in target_state
        if _is_backbone_key(name)
        or (transfer_head and name.startswith("region_tower.8."))
        or (transfer_alpha and name == "alpha")
    ]

    missing_keys = [name for name in keys_to_load if name not in source_state]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"pretrained model is missing parameters: {missing}")

    incompatible_shapes = [
        (
            name,
            tuple(source_state[name].shape),
            tuple(target_state[name].shape),
        )
        for name in keys_to_load
        if source_state[name].shape != target_state[name].shape
    ]
    if incompatible_shapes:
        details = "; ".join(
            f"{name}: source {source_shape}, target {target_shape}"
            for name, source_shape, target_shape in incompatible_shapes
        )
        raise ValueError(f"incompatible model architectures: {details}")

    with torch.no_grad():
        for name in keys_to_load:
            target_state[name].copy_(
                source_state[name].to(
                    device=target_state[name].device,
                    dtype=target_state[name].dtype,
                )
            )

    frozen_keys: list[str] = []
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if _is_backbone_key(name):
                parameter.requires_grad_(False)
                frozen_keys.append(name)

    return TransferReport(
        loaded_keys=tuple(keys_to_load),
        frozen_keys=tuple(frozen_keys),
    )


def set_single_target_backbone_trainable(
    model: RegionalResidualScorer,
    trainable: bool,
) -> tuple[str, ...]:
    """Заморозить или разморозить embedding и скрытый backbone."""
    changed_keys: list[str] = []
    for name, parameter in model.named_parameters():
        if _is_backbone_key(name):
            parameter.requires_grad_(trainable)
            changed_keys.append(name)
    return tuple(changed_keys)

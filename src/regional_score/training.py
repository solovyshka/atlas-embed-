from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

Batch = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True)
class ParameterStats:
    epoch: int
    mean: float
    std: float
    l2_norm: float
    max_abs: float


class Callback:
    """Base class for callbacks used by :func:`fit`."""

    stop_training = False

    def on_train_begin(self, model: nn.Module) -> None:
        pass

    def on_epoch_end(self, model: nn.Module, metrics: EpochMetrics) -> None:
        pass

    def on_train_end(
        self, model: nn.Module, history: Sequence[EpochMetrics]
    ) -> None:
        pass


class EarlyStopping(Callback):
    """Stop when validation loss has not improved for ``patience`` epochs."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_epoch: int | None = None
        self.best_value = math.inf
        self.stopped_epoch: int | None = None
        self.stop_training = False
        self._epochs_without_improvement = 0
        self._best_state: dict[str, Tensor] | None = None

    def on_train_begin(self, model: nn.Module) -> None:
        self.best_epoch = None
        self.best_value = math.inf
        self.stopped_epoch = None
        self.stop_training = False
        self._epochs_without_improvement = 0
        self._best_state = None

    def on_epoch_end(self, model: nn.Module, metrics: EpochMetrics) -> None:
        if metrics.val_loss < self.best_value - self.min_delta:
            self.best_value = metrics.val_loss
            self.best_epoch = metrics.epoch
            self._epochs_without_improvement = 0
            if self.restore_best_weights:
                self._best_state = copy.deepcopy(model.state_dict())
            return

        self._epochs_without_improvement += 1
        if self._epochs_without_improvement >= self.patience:
            self.stopped_epoch = metrics.epoch
            self.stop_training = True

    def on_train_end(
        self, model: nn.Module, history: Sequence[EpochMetrics]
    ) -> None:
        if self.restore_best_weights and self._best_state is not None:
            model.load_state_dict(self._best_state)


class WeightMonitor(Callback):
    """Collect per-parameter weight statistics after every epoch."""

    def __init__(self, parameter_names: Iterable[str] | None = None) -> None:
        self.parameter_names = (
            None if parameter_names is None else frozenset(parameter_names)
        )
        self.history: dict[str, list[ParameterStats]] = {}

    def on_train_begin(self, model: nn.Module) -> None:
        available_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if self.parameter_names is not None:
            unknown_names = self.parameter_names - available_names
            if unknown_names:
                names = ", ".join(sorted(unknown_names))
                raise ValueError(f"unknown parameter names: {names}")
            available_names &= self.parameter_names

        self.history = {name: [] for name in sorted(available_names)}

    def on_epoch_end(self, model: nn.Module, metrics: EpochMetrics) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in self.history:
                    continue
                values = parameter.detach().float()
                self.history[name].append(
                    ParameterStats(
                        epoch=metrics.epoch,
                        mean=values.mean().item(),
                        std=values.std(unbiased=False).item(),
                        l2_norm=values.norm().item(),
                        max_abs=values.abs().max().item(),
                    )
                )


class ModelCheckpoint(Callback):
    """Save a portable model checkpoint at a fixed epoch interval."""

    def __init__(
        self,
        directory: str | Path = "checkpoints",
        every_n_epochs: int = 5,
    ) -> None:
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")

        self.directory = Path(directory)
        self.every_n_epochs = every_n_epochs
        self.saved_paths: list[Path] = []

    def on_train_begin(self, model: nn.Module) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.saved_paths = []

    def on_epoch_end(self, model: nn.Module, metrics: EpochMetrics) -> None:
        if metrics.epoch % self.every_n_epochs != 0:
            return

        path = self.directory / f"epoch_{metrics.epoch:04d}.pt"
        model_state = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        }
        torch.save(
            {
                "epoch": metrics.epoch,
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "model_state_dict": model_state,
            },
            path,
        )
        self.saved_paths.append(path)


def _run_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_weight = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for base_logits, region_ids, targets in batches:
            base_logits = base_logits.to(device)
            region_ids = region_ids.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(base_logits, region_ids)
            targets = targets.reshape_as(logits).to(dtype=logits.dtype)
            loss = criterion(logits, targets)
            if loss.ndim != 0:
                raise ValueError("criterion must return a scalar loss")

            if training:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            aggregation_weight = getattr(criterion, "aggregation_weight", None)
            if callable(aggregation_weight):
                batch_weight_value = aggregation_weight(targets)
                batch_weight = float(
                    batch_weight_value.detach().item()
                    if isinstance(batch_weight_value, Tensor)
                    else batch_weight_value
                )
            else:
                batch_weight = float(targets.shape[0])

            total_loss += loss.detach().item() * batch_weight
            total_weight += batch_weight

    if total_weight == 0:
        raise ValueError("data loader must contain at least one example")
    return total_loss / total_weight


def fit(
    model: nn.Module,
    train_batches: Iterable[Batch],
    val_batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    callbacks: Sequence[Callback] = (),
    device: torch.device | str | None = None,
    max_grad_norm: float | None = None,
) -> list[EpochMetrics]:
    """Обучить single-target или multi-target региональную модель.

    Каждый элемент ``train_batches`` и ``val_batches`` — тройка:

    - ``base_logits``: ``[batch]`` для single-target или
      ``[batch, num_targets]`` для multi-target;
    - ``region_ids``: long-тензор ``[batch, 3]`` в порядке
      registration, birth, application;
    - ``targets``: бинарный float-тензор той же формы, что выход модели.
      Для ``MaskedMultiTargetBCELoss`` незрелые таргеты передаются как ``NaN``.

    Функция возвращает общий train/validation loss по каждой эпохе. Детальную
    статистику весов и остановку обучения обеспечивают callbacks.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    target_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    model.to(target_device)
    callback_list = list(callbacks)
    history: list[EpochMetrics] = []

    for callback in callback_list:
        callback.on_train_begin(model)

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model,
            train_batches,
            criterion,
            target_device,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
        )
        val_loss = _run_epoch(model, val_batches, criterion, target_device)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        history.append(metrics)

        for callback in callback_list:
            callback.on_epoch_end(model, metrics)
        if any(callback.stop_training for callback in callback_list):
            break

    for callback in callback_list:
        callback.on_train_end(model, history)

    return history

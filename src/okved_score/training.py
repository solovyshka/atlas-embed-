from __future__ import annotations

from collections.abc import Iterable, Sequence
import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

Batch = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float


class Callback:
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
    """Stop training when validation loss stops improving."""

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
        self._epochs_without_improvement = 0
        self._best_state: dict[str, Tensor] | None = None

    def on_train_begin(self, model: nn.Module) -> None:
        self.stop_training = False
        self.best_epoch = None
        self.best_value = math.inf
        self.stopped_epoch = None
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


def _run_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in batches:
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise ValueError(
                    "each batch must contain boost_score, okved_ids, and target"
                )
            boost_score, okved_ids, target = (
                value.to(device) for value in batch
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(boost_score, okved_ids)
            target = target.reshape_as(logits).to(dtype=logits.dtype)
            loss = criterion(logits, target)
            if loss.ndim != 0:
                raise ValueError("criterion must return a scalar loss")

            if training:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            batch_size = target.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("data loader must contain at least one example")
    return total_loss / total_examples


def fit_okved(
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
    """Train an OKVED residual scorer and return per-epoch losses."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    try:
        model_device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have trainable or frozen parameters") from error
    target_device = torch.device(device) if device is not None else model_device
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


__all__ = [
    "Callback",
    "EarlyStopping",
    "EpochMetrics",
    "fit_okved",
]

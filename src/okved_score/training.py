from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn

from regional_score.training import Callback, EarlyStopping, EpochMetrics

Batch = tuple[Tensor, Tensor, Tensor]


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

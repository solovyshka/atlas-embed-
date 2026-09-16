import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from okved_score.model import FlatOkvedResidualScorer
from okved_score.training import (
    Callback,
    EarlyStopping,
    EpochMetrics,
    fit_okved,
)


def make_batches() -> DataLoader:
    dataset = TensorDataset(
        torch.tensor([0.2, 0.4, 0.8, 0.5]),
        torch.tensor([1, 2, 3, 0], dtype=torch.long),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    return DataLoader(dataset, batch_size=2, shuffle=False)


class RecordingCallback(Callback):
    def __init__(self) -> None:
        self.began = False
        self.ended = False
        self.metrics: list[EpochMetrics] = []

    def on_train_begin(self, model: nn.Module) -> None:
        self.began = True

    def on_epoch_end(
        self, model: nn.Module, metrics: EpochMetrics
    ) -> None:
        self.metrics.append(metrics)

    def on_train_end(
        self, model: nn.Module, history: list[EpochMetrics]
    ) -> None:
        self.ended = True


def test_training_exposes_own_callback_contract() -> None:
    assert Callback.__module__ == "okved_score.training"
    assert EpochMetrics.__module__ == "okved_score.training"


def test_fit_okved_reports_losses_and_callbacks() -> None:
    model = FlatOkvedResidualScorer(vocab_size=5, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    recorder = RecordingCallback()

    history = fit_okved(
        model=model,
        train_batches=make_batches(),
        val_batches=make_batches(),
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        epochs=2,
        callbacks=[recorder],
        device="cpu",
        max_grad_norm=1.0,
    )

    assert recorder.began and recorder.ended
    assert recorder.metrics == history
    assert [metrics.epoch for metrics in history] == [1, 2]
    assert all(
        math.isfinite(metrics.train_loss) and math.isfinite(metrics.val_loss)
        for metrics in history
    )
    assert next(model.parameters()).device.type == "cpu"


def test_fit_okved_honors_early_stopping() -> None:
    model = FlatOkvedResidualScorer(vocab_size=5, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    early_stopping = EarlyStopping(patience=2)

    history = fit_okved(
        model=model,
        train_batches=make_batches(),
        val_batches=make_batches(),
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        epochs=10,
        callbacks=[early_stopping],
    )

    assert len(history) == 3
    assert early_stopping.best_epoch == 1
    assert early_stopping.stopped_epoch == 3
    assert early_stopping.stop_training


def test_fit_okved_applies_gradient_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FlatOkvedResidualScorer(vocab_size=5, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    observed_norms: list[float] = []
    original_clip = nn.utils.clip_grad_norm_

    def recording_clip(
        parameters: object, max_norm: float, *args: object, **kwargs: object
    ) -> torch.Tensor:
        observed_norms.append(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", recording_clip)
    fit_okved(
        model,
        make_batches(),
        make_batches(),
        optimizer,
        nn.BCEWithLogitsLoss(),
        epochs=1,
        max_grad_norm=0.5,
    )

    assert observed_norms == [0.5, 0.5]


@pytest.mark.parametrize(
    ("epochs", "max_grad_norm"),
    [(0, None), (True, None), (1, 0.0), (1, -1.0)],
)
def test_fit_okved_rejects_invalid_options(
    epochs: int, max_grad_norm: float | None
) -> None:
    model = FlatOkvedResidualScorer(vocab_size=5)

    with pytest.raises(ValueError):
        fit_okved(
            model,
            make_batches(),
            make_batches(),
            torch.optim.SGD(model.parameters(), lr=0.01),
            nn.BCEWithLogitsLoss(),
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )


def test_fit_okved_rejects_empty_batches() -> None:
    model = FlatOkvedResidualScorer(vocab_size=5)

    with pytest.raises(ValueError, match="at least one example"):
        fit_okved(
            model,
            [],
            make_batches(),
            torch.optim.SGD(model.parameters(), lr=0.01),
            nn.BCEWithLogitsLoss(),
            epochs=1,
        )

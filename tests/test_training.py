import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from regional_score import (
    EarlyStopping,
    EpochMetrics,
    ModelCheckpoint,
    RegionalResidualScorer,
    WeightMonitor,
    fit,
)


def make_batches() -> DataLoader:
    dataset = TensorDataset(
        torch.tensor([0.2, 0.4, 0.8, 0.5]),
        torch.tensor(
            [[0, 0], [12, 7], [69, 4], [3, 9]],
            dtype=torch.long,
        ),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_fit_stops_early_and_monitors_each_parameter() -> None:
    model = RegionalResidualScorer(dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    early_stopping = EarlyStopping(patience=2)
    weight_monitor = WeightMonitor()

    history = fit(
        model=model,
        train_batches=make_batches(),
        val_batches=make_batches(),
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        epochs=10,
        callbacks=[early_stopping, weight_monitor],
    )

    assert len(history) == 3
    assert early_stopping.best_epoch == 1
    assert early_stopping.stopped_epoch == 3
    assert set(weight_monitor.history) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert len(weight_monitor.history["embedding.weight"]) == 3
    assert weight_monitor.history["alpha"][0].std == 0.0


def test_early_stopping_restores_best_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    callback = EarlyStopping(patience=1, restore_best_weights=True)
    callback.on_train_begin(model)

    with torch.no_grad():
        model.weight.fill_(1.0)
    callback.on_epoch_end(model, EpochMetrics(1, train_loss=1.0, val_loss=0.5))

    with torch.no_grad():
        model.weight.fill_(2.0)
    callback.on_epoch_end(model, EpochMetrics(2, train_loss=0.8, val_loss=0.7))
    callback.on_train_end(model, [])

    assert callback.stop_training
    assert model.weight.item() == pytest.approx(1.0)


def test_weight_monitor_rejects_unknown_parameter_names() -> None:
    monitor = WeightMonitor(["missing.weight"])

    with pytest.raises(ValueError, match="unknown parameter names"):
        monitor.on_train_begin(RegionalResidualScorer())


def test_model_checkpoint_saves_every_fifth_epoch(tmp_path) -> None:
    model = RegionalResidualScorer()
    callback = ModelCheckpoint(tmp_path, every_n_epochs=5)
    callback.on_train_begin(model)

    for epoch in range(1, 11):
        callback.on_epoch_end(
            model,
            EpochMetrics(epoch, train_loss=1.0 / epoch, val_loss=2.0 / epoch),
        )

    assert [path.name for path in callback.saved_paths] == [
        "epoch_0005.pt",
        "epoch_0010.pt",
    ]
    checkpoint = torch.load(callback.saved_paths[0], weights_only=True)
    assert checkpoint["epoch"] == 5
    assert checkpoint["model_state_dict"]["embedding.weight"].shape == (70, 4)

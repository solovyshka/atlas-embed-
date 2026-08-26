import math

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from regional_score import (
    MaskedMultiTargetBCELoss,
    MultiTargetRegionalResidualScorer,
    fit,
)

TARGETS = ("30@3", "30@6", "30@9")


def test_multi_target_model_has_independent_heads_and_gates() -> None:
    model = MultiTargetRegionalResidualScorer(TARGETS)
    base_logits = torch.zeros(4, 3)
    region_ids = torch.tensor(
        [[0, 0, 0], [12, 7, 12], [70, 4, 71], [3, 3, 9]],
        dtype=torch.long,
    )

    logits = model(base_logits, region_ids)
    logits.sum().backward()

    assert logits.shape == (4, 3)
    for name in TARGETS:
        assert model.target_heads[name].weight.grad is not None
        assert model.alpha[name].grad is not None


def test_multi_target_model_requires_one_base_logit_per_target() -> None:
    model = MultiTargetRegionalResidualScorer(TARGETS)
    region_ids = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(ValueError, match=r"\[batch, 3\]"):
        model(torch.zeros(2), region_ids)


def test_masked_multi_target_loss_ignores_nan_labels() -> None:
    criterion = MaskedMultiTargetBCELoss(
        TARGETS,
        target_weights={"30@3": 1.0, "30@6": 2.0, "30@9": 3.0},
    )
    logits = torch.zeros(2, 3, requires_grad=True)
    targets = torch.tensor(
        [[0.0, 1.0, float("nan")], [1.0, float("nan"), 1.0]]
    )

    per_target = criterion.per_target(logits, targets)
    loss = criterion(logits, targets)
    loss.backward()

    assert per_target.tolist() == pytest.approx([math.log(2)] * 3)
    assert loss.item() == pytest.approx(math.log(2))
    assert logits.grad is not None
    assert logits.grad[0, 2] == 0
    assert logits.grad[1, 1] == 0


def test_masked_multi_target_loss_rejects_batch_without_labels() -> None:
    criterion = MaskedMultiTargetBCELoss(TARGETS)

    with pytest.raises(ValueError, match="no weighted target observations"):
        criterion(torch.zeros(2, 3), torch.full((2, 3), float("nan")))


def test_masked_loss_is_normalized_by_weighted_valid_observations() -> None:
    criterion = MaskedMultiTargetBCELoss(
        TARGETS,
        target_weights={"30@3": 1.0, "30@6": 2.0, "30@9": 4.0},
    )
    logits = torch.tensor([[2.0, -1.0, 0.0], [0.0, 3.0, -2.0]])
    targets = torch.tensor(
        [[1.0, 0.0, float("nan")], [0.0, float("nan"), 1.0]]
    )
    observed = ~torch.isnan(targets)
    weights = torch.tensor([1.0, 2.0, 4.0])
    element_losses = F.binary_cross_entropy_with_logits(
        logits,
        torch.nan_to_num(targets),
        reduction="none",
    )
    expected = (element_losses * observed * weights).sum() / (
        observed * weights
    ).sum()

    assert criterion(logits, targets).item() == pytest.approx(expected.item())
    assert criterion.aggregation_weight(targets).item() == pytest.approx(8.0)


def test_fit_aggregates_masked_loss_by_valid_observation_weight() -> None:
    model = MultiTargetRegionalResidualScorer(TARGETS, dropout=0.0)
    base_logits = torch.tensor(
        [[0.2, -0.1, 0.5], [-0.4, 0.3, 0.1], [0.8, -0.2, -0.6]]
    )
    region_ids = torch.tensor(
        [[0, 0, 0], [12, 7, 12], [70, 4, 71]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [1.0, float("nan"), float("nan")],
            [0.0, 1.0, float("nan")],
            [float("nan"), float("nan"), 1.0],
        ]
    )
    criterion = MaskedMultiTargetBCELoss(
        TARGETS,
        target_weights={"30@3": 1.0, "30@6": 2.0, "30@9": 3.0},
    )
    batches = DataLoader(
        TensorDataset(base_logits, region_ids, targets),
        batch_size=2,
        shuffle=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    with torch.inference_mode():
        expected = criterion(model(base_logits, region_ids), targets).item()
    history = fit(
        model=model,
        train_batches=batches,
        val_batches=batches,
        optimizer=optimizer,
        criterion=criterion,
        epochs=1,
    )

    assert history[0].val_loss == pytest.approx(expected)

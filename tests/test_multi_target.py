import math

import pytest
import torch

from regional_score import (
    MaskedMultiTargetBCELoss,
    MultiTargetRegionalResidualScorer,
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

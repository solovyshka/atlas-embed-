import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from regional_score import RegionalResidualScorer
from sweep_single_target import (
    baseline_config,
    build_trial_configs,
    evaluate_validation_auc,
    resolve_pos_weight,
)


def test_trial_plan_starts_with_epoch_baselines_and_is_deterministic() -> None:
    first = build_trial_configs(8, (5, 10, 20, 40), seed=42)
    second = build_trial_configs(8, (5, 10, 20, 40), seed=42)

    assert first == second
    assert first[:4] == [
        baseline_config(5),
        baseline_config(10),
        baseline_config(20),
        baseline_config(40),
    ]
    assert len(set(first)) == 8


def test_pos_weight_modes() -> None:
    targets = torch.tensor([0.0, 0.0, 0.0, 1.0])
    device = torch.device("cpu")

    assert resolve_pos_weight(targets, "none", device) is None
    assert resolve_pos_weight(targets, "sqrt", device).item() == pytest.approx(
        3**0.5
    )
    assert resolve_pos_weight(
        targets, "balanced", device
    ).item() == pytest.approx(3.0)


def test_validation_auc_uses_final_sigmoid_probability() -> None:
    model = RegionalResidualScorer(num_regions=4, dropout=0.0)
    base_scores = torch.tensor([0.1, 0.8, 0.2, 0.9])
    region_ids = torch.tensor([[0, 0], [1, 2], [2, 2], [3, 1]])
    equal_flags = torch.tensor([1.0, 0.0, 1.0, 0.0])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loader = DataLoader(
        TensorDataset(base_scores, region_ids, equal_flags, targets),
        batch_size=2,
    )

    base_auc, final_auc = evaluate_validation_auc(
        model,
        loader,
        torch.device("cpu"),
    )

    # Последний слой изначально нулевой, поэтому residual-модель возвращает
    # исходную вероятность base_score.
    assert base_auc == pytest.approx(1.0)
    assert final_auc == pytest.approx(base_auc)

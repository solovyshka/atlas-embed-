import numpy as np
import pytest

from okved_score.metrics import binary_metrics, client_bootstrap_auc_delta


def test_binary_metrics_return_expected_auc_and_gini() -> None:
    metrics = binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert set(metrics) == {"roc_auc", "gini", "pr_auc", "log_loss", "brier"}


def test_client_bootstrap_auc_delta_is_deterministic() -> None:
    target = np.tile([0, 1], 20)
    baseline = np.tile([0.4, 0.6], 20)
    candidate = np.tile([0.2, 0.8], 20)
    clients = np.repeat(np.arange(20), 2)

    first = client_bootstrap_auc_delta(
        target, candidate, baseline, clients, samples=20, seed=7
    )
    second = client_bootstrap_auc_delta(
        target, candidate, baseline, clients, samples=20, seed=7
    )

    assert first == second
    assert first["samples"] == 20
    assert first["lower"] == pytest.approx(0.0)
    assert first["upper"] == pytest.approx(0.0)


def test_binary_metrics_reject_single_class_target() -> None:
    with pytest.raises(ValueError, match="both classes"):
        binary_metrics([0, 0], [0.1, 0.2])

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _as_1d(values: Sequence[object] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return array


def binary_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Calculate ranking and calibration metrics for a binary target."""
    y_true = _as_1d(targets, "targets").astype(np.int64)
    y_score = _as_1d(probabilities, "probabilities").astype(np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("targets and probabilities must have the same shape")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("targets must contain only 0 and 1")
    if not np.isfinite(y_score).all() or np.any((y_score < 0) | (y_score > 1)):
        raise ValueError("probabilities must be finite values between 0 and 1")
    if np.unique(y_true).size != 2:
        raise ValueError("targets must contain both classes")

    auc = float(roc_auc_score(y_true, y_score))
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def client_bootstrap_auc_delta(
    targets: Sequence[int] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    client_ids: Sequence[object] | np.ndarray,
    *,
    samples: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Estimate an AUC-delta interval by resampling clients with replacement."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    y_true = _as_1d(targets, "targets").astype(np.int64)
    candidate_score = _as_1d(candidate, "candidate").astype(np.float64)
    baseline_score = _as_1d(baseline, "baseline").astype(np.float64)
    clients = _as_1d(client_ids, "client_ids")
    if not (
        y_true.shape
        == candidate_score.shape
        == baseline_score.shape
        == clients.shape
    ):
        raise ValueError("all bootstrap inputs must have the same shape")

    unique_clients, inverse = np.unique(clients, return_inverse=True)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sampled = rng.integers(0, unique_clients.size, size=unique_clients.size)
        client_weights = np.bincount(
            sampled, minlength=unique_clients.size
        ).astype(np.float64)
        row_weights = client_weights[inverse]
        weighted_positives = float(np.dot(row_weights, y_true))
        weighted_total = float(row_weights.sum())
        if weighted_positives == 0 or weighted_positives == weighted_total:
            continue
        deltas.append(
            float(
                roc_auc_score(
                    y_true, candidate_score, sample_weight=row_weights
                )
                - roc_auc_score(
                    y_true, baseline_score, sample_weight=row_weights
                )
            )
        )

    if not deltas:
        raise ValueError("bootstrap samples did not contain both target classes")
    alpha = (1 - confidence) / 2
    values = np.asarray(deltas)
    return {
        "samples": len(deltas),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
    }

"""Обучение flat и hierarchical ОКВЭД-моделей поверх готового бустинг-скора."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from okved_score import (
    RARE_INDEX,
    UNK_INDEX,
    FlatOkvedResidualScorer,
    FlatOkvedVocabulary,
    HierarchicalOkvedResidualScorer,
    HierarchicalOkvedVocabulary,
    binary_metrics,
    client_bootstrap_auc_delta,
    count_trainable_parameters,
    fit_okved,
    load_applications_csv,
    temporal_split,
)
from okved_score.data import OkvedApplications
from okved_score.training import EarlyStopping

DATA_PATH = Path("synthetic_data/okved_applications.csv")
OUTPUT_DIR = Path("checkpoints/okved_benchmark")
TRAIN_END = "2025-12-31"
VALIDATION_END = "2026-04-30"

RARE_THRESHOLD = 20
MAX_LEVELS = 3
EMBEDDING_DIM = 8
HIDDEN_DIM = 16
OUTPUT_DIM = 8
MATCH_PARAMETER_BUDGET = False
DROPOUT = 0.1

EPOCHS = 20
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4
BOOTSTRAP_SAMPLES = 200
SEED = 42


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ids_tensor(
    data: OkvedApplications,
    vocabulary: FlatOkvedVocabulary | HierarchicalOkvedVocabulary,
) -> torch.Tensor:
    encoded = vocabulary.transform(data.primary_okved)
    return torch.tensor(encoded, dtype=torch.long)


def _dataset(
    data: OkvedApplications,
    vocabulary: FlatOkvedVocabulary | HierarchicalOkvedVocabulary,
) -> TensorDataset:
    return TensorDataset(
        data.boost_scores,
        _ids_tensor(data, vocabulary),
        data.targets,
    )


def _loader(
    dataset: TensorDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _predict(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for boost_score, okved_ids, _ in _loader(
            dataset, batch_size, shuffle=False, seed=0
        ):
            logits = model(boost_score.to(device), okved_ids.to(device))
            predictions.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    return np.concatenate(predictions)


def _train_model(
    name: str,
    model: nn.Module,
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=1e-5,
        restore_best_weights=True,
    )
    history = fit_okved(
        model=model,
        train_batches=_loader(
            train_dataset,
            BATCH_SIZE,
            shuffle=True,
            seed=SEED,
        ),
        val_batches=_loader(
            validation_dataset,
            BATCH_SIZE,
            shuffle=False,
            seed=SEED,
        ),
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        epochs=EPOCHS,
        callbacks=[early_stopping],
        device=device,
        max_grad_norm=1.0,
    )
    print(
        f"{name}: epochs={len(history)} "
        f"best_epoch={early_stopping.best_epoch} "
        f"val_loss={early_stopping.best_value:.6f}"
    )
    return model, [asdict(metrics) for metrics in history]


def _assert_zero_initialized(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> None:
    boost_score, okved_ids, _ = next(
        iter(_loader(dataset, batch_size, shuffle=False, seed=0))
    )
    model.to(device).eval()
    with torch.inference_mode():
        probability = torch.sigmoid(
            model(boost_score.to(device), okved_ids.to(device))
        ).squeeze(1)
    torch.testing.assert_close(
        probability.cpu(),
        boost_score,
        rtol=1e-5,
        atol=1e-6,
        msg="zero-initialized residual model must reproduce boost_score",
    )


def _safe_segment_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    segment_targets = targets[mask]
    result: dict[str, Any] = {"rows": int(mask.sum())}
    if mask.sum() and np.unique(segment_targets).size == 2:
        result.update(binary_metrics(segment_targets, probabilities[mask]))
    return result


def _arm_result(
    name: str,
    targets: np.ndarray,
    probabilities: np.ndarray,
    baseline: np.ndarray,
    client_ids: tuple[str, ...],
    leaf_ids: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics = binary_metrics(targets, probabilities)
    baseline_auc = binary_metrics(targets, baseline)["roc_auc"]
    result: dict[str, Any] = {
        "arm": name,
        **metrics,
        "auc_delta_vs_boost": metrics["roc_auc"] - baseline_auc,
        "segments": {
            "frequent": _safe_segment_metrics(
                targets, probabilities, leaf_ids >= 3
            ),
            "rare": _safe_segment_metrics(
                targets, probabilities, leaf_ids == RARE_INDEX
            ),
            "oov": _safe_segment_metrics(
                targets, probabilities, leaf_ids == UNK_INDEX
            ),
        },
    }
    if name != "boost_score":
        result["auc_delta_bootstrap"] = client_bootstrap_auc_delta(
            targets,
            probabilities,
            baseline,
            np.asarray(client_ids),
            samples=bootstrap_samples,
            seed=seed,
        )
    return result


def _checkpoint(
    path: Path,
    model: nn.Module,
    model_config: dict[str, Any],
    vocabulary: FlatOkvedVocabulary | HierarchicalOkvedVocabulary,
    history: list[dict[str, float | int]],
) -> None:
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "model_config": model_config,
            "vocabulary": vocabulary.to_dict(),
            "history": history,
            "split": {
                "train_end": TRAIN_END,
                "validation_end": VALIDATION_END,
            },
            "base_score": {
                "column": "boost_score",
                "train_provenance": "OOF",
                "validation_test_provenance": "frozen",
            },
        },
        path,
    )


def run_benchmark() -> dict[str, Any]:
    """Обучить две ОКВЭД-сети и сравнить их с готовым boost score."""
    device = resolve_device()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    applications = load_applications_csv(DATA_PATH)
    splits = temporal_split(
        applications,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
    )
    flat_vocabulary = FlatOkvedVocabulary.fit(
        splits.train.primary_okved,
        rare_threshold=RARE_THRESHOLD,
        max_levels=MAX_LEVELS,
    )
    hierarchical_vocabulary = HierarchicalOkvedVocabulary.fit(
        splits.train.primary_okved,
        rare_threshold=RARE_THRESHOLD,
        max_levels=MAX_LEVELS,
    )

    flat_train = _dataset(splits.train, flat_vocabulary)
    flat_validation = _dataset(splits.validation, flat_vocabulary)
    flat_test = _dataset(splits.test, flat_vocabulary)
    hierarchical_train = _dataset(splits.train, hierarchical_vocabulary)
    hierarchical_validation = _dataset(
        splits.validation, hierarchical_vocabulary
    )
    hierarchical_test = _dataset(splits.test, hierarchical_vocabulary)

    flat_config = {
        "vocab_size": flat_vocabulary.vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT,
    }
    hierarchical_embedding_dim = EMBEDDING_DIM
    if MATCH_PARAMETER_BUDGET:
        flat_budget = EMBEDDING_DIM * (
            flat_vocabulary.vocab_size + HIDDEN_DIM
        )
        hierarchical_width = sum(hierarchical_vocabulary.vocab_sizes)
        hierarchical_embedding_dim = max(
            1,
            round(flat_budget / (hierarchical_width + HIDDEN_DIM)),
        )
    hierarchical_config = {
        "vocab_sizes": hierarchical_vocabulary.vocab_sizes,
        "embedding_dim": hierarchical_embedding_dim,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT,
    }
    torch.manual_seed(SEED)
    flat_model = FlatOkvedResidualScorer(**flat_config)
    torch.manual_seed(SEED)
    hierarchical_model = HierarchicalOkvedResidualScorer(**hierarchical_config)
    _assert_zero_initialized(
        flat_model, flat_train, BATCH_SIZE, device
    )
    _assert_zero_initialized(
        hierarchical_model, hierarchical_train, BATCH_SIZE, device
    )
    flat_model, flat_history = _train_model(
        "flat",
        flat_model,
        flat_train,
        flat_validation,
        device,
    )
    hierarchical_model, hierarchical_history = _train_model(
        "hierarchical",
        hierarchical_model,
        hierarchical_train,
        hierarchical_validation,
        device,
    )

    target = splits.test.targets.numpy().astype(np.int64)
    baseline = splits.test.boost_scores.numpy()
    flat_probability = _predict(
        flat_model, flat_test, BATCH_SIZE, device
    )
    hierarchical_probability = _predict(
        hierarchical_model, hierarchical_test, BATCH_SIZE, device
    )
    leaf_ids = np.asarray(flat_vocabulary.transform(splits.test.primary_okved))
    arms = [
        _arm_result(
            "boost_score",
            target,
            baseline,
            baseline,
            splits.test.client_ids,
            leaf_ids,
            BOOTSTRAP_SAMPLES,
            SEED,
        ),
        _arm_result(
            "flat_residual_nn",
            target,
            flat_probability,
            baseline,
            splits.test.client_ids,
            leaf_ids,
            BOOTSTRAP_SAMPLES,
            SEED,
        ),
        _arm_result(
            "hierarchical_residual_nn",
            target,
            hierarchical_probability,
            baseline,
            splits.test.client_ids,
            leaf_ids,
            BOOTSTRAP_SAMPLES,
            SEED,
        ),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _checkpoint(
        OUTPUT_DIR / "flat_model.pt",
        flat_model,
        flat_config,
        flat_vocabulary,
        flat_history,
    )
    _checkpoint(
        OUTPUT_DIR / "hierarchical_model.pt",
        hierarchical_model,
        hierarchical_config,
        hierarchical_vocabulary,
        hierarchical_history,
    )
    result = {
        "device": str(device),
        "seed": SEED,
        "rows": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "config": {
            "data": str(DATA_PATH),
            "output_dir": str(OUTPUT_DIR),
            "train_end": TRAIN_END,
            "validation_end": VALIDATION_END,
            "rare_threshold": RARE_THRESHOLD,
            "max_levels": MAX_LEVELS,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "dropout": DROPOUT,
            "match_parameter_budget": MATCH_PARAMETER_BUDGET,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "seed": SEED,
        },
        "coverage": {
            "frequent": int((leaf_ids >= 3).sum()),
            "rare": int((leaf_ids == RARE_INDEX).sum()),
            "oov": int((leaf_ids == UNK_INDEX).sum()),
        },
        "alpha": {
            "flat": float(flat_model.alpha.detach().cpu()),
            "hierarchical": float(
                hierarchical_model.alpha.detach().cpu()
            ),
        },
        "trainable_parameters": {
            "flat": count_trainable_parameters(flat_model),
            "hierarchical": count_trainable_parameters(hierarchical_model),
        },
        "arms": arms,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in arm.items()
                if key not in {"segments", "auc_delta_bootstrap"}
            }
            for arm in arms
        ]
    ).to_csv(OUTPUT_DIR / "results.csv", index=False)
    return result


def main() -> None:
    result = run_benchmark()
    summary = [
        {
            "arm": arm["arm"],
            "roc_auc": round(arm["roc_auc"], 6),
            "auc_delta_vs_boost": round(arm["auc_delta_vs_boost"], 6),
        }
        for arm in result["arms"]
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

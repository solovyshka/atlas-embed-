"""Compare flat and hierarchical OKVED residual networks over a ready score."""

from __future__ import annotations

import argparse
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
from regional_score import EarlyStopping


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("synthetic_data/okved_applications.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/okved_benchmark"),
    )
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--validation-end", default="2026-04-30")
    parser.add_argument("--rare-threshold", type=int, default=20)
    parser.add_argument("--max-levels", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument(
        "--match-parameter-budget",
        action="store_true",
        help="Reduce hierarchy embedding width to approximate the flat parameter budget.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "rare_threshold",
        "max_levels",
        "embedding_dim",
        "hidden_dim",
        "output_dim",
        "epochs",
        "batch_size",
        "patience",
        "bootstrap_samples",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")


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
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=1e-5,
        restore_best_weights=True,
    )
    history = fit_okved(
        model=model,
        train_batches=_loader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            seed=args.seed,
        ),
        val_batches=_loader(
            validation_dataset,
            args.batch_size,
            shuffle=False,
            seed=args.seed,
        ),
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        epochs=args.epochs,
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
    args: argparse.Namespace,
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
                "train_end": args.train_end,
                "validation_end": args.validation_end,
            },
            "base_score": {
                "column": "boost_score",
                "train_provenance": "OOF",
                "validation_test_provenance": "frozen",
            },
        },
        path,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    applications = load_applications_csv(args.data)
    splits = temporal_split(
        applications,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    flat_vocabulary = FlatOkvedVocabulary.fit(
        splits.train.primary_okved,
        rare_threshold=args.rare_threshold,
        max_levels=args.max_levels,
    )
    hierarchical_vocabulary = HierarchicalOkvedVocabulary.fit(
        splits.train.primary_okved,
        rare_threshold=args.rare_threshold,
        max_levels=args.max_levels,
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
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "dropout": args.dropout,
    }
    hierarchical_embedding_dim = args.embedding_dim
    if args.match_parameter_budget:
        flat_budget = args.embedding_dim * (
            flat_vocabulary.vocab_size + args.hidden_dim
        )
        hierarchical_width = sum(hierarchical_vocabulary.vocab_sizes)
        hierarchical_embedding_dim = max(
            1,
            round(flat_budget / (hierarchical_width + args.hidden_dim)),
        )
    hierarchical_config = {
        "vocab_sizes": hierarchical_vocabulary.vocab_sizes,
        "embedding_dim": hierarchical_embedding_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "dropout": args.dropout,
    }
    torch.manual_seed(args.seed)
    flat_model = FlatOkvedResidualScorer(**flat_config)
    torch.manual_seed(args.seed)
    hierarchical_model = HierarchicalOkvedResidualScorer(**hierarchical_config)
    _assert_zero_initialized(
        flat_model, flat_train, args.batch_size, device
    )
    _assert_zero_initialized(
        hierarchical_model, hierarchical_train, args.batch_size, device
    )
    flat_model, flat_history = _train_model(
        "flat",
        flat_model,
        flat_train,
        flat_validation,
        args,
        device,
    )
    hierarchical_model, hierarchical_history = _train_model(
        "hierarchical",
        hierarchical_model,
        hierarchical_train,
        hierarchical_validation,
        args,
        device,
    )

    target = splits.test.targets.numpy().astype(np.int64)
    baseline = splits.test.boost_scores.numpy()
    flat_probability = _predict(
        flat_model, flat_test, args.batch_size, device
    )
    hierarchical_probability = _predict(
        hierarchical_model, hierarchical_test, args.batch_size, device
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
            args.bootstrap_samples,
            args.seed,
        ),
        _arm_result(
            "flat_residual_nn",
            target,
            flat_probability,
            baseline,
            splits.test.client_ids,
            leaf_ids,
            args.bootstrap_samples,
            args.seed,
        ),
        _arm_result(
            "hierarchical_residual_nn",
            target,
            hierarchical_probability,
            baseline,
            splits.test.client_ids,
            leaf_ids,
            args.bootstrap_samples,
            args.seed,
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint(
        args.output_dir / "flat_model.pt",
        flat_model,
        flat_config,
        flat_vocabulary,
        flat_history,
        args,
    )
    _checkpoint(
        args.output_dir / "hierarchical_model.pt",
        hierarchical_model,
        hierarchical_config,
        hierarchical_vocabulary,
        hierarchical_history,
        args,
    )
    result = {
        "device": str(device),
        "seed": args.seed,
        "rows": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
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
    (args.output_dir / "results.json").write_text(
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
    ).to_csv(args.output_dir / "results.csv", index=False)
    return result


def main(argv: list[str] | None = None) -> None:
    result = run_benchmark(parse_args(argv))
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

"""Подбор гиперпараметров single-target residual-модели.

Каждый trial обучается ровно заданное число эпох без early stopping. Метрика
подбора — ROC AUC итоговой вероятности полной модели на validation:

    sigmoid(logit(score_dubai) + alpha * region_delta_logit)

Все trial-модели сохраняются отдельно. Test-выборка не используется при
подборе гиперпараметров.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from regional_score import (
    RegionalResidualScorer,
    fit,
    load_applications_csv,
    temporal_split,
)

EMBEDDING_DIMS = (2, 4, 8)
HIDDEN_DIMS = ((8, 4), (16, 8), (24, 8), (32, 16))
DROPOUTS = (0.0, 0.05, 0.1, 0.15, 0.25)
BATCH_SIZES = (512, 2048, 8192)
LEARNING_RATES = (3e-4, 1e-3, 3e-3)
WEIGHT_DECAYS = (0.0, 1e-5, 1e-4, 1e-3)
POS_WEIGHT_MODES = ("none", "sqrt", "balanced")
DEFAULT_EPOCH_OPTIONS = (5, 10, 20, 40)


@dataclass(frozen=True)
class TrialConfig:
    embedding_dim: int
    hidden_dims: tuple[int, int]
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    pos_weight_mode: str
    epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a randomized hyperparameter sweep for the single-target model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("synthetic_data/applications.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/hyperparameter_sweep"),
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--epoch-options",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPOCH_OPTIONS),
    )
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--validation-end", default="2026-04-30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
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


def baseline_config(epochs: int) -> TrialConfig:
    """Текущая конфигурация проекта для чистого сравнения числа эпох."""
    return TrialConfig(
        embedding_dim=4,
        hidden_dims=(24, 8),
        dropout=0.15,
        batch_size=8192,
        learning_rate=3e-3,
        weight_decay=1e-4,
        pos_weight_mode="balanced",
        epochs=epochs,
    )


def build_trial_configs(
    trials: int,
    epoch_options: tuple[int, ...],
    seed: int,
) -> list[TrialConfig]:
    """Покрыть все epoch options baseline-моделью и дополнить random search."""
    if not epoch_options or any(epochs < 1 for epochs in epoch_options):
        raise ValueError("epoch options must contain positive values")
    if trials < len(epoch_options):
        raise ValueError("trials must be at least the number of epoch options")

    baselines = [baseline_config(epochs) for epochs in epoch_options]
    baseline_set = set(baselines)
    pool = [
        TrialConfig(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_weight_mode=pos_weight_mode,
            epochs=epochs,
        )
        for (
            embedding_dim,
            hidden_dims,
            dropout,
            batch_size,
            learning_rate,
            weight_decay,
            pos_weight_mode,
            epochs,
        ) in itertools.product(
            EMBEDDING_DIMS,
            HIDDEN_DIMS,
            DROPOUTS,
            BATCH_SIZES,
            LEARNING_RATES,
            WEIGHT_DECAYS,
            POS_WEIGHT_MODES,
            epoch_options,
        )
        if TrialConfig(
            embedding_dim,
            hidden_dims,
            dropout,
            batch_size,
            learning_rate,
            weight_decay,
            pos_weight_mode,
            epochs,
        )
        not in baseline_set
    ]
    random.Random(seed).shuffle(pool)
    return baselines + pool[: trials - len(baselines)]


def resolve_pos_weight(
    targets: torch.Tensor,
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives.item() == 0 or negatives.item() == 0:
        raise ValueError("training target must contain both classes")

    imbalance = negatives / positives
    if mode == "none":
        return None
    if mode == "sqrt":
        return imbalance.sqrt().to(device)
    if mode == "balanced":
        return imbalance.to(device)
    raise ValueError(f"unknown pos_weight mode: {mode}")


def evaluate_validation_auc(
    model: RegionalResidualScorer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Вернуть ROC AUC исходного score и итоговой sigmoid(logit) модели."""
    model.eval()
    targets: list[torch.Tensor] = []
    base_scores: list[torch.Tensor] = []
    final_probabilities: list[torch.Tensor] = []

    with torch.inference_mode():
        for base_score, region_ids, equal_flag, target in loader:
            final_logits = model(
                base_score.to(device, non_blocking=True),
                region_ids.to(device, non_blocking=True),
                equal_flag.to(device, non_blocking=True),
            )
            targets.append(target.cpu())
            base_scores.append(base_score.cpu())
            final_probabilities.append(
                torch.sigmoid(final_logits).squeeze(1).cpu()
            )

    target_values = torch.cat(targets).numpy()
    base_values = torch.cat(base_scores).numpy()
    final_values = torch.cat(final_probabilities).numpy()
    return (
        float(roc_auc_score(target_values, base_values)),
        float(roc_auc_score(target_values, final_values)),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_row(
    trial_id: int,
    config: TrialConfig,
    checkpoint: Path,
    base_auc: float,
    final_auc: float,
    final_train_loss: float,
    final_val_loss: float,
    duration_seconds: float,
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "val_final_auc": final_auc,
        "val_base_auc": base_auc,
        "auc_lift": final_auc - base_auc,
        "embedding_dim": config.embedding_dim,
        "hidden_dim_1": config.hidden_dims[0],
        "hidden_dim_2": config.hidden_dims[1],
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "pos_weight_mode": config.pos_weight_mode,
        "epochs": config.epochs,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "duration_seconds": round(duration_seconds, 2),
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    args = parse_args()
    epoch_options = tuple(dict.fromkeys(args.epoch_options))
    configs = build_trial_configs(args.trials, epoch_options, args.seed)
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    data = load_applications_csv(args.data)
    datasets = temporal_split(
        data,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    num_regions = int(data.region_ids.max().item()) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search_metadata = {
        "data": str(args.data),
        "device": str(device),
        "seed": args.seed,
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "rows": {
            "train": len(datasets.train),
            "validation": len(datasets.validation),
            "test": len(datasets.test),
        },
        "trials": [asdict(config) for config in configs],
    }
    (args.output_dir / "search_plan.json").write_text(
        json.dumps(search_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results: list[dict[str, object]] = []
    for trial_id, config in enumerate(configs, start=1):
        # Одинаковый seed делает сравнение конфигураций менее шумным.
        torch.manual_seed(args.seed)
        model = RegionalResidualScorer(
            num_regions=num_regions,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        pos_weight = resolve_pos_weight(
            datasets.train.tensors[-1],
            config.pos_weight_mode,
            device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        train_loader = make_loader(
            datasets.train,
            config.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        validation_loader = make_loader(
            datasets.validation,
            config.batch_size,
            shuffle=False,
            seed=args.seed,
        )

        print(
            f"[{trial_id:03d}/{len(configs):03d}] "
            f"epochs={config.epochs} emb={config.embedding_dim} "
            f"hidden={config.hidden_dims} dropout={config.dropout} "
            f"batch={config.batch_size} lr={config.learning_rate:g} "
            f"wd={config.weight_decay:g} pos_weight={config.pos_weight_mode}"
        )
        started_at = time.perf_counter()
        history = fit(
            model=model,
            train_batches=train_loader,
            val_batches=validation_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config.epochs,
            callbacks=(),
            device=device,
            max_grad_norm=1.0,
        )
        base_auc, final_auc = evaluate_validation_auc(
            model,
            validation_loader,
            device,
        )
        duration_seconds = time.perf_counter() - started_at
        checkpoint = args.output_dir / (
            f"trial_{trial_id:03d}_val_auc_{final_auc:.6f}.pt"
        )
        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "model_config": {
                    "num_regions": num_regions,
                    "embedding_dim": config.embedding_dim,
                    "hidden_dims": config.hidden_dims,
                    "dropout": config.dropout,
                    "initial_alpha": 1.0,
                },
                "training_config": asdict(config),
                "target": "flag_6m_30p",
                "base_score": "score_dubai",
                "train_end": args.train_end,
                "validation_end": args.validation_end,
                "val_base_auc": base_auc,
                "val_final_auc": final_auc,
                "auc_lift": final_auc - base_auc,
                "history": [asdict(metrics) for metrics in history],
            },
            checkpoint,
        )
        row = result_row(
            trial_id=trial_id,
            config=config,
            checkpoint=checkpoint,
            base_auc=base_auc,
            final_auc=final_auc,
            final_train_loss=history[-1].train_loss,
            final_val_loss=history[-1].val_loss,
            duration_seconds=duration_seconds,
        )
        results.append(row)
        write_csv(args.output_dir / "results.csv", results)
        print(
            f"  val_auc={final_auc:.6f} "
            f"lift={final_auc - base_auc:+.6f} "
            f"time={duration_seconds:.1f}s"
        )

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    leaderboard = sorted(
        results,
        key=lambda row: float(row["val_final_auc"]),
        reverse=True,
    )
    write_csv(args.output_dir / "leaderboard.csv", leaderboard)
    best_checkpoint = Path(str(leaderboard[0]["checkpoint"]))
    shutil.copy2(best_checkpoint, args.output_dir / "best_model.pt")

    best = leaderboard[0]
    print(
        json.dumps(
            {
                "best_trial": best["trial_id"],
                "val_final_auc": round(float(best["val_final_auc"]), 6),
                "val_base_auc": round(float(best["val_base_auc"]), 6),
                "auc_lift": round(float(best["auc_lift"]), 6),
                "best_model": str(args.output_dir / "best_model.pt"),
                "leaderboard": str(args.output_dir / "leaderboard.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

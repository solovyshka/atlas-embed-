from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_applications_csv, temporal_split
from .model import RegionalResidualScorer
from .training import EarlyStopping, fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regional residual model on applications.csv."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("synthetic_data/applications.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/regional_residual.pt"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
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
    )


def evaluate_auc(
    model: RegionalResidualScorer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    targets: list[torch.Tensor] = []
    base_scores: list[torch.Tensor] = []
    final_scores: list[torch.Tensor] = []

    with torch.inference_mode():
        for base_score, region_ids, equal_flag, target in loader:
            logits = model(
                base_score.to(device),
                region_ids.to(device),
                equal_flag.to(device),
            )
            targets.append(target)
            base_scores.append(base_score)
            final_scores.append(torch.sigmoid(logits).cpu().squeeze(1))

    target_values = torch.cat(targets).numpy()
    base_values = torch.cat(base_scores).numpy()
    final_values = torch.cat(final_scores).numpy()
    return (
        float(roc_auc_score(target_values, base_values)),
        float(roc_auc_score(target_values, final_values)),
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    data = load_applications_csv(args.data)
    datasets = temporal_split(data)

    train_loader = make_loader(
        datasets.train,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = make_loader(
        datasets.validation,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = make_loader(
        datasets.test,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    num_regions = int(data.region_ids.max().item()) + 1
    model = RegionalResidualScorer(num_regions=num_regions)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_targets = datasets.train.tensors[-1]
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(negatives / positives).to(device)
    )
    early_stopping = EarlyStopping(
        patience=3,
        min_delta=1e-4,
        restore_best_weights=True,
    )

    history = fit(
        model=model,
        train_batches=train_loader,
        val_batches=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        callbacks=[early_stopping],
        device=device,
        max_grad_norm=1.0,
    )
    base_auc, final_auc = evaluate_auc(model, test_loader, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "model_config": {"num_regions": num_regions},
            "target": "flag_6m_30p",
            "base_score": "score_dubai",
            "train_end": "2025-12-31",
            "validation_end": "2026-04-30",
            "epochs_trained": len(history),
            "test_base_auc": base_auc,
            "test_final_auc": final_auc,
        },
        args.output,
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "rows": {
                    "train": len(datasets.train),
                    "validation": len(datasets.validation),
                    "test": len(datasets.test),
                },
                "epochs_trained": len(history),
                "test_base_auc": round(base_auc, 6),
                "test_final_auc": round(final_auc, 6),
                "auc_lift": round(final_auc - base_auc, 6),
                "checkpoint": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

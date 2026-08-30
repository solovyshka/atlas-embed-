"""Обучение single-target региональной модели и снятие эмбеддингов.

Сеть учится residual-поправкой к logit(score_dubai). На практике голова
не нужна: забираем 8-мерные эмбеддинги и подаём их вместе со score_dubai
в бустинг или логистическую регрессию.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from regional_score import (
    EarlyStopping,
    RegionalResidualScorer,
    fit,
    load_applications_csv,
    temporal_split,
)

DATA_PATH = Path("synthetic_data/applications.csv")
CHECKPOINT_PATH = Path("checkpoints/regional_residual.pt")
TRAIN_END = "2025-12-31"
VALIDATION_END = "2026-04-30"

EPOCHS = 12
BATCH_SIZE = 8192
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset: TensorDataset, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def collect_features(
    model: RegionalResidualScorer,
    dataset: TensorDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Собрать score_dubai, региональные эмбеддинги и таргет."""
    model.eval()
    scores: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.inference_mode():
        for base_score, region_ids, equal_flag, target in DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False
        ):
            encoded = model.encode(region_ids.to(device), equal_flag.to(device))
            scores.append(base_score.reshape(-1).cpu().numpy())
            embeddings.append(encoded.cpu().numpy())
            targets.append(target.reshape(-1).cpu().numpy())

    return (
        np.concatenate(scores),
        np.concatenate(embeddings),
        np.concatenate(targets),
    )


def residual_auc(
    model: RegionalResidualScorer,
    dataset: TensorDataset,
    device: torch.device,
) -> float:
    """AUC полной residual-модели — только для сравнения, на проде не используем."""
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for base_score, region_ids, equal_flag, target in DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False
        ):
            logits = model(
                base_score.to(device),
                region_ids.to(device),
                equal_flag.to(device),
            )
            scores.append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
            targets.append(target.reshape(-1).cpu().numpy())
    return float(roc_auc_score(np.concatenate(targets), np.concatenate(scores)))


def main() -> None:
    torch.manual_seed(SEED)
    device = resolve_device()

    data = load_applications_csv(DATA_PATH)
    datasets = temporal_split(
        data,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
    )
    print(
        f"device={device}  "
        f"train={len(datasets.train):,}  "
        f"val={len(datasets.validation):,}  "
        f"test={len(datasets.test):,}"
    )

    num_regions = int(data.region_ids.max().item()) + 1
    model = RegionalResidualScorer(num_regions=num_regions)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_targets = datasets.train.tensors[-1]
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives).to(device))

    history = fit(
        model=model,
        train_batches=make_loader(datasets.train, shuffle=True),
        val_batches=make_loader(datasets.validation, shuffle=False),
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        callbacks=[EarlyStopping(patience=3, min_delta=1e-4, restore_best_weights=True)],
        device=device,
        max_grad_norm=1.0,
    )
    for row in history:
        print(
            f"epoch {row.epoch:02d}  "
            f"train_loss={row.train_loss:.4f}  "
            f"val_loss={row.val_loss:.4f}"
        )

    train_score, train_emb, train_y = collect_features(model, datasets.train, device)
    test_score, test_emb, test_y = collect_features(model, datasets.test, device)

    # Так эмбеддинги будут использоваться в проде: score_dubai + encode(...).
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(np.column_stack([train_score, train_emb]), train_y)
    logreg_proba = logreg.predict_proba(np.column_stack([test_score, test_emb]))[:, 1]

    metrics = {
        "test_dubai_auc": float(roc_auc_score(test_y, test_score)),
        "test_logreg_dubai_plus_emb_auc": float(roc_auc_score(test_y, logreg_proba)),
        "test_residual_nn_auc": residual_auc(model, datasets.test, device),
        "embedding_dim": int(train_emb.shape[1]),
    }
    print(
        "test AUC  "
        f"dubai={metrics['test_dubai_auc']:.6f}  "
        f"logreg(dubai+emb)={metrics['test_logreg_dubai_plus_emb_auc']:.6f}  "
        f"residual_nn={metrics['test_residual_nn_auc']:.6f}"
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "model_config": {"num_regions": num_regions},
            "target": "flag_6m_30p",
            "base_score": "score_dubai",
            "train_end": TRAIN_END,
            "validation_end": VALIDATION_END,
            "epochs_trained": len(history),
            **metrics,
        },
        CHECKPOINT_PATH,
    )
    print(f"checkpoint={CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

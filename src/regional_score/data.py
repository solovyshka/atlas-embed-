from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset


REQUIRED_COLUMNS = (
    "reg_region_code",
    "fact_region_code",
    "equal_flag",
    "flag_6m_30p",
    "score_dubai",
    "issue_dt",
)


@dataclass(frozen=True)
class ApplicationTensors:
    base_scores: Tensor
    region_ids: Tensor
    equal_flags: Tensor
    targets: Tensor
    issue_days: Tensor

    def select(self, mask: Tensor) -> TensorDataset:
        return TensorDataset(
            self.base_scores[mask],
            self.region_ids[mask],
            self.equal_flags[mask],
            self.targets[mask],
        )


@dataclass(frozen=True)
class TemporalDatasets:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


def load_applications_csv(path: str | Path) -> ApplicationTensors:
    frame = pd.read_csv(path, usecols=REQUIRED_COLUMNS, parse_dates=["issue_dt"])
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {names}")
    if frame.empty:
        raise ValueError("CSV must contain at least one row")
    if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("required columns must not contain missing values")
    if not frame["score_dubai"].between(0, 1, inclusive="both").all():
        raise ValueError("score_dubai must be between 0 and 1")

    region_ids = torch.from_numpy(
        frame[["reg_region_code", "fact_region_code"]].to_numpy(copy=True)
    ).long()
    equal_flags = torch.from_numpy(frame["equal_flag"].to_numpy(copy=True)).float()
    targets = torch.from_numpy(frame["flag_6m_30p"].to_numpy(copy=True)).float()
    base_scores = torch.from_numpy(frame["score_dubai"].to_numpy(copy=True)).float()
    issue_days = torch.from_numpy(
        frame["issue_dt"].to_numpy(dtype="datetime64[D]").astype("int64")
    )

    if not torch.all((equal_flags == 0) | (equal_flags == 1)):
        raise ValueError("equal_flag must contain only 0 and 1")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("flag_6m_30p must contain only 0 and 1")

    return ApplicationTensors(
        base_scores=base_scores,
        region_ids=region_ids,
        equal_flags=equal_flags,
        targets=targets,
        issue_days=issue_days,
    )


def temporal_split(
    data: ApplicationTensors,
    train_end: str = "2025-12-31",
    validation_end: str = "2026-04-30",
) -> TemporalDatasets:
    train_boundary = torch.tensor(
        pd.Timestamp(train_end).to_datetime64().astype("datetime64[D]").astype("int64")
    )
    validation_boundary = torch.tensor(
        pd.Timestamp(validation_end)
        .to_datetime64()
        .astype("datetime64[D]")
        .astype("int64")
    )
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must be before validation_end")

    train_mask = data.issue_days <= train_boundary
    validation_mask = (
        (data.issue_days > train_boundary)
        & (data.issue_days <= validation_boundary)
    )
    test_mask = data.issue_days > validation_boundary
    sizes = (int(train_mask.sum()), int(validation_mask.sum()), int(test_mask.sum()))
    if any(size == 0 for size in sizes):
        raise ValueError("temporal split produced an empty dataset")

    return TemporalDatasets(
        train=data.select(train_mask),
        validation=data.select(validation_mask),
        test=data.select(test_mask),
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from .hierarchy import normalize_okved


REQUIRED_COLUMNS = (
    "client_id",
    "issue_dt",
    "primary_okved",
    "boost_score",
    "target",
)


@dataclass(frozen=True)
class OkvedApplications:
    client_ids: tuple[str, ...]
    issue_days: Tensor
    primary_okved: tuple[str, ...]
    boost_scores: Tensor
    targets: Tensor

    def select(self, mask: Tensor) -> OkvedApplications:
        indices = mask.nonzero(as_tuple=False).flatten().tolist()
        return OkvedApplications(
            client_ids=tuple(self.client_ids[index] for index in indices),
            issue_days=self.issue_days[mask],
            primary_okved=tuple(self.primary_okved[index] for index in indices),
            boost_scores=self.boost_scores[mask],
            targets=self.targets[mask],
        )

    def __len__(self) -> int:
        return len(self.client_ids)


@dataclass(frozen=True)
class OkvedRawSplits:
    train: OkvedApplications
    validation: OkvedApplications
    test: OkvedApplications


def load_applications_csv(path: str | Path) -> OkvedApplications:
    frame = pd.read_csv(
        path,
        dtype={"client_id": "string", "primary_okved": "string"},
    )
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {names}")
    if frame.empty:
        raise ValueError("CSV must contain at least one row")

    required = frame[list(REQUIRED_COLUMNS)]
    if required.isnull().any().any():
        raise ValueError("required columns must not contain missing values")

    client_ids = tuple(value.strip() for value in frame["client_id"].astype(str))
    if any(not client_id for client_id in client_ids):
        raise ValueError("client_id must not be empty")

    try:
        primary_okved = tuple(normalize_okved(code) for code in frame["primary_okved"])
    except ValueError as error:
        raise ValueError(f"invalid primary_okved: {error}") from error

    issue_dates = pd.to_datetime(frame["issue_dt"], errors="coerce")
    if issue_dates.isnull().any():
        raise ValueError("issue_dt must contain valid dates")

    boost_scores = pd.to_numeric(frame["boost_score"], errors="coerce")
    if boost_scores.isnull().any():
        raise ValueError("boost_score must be numeric")
    boost_tensor = torch.from_numpy(boost_scores.to_numpy(dtype="float32", copy=True))
    if not torch.isfinite(boost_tensor).all() or not torch.all(
        (boost_tensor >= 0) & (boost_tensor <= 1)
    ):
        raise ValueError("boost_score must be between 0 and 1")

    targets = pd.to_numeric(frame["target"], errors="coerce")
    if targets.isnull().any():
        raise ValueError("target must be numeric")
    target_tensor = torch.from_numpy(targets.to_numpy(dtype="float32", copy=True))
    if not torch.all((target_tensor == 0) | (target_tensor == 1)):
        raise ValueError("target must contain only 0 and 1")

    issue_days = torch.from_numpy(
        issue_dates.to_numpy(dtype="datetime64[D]").astype("int64", copy=True)
    )
    return OkvedApplications(
        client_ids=client_ids,
        issue_days=issue_days,
        primary_okved=primary_okved,
        boost_scores=boost_tensor,
        targets=target_tensor,
    )


def temporal_split(
    data: OkvedApplications,
    train_end: str = "2025-12-31",
    validation_end: str = "2026-04-30",
) -> OkvedRawSplits:
    train_boundary = _date_to_day(train_end, "train_end")
    validation_boundary = _date_to_day(validation_end, "validation_end")
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

    return OkvedRawSplits(
        train=data.select(train_mask),
        validation=data.select(validation_mask),
        test=data.select(test_mask),
    )


def _date_to_day(value: str, name: str) -> Tensor:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid date") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    return torch.tensor(
        timestamp.to_datetime64().astype("datetime64[D]").astype("int64")
    )

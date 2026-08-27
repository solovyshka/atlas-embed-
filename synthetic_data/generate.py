"""Generate synthetic application data for region-embedding experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_ROWS = 1_000_000
DEFAULT_CLIENTS = 500_000
DEFAULT_REGIONS = 70
DEFAULT_START_DATE = date(2024, 1, 1)
DEFAULT_END_DATE = date(2026, 8, 31)
DEFAULT_BAD_FLAG_RATE = 0.02
DEFAULT_REGION_MATCH_RATE = 0.70
DEFAULT_EQUAL_FALSE_NEGATIVE_RATE = 0.10
DEFAULT_SCORE_AUC = 0.75
DEFAULT_REGION_RISK_STRENGTH = 0.8
DEFAULT_SEED = 42
CHUNK_SIZE = 100_000

HEADER = (
    "app_id",
    "client_id",
    "reg_region_code",
    "fact_region_code",
    "equal_flag",
    "flag_6m_30p",
    "score_dubai",
    "issue_dt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--clients", type=int, default=DEFAULT_CLIENTS)
    parser.add_argument("--regions", type=int, default=DEFAULT_REGIONS)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--bad-flag-rate", type=float, default=DEFAULT_BAD_FLAG_RATE)
    parser.add_argument(
        "--region-match-rate", type=float, default=DEFAULT_REGION_MATCH_RATE
    )
    parser.add_argument(
        "--equal-false-negative-rate",
        type=float,
        default=DEFAULT_EQUAL_FALSE_NEGATIVE_RATE,
        help="Share of matching region pairs whose equal_flag is forced to 0.",
    )
    parser.add_argument("--score-auc", type=float, default=DEFAULT_SCORE_AUC)
    parser.add_argument(
        "--region-risk-strength",
        type=float,
        default=DEFAULT_REGION_RISK_STRENGTH,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("applications.csv"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rows < 1 or args.clients < 1:
        raise ValueError("rows and clients must be positive")
    if args.regions < 2:
        raise ValueError("regions must be at least 2")
    if args.start_date > args.end_date:
        raise ValueError("start-date must not be after end-date")
    positive_count = round(args.rows * args.bad_flag_rate)
    if not 0 < positive_count < args.rows:
        raise ValueError("bad-flag-rate must produce both target classes")
    if not 0.5 < args.score_auc < 1:
        raise ValueError("score-auc must be between 0.5 and 1")
    if args.region_risk_strength < 0:
        raise ValueError("region-risk-strength must be non-negative")

    rates = {
        "bad-flag-rate": args.bad_flag_rate,
        "region-match-rate": args.region_match_rate,
        "equal-false-negative-rate": args.equal_false_negative_rate,
    }
    for name, value in rates.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")


def make_score(
    target: np.ndarray,
    rng: np.random.Generator,
    target_auc: float,
) -> tuple[np.ndarray, float]:
    noise = rng.normal(size=target.size)
    low, high = 0.0, 5.0

    for _ in range(18):
        shift = (low + high) / 2
        auc = roc_auc_score(target, noise + shift * target)
        if auc < target_auc:
            low = shift
        else:
            high = shift

    raw_score = noise + ((low + high) / 2) * target
    score = 1 / (1 + np.exp(-raw_score))
    return score, float(roc_auc_score(target, score))


def generate(args: argparse.Namespace) -> None:
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    app_ids = np.arange(1, args.rows + 1, dtype=np.int64)
    client_ids = rng.integers(1, args.clients + 1, size=args.rows)
    reg_regions = rng.integers(0, args.regions, size=args.rows)
    regions_match = rng.random(args.rows) < args.region_match_rate
    mismatch_offsets = rng.integers(1, args.regions, size=args.rows)
    fact_regions = np.where(
        regions_match,
        reg_regions,
        (reg_regions + mismatch_offsets) % args.regions,
    )
    equal_flags = (
        regions_match
        & (rng.random(args.rows) >= args.equal_false_negative_rate)
    ).astype(np.uint8)

    region_effects = rng.normal(
        loc=0.0,
        scale=args.region_risk_strength,
        size=args.regions,
    )
    regional_log_risk = (
        0.6 * region_effects[reg_regions]
        + 0.4 * region_effects[fact_regions]
        - 0.25 * equal_flags
    )
    event_weights = np.exp(regional_log_risk - regional_log_risk.max())
    event_probabilities = event_weights / event_weights.sum()

    bad_flags = np.zeros(args.rows, dtype=np.uint8)
    positive_rows = rng.choice(
        args.rows,
        size=round(args.rows * args.bad_flag_rate),
        replace=False,
        p=event_probabilities,
    )
    bad_flags[positive_rows] = 1
    score_dubai, actual_auc = make_score(bad_flags, rng, args.score_auc)

    day_count = (args.end_date - args.start_date).days + 1
    issue_offsets = rng.integers(0, day_count, size=args.rows)
    issue_dates = (
        np.datetime64(args.start_date)
        + issue_offsets.astype("timedelta64[D]")
    ).astype(str)

    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(HEADER)

        for offset in range(0, args.rows, CHUNK_SIZE):
            end = min(offset + CHUNK_SIZE, args.rows)
            formatted_scores = np.char.mod("%.8f", score_dubai[offset:end])
            writer.writerows(
                zip(
                    app_ids[offset:end],
                    client_ids[offset:end],
                    reg_regions[offset:end],
                    fact_regions[offset:end],
                    equal_flags[offset:end],
                    bad_flags[offset:end],
                    formatted_scores,
                    issue_dates[offset:end],
                )
            )

    print(f"Generated {args.rows:,} rows: {args.output}")
    print(f"flag_6m_30p rate: {bad_flags.mean():.4%}")
    print(f"score_dubai ROC AUC: {actual_auc:.6f}")


if __name__ == "__main__":
    generate(parse_args())

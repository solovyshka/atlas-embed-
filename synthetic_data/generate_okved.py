"""Generate synthetic applications with hierarchical OKVED codes."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HEADER = (
    "app_id",
    "client_id",
    "primary_okved",
    "target",
    "boost_score",
    "issue_dt",
)
REGIMES = ("no_signal", "flat", "hierarchical", "mixed")

DEFAULT_CODES = (
    "01.11",
    "01.13.1",
    "10.71",
    "10.89.9",
    "41.20",
    "43.21",
    "45.20.1",
    "46.46",
    "46.74.2",
    "47.11",
    "47.91.2",
    "49.41",
    "56.10.1",
    "62.01",
    "62.09",
    "64.19",
    "68.20",
    "69.10",
    "70.22",
    "73.11",
    "85.41",
    "86.21",
    "86.90.9",
    "95.11",
)
DEFAULT_OOV_CODES = ("01.11.9", "46.74.9", "62.01.9", "86.90.1")

DEFAULTS = {
    "rows": 20_000,
    "clients": 10_000,
    "start_date": date(2024, 1, 1),
    "end_date": date(2026, 8, 31),
    "bad_rate": 0.05,
    "score_logit_std": 1.35,
    "regime": "mixed",
    "effect_strength": 0.55,
    "temporal_drift": 0.8,
    "oov_rate": 0.0,
    "seed": 42,
}


def load_stats_json(path: str | Path) -> dict[str, Any]:
    """Load aggregate-safe generator statistics from JSON."""
    with Path(path).open(encoding="utf-8") as source:
        stats = json.load(source)
    if not isinstance(stats, dict):
        raise ValueError("stats JSON root must be an object")
    return stats


def _as_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _stat_value(
    explicit: Any,
    stats: Mapping[str, Any],
    section: str,
    key: str,
    default: Any,
) -> Any:
    if explicit is not None:
        return explicit
    return stats.get(section, {}).get(key, default)


def _full_code_distribution(
    stats: Mapping[str, Any],
) -> tuple[tuple[str, ...], np.ndarray]:
    entries = stats.get("okved", {}).get("counts_and_bads_by_full", [])
    valid = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("code"), str)
        and float(entry.get("count", 0)) > 0
    ]
    if valid:
        codes = tuple(entry["code"] for entry in valid)
        weights = np.asarray([float(entry["count"]) for entry in valid])
    else:
        codes = DEFAULT_CODES
        # A Zipf-like head and a substantial tail.
        weights = np.arange(1, len(codes) + 1, dtype=float) ** -1.25
    weights /= weights.sum()
    return codes, weights


def _ancestors(code: str) -> tuple[str, str]:
    parts = code.split(".")
    l1 = parts[0]
    l2 = ".".join(parts[:2]) if len(parts) > 1 else l1
    return l1, l2


def _aggregate_logit_effects(
    entries: Sequence[Mapping[str, Any]],
    default_rate: float,
    *,
    prior_strength: float = 50.0,
) -> dict[str, float]:
    """Estimate shrunk residual log-odds from aggregate bad/expected counts."""
    effects: dict[str, float] = {}
    for entry in entries:
        code = entry.get("code")
        count = float(entry.get("count", 0))
        bads = float(entry.get("bads", -1))
        expected_bads = float(entry.get("expected_bads", count * default_rate))
        if (
            not isinstance(code, str)
            or count <= 0
            or not 0 <= bads <= count
            or not 0 <= expected_bads <= count
        ):
            continue
        expected_rate = np.clip(expected_bads / count, 1e-6, 1 - 1e-6)
        observed_rate = np.clip(
            (bads + prior_strength * expected_rate) / (count + prior_strength),
            1e-6,
            1 - 1e-6,
        )
        effects[code] = float(
            np.log(observed_rate / (1 - observed_rate))
            - np.log(expected_rate / (1 - expected_rate))
        )
    return effects


def _sample_codes(
    issue_offsets: np.ndarray,
    day_count: int,
    core_codes: Sequence[str],
    core_weights: np.ndarray,
    oov_codes: Sequence[str],
    oov_cutoff_offset: int | None,
    oov_rate: float,
    temporal_drift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rows = issue_offsets.size
    result = np.empty(rows, dtype=object)
    eligible_oov = (
        np.zeros(rows, dtype=bool)
        if oov_cutoff_offset is None or not oov_codes or oov_rate == 0
        else issue_offsets >= oov_cutoff_offset
    )
    use_oov = eligible_oov & (rng.random(rows) < oov_rate)
    if use_oov.any():
        result[use_oov] = rng.choice(oov_codes, size=int(use_oov.sum()))

    core_rows = ~use_oov
    # Sampling in a few temporal buckets avoids a rows-by-codes allocation.
    bucket_count = min(16, day_count)
    buckets = np.minimum(
        issue_offsets * bucket_count // max(day_count, 1),
        bucket_count - 1,
    )
    code_trend = np.linspace(-1.0, 1.0, len(core_codes))
    for bucket in np.unique(buckets[core_rows]):
        mask = core_rows & (buckets == bucket)
        time_position = (float(bucket) + 0.5) / bucket_count - 0.5
        weights = core_weights * np.exp(
            temporal_drift * code_trend * time_position
        )
        weights /= weights.sum()
        result[mask] = rng.choice(
            core_codes,
            size=int(mask.sum()),
            p=weights,
        )
    return result.astype(str)


def _okved_delta(
    codes: np.ndarray,
    regime: str,
    effect_strength: float,
    rng: np.random.Generator,
    stats: Mapping[str, Any] | None = None,
    bad_rate: float = DEFAULTS["bad_rate"],
) -> np.ndarray:
    if regime == "no_signal":
        return np.zeros(codes.size)

    unique_codes = sorted(set(codes))
    l1_values = sorted({_ancestors(code)[0] for code in unique_codes})
    l2_values = sorted({_ancestors(code)[1] for code in unique_codes})
    l1_effect = dict(
        zip(l1_values, rng.normal(0, effect_strength, len(l1_values)))
    )
    l2_effect = dict(
        zip(l2_values, rng.normal(0, effect_strength * 0.65, len(l2_values)))
    )
    leaf_effect = dict(
        zip(unique_codes, rng.normal(0, effect_strength, len(unique_codes)))
    )
    okved_stats = (stats or {}).get("okved", {})
    aggregate_l1 = _aggregate_logit_effects(
        okved_stats.get("counts_and_bads_by_l1", []), bad_rate
    )
    aggregate_l2 = _aggregate_logit_effects(
        okved_stats.get("counts_and_bads_by_l2", []), bad_rate
    )
    aggregate_full = _aggregate_logit_effects(
        okved_stats.get("counts_and_bads_by_full", []), bad_rate
    )
    has_aggregate_effects = bool(aggregate_l1 or aggregate_l2 or aggregate_full)

    hierarchy_total: dict[str, float] = {}
    full_total: dict[str, float] = {}
    for code in unique_codes:
        l1, l2 = _ancestors(code)
        l1_value = aggregate_l1.get(l1, l1_effect[l1])
        l2_value = aggregate_l2.get(l2, l1_value + l2_effect[l2])
        hierarchy_total[code] = l2_value
        full_total[code] = aggregate_full.get(
            code, l2_value + leaf_effect[code]
        )

    if regime == "hierarchical":
        delta = np.fromiter(
            (hierarchy_total[code] for code in codes),
            dtype=float,
            count=codes.size,
        )
    elif regime == "flat":
        delta = np.fromiter(
            (
                aggregate_full.get(code, leaf_effect[code])
                for code in codes
            ),
            dtype=float,
            count=codes.size,
        )
    else:
        delta = np.fromiter(
            (full_total[code] for code in codes),
            dtype=float,
            count=codes.size,
        )
    if regime == "mixed" and not has_aggregate_effects:
        delta *= 0.7
    return delta


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value)
    return np.where(
        value >= 0,
        1 / (1 + np.exp(-value)),
        np.exp(value) / (1 + np.exp(value)),
    )


def _calibrate_intercept(
    score_component: np.ndarray,
    delta: np.ndarray,
    bad_rate: float,
) -> float:
    low, high = -20.0, 20.0
    for _ in range(60):
        midpoint = (low + high) / 2
        if _sigmoid(midpoint + score_component + delta).mean() < bad_rate:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def post_check_metadata(
    data: pd.DataFrame,
    *,
    oov_codes: Sequence[str] = (),
    oov_cutoff: str | date | None = None,
) -> dict[str, Any]:
    """Return JSON-safe quality checks and hierarchy counts."""
    missing = set(HEADER) - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    codes = data["primary_okved"].astype(str)
    targets = data["target"].to_numpy()
    scores = data["boost_score"].to_numpy()
    auc = (
        float(roc_auc_score(targets, scores))
        if np.unique(targets).size == 2
        else None
    )
    l1 = codes.str.split(".").str[0]
    l2 = codes.str.split(".").str[:2].str.join(".")
    oov_mask = codes.isin(set(oov_codes))
    issue_dates = pd.to_datetime(data["issue_dt"])
    cutoff = pd.Timestamp(_as_date(oov_cutoff)) if oov_cutoff else None

    def counts(series: pd.Series) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in series.value_counts().sort_index().items()
        }

    metadata: dict[str, Any] = {
        "rows": int(len(data)),
        "clients": int(data["client_id"].nunique()),
        "achieved_bad_rate": float(np.mean(targets)),
        "base_auc": auc,
        "counts": {
            "l1": counts(l1),
            "l2": counts(l2),
            "full": counts(codes),
        },
        "depth_counts": counts(codes.str.count(r"\.") + 1),
        "oov": {
            "count": int(oov_mask.sum()),
            "rate": float(oov_mask.mean()),
            "before_cutoff_count": 0,
            "after_cutoff_count": int(oov_mask.sum()),
        },
    }
    if cutoff is not None:
        metadata["oov"]["before_cutoff_count"] = int(
            (oov_mask & (issue_dates < cutoff)).sum()
        )
        metadata["oov"]["after_cutoff_count"] = int(
            (oov_mask & (issue_dates >= cutoff)).sum()
        )
        metadata["oov"]["cutoff"] = cutoff.date().isoformat()
    return metadata


def generate_okved_data(
    *,
    rows: int | None = None,
    clients: int | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    bad_rate: float | None = None,
    score_logit_std: float | None = None,
    regime: str | None = None,
    effect_strength: float | None = None,
    temporal_drift: float | None = None,
    oov_rate: float | None = None,
    oov_cutoff: str | date | None = None,
    seed: int | None = None,
    stats: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a deterministic synthetic dataset and post-check metadata."""
    stats = stats or {}
    rows = int(_stat_value(rows, stats, "dataset", "rows", DEFAULTS["rows"]))
    clients = int(
        _stat_value(clients, stats, "dataset", "clients", DEFAULTS["clients"])
    )
    start_date = _as_date(
        _stat_value(
            start_date, stats, "dataset", "start_date", DEFAULTS["start_date"]
        )
    )
    end_date = _as_date(
        _stat_value(end_date, stats, "dataset", "end_date", DEFAULTS["end_date"])
    )
    bad_rate = float(
        _stat_value(bad_rate, stats, "dataset", "bad_rate", DEFAULTS["bad_rate"])
    )
    score_distribution = stats.get("score", {}).get("distribution", {})
    if score_logit_std is None:
        score_logit_std = float(
            score_distribution.get("logit_std", DEFAULTS["score_logit_std"])
        )
    regime = regime or DEFAULTS["regime"]
    effect_strength = float(
        DEFAULTS["effect_strength"] if effect_strength is None else effect_strength
    )
    temporal = stats.get("temporal", {})
    temporal_drift = float(
        temporal.get("drift_strength", DEFAULTS["temporal_drift"])
        if temporal_drift is None
        else temporal_drift
    )
    oov_rate = float(
        temporal.get("new_code_rate_after_cutoff", DEFAULTS["oov_rate"])
        if oov_rate is None
        else oov_rate
    )
    if oov_cutoff is None:
        oov_cutoff = temporal.get("oov_cutoff")
    oov_cutoff = _as_date(oov_cutoff)
    seed = int(DEFAULTS["seed"] if seed is None else seed)

    if rows < 2 or clients < 1:
        raise ValueError("rows must be at least 2 and clients must be positive")
    if start_date is None or end_date is None or start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if not 0 < bad_rate < 1:
        raise ValueError("bad_rate must be between 0 and 1")
    if score_logit_std <= 0 or effect_strength < 0 or temporal_drift < 0:
        raise ValueError("dispersion, effect strength and drift must be non-negative")
    if not 0 <= oov_rate <= 1:
        raise ValueError("oov_rate must be between 0 and 1")
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    if oov_rate > 0 and oov_cutoff is None:
        raise ValueError("oov_cutoff is required when oov_rate is positive")
    if oov_cutoff is not None and not start_date <= oov_cutoff <= end_date:
        raise ValueError("oov_cutoff must fall within the date range")

    rng = np.random.default_rng(seed)
    day_count = (end_date - start_date).days + 1
    issue_offsets = np.sort(rng.integers(0, day_count, size=rows))
    issue_dates = (
        np.datetime64(start_date)
        + issue_offsets.astype("timedelta64[D]")
    ).astype(str)

    core_codes, core_weights = _full_code_distribution(stats)
    configured_oov = temporal.get("oov_leaf_codes", DEFAULT_OOV_CODES)
    oov_codes = tuple(code for code in configured_oov if code not in core_codes)
    cutoff_offset = (
        (oov_cutoff - start_date).days if oov_cutoff is not None else None
    )
    codes = _sample_codes(
        issue_offsets,
        day_count,
        core_codes,
        core_weights,
        oov_codes,
        cutoff_offset,
        oov_rate,
        temporal_drift,
        rng,
    )
    delta = _okved_delta(
        codes,
        regime,
        effect_strength,
        rng,
        stats=stats,
        bad_rate=bad_rate,
    )

    # The ready score is generated before the target. Only then is OKVED risk
    # added to its logit and the Bernoulli target sampled.
    score_component = rng.normal(0, score_logit_std, size=rows)
    intercept = _calibrate_intercept(score_component, delta, bad_rate)
    boost_score = _sigmoid(intercept + score_component)
    target_probability = _sigmoid(
        np.log(boost_score / (1 - boost_score)) + delta
    )
    target = rng.binomial(1, target_probability).astype(np.uint8)

    data = pd.DataFrame(
        {
            "app_id": np.arange(1, rows + 1, dtype=np.int64),
            "client_id": rng.integers(1, clients + 1, size=rows),
            "primary_okved": codes,
            "target": target,
            "boost_score": boost_score,
            "issue_dt": issue_dates,
        },
        columns=HEADER,
    )
    metadata = post_check_metadata(
        data,
        oov_codes=oov_codes,
        oov_cutoff=oov_cutoff,
    )
    metadata.update(
        {
            "requested_bad_rate": bad_rate,
            "regime": regime,
            "seed": seed,
            "calibrated_intercept": float(intercept),
            "mean_target_probability": float(target_probability.mean()),
        }
    )
    return data, metadata


# Short alias for callers that already know the module's domain.
generate_data = generate_okved_data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--clients", type=int)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--bad-rate", type=float)
    parser.add_argument("--score-logit-std", type=float)
    parser.add_argument("--regime", choices=REGIMES)
    parser.add_argument("--effect-strength", type=float)
    parser.add_argument("--temporal-drift", type=float)
    parser.add_argument("--oov-rate", type=float)
    parser.add_argument("--oov-cutoff")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stats-json", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("okved_applications.csv"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stats = load_stats_json(args.stats_json) if args.stats_json else None
    data, metadata = generate_okved_data(
        rows=args.rows,
        clients=args.clients,
        start_date=args.start_date,
        end_date=args.end_date,
        bad_rate=args.bad_rate,
        score_logit_std=args.score_logit_std,
        regime=args.regime,
        effect_strength=args.effect_strength,
        temporal_drift=args.temporal_drift,
        oov_rate=args.oov_rate,
        oov_cutoff=args.oov_cutoff,
        seed=args.seed,
        stats=stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, float_format="%.8f")
    print(f"Generated {len(data):,} rows: {args.output}")
    print(f"target rate: {metadata['achieved_bad_rate']:.4%}")
    print(f"boost_score ROC AUC: {metadata['base_auc']:.6f}")


if __name__ == "__main__":
    main()

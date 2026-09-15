import re

import numpy as np
import pandas as pd
import pytest

from synthetic_data.generate_okved import HEADER, generate_okved_data


def test_generation_is_deterministic() -> None:
    kwargs = {
        "rows": 1_000,
        "clients": 300,
        "start_date": "2025-01-01",
        "end_date": "2025-06-30",
        "seed": 17,
    }
    first, first_metadata = generate_okved_data(**kwargs)
    second, second_metadata = generate_okved_data(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert first_metadata == second_metadata


def test_schema_and_formats() -> None:
    data, metadata = generate_okved_data(rows=2_000, seed=3)

    assert tuple(data.columns) == HEADER
    assert data["app_id"].tolist() == list(range(1, len(data) + 1))
    assert set(data["target"].unique()) <= {0, 1}
    assert data["boost_score"].between(0, 1, inclusive="neither").all()
    assert data["primary_okved"].map(
        lambda code: bool(re.fullmatch(r"\d{2}(?:\.\d{2})(?:\.\d+)?", code))
    ).all()
    assert pd.to_datetime(data["issue_dt"], format="%Y-%m-%d").notna().all()
    assert set(metadata["depth_counts"]) >= {"2", "3"}
    assert metadata["base_auc"] > 0.6


def test_achieved_bad_rate_is_close_to_request() -> None:
    requested_rate = 0.08
    _, metadata = generate_okved_data(
        rows=30_000,
        bad_rate=requested_rate,
        regime="mixed",
        seed=11,
    )

    assert metadata["mean_target_probability"] == pytest.approx(
        requested_rate, abs=1e-10
    )
    assert metadata["achieved_bad_rate"] == pytest.approx(
        requested_rate, abs=0.008
    )


def test_no_signal_has_no_hierarchy_residual() -> None:
    data, _ = generate_okved_data(
        rows=40_000,
        bad_rate=0.12,
        regime="no_signal",
        temporal_drift=1.2,
        seed=29,
    )
    l1 = data["primary_okved"].str.split(".").str[0]
    residual = data["target"] - data["boost_score"]
    grouped = pd.DataFrame({"l1": l1, "residual": residual}).groupby("l1")[
        "residual"
    ]
    sufficiently_large = grouped.count() >= 500

    assert np.abs(grouped.mean()[sufficiently_large]).max() < 0.035


def test_oov_codes_appear_only_after_cutoff() -> None:
    cutoff = "2025-09-01"
    data, metadata = generate_okved_data(
        rows=8_000,
        start_date="2025-01-01",
        end_date="2025-12-31",
        oov_rate=0.25,
        oov_cutoff=cutoff,
        seed=7,
    )
    known_oov = {"01.11.9", "46.74.9", "62.01.9", "86.90.1"}
    oov_rows = data["primary_okved"].isin(known_oov)

    assert oov_rows.any()
    assert (data.loc[oov_rows, "issue_dt"] >= cutoff).all()
    assert metadata["oov"]["before_cutoff_count"] == 0
    assert metadata["oov"]["after_cutoff_count"] == int(oov_rows.sum())


def test_aggregate_expected_bads_control_flat_residual_signal() -> None:
    stats = {
        "okved": {
            "counts_and_bads_by_full": [
                {
                    "code": "46.74.2",
                    "count": 2_000,
                    "bads": 400,
                    "expected_bads": 160,
                },
                {
                    "code": "69.10",
                    "count": 2_000,
                    "bads": 40,
                    "expected_bads": 160,
                },
            ]
        }
    }
    data, _ = generate_okved_data(
        rows=30_000,
        bad_rate=0.08,
        regime="flat",
        effect_strength=0.0,
        temporal_drift=0.0,
        stats=stats,
        seed=31,
    )
    residual_by_code = (
        data.assign(residual=data["target"] - data["boost_score"])
        .groupby("primary_okved")["residual"]
        .mean()
    )

    assert residual_by_code["46.74.2"] > residual_by_code["69.10"] + 0.05

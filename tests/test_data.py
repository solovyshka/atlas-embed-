from pathlib import Path

import pytest

from regional_score import load_applications_csv, temporal_split


def write_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "reg_region_code,fact_region_code,equal_flag,flag_6m_30p,score_dubai,issue_dt",
                "1,1,1,0,0.10,2025-12-31",
                "2,3,0,1,0.80,2026-01-01",
                "4,4,0,0,0.20,2026-04-30",
                "5,6,0,1,0.90,2026-05-01",
            )
        ),
        encoding="utf-8",
    )


def test_load_and_temporal_split(tmp_path: Path) -> None:
    csv_path = tmp_path / "applications.csv"
    write_csv(csv_path)

    data = load_applications_csv(csv_path)
    datasets = temporal_split(data)

    assert data.region_ids.shape == (4, 2)
    assert data.base_scores.tolist() == pytest.approx([0.1, 0.8, 0.2, 0.9])
    assert len(datasets.train) == 1
    assert len(datasets.validation) == 2
    assert len(datasets.test) == 1


def test_loader_rejects_non_binary_equal_flag(tmp_path: Path) -> None:
    csv_path = tmp_path / "applications.csv"
    write_csv(csv_path)
    contents = csv_path.read_text(encoding="utf-8").replace(
        "2,3,0,1,0.80",
        "2,3,2,1,0.80",
    )
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="equal_flag"):
        load_applications_csv(csv_path)

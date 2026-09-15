from pathlib import Path

import pytest

from okved_score import (
    UNK_INDEX,
    HierarchicalOkvedVocabulary,
    load_applications_csv,
    temporal_split,
)


def write_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "client_id,issue_dt,primary_okved,boost_score,target",
                "001,2025-12-31, 46.74.2 ,0.10,0",
                "002,2026-01-01,46.74.9,0.80,1",
                "002,2026-04-30,46.74,0.20,0",
                "003,2026-05-01,69.10,0.90,1",
            )
        ),
        encoding="utf-8",
    )


def test_load_and_temporal_split_preserve_raw_codes_and_client_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "applications.csv"
    write_csv(csv_path)

    data = load_applications_csv(csv_path)
    splits = temporal_split(data)

    assert data.client_ids == ("001", "002", "002", "003")
    assert data.primary_okved == ("46.74.2", "46.74.9", "46.74", "69.10")
    assert data.boost_scores.tolist() == pytest.approx([0.1, 0.8, 0.2, 0.9])
    assert len(splits.train) == 1
    assert len(splits.validation) == 2
    assert len(splits.test) == 1


def test_vocabulary_is_fit_after_split_without_validation_leakage(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "applications.csv"
    write_csv(csv_path)
    splits = temporal_split(load_applications_csv(csv_path))

    vocabulary = HierarchicalOkvedVocabulary.fit(
        splits.train.primary_okved,
        max_levels=3,
    )
    train_id = vocabulary.encode(splits.train.primary_okved[0])
    validation_id = vocabulary.encode(splits.validation.primary_okved[0])

    assert validation_id[:2] == train_id[:2]
    assert validation_id[2] == UNK_INDEX


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("0.80,1", "1.20,1", "boost_score"),
        ("0.80,1", "0.80,2", "target"),
        ("2026-01-01", "not-a-date", "issue_dt"),
        ("46.74.9", "46..9", "primary_okved"),
        ("002,2026-01-01", "   ,2026-01-01", "client_id"),
    ),
)
def test_loader_rejects_invalid_values(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    csv_path = tmp_path / "applications.csv"
    write_csv(csv_path)
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace(old, new, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_applications_csv(csv_path)


def test_loader_rejects_missing_columns_and_empty_files(tmp_path: Path) -> None:
    missing_column = tmp_path / "missing.csv"
    missing_column.write_text(
        "client_id,issue_dt,primary_okved,boost_score\n"
        "1,2025-01-01,69.10,0.5\n",
        encoding="utf-8",
    )
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "client_id,issue_dt,primary_okved,boost_score,target\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target"):
        load_applications_csv(missing_column)
    with pytest.raises(ValueError, match="at least one row"):
        load_applications_csv(empty)

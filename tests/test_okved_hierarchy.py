import pytest

from okved_score import infer_max_levels, normalize_okved, okved_path


def test_normalize_and_build_cumulative_path() -> None:
    assert normalize_okved(" 046 . 07 . 02 ") == "046.07.02"
    assert okved_path("69.10") == ("69", "69.10")
    assert okved_path("46.74.2") == ("46", "46.74", "46.74.2")


@pytest.mark.parametrize(
    "code",
    ("", " ", ".", ".69", "69.", "69..10", "6 9.10", "69.A", "69.-10"),
)
def test_rejects_malformed_codes(code: str) -> None:
    with pytest.raises(ValueError, match="malformed"):
        normalize_okved(code)


def test_max_levels_can_be_inferred_or_enforced() -> None:
    assert infer_max_levels(code for code in ("69.10", "46.74.2")) == 3

    with pytest.raises(ValueError, match="at most 2"):
        okved_path("46.74.2", max_levels=2)
    with pytest.raises(ValueError, match="positive"):
        okved_path("69.10", max_levels=0)

from __future__ import annotations

from collections.abc import Iterable


def normalize_okved(code: str) -> str:
    if not isinstance(code, str):
        raise ValueError("OKVED code must be a string")

    segments = code.strip().split(".")
    normalized = tuple(segment.strip() for segment in segments)
    if not normalized or any(
        not segment or not segment.isascii() or not segment.isdigit()
        for segment in normalized
    ):
        raise ValueError(f"malformed OKVED code: {code!r}")
    return ".".join(normalized)


def okved_path(code: str, max_levels: int | None = None) -> tuple[str, ...]:
    normalized = normalize_okved(code)
    segments = normalized.split(".")
    if max_levels is not None:
        if max_levels < 1:
            raise ValueError("max_levels must be positive")
        if len(segments) > max_levels:
            raise ValueError(
                f"OKVED code has {len(segments)} levels, expected at most {max_levels}"
            )
    return tuple(".".join(segments[:index]) for index in range(1, len(segments) + 1))


def infer_max_levels(codes: Iterable[str]) -> int:
    depths = [len(okved_path(code)) for code in codes]
    if not depths:
        raise ValueError("at least one OKVED code is required")
    return max(depths)

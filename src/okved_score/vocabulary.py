from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .hierarchy import infer_max_levels, normalize_okved, okved_path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
RARE_TOKEN = "<RARE>"
PAD_INDEX = 0
UNK_INDEX = 1
RARE_INDEX = 2
SPECIAL_TOKENS = {
    PAD_TOKEN: PAD_INDEX,
    UNK_TOKEN: UNK_INDEX,
    RARE_TOKEN: RARE_INDEX,
}


def _validate_rare_threshold(rare_threshold: int) -> None:
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be positive")


def _build_level(
    values: Iterable[str],
    rare_threshold: int,
) -> tuple[dict[str, int], frozenset[str]]:
    counts = Counter(values)
    frequent = sorted(code for code, count in counts.items() if count >= rare_threshold)
    rare = frozenset(code for code, count in counts.items() if count < rare_threshold)
    mapping = dict(SPECIAL_TOKENS)
    mapping.update((code, index) for index, code in enumerate(frequent, start=3))
    return mapping, rare


def _encode(code: str, mapping: Mapping[str, int], rare_codes: frozenset[str]) -> int:
    if code in mapping:
        return mapping[code]
    if code in rare_codes:
        return RARE_INDEX
    return UNK_INDEX


@dataclass(frozen=True)
class FlatOkvedVocabulary:
    token_to_index: dict[str, int]
    rare_codes: frozenset[str]
    rare_threshold: int
    max_levels: int

    @classmethod
    def fit(
        cls,
        codes: Iterable[str],
        rare_threshold: int = 1,
        max_levels: int | None = None,
    ) -> FlatOkvedVocabulary:
        _validate_rare_threshold(rare_threshold)
        normalized = [normalize_okved(code) for code in codes]
        if not normalized:
            raise ValueError("at least one train OKVED code is required")

        inferred_levels = infer_max_levels(normalized)
        levels = inferred_levels if max_levels is None else max_levels
        for code in normalized:
            okved_path(code, levels)

        mapping, rare_codes = _build_level(normalized, rare_threshold)
        return cls(mapping, rare_codes, rare_threshold, levels)

    def encode(self, code: str) -> int:
        normalized = normalize_okved(code)
        okved_path(normalized, self.max_levels)
        return _encode(normalized, self.token_to_index, self.rare_codes)

    def transform(self, codes: Iterable[str]) -> list[int]:
        return [self.encode(code) for code in codes]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_index)

    def __len__(self) -> int:
        return self.vocab_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "flat",
            "rare_threshold": self.rare_threshold,
            "max_levels": self.max_levels,
            "token_to_index": dict(self.token_to_index),
            "rare_codes": sorted(self.rare_codes),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> FlatOkvedVocabulary:
        if state.get("type") != "flat":
            raise ValueError("state does not contain a flat OKVED vocabulary")
        return cls(
            token_to_index={
                str(token): int(index)
                for token, index in state["token_to_index"].items()
            },
            rare_codes=frozenset(str(code) for code in state["rare_codes"]),
            rare_threshold=int(state["rare_threshold"]),
            max_levels=int(state["max_levels"]),
        )


@dataclass(frozen=True)
class HierarchicalOkvedVocabulary:
    level_token_to_index: tuple[dict[str, int], ...]
    level_rare_codes: tuple[frozenset[str], ...]
    rare_threshold: int
    max_levels: int

    @classmethod
    def fit(
        cls,
        codes: Iterable[str],
        rare_threshold: int = 1,
        max_levels: int | None = None,
    ) -> HierarchicalOkvedVocabulary:
        _validate_rare_threshold(rare_threshold)
        normalized = [normalize_okved(code) for code in codes]
        if not normalized:
            raise ValueError("at least one train OKVED code is required")

        inferred_levels = infer_max_levels(normalized)
        levels = inferred_levels if max_levels is None else max_levels
        paths = [okved_path(code, levels) for code in normalized]
        level_values = (
            (path[level] for path in paths if len(path) > level)
            for level in range(levels)
        )
        built_levels = [_build_level(values, rare_threshold) for values in level_values]
        return cls(
            level_token_to_index=tuple(mapping for mapping, _ in built_levels),
            level_rare_codes=tuple(rare for _, rare in built_levels),
            rare_threshold=rare_threshold,
            max_levels=levels,
        )

    def encode(self, code: str) -> tuple[int, ...]:
        path = okved_path(code, self.max_levels)
        encoded = []
        for level in range(self.max_levels):
            if level >= len(path):
                encoded.append(PAD_INDEX)
                continue
            encoded.append(
                _encode(
                    path[level],
                    self.level_token_to_index[level],
                    self.level_rare_codes[level],
                )
            )
        return tuple(encoded)

    def transform(self, codes: Iterable[str]) -> list[tuple[int, ...]]:
        return [self.encode(code) for code in codes]

    @property
    def vocab_sizes(self) -> tuple[int, ...]:
        return tuple(len(mapping) for mapping in self.level_token_to_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "hierarchical",
            "rare_threshold": self.rare_threshold,
            "max_levels": self.max_levels,
            "level_token_to_index": [
                dict(mapping) for mapping in self.level_token_to_index
            ],
            "level_rare_codes": [
                sorted(codes) for codes in self.level_rare_codes
            ],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> HierarchicalOkvedVocabulary:
        if state.get("type") != "hierarchical":
            raise ValueError("state does not contain a hierarchical OKVED vocabulary")
        return cls(
            level_token_to_index=tuple(
                {
                    str(token): int(index)
                    for token, index in mapping.items()
                }
                for mapping in state["level_token_to_index"]
            ),
            level_rare_codes=tuple(
                frozenset(str(code) for code in codes)
                for codes in state["level_rare_codes"]
            ),
            rare_threshold=int(state["rare_threshold"]),
            max_levels=int(state["max_levels"]),
        )

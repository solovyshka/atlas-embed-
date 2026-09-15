import json

import pytest

from okved_score import (
    PAD_INDEX,
    RARE_INDEX,
    UNK_INDEX,
    FlatOkvedVocabulary,
    HierarchicalOkvedVocabulary,
)


TRAIN_CODES = ("46.74.2", "46.74.2", "46.75.1", "69.10")


def test_flat_vocabulary_distinguishes_rare_and_unseen_codes() -> None:
    vocabulary = FlatOkvedVocabulary.fit(TRAIN_CODES, rare_threshold=2)

    frequent_id = vocabulary.encode("46.74.2")
    assert frequent_id >= 3
    assert vocabulary.encode("46.75.1") == RARE_INDEX
    assert vocabulary.encode("46.74.9") == UNK_INDEX
    assert vocabulary.max_levels == 3
    assert vocabulary.vocab_size == len(vocabulary)


def test_hierarchical_vocabulary_preserves_known_ancestors_and_padding() -> None:
    vocabulary = HierarchicalOkvedVocabulary.fit(
        TRAIN_CODES,
        rare_threshold=2,
    )

    known = vocabulary.encode("46.74.2")
    unseen_leaf = vocabulary.encode("46.74.9")
    short_code = vocabulary.encode("46.74")

    assert known[0] >= 3
    assert known[1] >= 3
    assert known[2] >= 3
    assert unseen_leaf[:2] == known[:2]
    assert unseen_leaf[2] == UNK_INDEX
    assert short_code == (*known[:2], PAD_INDEX)
    assert vocabulary.encode("46.75.1")[1:] == (RARE_INDEX, RARE_INDEX)
    assert vocabulary.vocab_sizes == tuple(
        len(mapping) for mapping in vocabulary.level_token_to_index
    )


def test_vocabularies_round_trip_through_plain_dicts() -> None:
    flat = FlatOkvedVocabulary.fit(TRAIN_CODES, rare_threshold=2)
    hierarchical = HierarchicalOkvedVocabulary.fit(TRAIN_CODES, rare_threshold=2)

    flat_state = json.loads(json.dumps(flat.to_dict()))
    hierarchical_state = json.loads(json.dumps(hierarchical.to_dict()))

    restored_flat = FlatOkvedVocabulary.from_dict(flat_state)
    restored_hierarchical = HierarchicalOkvedVocabulary.from_dict(
        hierarchical_state
    )
    assert restored_flat.transform(TRAIN_CODES) == flat.transform(TRAIN_CODES)
    assert restored_hierarchical.transform(TRAIN_CODES) == hierarchical.transform(
        TRAIN_CODES
    )


def test_explicit_max_levels_validates_train_schema() -> None:
    vocabulary = HierarchicalOkvedVocabulary.fit(("69.10",), max_levels=3)
    assert vocabulary.encode("69.10")[-1] == PAD_INDEX

    with pytest.raises(ValueError, match="at most 2"):
        FlatOkvedVocabulary.fit(("46.74.2",), max_levels=2)

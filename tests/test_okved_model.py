import pytest
import torch

from okved_score.model import (
    FlatOkvedResidualScorer,
    HierarchicalOkvedResidualScorer,
    count_trainable_parameters,
)


def test_flat_forward_is_zero_initialized_and_backpropagates() -> None:
    model = FlatOkvedResidualScorer(vocab_size=12, dropout=0.0)
    boost_score = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    okved_ids = torch.tensor([0, 1, 3, 7, 11], dtype=torch.long)

    logits = model(boost_score, okved_ids)
    logits[1:-1].sum().backward()

    assert logits.shape == (5, 1)
    torch.testing.assert_close(
        torch.sigmoid(logits).squeeze(1), boost_score
    )
    assert torch.count_nonzero(model.residual_head.weight) == 0
    assert torch.count_nonzero(model.residual_head.bias) == 0
    assert model.leaf_embedding.padding_idx == 0
    assert model.leaf_embedding.weight.grad is not None
    assert model.residual_head.weight.grad is not None
    assert model.alpha.grad is not None


def test_hierarchical_forward_accepts_column_boost_score() -> None:
    model = HierarchicalOkvedResidualScorer([5, 7, 9], dropout=0.0)
    boost_score = torch.tensor([[0.25], [0.75]])
    okved_ids = torch.tensor([[1, 2, 3], [4, 0, 0]], dtype=torch.long)

    logits = model(boost_score, okved_ids)

    assert logits.shape == (2, 1)
    torch.testing.assert_close(torch.sigmoid(logits), boost_score)
    assert all(table.padding_idx == 0 for table in model.level_embeddings)


def test_hierarchy_fuses_only_non_pad_levels() -> None:
    model = HierarchicalOkvedResidualScorer(
        [3, 4], embedding_dim=2, hidden_dim=4, output_dim=2, dropout=0.0
    )
    with torch.no_grad():
        for table in model.level_embeddings:
            table.weight.zero_()
        model.level_embeddings[0].weight[1].fill_(2.0)
        model.level_embeddings[1].weight[2].fill_(4.0)

    features = model.okved_features(
        torch.tensor([[1, 2], [1, 0]], dtype=torch.long)
    )

    torch.testing.assert_close(
        features, torch.tensor([[3.0, 3.0], [2.0, 2.0]])
    )


def test_encode_has_compact_shape_and_is_deterministic_in_eval() -> None:
    model = FlatOkvedResidualScorer(
        vocab_size=8, output_dim=6, dropout=0.8
    ).eval()
    okved_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    first = model.encode(okved_ids)
    second = model.encode(okved_ids)

    assert first.shape == (3, 6)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_flat_and_hierarchical_towers_have_fair_parameter_counts() -> None:
    flat = FlatOkvedResidualScorer(vocab_size=20)
    hierarchical = HierarchicalOkvedResidualScorer([3, 7, 10])

    assert count_trainable_parameters(flat) == count_trainable_parameters(
        hierarchical
    )
    assert type(flat.encoder) is type(hierarchical.encoder)
    assert flat.residual_head.weight.shape == hierarchical.residual_head.weight.shape


@pytest.mark.parametrize(
    ("ids", "error"),
    [
        (torch.tensor([[1]], dtype=torch.long), ValueError),
        (torch.tensor([1.0]), TypeError),
        (torch.tensor([-1], dtype=torch.long), ValueError),
        (torch.tensor([5], dtype=torch.long), ValueError),
    ],
)
def test_flat_rejects_invalid_ids(
    ids: torch.Tensor, error: type[Exception]
) -> None:
    model = FlatOkvedResidualScorer(vocab_size=5)

    with pytest.raises(error):
        model.encode(ids)


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([[0, 0]], dtype=torch.long),
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[1, 4]], dtype=torch.long),
        torch.tensor([[1.0, 2.0]]),
    ],
)
def test_hierarchy_rejects_invalid_ids(ids: torch.Tensor) -> None:
    model = HierarchicalOkvedResidualScorer([3, 4])

    with pytest.raises((TypeError, ValueError)):
        model.encode(ids)


@pytest.mark.parametrize(
    "boost_score",
    [
        torch.tensor([1]),
        torch.tensor([[0.2, 0.3]]),
        torch.tensor([-0.1]),
        torch.tensor([1.1]),
        torch.tensor([float("nan")]),
    ],
)
def test_forward_rejects_invalid_boost_scores(
    boost_score: torch.Tensor,
) -> None:
    model = FlatOkvedResidualScorer(vocab_size=5)

    with pytest.raises((TypeError, ValueError)):
        model(boost_score, torch.tensor([1], dtype=torch.long))


def test_forward_rejects_different_batch_sizes() -> None:
    model = FlatOkvedResidualScorer(vocab_size=5)

    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.tensor([0.2, 0.3]), torch.tensor([1], dtype=torch.long))

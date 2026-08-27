import pytest
import torch

from regional_score import RegionalResidualScorer, count_trainable_parameters


def test_default_model_has_expected_parameter_count() -> None:
    model = RegionalResidualScorer()

    assert count_trainable_parameters(model) == 986
    assert model.embedding.weight.shape == (70, 4)


def test_forward_preserves_batch_and_supports_backpropagation() -> None:
    model = RegionalResidualScorer()
    base_scores = torch.tensor([0.2, 0.4, 0.8])
    region_ids = torch.tensor(
        [[0, 0], [12, 7], [69, 4]], dtype=torch.long
    )
    equal_flags = torch.tensor([1.0, 0.0, 0.0])

    logits = model(base_scores, region_ids, equal_flags)
    logits.sum().backward()

    assert logits.shape == (3, 1)
    assert torch.sigmoid(logits).squeeze(1).tolist() == pytest.approx(
        base_scores.tolist()
    )
    assert model.embedding.weight.grad is not None
    assert model.alpha.grad is not None


def test_region_features_have_expected_width() -> None:
    model = RegionalResidualScorer()
    region_ids = torch.tensor([[1, 2], [4, 4]], dtype=torch.long)
    equal_flags = torch.tensor([0, 1])

    features = model.region_features(region_ids, equal_flags)

    assert features.shape == (2, 17)
    assert features[:, -1].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    ("region_ids", "error"),
    [
        (torch.tensor([1, 2]), ValueError),
        (torch.tensor([[1.0, 2.0]]), TypeError),
        (torch.tensor([[0, 70]]), ValueError),
    ],
)
def test_invalid_region_ids_are_rejected(
    region_ids: torch.Tensor, error: type[Exception]
) -> None:
    model = RegionalResidualScorer()

    with pytest.raises(error):
        model.region_features(region_ids, torch.tensor([0]))

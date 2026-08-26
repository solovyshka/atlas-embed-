import pytest
import torch

from regional_score import RegionalResidualScorer, count_trainable_parameters


def test_default_model_has_expected_parameter_count() -> None:
    model = RegionalResidualScorer()

    assert count_trainable_parameters(model) == 1_546
    assert model.embedding.weight.shape == (72, 4)


def test_forward_preserves_batch_and_supports_backpropagation() -> None:
    model = RegionalResidualScorer()
    base_logits = torch.tensor([0.2, -0.4, 1.1])
    region_ids = torch.tensor(
        [[0, 0, 0], [12, 7, 12], [70, 4, 71]], dtype=torch.long
    )

    logits = model(base_logits, region_ids)
    logits.sum().backward()

    assert logits.shape == (3, 1)
    assert model.embedding.weight.grad is not None
    assert model.alpha.grad is not None


def test_region_features_have_expected_width() -> None:
    model = RegionalResidualScorer()
    region_ids = torch.tensor([[1, 2, 3], [4, 4, 4]], dtype=torch.long)

    features = model.region_features(region_ids)

    assert features.shape == (2, 40)
    assert features[:, -1].tolist() == [3.0, 1.0]


@pytest.mark.parametrize(
    ("region_ids", "error"),
    [
        (torch.tensor([1, 2, 3]), ValueError),
        (torch.tensor([[1.0, 2.0, 3.0]]), TypeError),
        (torch.tensor([[0, 1, 72]]), ValueError),
    ],
)
def test_invalid_region_ids_are_rejected(
    region_ids: torch.Tensor, error: type[Exception]
) -> None:
    model = RegionalResidualScorer()

    with pytest.raises(error):
        model.region_features(region_ids)

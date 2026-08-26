import pytest
import torch

from regional_score import (
    RegionalResidualScorer,
    initialize_single_target_from_pretrained,
    set_single_target_backbone_trainable,
)


def fill_parameters(model: RegionalResidualScorer) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(float(index))


def test_transfer_keeps_new_target_head_and_alpha_by_default() -> None:
    source = RegionalResidualScorer()
    fill_parameters(source)
    target = RegionalResidualScorer()
    original_head = target.region_tower[8].weight.detach().clone()
    original_alpha = target.alpha.detach().clone()

    report = initialize_single_target_from_pretrained(target, source)

    assert torch.equal(target.embedding.weight, source.embedding.weight)
    assert torch.equal(target.region_tower[4].weight, source.region_tower[4].weight)
    assert torch.equal(target.region_tower[8].weight, original_head)
    assert torch.equal(target.alpha, original_alpha)
    assert "embedding.weight" in report.loaded_keys
    assert "region_tower.8.weight" not in report.loaded_keys
    assert "alpha" not in report.loaded_keys


def test_transfer_can_include_target_head_and_alpha() -> None:
    source = RegionalResidualScorer()
    fill_parameters(source)
    target = RegionalResidualScorer()

    initialize_single_target_from_pretrained(
        target,
        source,
        transfer_head=True,
        transfer_alpha=True,
    )

    assert torch.equal(target.region_tower[8].weight, source.region_tower[8].weight)
    assert torch.equal(target.region_tower[8].bias, source.region_tower[8].bias)
    assert torch.equal(target.alpha, source.alpha)


def test_transfer_from_checkpoint_can_freeze_and_unfreeze_backbone(
    tmp_path,
) -> None:
    source = RegionalResidualScorer()
    fill_parameters(source)
    checkpoint = tmp_path / "pretrained.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    target = RegionalResidualScorer()

    report = initialize_single_target_from_pretrained(
        target,
        checkpoint,
        freeze_backbone=True,
    )

    parameters = dict(target.named_parameters())
    assert report.frozen_keys
    assert all(not parameters[name].requires_grad for name in report.frozen_keys)
    assert parameters["region_tower.8.weight"].requires_grad
    assert parameters["alpha"].requires_grad

    changed = set_single_target_backbone_trainable(target, True)

    assert set(changed) == set(report.frozen_keys)
    assert all(parameters[name].requires_grad for name in changed)


def test_transfer_rejects_incompatible_architecture() -> None:
    source = RegionalResidualScorer(embedding_dim=4)
    target = RegionalResidualScorer(embedding_dim=5)

    with pytest.raises(ValueError, match="incompatible model architectures"):
        initialize_single_target_from_pretrained(target, source)

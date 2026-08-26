from __future__ import annotations

import torch

from .model import RegionalResidualScorer, count_trainable_parameters


def main() -> None:
    model = RegionalResidualScorer()
    print("Trainable parameters")
    print("-" * 54)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"{name:<38} {parameter.numel():>8,}")
    print("-" * 54)
    print(f"{'TOTAL':<38} {count_trainable_parameters(model):>8,}")

    sample_regions = torch.tensor(
        [[0, 0, 0], [12, 7, 12], [70, 4, 71]], dtype=torch.long
    )
    sample_base_logits = torch.zeros(3)
    with torch.inference_mode():
        logits = model(sample_base_logits, sample_regions)
    print(f"\nSample output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

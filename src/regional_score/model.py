from __future__ import annotations

import torch
from torch import Tensor, nn


class RegionalResidualScorer(nn.Module):
    """Add a compact region-based correction to an existing model logit.

    Region columns must be ordered as registration, birth, application.
    IDs 0..69 represent known regions, 70 is MISSING, and 71 is UNKNOWN.
    """

    def __init__(
        self,
        num_regions: int = 72,
        embedding_dim: int = 4,
        hidden_dims: tuple[int, int] = (24, 8),
        dropout: float = 0.15,
        initial_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if num_regions < 1:
            raise ValueError("num_regions must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if len(hidden_dims) != 2 or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain two positive widths")

        self.num_regions = num_regions
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_regions, embedding_dim)

        # Three embeddings, three absolute differences, three products,
        # three equality flags, and one unique-region count.
        interaction_dim = 9 * embedding_dim + 4
        first_hidden, second_hidden = hidden_dims
        self.region_tower = nn.Sequential(
            nn.Linear(interaction_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def region_features(self, region_ids: Tensor) -> Tensor:
        """Build role-aware embedding interactions for a [batch, 3] ID tensor."""
        if region_ids.ndim != 2 or region_ids.shape[1] != 3:
            raise ValueError("region_ids must have shape [batch, 3]")
        if region_ids.dtype != torch.long:
            raise TypeError("region_ids must have dtype torch.long")
        if torch.any((region_ids < 0) | (region_ids >= self.num_regions)):
            raise ValueError(f"region IDs must be in [0, {self.num_regions - 1}]")

        embedded = self.embedding(region_ids)
        registration, birth, application = embedded.unbind(dim=1)
        pairs = (
            (registration, birth),
            (registration, application),
            (birth, application),
        )

        flat_embeddings = embedded.flatten(start_dim=1)
        absolute_differences = torch.cat(
            [torch.abs(left - right) for left, right in pairs], dim=1
        )
        products = torch.cat([left * right for left, right in pairs], dim=1)

        registration_id, birth_id, application_id = region_ids.unbind(dim=1)
        equality_flags = torch.stack(
            (
                registration_id == birth_id,
                registration_id == application_id,
                birth_id == application_id,
            ),
            dim=1,
        ).to(dtype=embedded.dtype)
        unique_count = (
            1
            + (birth_id != registration_id).to(dtype=embedded.dtype)
            + (
                (application_id != registration_id)
                & (application_id != birth_id)
            ).to(dtype=embedded.dtype)
        ).unsqueeze(1)

        return torch.cat(
            (
                flat_embeddings,
                absolute_differences,
                products,
                equality_flags,
                unique_count,
            ),
            dim=1,
        )

    def forward(self, base_logit: Tensor, region_ids: Tensor) -> Tensor:
        """Return final logits; apply sigmoid only for probability inference."""
        if base_logit.ndim == 1:
            base_logit = base_logit.unsqueeze(1)
        if base_logit.ndim != 2 or base_logit.shape[1] != 1:
            raise ValueError("base_logit must have shape [batch] or [batch, 1]")
        if base_logit.shape[0] != region_ids.shape[0]:
            raise ValueError("base_logit and region_ids batch sizes must match")

        delta_logit = self.region_tower(self.region_features(region_ids))
        return base_logit + self.alpha * delta_logit


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

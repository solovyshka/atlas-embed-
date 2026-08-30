from __future__ import annotations

import torch
from torch import Tensor, nn


class RegionalResidualScorer(nn.Module):
    """Add a compact region-based correction to an existing model score.

    Region columns must be ordered as registration and factual address.
    ``base_score`` is the probability produced by the primary scoring model.
    """

    def __init__(
        self,
        num_regions: int = 70,
        embedding_dim: int = 4,
        hidden_dims: tuple[int, int] = (24, 8),
        dropout: float = 0.15,
        initial_alpha: float = 1.0,
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

        # Two embeddings, their absolute difference and product, plus the
        # supplied address-equality flag (which is not derived from region IDs).
        interaction_dim = 4 * embedding_dim + 1
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
        nn.init.zeros_(self.region_tower[-1].weight)
        nn.init.zeros_(self.region_tower[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def region_features(self, region_ids: Tensor, equal_flag: Tensor) -> Tensor:
        """Build region interactions for a [batch, 2] ID tensor."""
        if region_ids.ndim != 2 or region_ids.shape[1] != 2:
            raise ValueError("region_ids must have shape [batch, 2]")
        if region_ids.dtype != torch.long:
            raise TypeError("region_ids must have dtype torch.long")
        if torch.any((region_ids < 0) | (region_ids >= self.num_regions)):
            raise ValueError(f"region IDs must be in [0, {self.num_regions - 1}]")

        if equal_flag.ndim == 1:
            equal_flag = equal_flag.unsqueeze(1)
        if equal_flag.ndim != 2 or equal_flag.shape != (region_ids.shape[0], 1):
            raise ValueError("equal_flag must have shape [batch] or [batch, 1]")
        if torch.any((equal_flag != 0) & (equal_flag != 1)):
            raise ValueError("equal_flag must contain only 0 and 1")

        embedded = self.embedding(region_ids)
        registration, factual = embedded.unbind(dim=1)
        flat_embeddings = embedded.flatten(start_dim=1)
        absolute_difference = torch.abs(registration - factual)
        product = registration * factual

        return torch.cat(
            (
                flat_embeddings,
                absolute_difference,
                product,
                equal_flag.to(dtype=embedded.dtype),
            ),
            dim=1,
        )

    def forward(
        self,
        base_score: Tensor,
        region_ids: Tensor,
        equal_flag: Tensor,
    ) -> Tensor:
        """Return final logits for ``flag_6m_30p``."""
        if base_score.ndim == 1:
            base_score = base_score.unsqueeze(1)
        if base_score.ndim != 2 or base_score.shape[1] != 1:
            raise ValueError("base_score must have shape [batch] or [batch, 1]")
        if base_score.shape[0] != region_ids.shape[0]:
            raise ValueError("base_score and region_ids batch sizes must match")
        if not torch.is_floating_point(base_score):
            raise TypeError("base_score must have a floating-point dtype")
        if torch.any((base_score < 0) | (base_score > 1)):
            raise ValueError("base_score must be between 0 and 1")

        epsilon = torch.finfo(base_score.dtype).eps
        base_logit = torch.logit(base_score.clamp(epsilon, 1 - epsilon))
        features = self.region_features(region_ids, equal_flag)
        delta_logit = self.region_tower(features)
        return base_logit + self.alpha * delta_logit

    def encode(self, region_ids: Tensor, equal_flag: Tensor) -> Tensor:
        """Return compact region embeddings for boosting or logistic regression.

        Shape is ``[batch, hidden_dims[1]]`` (8 by default). ``score_dubai``
        is not included: concatenate it later as a separate feature.
        """
        hidden = self.region_features(region_ids, equal_flag)
        for module in self.region_tower:
            if isinstance(module, nn.Dropout):
                continue
            if isinstance(module, nn.Linear) and module.out_features == 1:
                break
            hidden = module(hidden)
        return hidden


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

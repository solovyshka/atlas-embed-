from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_ids(ids: Tensor, *, ndim: int, name: str) -> None:
    if not isinstance(ids, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ids.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    if ids.ndim != ndim:
        shape = "[batch]" if ndim == 1 else "[batch, levels]"
        raise ValueError(f"{name} must have shape {shape}")
    if ids.shape[0] == 0:
        raise ValueError(f"{name} batch must not be empty")


class _OkvedResidualScorer(nn.Module):
    """Shared compact residual tower for OKVED representations."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        initial_alpha: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.output_dim = _positive_int(output_dim, "output_dim")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(initial_alpha, (int, float)) or not torch.isfinite(
            torch.tensor(float(initial_alpha))
        ):
            raise ValueError("initial_alpha must be finite")

        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.residual_head = nn.Linear(self.output_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def _okved_features(self, okved_ids: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _validate_boost_score(boost_score: Tensor) -> Tensor:
        if not isinstance(boost_score, Tensor):
            raise TypeError("boost_score must be a torch.Tensor")
        if not torch.is_floating_point(boost_score):
            raise TypeError("boost_score must have a floating-point dtype")
        if boost_score.ndim == 1:
            boost_score = boost_score.unsqueeze(1)
        if boost_score.ndim != 2 or boost_score.shape[1] != 1:
            raise ValueError("boost_score must have shape [batch] or [batch, 1]")
        if boost_score.shape[0] == 0:
            raise ValueError("boost_score batch must not be empty")
        if not torch.all(torch.isfinite(boost_score)):
            raise ValueError("boost_score must contain only finite values")
        if torch.any((boost_score < 0) | (boost_score > 1)):
            raise ValueError("boost_score must be between 0 and 1")
        return boost_score

    def encode(self, okved_ids: Tensor) -> Tensor:
        """Return the compact ``[batch, output_dim]`` OKVED representation."""
        return self.encoder(self._okved_features(okved_ids))

    def forward(self, boost_score: Tensor, okved_ids: Tensor) -> Tensor:
        """Return the base score logit plus a learned OKVED correction."""
        boost_score = self._validate_boost_score(boost_score)
        features = self._okved_features(okved_ids)
        if boost_score.shape[0] != features.shape[0]:
            raise ValueError("boost_score and okved_ids batch sizes must match")
        if boost_score.device != features.device:
            raise ValueError("boost_score and okved_ids must be on the same device")

        delta = self.residual_head(self.encoder(features))
        epsilon = torch.finfo(boost_score.dtype).eps
        base_logit = torch.logit(boost_score.clamp(epsilon, 1 - epsilon))
        return base_logit + self.alpha * delta


class FlatOkvedResidualScorer(_OkvedResidualScorer):
    """Residual scorer using one leaf OKVED ID per application."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 8,
        dropout: float = 0.1,
        initial_alpha: float = 1.0,
    ) -> None:
        vocab_size = _positive_int(vocab_size, "vocab_size")
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            initial_alpha=initial_alpha,
        )
        self.vocab_size = vocab_size
        self.leaf_embedding = nn.Embedding(
            vocab_size, self.embedding_dim, padding_idx=0
        )

    @property
    def embedding(self) -> nn.Embedding:
        """Alias matching the single-table terminology used by PyTorch."""
        return self.leaf_embedding

    def okved_features(self, okved_ids: Tensor) -> Tensor:
        """Look up leaf embeddings for a ``[batch]`` ID tensor."""
        _validate_ids(okved_ids, ndim=1, name="okved_ids")
        if torch.any((okved_ids < 0) | (okved_ids >= self.vocab_size)):
            raise ValueError(
                f"okved IDs must be in [0, {self.vocab_size - 1}]"
            )
        return self.leaf_embedding(okved_ids)

    def _okved_features(self, okved_ids: Tensor) -> Tensor:
        return self.okved_features(okved_ids)


class HierarchicalOkvedResidualScorer(_OkvedResidualScorer):
    """Residual scorer fusing one independently embedded ID per hierarchy level."""

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        embedding_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 8,
        dropout: float = 0.1,
        initial_alpha: float = 1.0,
    ) -> None:
        if isinstance(vocab_sizes, (str, bytes)) or not isinstance(
            vocab_sizes, Sequence
        ):
            raise TypeError("vocab_sizes must be a sequence of positive integers")
        if not vocab_sizes:
            raise ValueError("vocab_sizes must contain at least one level")
        checked_sizes = tuple(
            _positive_int(size, f"vocab_sizes[{level}]")
            for level, size in enumerate(vocab_sizes)
        )
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            initial_alpha=initial_alpha,
        )
        self.vocab_sizes = checked_sizes
        self.level_embeddings = nn.ModuleList(
            nn.Embedding(size, self.embedding_dim, padding_idx=0)
            for size in checked_sizes
        )

    @property
    def embeddings(self) -> nn.ModuleList:
        """Return the level-specific embedding tables."""
        return self.level_embeddings

    def okved_features(self, okved_ids: Tensor) -> Tensor:
        """Fuse non-PAD level embeddings with a masked mean."""
        _validate_ids(okved_ids, ndim=2, name="okved_ids")
        if okved_ids.shape[1] != len(self.vocab_sizes):
            raise ValueError(
                f"okved_ids must have {len(self.vocab_sizes)} hierarchy levels"
            )

        mask = okved_ids.ne(0)
        if torch.any(~mask.any(dim=1)):
            raise ValueError("each hierarchy row must contain at least one non-PAD ID")

        level_values: list[Tensor] = []
        for level, (ids, size, embedding) in enumerate(
            zip(okved_ids.unbind(dim=1), self.vocab_sizes, self.level_embeddings)
        ):
            if torch.any((ids < 0) | (ids >= size)):
                raise ValueError(
                    f"OKVED IDs at level {level} must be in [0, {size - 1}]"
                )
            level_values.append(embedding(ids))

        stacked = torch.stack(level_values, dim=1)
        float_mask = mask.unsqueeze(2).to(dtype=stacked.dtype)
        return (stacked * float_mask).sum(dim=1) / float_mask.sum(dim=1)

    def _okved_features(self, okved_ids: Tensor) -> Tensor:
        return self.okved_features(okved_ids)


def count_trainable_parameters(module: nn.Module) -> int:
    """Count parameters that will be updated by an optimizer."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

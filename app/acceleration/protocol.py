"""Shared contracts for FireLens compute backends."""

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt


Float32Array: TypeAlias = npt.NDArray[np.float32]
Int64Array: TypeAlias = npt.NDArray[np.int64]
SymbolCandidate: TypeAlias = tuple[str, str]
CapabilityName: TypeAlias = Literal["semantic", "top_k", "fuzzy", "exact"]


class AccelerationError(RuntimeError):
    """A compute backend failed while executing an available operation."""


@dataclass(frozen=True)
class RankedScores:
    """Selected candidate indexes and their aligned raw scores."""

    indices: Int64Array
    scores: Float32Array

    def __post_init__(self) -> None:
        if not isinstance(self.indices, np.ndarray):
            raise TypeError("Ranked indexes must be a NumPy array")
        if not isinstance(self.scores, np.ndarray):
            raise TypeError("Ranked scores must be a NumPy array")
        if self.indices.dtype != np.dtype(np.int64):
            raise TypeError("Ranked indexes must use the int64 dtype")
        if self.scores.dtype != np.dtype(np.float32):
            raise TypeError("Ranked scores must use the float32 dtype")
        if self.indices.ndim != 1 or self.scores.ndim != 1:
            raise ValueError("Ranked indexes and scores must be one-dimensional")
        if self.indices.shape != self.scores.shape:
            raise ValueError("Ranked indexes and scores must have the same shape")
        if np.any(self.indices < 0):
            raise ValueError("Ranked indexes must not be negative")
        if not np.all(np.isfinite(self.scores)):
            raise ValueError("Ranked scores must contain only finite values")


@runtime_checkable
class AccelerationBackend(Protocol):
    """Pure compute operations available to FireLens search orchestration."""

    name: str

    def supports(self, capability: CapabilityName) -> bool:
        """Return whether this backend implements a compute capability."""

        ...

    def semantic_top_k(
        self,
        matrix: Float32Array,
        query: Float32Array,
        top_k: int,
    ) -> RankedScores:
        """Return the highest clipped dot-product scores."""

        ...

    def top_k(self, scores: Float32Array, top_k: int) -> RankedScores:
        """Return the highest raw scores with stable tie ordering."""

        ...

    def fuzzy_scores(
        self,
        query: str,
        candidates: Sequence[SymbolCandidate],
        minimum_score: float,
    ) -> npt.NDArray[np.float64]:
        """Return one fuzzy relevance score for each candidate."""

        ...

    def exact_match_indices(
        self,
        query: str,
        candidates: Sequence[SymbolCandidate],
    ) -> Int64Array:
        """Return qualified-name matches followed by short-name matches."""

        ...

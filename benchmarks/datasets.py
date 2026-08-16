"""Deterministic in-memory datasets for acceleration benchmarks."""

from dataclasses import dataclass

import numpy as np


CandidateNames = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SemanticDataset:
    """A unit-normalized vector matrix and query used for semantic ranking."""

    matrix: np.ndarray
    query: np.ndarray
    top_k: int


@dataclass(frozen=True)
class FuzzyDataset:
    """A query and short/qualified symbol names used for fuzzy scoring."""

    query: str
    candidates: CandidateNames
    minimum_score: float


@dataclass(frozen=True)
class ExactDataset:
    """A query and short/qualified symbol names used for exact matching."""

    query: str
    candidates: CandidateNames


def make_semantic_dataset(
    candidate_count: int,
    dimension: int,
    *,
    top_k: int = 10,
    seed: int = 20260816,
) -> SemanticDataset:
    """Build a reproducible float32 semantic-ranking dataset."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be greater than 0")
    if dimension < 1:
        raise ValueError("dimension must be greater than 0")
    if top_k < 1:
        raise ValueError("top_k must be greater than 0")

    random_generator = np.random.default_rng(
        _case_seed(seed, candidate_count, dimension)
    )
    matrix = random_generator.standard_normal(
        (candidate_count, dimension),
        dtype=np.float32,
    )
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = row_norms[:, 0] == 0.0
    if np.any(zero_rows):
        matrix[zero_rows, 0] = 1.0
        row_norms[zero_rows, 0] = 1.0
    matrix = np.ascontiguousarray(matrix / row_norms, dtype=np.float32)

    query = random_generator.standard_normal(dimension, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        query[0] = 1.0
        query_norm = np.float32(1.0)
    query = np.ascontiguousarray(query / query_norm, dtype=np.float32)

    # Make the best result predictable without creating a floating-point tie.
    matrix[0] = query
    matrix.setflags(write=False)
    query.setflags(write=False)
    return SemanticDataset(
        matrix=matrix,
        query=query,
        top_k=min(top_k, candidate_count),
    )


def make_fuzzy_dataset(
    candidate_count: int,
    *,
    minimum_score: float = 0.55,
    seed: int = 20260816,
) -> FuzzyDataset:
    """Build reproducible names spanning fuzzy-score match categories."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be greater than 0")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")

    query = "semantic_search_index"
    special_candidates = (
        ("semantic_search_index", "app.search.semantic_search_index"),
        ("semantic_search_index_cache", "app.cache.semantic_search_index_cache"),
        ("load_semantic_search_index", "app.search.load_semantic_search_index"),
        ("semantic_serch_index", "app.search.semantic_serch_index"),
        ("SemanticSearchIndex", "app.search.SemanticSearchIndex"),
        ("search_index", "app.semantic.search_index"),
        ("index_symbols", "app.indexing.index_symbols"),
        ("unrelated_name", "package.module.unrelated_name"),
    )
    candidates: list[tuple[str, str]] = []
    for candidate_number in range(candidate_count):
        if candidate_number < len(special_candidates):
            candidates.append(special_candidates[candidate_number])
            continue

        token = _deterministic_token(seed, candidate_number)
        short_name = f"{token}_symbol_{candidate_number:06d}"
        qualified_name = (
            f"package_{candidate_number % 97}.module_{candidate_number % 31}."
            f"{short_name}"
        )
        candidates.append((short_name, qualified_name))

    return FuzzyDataset(
        query=query,
        candidates=tuple(candidates),
        minimum_score=minimum_score,
    )


def make_exact_dataset(
    candidate_count: int,
    *,
    seed: int = 20260816,
) -> ExactDataset:
    """Build names with deterministic short- and qualified-name matches."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be greater than 0")

    query = "TargetSymbol"
    candidates: list[tuple[str, str]] = []
    qualified_match_index = candidate_count // 2
    for candidate_number in range(candidate_count):
        token = _deterministic_token(seed, candidate_number)
        short_name = f"{token}Symbol{candidate_number:06d}"
        qualified_name = f"package.module.{short_name}"
        if candidate_number == 0:
            short_name = query
            qualified_name = f"package.module.{query}Alias"
        elif candidate_number == qualified_match_index:
            qualified_name = query
        candidates.append((short_name, qualified_name))

    return ExactDataset(query=query, candidates=tuple(candidates))


def _case_seed(seed: int, candidate_count: int, dimension: int) -> int:
    """Mix case parameters without relying on Python's randomized hash."""

    return (seed * 1_000_003 + candidate_count * 101 + dimension) % (2**63)


def _deterministic_token(seed: int, candidate_number: int) -> str:
    """Return a stable alphabetic token without allocating another RNG."""

    value = (seed + candidate_number * 2_654_435_761) & 0xFFFFFFFF
    characters = []
    for _ in range(8):
        characters.append(chr(ord("a") + (value % 26)))
        value //= 26
    return "".join(characters)

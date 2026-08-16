"""Reference Python implementation of FireLens compute kernels."""

import math
import re
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt

from app.acceleration.protocol import (
    CapabilityName,
    Float32Array,
    Int64Array,
    RankedScores,
    SymbolCandidate,
)
from app.search.limits import MAX_FUZZY_CANDIDATE_CHARS, MAX_FUZZY_QUERY_CHARS

DEFAULT_MINIMUM_FUZZY_SCORE = 0.55


class PythonBackend:
    """Compute search scores with Python and NumPy."""

    name = "python"

    def supports(self, capability: CapabilityName) -> bool:
        """The reference backend implements every defined capability."""

        return capability in {"semantic", "top_k", "fuzzy", "exact"}

    def semantic_top_k(
        self,
        matrix: Float32Array,
        query: Float32Array,
        top_k: int,
    ) -> RankedScores:
        """Compute clipped matrix-vector scores and return the highest values."""

        _validate_float32_array(matrix, name="matrix", dimensions=2)
        _validate_float32_array(query, name="query", dimensions=1)
        _validate_top_k(top_k)

        if query.shape[0] == 0:
            raise ValueError("Query vector must not be empty")
        if matrix.shape[1] != query.shape[0]:
            raise ValueError("Matrix and query dimensions must match")

        raw_scores = np.asarray(matrix @ query, dtype=np.float32)
        clipped_scores = np.clip(raw_scores, -1.0, 1.0)
        return self.top_k(clipped_scores, top_k)

    def top_k(self, scores: Float32Array, top_k: int) -> RankedScores:
        """Return stable descending indexes and their unchanged scores."""

        _validate_float32_array(scores, name="scores", dimensions=1)
        _validate_top_k(top_k)

        selected_count = min(top_k, scores.shape[0])
        if selected_count == 0:
            return RankedScores(
                indices=np.empty(0, dtype=np.int64),
                scores=np.empty(0, dtype=np.float32),
            )

        ranked_indices = np.argsort(-scores, kind="stable")[:selected_count]
        selected_indices = np.asarray(ranked_indices, dtype=np.int64)
        selected_scores = np.asarray(scores[selected_indices], dtype=np.float32)
        return RankedScores(indices=selected_indices, scores=selected_scores)

    def fuzzy_scores(
        self,
        query: str,
        candidates: Sequence[SymbolCandidate],
        minimum_score: float = DEFAULT_MINIMUM_FUZZY_SCORE,
    ) -> npt.NDArray[np.float64]:
        """Score each candidate's short and qualified names in one batch."""

        _validate_query(query)
        validated_candidates = _validate_candidates(candidates)
        threshold = _validate_minimum_score(minimum_score)

        scores = np.zeros(len(validated_candidates), dtype=np.float64)
        if len(query) > MAX_FUZZY_QUERY_CHARS:
            return scores

        normalized_query = normalize_identifier(query)
        if normalized_query == "":
            return scores

        for index, (short_name, qualified_name) in enumerate(validated_candidates):
            short_score = _fuzzy_score_for_normalized_query(
                normalized_query,
                short_name,
                threshold,
            )
            qualified_score = _fuzzy_score_for_normalized_query(
                normalized_query,
                qualified_name,
                threshold,
            )
            scores[index] = max(short_score, qualified_score)

        return scores

    def exact_match_indices(
        self,
        query: str,
        candidates: Sequence[SymbolCandidate],
    ) -> Int64Array:
        """Return qualified matches first, then remaining short-name matches."""

        _validate_query(query)
        validated_candidates = _validate_candidates(candidates)
        normalized_query = query.strip()

        if normalized_query == "":
            return np.empty(0, dtype=np.int64)

        qualified_matches: list[int] = []
        short_matches: list[int] = []
        for index, (short_name, qualified_name) in enumerate(validated_candidates):
            if qualified_name == normalized_query:
                qualified_matches.append(index)
            elif short_name == normalized_query:
                short_matches.append(index)

        return np.asarray(qualified_matches + short_matches, dtype=np.int64)


def fuzzy_score(
    query: str,
    candidate: str,
    minimum_score: float = DEFAULT_MINIMUM_FUZZY_SCORE,
) -> float:
    """Return normalized fuzzy relevance in the range 0.0 to 1.0."""

    _validate_query(query)
    if not isinstance(candidate, str):
        raise TypeError("Fuzzy candidate must be a string")
    threshold = _validate_minimum_score(minimum_score)
    if len(query) > MAX_FUZZY_QUERY_CHARS:
        return 0.0

    normalized_query = normalize_identifier(query)
    return _fuzzy_score_for_normalized_query(
        normalized_query,
        candidate,
        threshold,
    )


def _fuzzy_score_for_normalized_query(
    normalized_query: str,
    candidate: str,
    minimum_score: float,
) -> float:
    if len(candidate) > MAX_FUZZY_CANDIDATE_CHARS:
        return 0.0

    normalized_candidate = normalize_identifier(candidate)

    if normalized_query == "" or normalized_candidate == "":
        return 0.0
    if normalized_query == normalized_candidate:
        return 1.0
    if normalized_candidate.startswith(normalized_query):
        return 0.95
    if normalized_query in normalized_candidate:
        return 0.85

    max_length = max(len(normalized_query), len(normalized_candidate))
    maximum_distance = math.floor(
        ((1.0 - minimum_score) * max_length) + 1e-12
    )
    distance = levenshtein_distance(
        normalized_query,
        normalized_candidate,
        max_distance=maximum_distance,
    )
    if distance > maximum_distance:
        return 0.0

    return max(0.0, 1.0 - (distance / max_length))


def normalize_identifier(value: str) -> str:
    """Normalize code identifiers before fuzzy comparison."""

    if not isinstance(value, str):
        raise TypeError("Identifier must be a string")
    value = split_camel_case(value.strip())
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = value.replace(".", " ")
    value = " ".join(value.split())
    return value.lower()


def split_camel_case(value: str) -> str:
    """Insert spaces at camel-case boundaries without changing characters."""

    if not isinstance(value, str):
        raise TypeError("Identifier must be a string")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)


def levenshtein_distance(
    word1: str,
    word2: str,
    max_distance: int | None = None,
) -> int:
    """Return edit distance, stopping when a configured bound is impossible."""

    if not isinstance(word1, str) or not isinstance(word2, str):
        raise TypeError("Levenshtein inputs must be strings")
    if max_distance is None:
        max_distance = max(len(word1), len(word2))
    if not isinstance(max_distance, Integral) or isinstance(max_distance, bool):
        raise TypeError("max_distance must be an integer")
    max_distance = int(max_distance)
    if max_distance < 0:
        raise ValueError("max_distance must not be negative")
    if abs(len(word1) - len(word2)) > max_distance:
        return max_distance + 1
    if word1 == word2:
        return 0

    if len(word1) < len(word2):
        longer_word, shorter_word = word2, word1
    else:
        longer_word, shorter_word = word1, word2

    outside_bound = max_distance + 1
    previous_row = {
        column: column
        for column in range(min(len(shorter_word), max_distance) + 1)
    }
    for row_number, longer_character in enumerate(longer_word, start=1):
        current_row: dict[int, int] = {}
        if row_number <= max_distance:
            current_row[0] = row_number

        first_column = max(1, row_number - max_distance)
        last_column = min(len(shorter_word), row_number + max_distance)
        for column_number in range(first_column, last_column + 1):
            shorter_character = shorter_word[column_number - 1]
            insertion_cost = current_row.get(column_number - 1, outside_bound) + 1
            deletion_cost = previous_row.get(column_number, outside_bound) + 1
            substitution_cost = previous_row.get(
                column_number - 1,
                outside_bound,
            )
            if longer_character != shorter_character:
                substitution_cost += 1
            distance = min(insertion_cost, deletion_cost, substitution_cost)
            if distance <= max_distance:
                current_row[column_number] = distance

        if not current_row:
            return outside_bound
        previous_row = current_row

    return previous_row.get(len(shorter_word), outside_bound)


def _validate_float32_array(
    value: object,
    *,
    name: str,
    dimensions: int,
) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name.capitalize()} must be a NumPy array")
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name.capitalize()} must use the float32 dtype")
    if value.ndim != dimensions:
        raise ValueError(f"{name.capitalize()} must be {dimensions}-dimensional")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name.capitalize()} must be C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name.capitalize()} must contain only finite values")


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, Integral) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k < 1:
        raise ValueError("top_k must be greater than 0")


def _validate_query(query: object) -> None:
    if not isinstance(query, str):
        raise TypeError("Query must be a string")


def _validate_candidates(
    candidates: Sequence[SymbolCandidate],
) -> list[SymbolCandidate]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("Candidates must be a sequence of name pairs")

    validated_candidates: list[SymbolCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, tuple) or len(candidate) != 2:
            raise TypeError("Each candidate must be a short and qualified name tuple")
        short_name, qualified_name = candidate
        if not isinstance(short_name, str) or not isinstance(qualified_name, str):
            raise TypeError("Candidate names must be strings")
        validated_candidates.append((short_name, qualified_name))
    return validated_candidates


def _validate_minimum_score(minimum_score: float) -> float:
    if not isinstance(minimum_score, Real) or isinstance(minimum_score, bool):
        raise TypeError("minimum_score must be a real number")
    threshold = float(minimum_score)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")
    return threshold

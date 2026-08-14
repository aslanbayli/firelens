"""Python fuzzy-search flow.

This module owns request handling, timing, and conversion from
stored Symbol records into SearchResult records. It does not contain SQL.
"""

import math
import re
import time
import uuid

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.search.limits import (
    MAX_FUZZY_CANDIDATE_CHARS,
    MAX_FUZZY_QUERY_CHARS,
    bounded_symbol_name,
)
from app.storage.database import SQLiteIndexStore

MIN_FUZZY_SCORE = 0.55


def fuzzy_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
    minimum_score: float = MIN_FUZZY_SCORE,
    max_candidates: int = 512,
    cancellation_callback: CancellationCallback | None = None,
) -> SearchResponse:
    # Trim user input before fuzzy-specific normalization. Keep the original
    # string in the response for callers and logs.
    raise_if_cancelled(cancellation_callback)
    query = request.query.strip()

    # Empty fuzzy queries should return no results. They should not fall
    # through into a broad storage query.
    if query == "":
        return SearchResponse(
            original_query=request.query,
            requested_mode=request.request_mode,
            mode="fuzzy",
            requested_backend=request.backend,
            backend="python",
            elapsed_time=0.0,
            ranked_results=[],
            warnings=[],
        )
    if len(query) > MAX_FUZZY_QUERY_CHARS:
        raise ValueError(
            f"Fuzzy queries must be at most {MAX_FUZZY_QUERY_CHARS} characters"
        )
    if max_candidates < 1:
        raise ValueError("max_candidates must be greater than 0")

    # Start timing after cheap validation so elapsed_time represents the
    # retrieval path.
    start_time = time.perf_counter()

    # Load only ranking metadata. Full snippets are fetched after top-k
    # selection so a fuzzy query cannot materialize every source snippet.
    raise_if_cancelled(cancellation_callback)
    candidates = store.load_symbol_candidates(
        repository_id=repository_id,
        path_filter=request.path,
        limit=max_candidates,
        candidate_char_limit=MAX_FUZZY_CANDIDATE_CHARS,
    )
    raise_if_cancelled(cancellation_callback)

    # Rank the retrieved results based on normalized similarity. Levenshtein
    # distance is lower-is-better, so convert it to a higher-is-better score.
    scored_symbols = []

    for candidate in candidates:
        raise_if_cancelled(cancellation_callback)
        name_score = fuzzy_score(query, candidate.name, minimum_score)
        qualified_name_score = fuzzy_score(
            query,
            candidate.qualified_name,
            minimum_score,
        )

        score = max(name_score, qualified_name_score)
        if score < minimum_score:
            continue

        scored_symbols.append((score, candidate))

    # Sort by relevance first, then stable code-location fields so repeated
    # MCP calls produce deterministic context.
    scored_symbols.sort(
        key=lambda scored: (
            -scored[0],
            len(scored[1].qualified_name),
            scored[1].relative_path,
            scored[1].qualified_name,
            scored[1].start_line,
        )
    )
    raise_if_cancelled(cancellation_callback)

    selected_candidates = scored_symbols[: request.top_k]
    symbols_by_id = store.load_symbols_by_ids(
        (candidate.id for _score, candidate in selected_candidates),
        max_snippet_chars=request.max_snippet_chars,
    )
    raise_if_cancelled(cancellation_callback)

    results: list[SearchResult] = []
    for score, candidate in selected_candidates:
        raise_if_cancelled(cancellation_callback)
        symbol = symbols_by_id.get(candidate.id)
        if symbol is None:
            raise ValueError("Ranked fuzzy symbol was not found")
        snippet = symbol.source_snippet[: request.max_snippet_chars]
        results.append(
            SearchResult(
                id=symbol.id,
                result_type="symbol",
                file_path=symbol.relative_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol_name=bounded_symbol_name(symbol.qualified_name),
                snippet=snippet,
                snippet_truncated=len(snippet) < len(symbol.source_snippet),
                score=score,
                mode="fuzzy",
                backend="python",
            )
        )

    return SearchResponse(
        original_query=request.query,
        requested_mode=request.request_mode,
        mode="fuzzy",
        requested_backend=request.backend,
        backend="python",
        elapsed_time=time.perf_counter() - start_time,
        ranked_results=results,
        warnings=[],
    )


def fuzzy_score(
    query: str,
    candidate: str,
    minimum_score: float = MIN_FUZZY_SCORE,
) -> float:
    """Return a normalized fuzzy relevance score in the range 0.0 to 1.0."""

    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")
    if (
        len(query) > MAX_FUZZY_QUERY_CHARS
        or len(candidate) > MAX_FUZZY_CANDIDATE_CHARS
    ):
        return 0.0

    normalized_query = normalize_identifier(query)
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

    value = split_camel_case(value.strip())
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = value.replace(".", " ")
    value = " ".join(value.split())

    return value.lower()


def split_camel_case(value: str) -> str:
    """Insert spaces at camel-case boundaries without changing characters."""

    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)


def levenshtein_distance(
    word1: str,
    word2: str,
    max_distance: int | None = None,
) -> int:
    """Return edit distance, stopping once a configured bound is impossible.

    When the true distance exceeds ``max_distance``, the function returns
    ``max_distance + 1``. Restricting work to that diagonal band prevents
    pathological fuzzy queries from filling the complete edit-distance matrix.
    """

    if max_distance is None:
        max_distance = max(len(word1), len(word2))
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

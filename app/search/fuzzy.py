"""Python fuzzy-search flow.

This module owns request handling, timing, and conversion from
stored Symbol records into SearchResult records. It does not contain SQL.
"""

import re
import time
import uuid

from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.storage.database import SQLiteIndexStore

MIN_FUZZY_SCORE = 0.55


def fuzzy_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
) -> SearchResponse:
    # Trim user input before fuzzy-specific normalization. Keep the original
    # string in the response for callers and logs.
    query = request.query.strip()

    # Empty fuzzy queries should return no results. They should not fall
    # through into a broad storage query.
    if query == "":
        return SearchResponse(
            original_query=request.query,
            mode="fuzzy",
            backend="python",
            elapsed_time=0.0,
            ranked_results=[],
            warnings=[],
        )

    # Start timing after cheap validation so elapsed_time represents the
    # retrieval path.
    start_time = time.time()

    # Get all symbols in the database for the given repo and optionally path
    all_symbols = store.load_all_symbols(
        repository_id=repository_id,
        path_filter=request.path,
    )

    # Rank the retrieved results based on normalized similarity. Levenshtein
    # distance is lower-is-better, so convert it to a higher-is-better score.
    scored_symbols = []

    for symbol in all_symbols:
        name_score = fuzzy_score(query, symbol.name)
        qualified_name_score = fuzzy_score(query, symbol.qualified_name)

        score = max(name_score, qualified_name_score)
        if score < MIN_FUZZY_SCORE:
            continue

        scored_symbols.append((score, symbol))

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

    results: list[SearchResult] = []
    for score, symbol in scored_symbols[: request.top_k]:
        results.append(
            SearchResult(
                id=symbol.id,
                result_type="symbol",
                file_path=symbol.relative_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol_name=symbol.qualified_name,
                snippet=symbol.source_snippet,
                score=score,
                mode="fuzzy",
                backend="python",
            )
        )

    return SearchResponse(
        original_query=request.query,
        mode="fuzzy",
        backend="python",
        elapsed_time=time.time() - start_time,
        ranked_results=results,
        warnings=[],
    )


def fuzzy_score(query: str, candidate: str) -> float:
    """Return a normalized fuzzy relevance score in the range 0.0 to 1.0."""

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

    distance = levenshtein_distance(normalized_query, normalized_candidate)
    max_length = max(len(normalized_query), len(normalized_candidate))

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


def levenshtein_distance(word1: str, word2: str) -> float:
    m, n = len(word1), len(word2)
    dp = [[float("inf") for _ in range(n + 1)] for _ in range(m + 1)]

    # base cases
    for i in range(m + 1):
        dp[i][n] = m - i
    for j in range(n + 1):
        dp[m][j] = n - j

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if word1[i] == word2[j]:
                dp[i][j] = dp[i + 1][j + 1]
            else:
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])

    return dp[0][0]

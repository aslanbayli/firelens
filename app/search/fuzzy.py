"""Python fuzzy-search flow.

This module owns request handling, timing, and conversion from
stored Symbol records into SearchResult records. It does not contain SQL.
"""

import time
import uuid

from app.acceleration.protocol import AccelerationBackend, AccelerationError
from app.acceleration.python_backend import (
    DEFAULT_MINIMUM_FUZZY_SCORE,
    PythonBackend,
    fuzzy_score,
    levenshtein_distance,
    normalize_identifier,
    split_camel_case,
)
from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.search.limits import (
    MAX_FUZZY_CANDIDATE_CHARS,
    MAX_FUZZY_QUERY_CHARS,
    bounded_symbol_name,
)
from app.storage.database import SQLiteIndexStore

MIN_FUZZY_SCORE = DEFAULT_MINIMUM_FUZZY_SCORE
_PYTHON_BACKEND = PythonBackend()


def fuzzy_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
    minimum_score: float = MIN_FUZZY_SCORE,
    max_candidates: int = 512,
    cancellation_callback: CancellationCallback | None = None,
    backend: AccelerationBackend = _PYTHON_BACKEND,
    fallback_backend: AccelerationBackend | None = None,
    minimum_accelerated_candidates: int = 1,
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
            backend=backend.name,
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
    if minimum_accelerated_candidates < 1:
        raise ValueError("minimum_accelerated_candidates must be greater than 0")

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

    # Score all short and qualified names in one backend call. Full symbol
    # records and snippets remain in Python and are fetched only after ranking.
    candidate_names = [
        (candidate.name, candidate.qualified_name) for candidate in candidates
    ]
    active_backend = backend
    if (
        fallback_backend is not None
        and len(candidate_names) < minimum_accelerated_candidates
    ):
        active_backend = fallback_backend
    warnings: list[str] = []
    try:
        scores = active_backend.fuzzy_scores(
            query,
            candidate_names,
            minimum_score,
        )
    except AccelerationError:
        if fallback_backend is None:
            raise
        active_backend = fallback_backend
        scores = active_backend.fuzzy_scores(
            query,
            candidate_names,
            minimum_score,
        )
        warnings.append("Mojo fuzzy acceleration failed; using Python")
    raise_if_cancelled(cancellation_callback)

    scored_symbols = []
    for score_value, candidate in zip(scores, candidates, strict=True):
        score = float(score_value)
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
                backend=active_backend.name,
            )
        )

    return SearchResponse(
        original_query=request.query,
        requested_mode=request.request_mode,
        mode="fuzzy",
        requested_backend=request.backend,
        backend=active_backend.name,
        elapsed_time=time.perf_counter() - start_time,
        ranked_results=results,
        warnings=warnings,
    )

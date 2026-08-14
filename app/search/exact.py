"""Python exact-search flow.

This module owns request handling, timing, and conversion from
stored Symbol records into SearchResult records. It does not contain SQL.
"""

import time
import uuid

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.search.limits import bounded_symbol_name
from app.storage.database import SQLiteIndexStore


def exact_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
    cancellation_callback: CancellationCallback | None = None,
) -> SearchResponse:
    # Trim user input according to the explicit exact-search normalization
    # rule. Start with whitespace only; do not lowercase unless exact search
    # becomes intentionally case-insensitive across the product.
    raise_if_cancelled(cancellation_callback)
    query = request.query.strip()

    # Empty exact queries should return no results. They should not fall
    # through into a broad storage query.
    if query == "":
        return SearchResponse(
            original_query=request.query,
            requested_mode=request.request_mode,
            mode="exact",
            requested_backend=request.backend,
            backend="python",
            elapsed_time=0.0,
            ranked_results=[],
            warnings=[],
        )

    # Start timing after cheap validation so elapsed_time represents the
    # retrieval path.
    start_time = time.perf_counter()

    # Ask the storage layer for already-ordered exact matches. Storage owns
    # raw SQL and should rank qualified-name matches before short-name matches.
    raise_if_cancelled(cancellation_callback)
    symbols = store.exact_search_symbols(
        repository_id=repository_id,
        query=query,
        path_filter=request.path,
        limit=request.top_k,
        max_snippet_chars=request.max_snippet_chars,
    )
    raise_if_cancelled(cancellation_callback)

    # Convert each Symbol into the public result contract. Exact symbol
    # matches can start with score 1.0 because ordering is deterministic and
    # all returned rows are literal matches.
    results: list[SearchResult] = []
    for symbol in symbols:
        raise_if_cancelled(cancellation_callback)
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
                score=1.0,
                mode="exact",
                backend="python",
            )
        )

    return SearchResponse(
        original_query=request.query,
        requested_mode=request.request_mode,
        mode="exact",
        requested_backend=request.backend,
        backend="python",
        elapsed_time=time.perf_counter() - start_time,
        ranked_results=results,
        warnings=[],
    )

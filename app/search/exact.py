"""Pseudocode for the Python exact-search flow.

This module should eventually own request handling, timing, and conversion from
stored Symbol records into SearchResult records. It should not contain SQL.
"""

import time
import uuid

from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.storage.database import SQLiteIndexStore


def search_exact(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
) -> SearchResponse:
    # Trim user input according to the explicit exact-search normalization
    # rule. Start with whitespace only; do not lowercase unless exact search
    # becomes intentionally case-insensitive across the product.
    query = request.query.strip()

    # Empty exact queries should return no results. They should not fall
    # through into a broad storage query.
    if query == "":
        return SearchResponse(
            original_query=request.query,
            mode="exact",
            backend="python",
            elapsed_time=0.0,
            ranked_results=[],
            warnings=[],
        )

    # Start timing after cheap validation so elapsed_time represents the
    # retrieval path.
    start_time = time.time()

    # Ask the storage layer for already-ordered exact matches. Storage owns
    # raw SQL and should rank qualified-name matches before short-name matches.
    symbols = store.find_exact_symbols(
        repository_id=repository_id,
        query=query,
        path_filter=request.path,
        limit=request.top_k,
    )

    # Convert each Symbol into the public result contract. Exact symbol
    # matches can start with score 1.0 because ordering is deterministic and
    # all returned rows are literal matches.
    results: list[SearchResult] = []
    for symbol in symbols:
        results.append(
            SearchResult(
                id=symbol.id,
                result_type="symbol",
                file_path=symbol.relative_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol_name=symbol.qualified_name,
                snippet=symbol.source_snippet,
                score=1.0,
                mode="exact",
                backend="python",
            )
        )

    return SearchResponse(
        original_query=request.query,
        mode="exact",
        backend="python",
        elapsed_time=time.time() - start_time,
        ranked_results=results,
        warnings=[],
    )

"""Python semantic-search flow."""

import time
import uuid

import numpy as np

from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.indexing.embedder import Embedder
from app.storage.database import SQLiteIndexStore


def semantic_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
    embedder: Embedder,
) -> SearchResponse:
    """Search stored code chunks by cosine similarity."""

    query = request.query.strip()

    if query == "":
        return SearchResponse(
            original_query=request.query,
            mode="semantic",
            backend="python",
            elapsed_time=0.0,
            ranked_results=[],
            warnings=[],
        )

    if request.top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    repository = store.load_repository(repository_id)
    if repository is None:
        raise ValueError("Repository index was not found")

    if repository.embedding_model != embedder.model:
        raise ValueError("Embedding model does not match the repository index")

    if repository.embedding_dim != embedder.dimension:
        raise ValueError("Embedding dimension does not match the repository index")

    start_time = time.perf_counter()

    candidates = store.load_semantic_candidates(
        repository_id=repository_id,
        path_filter=request.path,
    )

    if not candidates:
        return SearchResponse(
            original_query=request.query,
            mode="semantic",
            backend="python",
            elapsed_time=time.perf_counter() - start_time,
            ranked_results=[],
            warnings=[],
        )

    matrix = np.asarray(
        [candidate.vector for candidate in candidates],
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError("Stored embedding matrix must be two-dimensional")

    if matrix.shape[0] != len(candidates):
        raise ValueError("Embedding matrix row count does not match candidates")

    if matrix.shape[1] != repository.embedding_dim:
        raise ValueError("Stored embedding dimension does not match repository")

    query_vector = np.asarray(embedder.embed_query(query), dtype=np.float32)

    if query_vector.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional")

    if query_vector.shape[0] != repository.embedding_dim:
        raise ValueError("Query embedding dimension does not match repository")

    if not np.all(np.isfinite(query_vector)):
        raise ValueError("Query embedding contains non-finite values")

    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Query embedding cannot be a zero vector")

    normalized_query = query_vector / query_norm

    # Embeddings are validated as finite, nonzero unit vectors before storage,
    # so the persisted matrix does not need another full validation pass here.
    raw_scores = matrix @ normalized_query
    raw_scores = np.clip(raw_scores, -1.0, 1.0)

    # Sort score indices, not scores themselves. Each index remains connected
    # to the candidate that supplies the SearchResult metadata.
    ranked_indices = np.argsort(-raw_scores, kind="stable")
    selected_indices = ranked_indices[: request.top_k]

    results: list[SearchResult] = []
    for candidate_index in selected_indices:
        index = int(candidate_index)
        candidate = candidates[index]
        chunk = candidate.chunk

        public_score = float(np.clip((raw_scores[index] + 1.0) / 2.0, 0.0, 1.0))

        results.append(
            SearchResult(
                id=chunk.id,
                result_type="chunk",
                file_path=chunk.relative_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                symbol_name=candidate.qualified_symbol_name,
                snippet=chunk.raw_text,
                score=public_score,
                mode="semantic",
                backend="python",
            )
        )

    return SearchResponse(
        original_query=request.query,
        mode="semantic",
        backend="python",
        elapsed_time=time.perf_counter() - start_time,
        ranked_results=results,
        warnings=[],
    )

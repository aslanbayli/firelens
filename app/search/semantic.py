"""Python semantic-search flow."""

import time
import uuid
from dataclasses import dataclass

import numpy as np

from app.acceleration.protocol import AccelerationBackend, AccelerationError
from app.acceleration.python_backend import PythonBackend
from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.indexing.embedder import Embedder
from app.search.limits import bounded_symbol_name
from app.storage.database import (
    SQLiteIndexStore,
    SemanticCandidateSummary,
    StoredSemanticCandidate,
)


@dataclass(frozen=True)
class SemanticSearchIndex:
    """Candidate metadata and its aligned, cached vector matrix."""

    candidates: list[StoredSemanticCandidate]
    matrix: np.ndarray


_PYTHON_BACKEND = PythonBackend()


def load_semantic_search_index(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    path_filter: str | None = None,
    *,
    max_candidates: int,
    max_vector_bytes: int,
    cancellation_callback: CancellationCallback | None = None,
    summary: SemanticCandidateSummary | None = None,
) -> SemanticSearchIndex:
    """Load a bounded candidate list and contiguous matrix from SQLite."""

    raise_if_cancelled(cancellation_callback)
    if summary is None:
        summary = summarize_semantic_search_index(
            store,
            repository_id,
            path_filter,
            max_candidates=max_candidates,
            max_vector_bytes=max_vector_bytes,
            cancellation_callback=cancellation_callback,
        )
    else:
        _validate_semantic_summary(
            summary,
            max_candidates=max_candidates,
            max_vector_bytes=max_vector_bytes,
        )

    if summary.count == 0:
        matrix = np.empty((0, 0), dtype=np.float32)
        matrix.setflags(write=False)
        return SemanticSearchIndex(candidates=[], matrix=matrix)

    if summary.dimension is None or summary.dimension < 1:
        raise ValueError("Stored embedding dimension is invalid")

    expected_vector_bytes = summary.count * summary.dimension * np.dtype(
        np.float32
    ).itemsize
    if summary.vector_bytes != expected_vector_bytes:
        raise ValueError("Stored embedding byte size does not match its dimension")

    raise_if_cancelled(cancellation_callback)
    matrix = np.empty((summary.count, summary.dimension), dtype=np.float32)
    candidates: list[StoredSemanticCandidate] = []
    for row_number, stored_row in enumerate(
        store.iter_semantic_candidate_rows(
            repository_id=repository_id,
            path_filter=path_filter,
        )
    ):
        raise_if_cancelled(cancellation_callback)
        if row_number >= summary.count:
            raise ValueError("Semantic candidate count changed while loading")
        vector = np.frombuffer(stored_row.vector_blob, dtype=np.float32)
        if vector.shape != (summary.dimension,):
            raise ValueError("Stored embedding dimension is inconsistent")
        matrix[row_number] = vector
        candidates.append(stored_row.candidate)

    raise_if_cancelled(cancellation_callback)
    if len(candidates) != summary.count:
        raise ValueError("Semantic candidate count changed while loading")

    matrix.setflags(write=False)
    return SemanticSearchIndex(candidates=candidates, matrix=matrix)


def summarize_semantic_search_index(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    path_filter: str | None = None,
    *,
    max_candidates: int,
    max_vector_bytes: int,
    cancellation_callback: CancellationCallback | None = None,
) -> SemanticCandidateSummary:
    """Read bounded matrix dimensions before reserving search-cache memory."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be greater than 0")
    if max_vector_bytes < 1:
        raise ValueError("max_vector_bytes must be greater than 0")
    raise_if_cancelled(cancellation_callback)

    def check_cancellation() -> None:
        raise_if_cancelled(cancellation_callback)

    summary = store.semantic_candidate_summary(
        repository_id=repository_id,
        path_filter=path_filter,
        max_candidates=max_candidates,
        max_vector_bytes=max_vector_bytes,
        cancellation_check=(
            check_cancellation if cancellation_callback is not None else None
        ),
    )
    raise_if_cancelled(cancellation_callback)
    _validate_semantic_summary(
        summary,
        max_candidates=max_candidates,
        max_vector_bytes=max_vector_bytes,
    )
    return summary


def _validate_semantic_summary(
    summary: SemanticCandidateSummary,
    *,
    max_candidates: int,
    max_vector_bytes: int,
) -> None:
    """Reject a summary that exceeds allocation or shape limits."""

    if summary.count > max_candidates:
        raise ValueError(
            "Semantic search candidate limit was exceeded; narrow the path filter"
        )
    if summary.vector_bytes > max_vector_bytes:
        raise ValueError(
            "Semantic search vector memory limit was exceeded; narrow the path filter"
        )
    if summary.count > 0 and (summary.dimension is None or summary.dimension < 1):
        raise ValueError("Stored embedding dimension is invalid")
    if summary.count == 0:
        return
    expected_vector_bytes = (
        summary.count * summary.dimension * np.dtype(np.float32).itemsize
    )
    if summary.vector_bytes != int(expected_vector_bytes):
        raise ValueError("Stored embedding byte size does not match its dimension")


def semantic_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
    embedder: Embedder,
    search_index: SemanticSearchIndex | None = None,
    cancellation_callback: CancellationCallback | None = None,
    backend: AccelerationBackend = _PYTHON_BACKEND,
    fallback_backend: AccelerationBackend | None = None,
    score_floor: float | None = None,
) -> SearchResponse:
    """Search stored code chunks by cosine similarity."""

    raise_if_cancelled(cancellation_callback)
    query = request.query.strip()

    if query == "":
        return SearchResponse(
            original_query=request.query,
            requested_mode=request.request_mode,
            mode="semantic",
            requested_backend=request.backend,
            backend=backend.name,
            elapsed_time=0.0,
            ranked_results=[],
            warnings=[],
        )

    if request.top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    repository = store.load_repository(repository_id)
    raise_if_cancelled(cancellation_callback)
    if repository is None:
        raise ValueError("Repository index was not found")

    raise_if_cancelled(cancellation_callback)
    if repository.embedding_model != embedder.model:
        raise ValueError("Embedding model does not match the repository index")

    raise_if_cancelled(cancellation_callback)
    if repository.embedding_provider != embedder.provider:
        raise ValueError("Embedding provider does not match the repository index")

    raise_if_cancelled(cancellation_callback)
    if repository.embedding_dim != embedder.dimension:
        raise ValueError("Embedding dimension does not match the repository index")
    raise_if_cancelled(cancellation_callback)

    start_time = time.perf_counter()

    loaded_index = search_index or load_semantic_search_index(
        store,
        repository_id,
        path_filter=request.path,
        max_candidates=50_000,
        max_vector_bytes=192 * 1024 * 1024,
        cancellation_callback=cancellation_callback,
    )
    candidates = loaded_index.candidates

    if not candidates:
        return SearchResponse(
            original_query=request.query,
            requested_mode=request.request_mode,
            mode="semantic",
            requested_backend=request.backend,
            backend=backend.name,
            elapsed_time=time.perf_counter() - start_time,
            ranked_results=[],
            warnings=[],
        )

    matrix = loaded_index.matrix

    if matrix.ndim != 2:
        raise ValueError("Stored embedding matrix must be two-dimensional")

    if matrix.shape[0] != len(candidates):
        raise ValueError("Embedding matrix row count does not match candidates")

    if matrix.shape[1] != repository.embedding_dim:
        raise ValueError("Stored embedding dimension does not match repository")

    raise_if_cancelled(cancellation_callback)
    query_vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
    raise_if_cancelled(cancellation_callback)

    if query_vector.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional")

    if query_vector.shape[0] != repository.embedding_dim:
        raise ValueError("Query embedding dimension does not match repository")

    if not np.all(np.isfinite(query_vector)):
        raise ValueError("Query embedding contains non-finite values")

    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Query embedding cannot be a zero vector")

    normalized_query = np.ascontiguousarray(
        query_vector / query_norm,
        dtype=np.float32,
    )

    # Embeddings are validated as finite, nonzero unit vectors before storage.
    # The selected backend owns the final contiguous-array boundary checks.
    raise_if_cancelled(cancellation_callback)
    active_backend = backend
    warnings: list[str] = []
    try:
        ranked_scores = active_backend.semantic_top_k(
            matrix,
            normalized_query,
            request.top_k,
        )
    except AccelerationError:
        if fallback_backend is None:
            raise
        active_backend = fallback_backend
        ranked_scores = active_backend.semantic_top_k(
            matrix,
            normalized_query,
            request.top_k,
        )
        warnings.append("Mojo semantic acceleration failed; using Python")
    raise_if_cancelled(cancellation_callback)
    selected_indices = ranked_scores.indices
    selected_candidates = [candidates[int(index)] for index in selected_indices]
    chunk_texts = store.load_chunk_texts(
        (candidate.chunk_id for candidate in selected_candidates),
        max_chars=request.max_snippet_chars,
    )
    raise_if_cancelled(cancellation_callback)

    results: list[SearchResult] = []
    for raw_score, candidate in zip(
        ranked_scores.scores,
        selected_candidates,
        strict=True,
    ):
        raise_if_cancelled(cancellation_callback)
        raw_text = chunk_texts.get(candidate.chunk_id)
        if raw_text is None:
            raise ValueError("Ranked semantic chunk was not found")
        snippet = raw_text[: request.max_snippet_chars]

        public_score = float(np.clip((float(raw_score) + 1.0) / 2.0, 0.0, 1.0))
        if score_floor is not None and public_score < score_floor:
            continue

        results.append(
            SearchResult(
                id=candidate.chunk_id,
                result_type="chunk",
                file_path=candidate.relative_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                symbol_name=bounded_symbol_name(candidate.qualified_symbol_name),
                language=candidate.language,
                semantic_unit_kind=candidate.semantic_unit_kind,
                snippet=snippet,
                snippet_truncated=len(snippet) < len(raw_text),
                score=public_score,
                mode="semantic",
                backend=active_backend.name,
                retrieval_channels=["semantic"],
                retrieval_evidence=[
                    {
                        "channel": "semantic",
                        "score": public_score,
                        "rank": len(results) + 1,
                    }
                ],
            )
        )

    return SearchResponse(
        original_query=request.query,
        requested_mode=request.request_mode,
        mode="semantic",
        requested_backend=request.backend,
        backend=active_backend.name,
        elapsed_time=time.perf_counter() - start_time,
        ranked_results=results,
        warnings=warnings,
    )

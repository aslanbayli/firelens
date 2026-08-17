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
from app.search.relevance import is_result_kind_requested
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

SEMANTIC_RANKING_VERSION = "complete-context-v3"

# Semantic chunks are intentionally fine-grained so comments and docstrings can
# support symbol context or targeted searches. Ranking more matches than the
# caller requests leaves room to collapse overlapping chunks into one result.
_MATCH_POOL_FACTOR = 8

# A symbol-owned match can be returned with complete declaration context, so
# prefer it when vector similarities are close.
_SYMBOL_CONTEXT_BONUS = 0.03


@dataclass(frozen=True)
class _RankedSemanticMatch:
    """One vector match with its public relevance and stable source rank."""

    candidate: StoredSemanticCandidate
    source_rank: int
    raw_score: float
    similarity_score: float
    relevance_score: float


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
    candidate_pool_size: int | None = None,
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

    result_limit = (
        request.top_k if candidate_pool_size is None else candidate_pool_size
    )
    if not 1 <= result_limit <= 20:
        raise ValueError("candidate_pool_size must be between 1 and 20")

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
    # Rank a larger evidence pool because several high-scoring chunks may belong
    # to the same symbol and will collapse into one returned result.
    raise_if_cancelled(cancellation_callback)
    match_limit = min(len(candidates), result_limit * _MATCH_POOL_FACTOR)
    active_backend = backend
    warnings: list[str] = []
    try:
        ranked_scores = active_backend.semantic_top_k(
            matrix,
            normalized_query,
            match_limit,
        )
    except AccelerationError:
        if fallback_backend is None:
            raise
        active_backend = fallback_backend
        ranked_scores = active_backend.semantic_top_k(
            matrix,
            normalized_query,
            match_limit,
        )
        warnings.append("Mojo semantic acceleration failed; using Python")
    raise_if_cancelled(cancellation_callback)

    matches = _ranked_semantic_matches(
        candidates,
        ranked_scores.indices,
        ranked_scores.scores,
    )
    selected_matches = _select_semantic_matches(
        matches,
        query=query,
        result_limit=result_limit,
        score_floor=score_floor,
    )

    selected_symbol_ids = [
        match.candidate.symbol_id
        for match in selected_matches
        if match.candidate.symbol_id is not None
    ]
    symbols_by_id = store.load_symbols_by_ids(
        selected_symbol_ids,
        max_snippet_chars=request.max_snippet_chars,
    )
    selected_file_paths = [
        match.candidate.relative_path
        for match in selected_matches
        if match.candidate.symbol_id is None
    ]
    files_by_path = store.load_file_sources_by_paths(
        repository_id,
        selected_file_paths,
        max_snippet_chars=request.max_snippet_chars,
    )
    raise_if_cancelled(cancellation_callback)

    results: list[SearchResult] = []
    for result_rank, match in enumerate(selected_matches, start=1):
        raise_if_cancelled(cancellation_callback)
        candidate = match.candidate
        if candidate.symbol_id is not None:
            symbol = symbols_by_id.get(candidate.symbol_id)
            if symbol is None:
                raise ValueError("Ranked semantic symbol was not found")
            snippet = symbol.source_snippet[: request.max_snippet_chars]
            result = SearchResult(
                id=symbol.id,
                result_type="symbol",
                file_path=symbol.relative_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol_name=bounded_symbol_name(symbol.qualified_name),
                symbol_id=symbol.id,
                language=symbol.language,
                snippet=snippet,
                snippet_truncated=len(snippet) < len(symbol.source_snippet),
                score=match.relevance_score,
                mode="semantic",
                backend=active_backend.name,
                retrieval_channels=["semantic"],
                retrieval_evidence=[
                    {
                        "channel": "semantic",
                        "score": match.relevance_score,
                        "rank": result_rank,
                        "raw_score": match.raw_score,
                    }
                ],
            )
        else:
            source_file = files_by_path.get(candidate.relative_path)
            if source_file is None:
                raise ValueError("Ranked semantic file was not found")
            snippet = source_file.source_text[: request.max_snippet_chars]
            result = SearchResult(
                id=_file_result_id(repository_id, candidate.relative_path),
                result_type="file",
                file_path=source_file.relative_path,
                start_line=1,
                end_line=source_file.line_count,
                language=source_file.language,
                semantic_unit_kind=candidate.semantic_unit_kind,
                snippet=snippet,
                snippet_truncated=len(snippet) < len(source_file.source_text),
                score=match.relevance_score,
                mode="semantic",
                backend=active_backend.name,
                retrieval_channels=["semantic"],
                retrieval_evidence=[
                    {
                        "channel": "semantic",
                        "score": match.relevance_score,
                        "rank": result_rank,
                        "raw_score": match.raw_score,
                    }
                ],
            )

        results.append(result)

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


def _ranked_semantic_matches(
    candidates: list[StoredSemanticCandidate],
    ranked_indices: np.ndarray,
    raw_scores: np.ndarray,
) -> list[_RankedSemanticMatch]:
    """Attach context-aware public relevance to backend-ranked matches."""

    matches: list[_RankedSemanticMatch] = []
    for source_rank, (index, raw_score_value) in enumerate(
        zip(ranked_indices, raw_scores, strict=True),
        start=1,
    ):
        candidate = candidates[int(index)]
        raw_score = float(raw_score_value)
        similarity_score = float(np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0))
        relevance_score = _contextual_relevance_score(
            candidate,
            similarity_score,
        )
        matches.append(
            _RankedSemanticMatch(
                candidate=candidate,
                source_rank=source_rank,
                raw_score=raw_score,
                similarity_score=similarity_score,
                relevance_score=relevance_score,
            )
        )
    return matches


def _contextual_relevance_score(
    candidate: StoredSemanticCandidate,
    similarity_score: float,
) -> float:
    """Prefer matches that can be presented with useful source context."""

    if candidate.symbol_id is not None:
        adjustment = _SYMBOL_CONTEXT_BONUS
    else:
        adjustment = 0.0
    return float(np.clip(similarity_score + adjustment, 0.0, 1.0))


def _select_semantic_matches(
    matches: list[_RankedSemanticMatch],
    *,
    query: str,
    result_limit: int,
    score_floor: float | None,
) -> list[_RankedSemanticMatch]:
    """Drop unrequested fragments and collapse matches by source entity."""

    best_by_identity: dict[tuple[str, str], _RankedSemanticMatch] = {}
    for match in matches:
        if (
            match.candidate.symbol_id is None
            and not is_result_kind_requested(
                query,
                match.candidate.semantic_unit_kind,
            )
        ):
            continue
        identity = _semantic_match_identity(match.candidate)
        current = best_by_identity.get(identity)
        if current is None or _semantic_match_sort_key(
            match
        ) < _semantic_match_sort_key(current):
            best_by_identity[identity] = match

    ranked = sorted(best_by_identity.values(), key=_semantic_match_sort_key)
    if score_floor is not None:
        ranked = [
            match for match in ranked if match.relevance_score >= score_floor
        ]
    return ranked[:result_limit]


def _semantic_match_identity(
    candidate: StoredSemanticCandidate,
) -> tuple[str, str]:
    if candidate.symbol_id is not None:
        return "symbol", str(candidate.symbol_id)
    return "file", candidate.relative_path


def _file_result_id(repository_id: uuid.UUID, relative_path: str) -> uuid.UUID:
    """Return a stable public identity for one indexed file result."""

    return uuid.uuid5(repository_id, f"file:{relative_path}")


def _semantic_match_sort_key(
    match: _RankedSemanticMatch,
) -> tuple[float, float, int, str, int, int, str]:
    candidate = match.candidate
    return (
        -match.relevance_score,
        -match.similarity_score,
        match.source_rank,
        candidate.relative_path,
        candidate.start_line,
        candidate.end_line,
        str(candidate.chunk_id),
    )

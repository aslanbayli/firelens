"""Deterministic multi-channel lexical retrieval over SQLite FTS5."""

import re
import time
import uuid
from dataclasses import dataclass, field

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.config import Settings
from app.core.models import (
    RetrievalEvidence,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.indexing.adapters import DEFAULT_ADAPTER_REGISTRY
from app.search.fuzzy import fuzzy_search
from app.search.limits import bounded_symbol_name
from app.storage.database import (
    SQLiteIndexStore,
    SearchCandidateLimitError,
    StoredLexicalCandidate,
)


_FTS_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_PATH_LIKE = re.compile(r"[/\\]|(?:^|\s)[.\w-]+\.[A-Za-z0-9]{1,12}(?:$|\s)")
_CHANNEL_ORDER = {
    "exact_qualified": 0,
    "exact_short": 1,
    "path": 2,
    "identifier": 3,
    "bm25": 4,
    "fuzzy_symbol": 5,
}


@dataclass(frozen=True)
class LexicalSearchConfig:
    exact_qualified_bonus: float
    exact_short_bonus: float
    path_bonus: float
    identifier_bonus: float
    bm25_bonus: float
    fuzzy_bonus: float
    exact_limit: int
    path_limit: int
    identifier_limit: int
    bm25_limit: int
    fuzzy_limit: int
    maximum_documents_ranked: int
    minimum_fuzzy_score: float
    bm25_field_weights: tuple[float, float, float, float, float]

    @classmethod
    def from_settings(cls, settings: Settings) -> "LexicalSearchConfig":
        return cls(
            exact_qualified_bonus=settings.lexical_exact_qualified_bonus,
            exact_short_bonus=settings.lexical_exact_short_bonus,
            path_bonus=settings.lexical_path_bonus,
            identifier_bonus=settings.lexical_identifier_bonus,
            bm25_bonus=settings.lexical_bm25_bonus,
            fuzzy_bonus=settings.lexical_fuzzy_bonus,
            exact_limit=settings.lexical_exact_candidate_limit,
            path_limit=settings.lexical_path_candidate_limit,
            identifier_limit=settings.lexical_identifier_candidate_limit,
            bm25_limit=settings.lexical_bm25_candidate_limit,
            fuzzy_limit=settings.lexical_fuzzy_candidate_limit,
            maximum_documents_ranked=settings.max_lexical_documents_ranked,
            minimum_fuzzy_score=settings.fuzzy_threshold,
            bm25_field_weights=(
                settings.bm25_name_weight,
                settings.bm25_qualified_name_weight,
                settings.bm25_identifier_weight,
                settings.bm25_path_weight,
                settings.bm25_content_weight,
            ),
        )


@dataclass
class _MergedResult:
    result: SearchResult
    evidence: dict[str, RetrievalEvidence] = field(default_factory=dict)
    weighted_scores: dict[str, float] = field(default_factory=dict)


def safe_fts_terms(query: str, maximum_terms: int = 32) -> tuple[str, ...]:
    """Tokenize user text into bounded plain terms, never raw FTS syntax."""

    if maximum_terms < 1:
        raise ValueError("maximum_terms must be greater than 0")
    normalized_terms = DEFAULT_ADAPTER_REGISTRY.identifier_terms(query)
    terms: list[str] = []
    seen: set[str] = set()
    for normalized in normalized_terms:
        for match in _FTS_TOKEN.finditer(normalized):
            term = match.group(0).casefold()[:128]
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= maximum_terms:
                return tuple(terms)
    return tuple(terms)


def build_safe_fts_query(query: str, column: str) -> str:
    """Quote validated tokens and scope them to one allowed FTS column."""

    if column not in {"identifier_terms", "content"}:
        raise ValueError(f"Unsupported FTS query column: {column}")
    return " OR ".join(
        f'{column} : "{term}"' for term in safe_fts_terms(query)
    )


def is_path_like_query(query: str) -> bool:
    return bool(_PATH_LIKE.search(query.strip()))


def lexical_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
    config: LexicalSearchConfig,
    *,
    candidate_pool_size: int | None = None,
    retrieval_config: str = "default",
    cancellation_callback: CancellationCallback | None = None,
) -> SearchResponse:
    """Merge exact, path, identifier, BM25, and typo-recovery channels."""

    raise_if_cancelled(cancellation_callback)
    started_at = time.perf_counter()
    result_limit = (
        request.top_k if candidate_pool_size is None else candidate_pool_size
    )
    if not 1 <= result_limit <= 20:
        raise ValueError("candidate_pool_size must be between 1 and 20")
    query = request.query.strip()
    if not query:
        return _empty_response(request, retrieval_config)

    merged: dict[tuple[str, uuid.UUID], _MergedResult] = {}
    warnings: list[str] = []

    exact_candidates = store.exact_lexical_candidates(
        repository_id,
        query,
        request.path,
        limit=min(config.exact_limit, config.maximum_documents_ranked),
        max_snippet_chars=request.max_snippet_chars,
    )
    qualified_rank = 0
    short_rank = 0
    for candidate in exact_candidates:
        raise_if_cancelled(cancellation_callback)
        if candidate.qualified_name == query:
            qualified_rank += 1
            _add_candidate(
                merged,
                candidate,
                "exact_qualified",
                qualified_rank,
                config.exact_qualified_bonus,
                request.max_snippet_chars,
                config.maximum_documents_ranked,
            )
        if candidate.name == query:
            short_rank += 1
            _add_candidate(
                merged,
                candidate,
                "exact_short",
                short_rank,
                config.exact_short_bonus,
                request.max_snippet_chars,
                config.maximum_documents_ranked,
            )

    if is_path_like_query(query):
        normalized_path_query = query.replace("\\", "/").strip().removeprefix("./")
        for rank, candidate in enumerate(
            store.path_lexical_candidates(
                repository_id,
                normalized_path_query,
                request.path,
                limit=min(config.path_limit, config.maximum_documents_ranked),
                max_snippet_chars=request.max_snippet_chars,
            ),
            start=1,
        ):
            raise_if_cancelled(cancellation_callback)
            _add_candidate(
                merged,
                candidate,
                "path",
                rank,
                config.path_bonus,
                request.max_snippet_chars,
                config.maximum_documents_ranked,
            )

    identifier_query = build_safe_fts_query(query, "identifier_terms")
    for rank, candidate in enumerate(
        store.fts_lexical_candidates(
            repository_id,
            identifier_query,
            request.path,
            limit=min(config.identifier_limit, config.maximum_documents_ranked),
            max_snippet_chars=request.max_snippet_chars,
            field_weights=config.bm25_field_weights,
        ),
        start=1,
    ):
        raise_if_cancelled(cancellation_callback)
        _add_candidate(
            merged,
            candidate,
            "identifier",
            rank,
            config.identifier_bonus,
            request.max_snippet_chars,
            config.maximum_documents_ranked,
        )

    body_columns = ["name", "qualified_name", "identifier_terms", "content"]
    if is_path_like_query(query):
        body_columns.append("relative_path")
    body_query = _build_safe_multi_column_query(query, body_columns)
    for rank, candidate in enumerate(
        store.fts_lexical_candidates(
            repository_id,
            body_query,
            request.path,
            limit=min(config.bm25_limit, config.maximum_documents_ranked),
            max_snippet_chars=request.max_snippet_chars,
            field_weights=config.bm25_field_weights,
        ),
        start=1,
    ):
        raise_if_cancelled(cancellation_callback)
        _add_candidate(
            merged,
            candidate,
            "bm25",
            rank,
            config.bm25_bonus,
            request.max_snippet_chars,
            config.maximum_documents_ranked,
        )

    fuzzy_request = request.model_copy(
        update={"top_k": min(result_limit, config.fuzzy_limit, 20)}
    )
    try:
        fuzzy_response = fuzzy_search(
            store,
            repository_id,
            fuzzy_request,
            minimum_score=config.minimum_fuzzy_score,
            max_candidates=min(
                config.fuzzy_limit,
                config.maximum_documents_ranked,
            ),
            cancellation_callback=cancellation_callback,
        )
    except SearchCandidateLimitError:
        fuzzy_response = None
        warnings.append(
            "Lexical fuzzy channel skipped because its candidate limit was exceeded"
        )
    if fuzzy_response is not None:
        for rank, result in enumerate(fuzzy_response.ranked_results, start=1):
            _add_search_result(
                merged,
                result,
                "fuzzy_symbol",
                rank,
                config.fuzzy_bonus,
                config.maximum_documents_ranked,
            )

    ranked = [_finalize_result(item) for item in merged.values()]
    ranked.sort(key=_result_sort_key)
    return SearchResponse(
        original_query=request.query,
        requested_mode=request.request_mode,
        mode="lexical",
        requested_backend=request.backend,
        backend="python",
        elapsed_time=time.perf_counter() - started_at,
        ranked_results=ranked[:result_limit],
        warnings=warnings,
        retrieval_config=retrieval_config,
    )


def _add_candidate(
    merged: dict[tuple[str, uuid.UUID], _MergedResult],
    candidate: StoredLexicalCandidate,
    channel: str,
    rank: int,
    weight: float,
    max_snippet_chars: int,
    maximum_documents: int,
) -> None:
    snippet = candidate.snippet[:max_snippet_chars]
    result = SearchResult(
        id=candidate.record_id,
        result_type=candidate.result_type,
        file_path=candidate.relative_path,
        start_line=candidate.start_line,
        end_line=candidate.end_line,
        symbol_name=bounded_symbol_name(candidate.qualified_name),
        language=candidate.language,
        semantic_unit_kind=candidate.semantic_unit_kind,
        snippet=snippet,
        snippet_truncated=len(snippet) < len(candidate.snippet),
        score=0.0,
        mode="lexical",
        backend="python",
    )
    _merge_result(merged, result, channel, rank, weight, 1.0, maximum_documents)


def _add_search_result(
    merged: dict[tuple[str, uuid.UUID], _MergedResult],
    result: SearchResult,
    channel: str,
    rank: int,
    weight: float,
    maximum_documents: int,
) -> None:
    lexical_result = result.model_copy(
        update={
            "mode": "lexical",
            "backend": "python",
            "retrieval_channels": [],
            "retrieval_evidence": [],
        }
    )
    _merge_result(
        merged,
        lexical_result,
        channel,
        rank,
        weight,
        result.score,
        maximum_documents,
    )


def _merge_result(
    merged: dict[tuple[str, uuid.UUID], _MergedResult],
    result: SearchResult,
    channel: str,
    rank: int,
    weight: float,
    channel_quality: float,
    maximum_documents: int,
) -> None:
    key = (result.result_type, result.id)
    existing = merged.get(key)
    if existing is None:
        if len(merged) >= maximum_documents:
            return
        existing = _MergedResult(result=result)
        merged[key] = existing
    rank_score = 1.0 / rank
    evidence_score = max(0.0, min(1.0, rank_score * channel_quality))
    existing.evidence[channel] = RetrievalEvidence(
        channel=channel,
        score=evidence_score,
        rank=rank,
    )
    existing.weighted_scores[channel] = weight * evidence_score


def _finalize_result(merged: _MergedResult) -> SearchResult:
    channels = sorted(merged.evidence, key=lambda channel: _CHANNEL_ORDER[channel])
    weighted = sorted(merged.weighted_scores.values(), reverse=True)
    score = weighted[0] if weighted else 0.0
    if len(weighted) > 1:
        score += 0.05 * sum(weighted[1:])
    return merged.result.model_copy(
        update={
            "score": min(1.0, score),
            "retrieval_channels": channels,
            "retrieval_evidence": [merged.evidence[channel] for channel in channels],
        }
    )


def _result_sort_key(result: SearchResult) -> tuple[int, float, str, int, int, str]:
    primary_channel = min(
        (_CHANNEL_ORDER[channel] for channel in result.retrieval_channels),
        default=len(_CHANNEL_ORDER),
    )
    return (
        primary_channel,
        -result.score,
        result.file_path,
        result.start_line,
        result.end_line,
        str(result.id),
    )


def _empty_response(request: SearchRequest, retrieval_config: str) -> SearchResponse:
    return SearchResponse(
        original_query=request.query,
        requested_mode=request.request_mode,
        mode="lexical",
        requested_backend=request.backend,
        backend="python",
        elapsed_time=0.0,
        ranked_results=[],
        retrieval_config=retrieval_config,
    )


def _build_safe_multi_column_query(query: str, columns: list[str]) -> str:
    allowed_columns = {
        "name",
        "qualified_name",
        "identifier_terms",
        "relative_path",
        "content",
    }
    if not columns or any(column not in allowed_columns for column in columns):
        raise ValueError("Unsupported FTS query columns")
    expressions = []
    for term in safe_fts_terms(query):
        quoted = f'"{term}"'
        expressions.extend(f"{column} : {quoted}" for column in columns)
    return " OR ".join(expressions)

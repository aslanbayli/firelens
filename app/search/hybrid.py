"""Deterministic candidate identity and pure-Python hybrid fusion."""

import math
import uuid
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import Settings
from app.core.models import RetrievalEvidence, SearchResponse, SearchResult


RetrieverName = Literal["lexical", "semantic"]


class ReciprocalRankFusionConfig(BaseModel):
    """Serializable settings for the named ``hybrid_rrf`` mode."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["hybrid_rrf"] = "hybrid_rrf"
    lexical_pool_size: int = Field(ge=1, le=20)
    semantic_pool_size: int = Field(ge=1, le=20)
    rrf_k: float = Field(gt=0.0, allow_inf_nan=False)
    lexical_weight: float = Field(ge=0.0, allow_inf_nan=False)
    semantic_weight: float = Field(ge=0.0, allow_inf_nan=False)
    final_top_k: int = Field(ge=1, le=20)
    tie_breaking_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def normalize_weights(self) -> "ReciprocalRankFusionConfig":
        total = self.lexical_weight + self.semantic_weight
        if total <= 0.0:
            raise ValueError("RRF source weights must sum to a positive value")
        self.lexical_weight /= total
        self.semantic_weight /= total
        return self

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        final_top_k: int,
    ) -> "ReciprocalRankFusionConfig":
        return cls(
            lexical_pool_size=settings.hybrid_lexical_pool_size,
            semantic_pool_size=settings.hybrid_semantic_pool_size,
            rrf_k=settings.hybrid_rrf_k,
            lexical_weight=settings.hybrid_rrf_lexical_weight,
            semantic_weight=settings.hybrid_rrf_semantic_weight,
            final_top_k=final_top_k,
            tie_breaking_version=settings.hybrid_tie_breaking_version,
        )


class NormalizedWeightedFusionConfig(BaseModel):
    """Serializable settings for the named ``hybrid_weighted`` mode."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["hybrid_weighted"] = "hybrid_weighted"
    lexical_pool_size: int = Field(ge=1, le=20)
    semantic_pool_size: int = Field(ge=1, le=20)
    lexical_weight: float = Field(ge=0.0, allow_inf_nan=False)
    semantic_weight: float = Field(ge=0.0, allow_inf_nan=False)
    normalization_method: Literal["min_max"] = "min_max"
    missing_source_value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    final_top_k: int = Field(ge=1, le=20)
    tie_breaking_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def normalize_weights(self) -> "NormalizedWeightedFusionConfig":
        total = self.lexical_weight + self.semantic_weight
        if total <= 0.0:
            raise ValueError("weighted source weights must sum to a positive value")
        self.lexical_weight /= total
        self.semantic_weight /= total
        return self

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        final_top_k: int,
    ) -> "NormalizedWeightedFusionConfig":
        return cls(
            lexical_pool_size=settings.hybrid_lexical_pool_size,
            semantic_pool_size=settings.hybrid_semantic_pool_size,
            lexical_weight=settings.hybrid_weighted_lexical_weight,
            semantic_weight=settings.hybrid_weighted_semantic_weight,
            missing_source_value=settings.hybrid_weighted_missing_source_value,
            final_top_k=final_top_k,
            tie_breaking_version=settings.hybrid_tie_breaking_version,
        )


@dataclass(frozen=True)
class RetrievalCandidate:
    """One bounded result with stable identity and source-specific evidence."""

    repository_id: uuid.UUID
    stable_record_id: uuid.UUID
    file_path: str
    start_line: int
    end_line: int
    result_type: Literal["symbol", "chunk", "file"]
    semantic_unit_kind: str | None
    language: str
    symbol_id: uuid.UUID | None
    qualified_name: str | None
    retriever: RetrieverName
    source_rank: int
    raw_score: float
    result: SearchResult


@dataclass(frozen=True)
class FusedCandidate:
    """A public result paired with its internal, unnormalized fusion score."""

    result: SearchResult
    raw_fusion_score: float


@dataclass
class _CandidateGroup:
    candidates: list[RetrievalCandidate]
    representative: RetrievalCandidate

    def best_from(self, retriever: RetrieverName) -> RetrievalCandidate | None:
        matching = [
            candidate
            for candidate in self.candidates
            if candidate.retriever == retriever
        ]
        if not matching:
            return None
        return min(matching, key=_source_candidate_sort_key)


def response_candidates(
    repository_id: uuid.UUID,
    retriever: RetrieverName,
    response: SearchResponse,
) -> list[RetrievalCandidate]:
    """Adapt a bounded component response to the shared candidate contract."""

    candidates: list[RetrievalCandidate] = []
    for rank, result in enumerate(response.ranked_results, start=1):
        raw_score = float(result.score)
        if not math.isfinite(raw_score):
            raise ValueError(f"{retriever} candidate score must be finite")
        symbol_id = result.symbol_id
        if symbol_id is None and result.result_type == "symbol":
            symbol_id = result.id
        candidates.append(
            RetrievalCandidate(
                repository_id=repository_id,
                stable_record_id=result.id,
                file_path=result.file_path,
                start_line=result.start_line,
                end_line=result.end_line,
                result_type=result.result_type,
                semantic_unit_kind=result.semantic_unit_kind,
                language=result.language,
                symbol_id=symbol_id,
                qualified_name=result.symbol_name,
                retriever=retriever,
                source_rank=rank,
                raw_score=raw_score,
                result=result,
            )
        )
    return candidates


def deduplicate_candidates(
    candidates: list[RetrievalCandidate],
) -> list[_CandidateGroup]:
    """Collapse record, symbol, and equivalent-source identities."""

    if not candidates:
        return []

    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            if _same_candidate_identity(left, candidates[right_index]):
                union(left_index, right_index)

    grouped: dict[int, list[RetrievalCandidate]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)

    groups = []
    for group_candidates in grouped.values():
        representative = min(group_candidates, key=_representative_sort_key)
        groups.append(
            _CandidateGroup(
                candidates=group_candidates,
                representative=representative,
            )
        )
    groups.sort(key=_group_tie_break_key)
    return groups


def reciprocal_rank_fusion(
    candidates: list[RetrievalCandidate],
    config: ReciprocalRankFusionConfig,
) -> list[FusedCandidate]:
    """Fuse candidates using weighted reciprocal ranks and stable ties."""

    groups = deduplicate_candidates(candidates)
    normalized_scores = _normalized_component_scores(candidates)
    scored_groups: list[tuple[_CandidateGroup, float]] = []
    for group in groups:
        raw_score = 0.0
        lexical = group.best_from("lexical")
        semantic = group.best_from("semantic")
        if lexical is not None:
            raw_score += config.lexical_weight / (
                config.rrf_k + lexical.source_rank
            )
        if semantic is not None:
            raw_score += config.semantic_weight / (
                config.rrf_k + semantic.source_rank
            )
        scored_groups.append((group, raw_score))

    scored_groups.sort(key=lambda item: (-item[1], _group_tie_break_key(item[0])))
    maximum_score = max((score for _, score in scored_groups), default=0.0)
    fused = []
    for group, raw_score in scored_groups[: config.final_top_k]:
        public_score = raw_score / maximum_score if maximum_score > 0.0 else 0.0
        fused.append(
            FusedCandidate(
                result=_build_fused_result(
                    group,
                    normalized_scores,
                    public_score,
                    mode="hybrid_rrf",
                    fusion_method="rrf",
                ),
                raw_fusion_score=raw_score,
            )
        )
    return fused


def normalized_weighted_fusion(
    candidates: list[RetrievalCandidate],
    config: NormalizedWeightedFusionConfig,
) -> list[FusedCandidate]:
    """Fuse independently normalized component scores with explicit weights."""

    groups = deduplicate_candidates(candidates)
    normalized_scores = _normalized_component_scores(candidates)
    scored_groups: list[tuple[_CandidateGroup, float]] = []
    for group in groups:
        lexical = group.best_from("lexical")
        semantic = group.best_from("semantic")
        lexical_score = (
            normalized_scores[_candidate_key(lexical)]
            if lexical is not None
            else config.missing_source_value
        )
        semantic_score = (
            normalized_scores[_candidate_key(semantic)]
            if semantic is not None
            else config.missing_source_value
        )
        score = (
            config.lexical_weight * lexical_score
            + config.semantic_weight * semantic_score
        )
        scored_groups.append((group, score))

    scored_groups.sort(key=lambda item: (-item[1], _group_tie_break_key(item[0])))
    return [
        FusedCandidate(
            result=_build_fused_result(
                group,
                normalized_scores,
                score,
                mode="hybrid_weighted",
                fusion_method="normalized_weighted",
            ),
            raw_fusion_score=score,
        )
        for group, score in scored_groups[: config.final_top_k]
    ]


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize one source, treating constant ranges as tied maxima."""

    if not scores:
        return []
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("source scores must be finite")
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0] * len(scores)
    score_range = maximum - minimum
    return [(score - minimum) / score_range for score in scores]


def _same_candidate_identity(
    left: RetrievalCandidate,
    right: RetrievalCandidate,
) -> bool:
    if left.repository_id != right.repository_id:
        return False
    if left.stable_record_id == right.stable_record_id:
        return True
    if left.symbol_id is not None and left.symbol_id == right.symbol_id:
        return True
    if left.file_path != right.file_path:
        return False
    if left.start_line == right.start_line and left.end_line == right.end_line:
        return True
    same_named_unit = (
        left.qualified_name is not None
        and left.qualified_name == right.qualified_name
    )
    ranges_overlap = (
        left.start_line <= right.end_line and right.start_line <= left.end_line
    )
    return same_named_unit and ranges_overlap


def _normalized_component_scores(
    candidates: list[RetrievalCandidate],
) -> dict[tuple[RetrieverName, int, uuid.UUID], float]:
    normalized: dict[tuple[RetrieverName, int, uuid.UUID], float] = {}
    for retriever in ("lexical", "semantic"):
        source_candidates = [
            candidate
            for candidate in candidates
            if candidate.retriever == retriever
        ]
        source_scores = normalize_scores(
            [candidate.raw_score for candidate in source_candidates]
        )
        for candidate, score in zip(source_candidates, source_scores, strict=True):
            normalized[_candidate_key(candidate)] = score
    return normalized


def _build_fused_result(
    group: _CandidateGroup,
    normalized_scores: dict[tuple[RetrieverName, int, uuid.UUID], float],
    final_score: float,
    *,
    mode: Literal["hybrid_rrf", "hybrid_weighted"],
    fusion_method: Literal["rrf", "normalized_weighted"],
) -> SearchResult:
    representative = group.representative.result
    evidence: dict[str, RetrievalEvidence] = {}
    ordered_channels: list[str] = []

    for retriever in ("lexical", "semantic"):
        candidate = group.best_from(retriever)
        if candidate is None:
            continue
        ordered_channels.append(retriever)
        evidence[retriever] = RetrievalEvidence(
            channel=retriever,
            score=normalized_scores[_candidate_key(candidate)],
            rank=candidate.source_rank,
            raw_score=candidate.raw_score,
            backend=candidate.result.backend,
        )
        if retriever == "lexical":
            for lexical_candidate in sorted(
                (
                    item
                    for item in group.candidates
                    if item.retriever == "lexical"
                ),
                key=_source_candidate_sort_key,
            ):
                for inherited in lexical_candidate.result.retrieval_evidence:
                    if inherited.channel in {"lexical", "semantic"}:
                        continue
                    inherited_with_backend = inherited.model_copy(
                        update={
                            "backend": inherited.backend
                            or lexical_candidate.result.backend
                        }
                    )
                    current = evidence.get(inherited.channel)
                    if current is None or (
                        inherited_with_backend.rank,
                        -inherited_with_backend.score,
                    ) < (current.rank, -current.score):
                        evidence[inherited.channel] = inherited_with_backend
                    if inherited.channel not in ordered_channels:
                        ordered_channels.append(inherited.channel)

    symbol_id = representative.symbol_id
    if symbol_id is None:
        symbol_id = next(
            (
                candidate.symbol_id
                for candidate in group.candidates
                if candidate.symbol_id is not None
            ),
            None,
        )
    return representative.model_copy(
        update={
            "symbol_id": symbol_id,
            "score": min(1.0, max(0.0, final_score)),
            "mode": mode,
            "fusion_method": fusion_method,
            "backend": _result_backend(group),
            "retrieval_channels": ordered_channels,
            "retrieval_evidence": [
                evidence[channel] for channel in ordered_channels
            ],
        }
    )


def _result_backend(group: _CandidateGroup) -> Literal["python", "mojo"]:
    semantic = group.best_from("semantic")
    if semantic is not None:
        return semantic.result.backend
    return "python"


def _candidate_key(
    candidate: RetrievalCandidate,
) -> tuple[RetrieverName, int, uuid.UUID]:
    return candidate.retriever, candidate.source_rank, candidate.stable_record_id


def _representative_sort_key(
    candidate: RetrievalCandidate,
) -> tuple[int, bool, int, str, int, int, str, str]:
    metadata_count = sum(
        value is not None
        for value in (
            candidate.symbol_id,
            candidate.qualified_name,
            candidate.semantic_unit_kind,
        )
    )
    return (
        -len(candidate.result.snippet),
        candidate.result.snippet_truncated,
        -metadata_count,
        *_candidate_tie_break_key(candidate),
    )


def _source_candidate_sort_key(
    candidate: RetrievalCandidate,
) -> tuple[int, str, int, int, str, str]:
    return candidate.source_rank, *_candidate_tie_break_key(candidate)


def _candidate_tie_break_key(
    candidate: RetrievalCandidate,
) -> tuple[str, int, int, str, str]:
    return (
        candidate.file_path,
        candidate.start_line,
        candidate.end_line,
        candidate.qualified_name or "",
        str(candidate.stable_record_id),
    )


def _group_tie_break_key(
    group: _CandidateGroup,
) -> tuple[str, int, int, str, str]:
    return _candidate_tie_break_key(group.representative)

"""Bounded, deterministic graph expansion over persisted adjacency."""

import time
import uuid
from dataclasses import dataclass

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.config import Settings
from app.core.models import (
    GraphPathEvidence,
    RetrievalEvidence,
    RetrievalTiming,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.search.limits import bounded_symbol_name
from app.storage.database import SQLiteIndexStore, StoredGraphNeighbor


@dataclass(frozen=True)
class GraphSearchConfig:
    seed_mode: str
    seed_count: int
    max_hops: int
    maximum_neighbors_per_node: int
    maximum_expanded_nodes: int
    allowed_edge_kinds: tuple[str, ...]
    directions: tuple[str, ...]
    minimum_edge_confidence: float
    hop_decay: float
    edge_weights: dict[str, float]

    @classmethod
    def from_settings(cls, settings: Settings) -> "GraphSearchConfig":
        return cls(
            seed_mode=settings.graph_seed_mode,
            seed_count=settings.graph_seed_count,
            max_hops=settings.graph_max_hops,
            maximum_neighbors_per_node=settings.graph_max_neighbors_per_node,
            maximum_expanded_nodes=settings.graph_max_expanded_nodes,
            allowed_edge_kinds=tuple(settings.graph_allowed_edge_kinds),
            directions=tuple(settings.graph_directions),
            minimum_edge_confidence=settings.graph_min_edge_confidence,
            hop_decay=settings.graph_hop_decay,
            edge_weights={
                "calls": settings.graph_calls_weight,
                "imports": settings.graph_imports_weight,
                "inherits": settings.graph_inherits_weight,
                "references": settings.graph_references_weight,
                "depends_on": settings.graph_depends_on_weight,
                "tests": settings.graph_tests_weight,
            },
        )


@dataclass(frozen=True)
class _TraversalState:
    node_id: uuid.UUID
    seed: SearchResult
    path_factor: float
    hop_count: int


@dataclass(frozen=True)
class _Expansion:
    node_id: uuid.UUID
    seed: SearchResult
    neighbor: StoredGraphNeighbor
    hop_count: int
    contribution: float
    path_factor: float


def graph_search(
    store: SQLiteIndexStore,
    repository_id: uuid.UUID,
    request: SearchRequest,
    seed_response: SearchResponse,
    config: GraphSearchConfig,
    *,
    retrieval_config: str,
    cancellation_callback: CancellationCallback | None = None,
) -> SearchResponse:
    """Expand a bounded seed response by at most two graph hops."""

    _validate_config(config)
    raise_if_cancelled(cancellation_callback)
    started_at = time.perf_counter()
    seeds = seed_response.ranked_results[: config.seed_count]
    seed_symbol_ids = [
        result.symbol_id or result.id
        for result in seeds
        if result.symbol_id is not None or result.result_type == "symbol"
    ]
    seed_paths = [result.file_path for result in seeds]
    graph_nodes = store.load_graph_nodes_for_results(
        repository_id,
        seed_symbol_ids,
        seed_paths,
    )
    raise_if_cancelled(cancellation_callback)

    nodes_by_symbol: dict[uuid.UUID, list[uuid.UUID]] = {}
    nodes_by_path: dict[str, list[uuid.UUID]] = {}
    for node in graph_nodes:
        if node.symbol_id is not None:
            nodes_by_symbol.setdefault(node.symbol_id, []).append(node.id)
        elif node.relative_path is not None:
            nodes_by_path.setdefault(node.relative_path, []).append(node.id)

    frontier: list[_TraversalState] = []
    seed_node_ids: set[uuid.UUID] = set()
    for seed in seeds:
        symbol_id = seed.symbol_id or (seed.id if seed.result_type == "symbol" else None)
        matching_node_ids = [
            *(nodes_by_symbol.get(symbol_id, []) if symbol_id else []),
            *nodes_by_path.get(seed.file_path, []),
        ]
        for node_id in matching_node_ids:
            seed_node_ids.add(node_id)
            frontier.append(
                _TraversalState(
                    node_id=node_id,
                    seed=seed,
                    path_factor=1.0,
                    hop_count=0,
                )
            )

    best_expansion_by_node: dict[uuid.UUID, _Expansion] = {}
    best_path_score: dict[tuple[uuid.UUID, uuid.UUID], float] = {}
    expanded_node_ids: set[uuid.UUID] = set()
    for hop_count in range(1, config.max_hops + 1):
        raise_if_cancelled(cancellation_callback)
        if not frontier:
            break
        adjacency = store.load_graph_adjacency(
            repository_id,
            [state.node_id for state in frontier],
            edge_kinds=config.allowed_edge_kinds,
            directions=config.directions,
            minimum_confidence=config.minimum_edge_confidence,
            maximum_neighbors_per_node=config.maximum_neighbors_per_node,
        )
        adjacency_by_node: dict[uuid.UUID, list[StoredGraphNeighbor]] = {}
        for neighbor in adjacency:
            adjacency_by_node.setdefault(neighbor.current_node_id, []).append(neighbor)

        next_frontier_by_key: dict[tuple[uuid.UUID, uuid.UUID], _TraversalState] = {}
        for state in frontier:
            raise_if_cancelled(cancellation_callback)
            for neighbor in adjacency_by_node.get(state.node_id, []):
                edge_weight = config.edge_weights.get(neighbor.kind, 0.0)
                if edge_weight <= 0.0:
                    continue
                path_factor = state.path_factor * edge_weight * neighbor.confidence
                contribution = min(
                    1.0,
                    state.seed.score * path_factor * (config.hop_decay**hop_count),
                )
                path_key = (neighbor.neighbor_node_id, state.seed.id)
                previous_path_score = best_path_score.get(path_key, -1.0)
                if contribution <= previous_path_score:
                    continue

                is_new_node = (
                    neighbor.neighbor_node_id not in seed_node_ids
                    and neighbor.neighbor_node_id not in expanded_node_ids
                )
                if is_new_node:
                    if len(expanded_node_ids) >= config.maximum_expanded_nodes:
                        continue
                    expanded_node_ids.add(neighbor.neighbor_node_id)
                best_path_score[path_key] = contribution

                if neighbor.neighbor_node_id not in seed_node_ids:
                    expansion = _Expansion(
                        node_id=neighbor.neighbor_node_id,
                        seed=state.seed,
                        neighbor=neighbor,
                        hop_count=hop_count,
                        contribution=contribution,
                        path_factor=path_factor,
                    )
                    previous = best_expansion_by_node.get(neighbor.neighbor_node_id)
                    if previous is None or _expansion_sort_key(expansion) < _expansion_sort_key(
                        previous
                    ):
                        best_expansion_by_node[neighbor.neighbor_node_id] = expansion

                next_state = _TraversalState(
                    node_id=neighbor.neighbor_node_id,
                    seed=state.seed,
                    path_factor=path_factor,
                    hop_count=hop_count,
                )
                next_frontier_by_key[path_key] = next_state
        frontier = list(next_frontier_by_key.values())

    stored_results = store.load_graph_results(
        repository_id,
        best_expansion_by_node,
        max_snippet_chars=request.max_snippet_chars,
        path_filter=request.path,
    )
    raise_if_cancelled(cancellation_callback)

    candidates: dict[tuple[object, ...], tuple[bool, SearchResult]] = {}
    for rank, seed in enumerate(seeds, start=1):
        graph_seed = seed.model_copy(
            update={
                "mode": "graph",
                "retrieval_channels": [*seed.retrieval_channels, "graph_seed"],
                "retrieval_evidence": [
                    *seed.retrieval_evidence,
                    RetrievalEvidence(
                        channel="graph_seed",
                        score=seed.score,
                        rank=rank,
                        raw_score=seed.score,
                        backend=seed.backend,
                    ),
                ],
            }
        )
        candidates[_result_identity(graph_seed)] = (True, graph_seed)

    for node_id, stored in stored_results.items():
        expansion = best_expansion_by_node[node_id]
        snippet = stored.snippet[: request.max_snippet_chars]
        result = SearchResult(
            id=stored.record_id,
            result_type=stored.result_type,
            file_path=stored.relative_path,
            start_line=stored.start_line,
            end_line=stored.end_line,
            symbol_name=bounded_symbol_name(stored.qualified_name or stored.name),
            symbol_id=stored.symbol_id,
            language=stored.language,
            semantic_unit_kind=stored.semantic_unit_kind,
            snippet=snippet,
            snippet_truncated=len(stored.snippet) > len(snippet),
            score=expansion.contribution,
            mode="graph",
            backend="python",
            retrieval_channels=["graph"],
            retrieval_evidence=[
                RetrievalEvidence(
                    channel="graph",
                    score=expansion.contribution,
                    rank=1,
                    raw_score=expansion.contribution,
                    backend="python",
                )
            ],
            graph_evidence=[
                GraphPathEvidence(
                    originating_seed_id=expansion.seed.id,
                    originating_seed_path=expansion.seed.file_path,
                    edge_kind=expansion.neighbor.kind,
                    direction=expansion.neighbor.direction,
                    hop_count=expansion.hop_count,
                    edge_confidence=expansion.neighbor.confidence,
                    graph_contribution=expansion.contribution,
                )
            ],
        )
        identity = _result_identity(result)
        previous = candidates.get(identity)
        if previous is None or result.score > previous[1].score:
            candidates[identity] = (False, result)

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate[1].score,
            0 if candidate[0] else 1,
            candidate[1].file_path,
            candidate[1].start_line,
            candidate[1].end_line,
            str(candidate[1].id),
        ),
    )[: request.top_k]
    ranked_results: list[SearchResult] = []
    graph_rank = 0
    for _is_seed, result in ordered:
        if result.graph_evidence:
            graph_rank += 1
            evidence = result.retrieval_evidence[0].model_copy(
                update={"rank": graph_rank}
            )
            result = result.model_copy(update={"retrieval_evidence": [evidence]})
        ranked_results.append(result)

    graph_elapsed = time.perf_counter() - started_at
    return SearchResponse(
        original_query=request.query,
        requested_mode="graph",
        mode="graph",
        requested_backend=request.backend,
        backend=seed_response.backend,
        elapsed_time=seed_response.elapsed_time + graph_elapsed,
        ranked_results=ranked_results,
        warnings=[f"Graph seeds: {warning}" for warning in seed_response.warnings],
        retrieval_timings=[
            *seed_response.retrieval_timings,
            RetrievalTiming(
                component="graph",
                elapsed_time=graph_elapsed,
                backend="python",
            ),
        ],
        retrieval_config=retrieval_config,
    )


def _validate_config(config: GraphSearchConfig) -> None:
    if not 1 <= config.max_hops <= 2:
        raise ValueError("Graph traversal max_hops must be between 1 and 2")
    if config.seed_count < 1 or config.maximum_neighbors_per_node < 1:
        raise ValueError("Graph traversal bounds must be positive")
    if config.maximum_expanded_nodes < 1:
        raise ValueError("maximum_expanded_nodes must be positive")
    if not config.allowed_edge_kinds or not config.directions:
        raise ValueError("Graph edge kinds and directions must not be empty")


def _expansion_sort_key(expansion: _Expansion) -> tuple[object, ...]:
    return (
        -expansion.contribution,
        expansion.hop_count,
        expansion.neighbor.kind,
        expansion.neighbor.direction,
        str(expansion.seed.id),
        str(expansion.neighbor.edge_id),
    )


def _result_identity(result: SearchResult) -> tuple[object, ...]:
    if result.symbol_id is not None:
        return ("symbol", result.symbol_id)
    if result.result_type == "symbol":
        return ("symbol", result.id)
    return (
        "span",
        result.file_path,
        result.start_line,
        result.end_line,
        result.semantic_unit_kind,
    )

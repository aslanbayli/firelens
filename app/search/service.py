"""Unified code-search service shared by CLI, Streamlit, and MCP."""

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from app.acceleration.mojo_backend import MojoBackend
from app.acceleration.protocol import AccelerationBackend, CapabilityName
from app.acceleration.python_backend import PythonBackend
from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.config import Settings, settings as default_settings
from app.core.coordinator import RepositoryBusyError, RepositoryCoordinator
from app.core.models import (
    RetrievalTiming,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.core.repositories import RepositoryResolver, ResolvedRepository
from app.indexing.embedder import CodeRankEmbedder, Embedder
from app.indexing.service import INDEX_FORMAT_VERSION
from app.search.exact import exact_search
from app.search.fuzzy import fuzzy_search
from app.search.graph import GraphSearchConfig, graph_search
from app.search.hybrid import (
    NormalizedWeightedFusionConfig,
    ReciprocalRankFusionConfig,
    normalized_weighted_fusion,
    reciprocal_rank_fusion,
    response_candidates,
)
from app.search.lexical import LexicalSearchConfig, lexical_search
from app.search.router import classify_non_exact_query
from app.search.semantic import (
    SEMANTIC_RANKING_VERSION,
    SemanticSearchIndex,
    load_semantic_search_index,
    semantic_search,
    summarize_semantic_search_index,
)
from app.storage.database import (
    SQLiteIndexStore,
    SemanticCandidateSummary,
    SearchCandidateLimitError,
    default_database_path,
)
from app.storage.locking import DatabaseLockBusyError, shared_database_lock


class IndexNotFoundError(ValueError):
    """Raised when search is requested before a repository is indexed."""


class BackendUnavailableError(ValueError):
    """Raised when a caller explicitly requires an unavailable backend."""


EmbedderFactory = Callable[[], Embedder]
SemanticCacheKey = tuple[str, str, str | None, int, int, int]
MAX_SEMANTIC_CACHE_ENTRIES = 4
SEMANTIC_CACHE_WAIT_SECONDS = 0.05


class SearchService:
    """Validate, route, execute, cache, and bound repository searches."""

    def __init__(
        self,
        settings: Settings = default_settings,
        resolver: RepositoryResolver | None = None,
        coordinator: RepositoryCoordinator | None = None,
        embedder_factory: EmbedderFactory | None = None,
        mojo_backend: AccelerationBackend | None = None,
    ) -> None:
        self.settings = settings
        self.resolver = resolver or RepositoryResolver(settings)
        self.coordinator = coordinator or RepositoryCoordinator()
        self._embedder_factory = embedder_factory or self._create_default_embedder
        self._python_backend = PythonBackend()
        if mojo_backend is None:
            self._mojo_backend, self._mojo_unavailable_reason = (
                MojoBackend.try_create(settings.mojo_library_path)
            )
        else:
            self._mojo_backend = mojo_backend
            self._mojo_unavailable_reason = None
        self._embedder: Embedder | None = None
        self._embedder_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache_condition = threading.Condition(self._cache_lock)
        self._semantic_cache: OrderedDict[
            SemanticCacheKey,
            SemanticSearchIndex,
        ] = OrderedDict()
        self._semantic_loads: dict[SemanticCacheKey, threading.Event] = {}
        self._semantic_active_uses: dict[SemanticCacheKey, int] = {}
        self._semantic_inflight_vector_bytes = 0
        self._semantic_inflight_candidate_count = 0
        self._lexical_config = LexicalSearchConfig.from_settings(settings)
        self._graph_config = GraphSearchConfig.from_settings(settings)
        self._retrieval_config = _retrieval_config_identity(settings)

    def search(
        self,
        repository_path: str | Path,
        request: SearchRequest,
        cancellation_callback: CancellationCallback | None = None,
    ) -> SearchResponse:
        """Search an existing index without implicitly checking freshness."""

        raise_if_cancelled(cancellation_callback)
        started_at = time.perf_counter()
        resolved = self.resolver.resolve(repository_path)
        raise_if_cancelled(cancellation_callback)

        with self.coordinator.searching(resolved.root):
            raise_if_cancelled(cancellation_callback)
            return self._search_with_lease(
                resolved,
                request,
                started_at,
                cancellation_callback,
            )

    def _search_with_lease(
        self,
        resolved: ResolvedRepository,
        request: SearchRequest,
        started_at: float,
        cancellation_callback: CancellationCallback | None,
    ) -> SearchResponse:
        """Execute a search while the caller owns a repository read lease."""

        raise_if_cancelled(cancellation_callback)
        normalized_path = self.resolver.validate_path_filter(request.path)
        request = request.model_copy(update={"path": normalized_path})
        self._validate_runtime_limits(request)
        backend_warnings = self._select_backend(request)

        if not resolved.database_path.exists():
            raise IndexNotFoundError(
                "Repository index is missing; call index_repository first"
            )

        try:
            with shared_database_lock(resolved.database_path, blocking=False):
                raise_if_cancelled(cancellation_callback)
                return self._search_database(
                    resolved,
                    request,
                    started_at,
                    backend_warnings,
                    cancellation_callback,
                )
        except DatabaseLockBusyError as error:
            raise RepositoryBusyError(
                f"Repository is currently being indexed: {resolved.root}"
            ) from error

    def _search_database(
        self,
        resolved: ResolvedRepository,
        request: SearchRequest,
        started_at: float,
        backend_warnings: list[str],
        cancellation_callback: CancellationCallback | None,
    ) -> SearchResponse:
        """Read and search one database while holding its cross-process lock."""

        raise_if_cancelled(cancellation_callback)
        store = SQLiteIndexStore(resolved.database_path)
        repository = store.load_latest_repository(
            absolute_path=str(resolved.root),
            index_format_version=INDEX_FORMAT_VERSION,
        )
        if repository is None:
            raise IndexNotFoundError(
                "Repository index is missing; call index_repository first"
            )
        raise_if_cancelled(cancellation_callback)

        if request.request_mode == "auto":
            response = self._automatic_search(
                store,
                repository.id,
                request,
                cancellation_callback,
            )
        elif request.request_mode == "exact":
            self._select_compute_backend(request, "exact")
            response = exact_search(
                store,
                repository.id,
                request,
                cancellation_callback=cancellation_callback,
            )
        elif request.request_mode == "fuzzy":
            backend = self._select_compute_backend(request, "fuzzy")
            response = fuzzy_search(
                store,
                repository.id,
                request,
                minimum_score=self.settings.fuzzy_threshold,
                max_candidates=self.settings.max_fuzzy_candidates,
                cancellation_callback=cancellation_callback,
                backend=backend,
                fallback_backend=(
                    self._python_backend if request.backend == "auto" else None
                ),
                minimum_accelerated_candidates=(
                    self.settings.mojo_fuzzy_min_candidates
                ),
            )
        elif request.request_mode == "lexical":
            if request.backend == "mojo":
                raise BackendUnavailableError(
                    "Lexical search uses the SQLite/Python backend"
                )
            response = lexical_search(
                store,
                repository.id,
                request,
                self._lexical_config,
                retrieval_config=self._retrieval_config,
                cancellation_callback=cancellation_callback,
            )
        elif request.request_mode == "semantic":
            response = self._semantic_search(
                store,
                repository.id,
                request,
                resolved.database_path,
                cancellation_callback,
            )
        elif request.request_mode in {"hybrid_rrf", "hybrid_weighted"}:
            response = self._hybrid_search(
                store,
                repository.id,
                request,
                resolved.database_path,
                cancellation_callback,
            )
        else:
            response = self._graph_search(
                store,
                repository.id,
                request,
                resolved.database_path,
                cancellation_callback,
            )

        raise_if_cancelled(cancellation_callback)
        response = response.model_copy(
            update={
                "elapsed_time": time.perf_counter() - started_at,
                "warnings": [*response.warnings, *backend_warnings],
                "retrieval_config": (
                    response.retrieval_config
                    if request.request_mode
                    in {"hybrid_rrf", "hybrid_weighted", "graph"}
                    else self._retrieval_config
                ),
            }
        )
        return self._bound_snippets(
            response,
            request.max_snippet_chars,
            cancellation_callback,
        )

    def invalidate_repository(self, repository_root: str | Path) -> None:
        """Discard cached semantic matrices after an index replacement."""

        root = Path(repository_root).expanduser().resolve()
        database_key = str(
            default_database_path(root, data_directory=self.settings.data_dir)
        )
        with self._cache_condition:
            keys_to_remove = [
                key
                for key in self._semantic_cache
                if key[0] == database_key
                and self._semantic_active_uses.get(key, 0) == 0
            ]
            for key in keys_to_remove:
                del self._semantic_cache[key]
            self._cache_condition.notify_all()

    def _automatic_search(
        self,
        store: SQLiteIndexStore,
        repository_id: uuid.UUID,
        request: SearchRequest,
        cancellation_callback: CancellationCallback | None,
    ) -> SearchResponse:
        raise_if_cancelled(cancellation_callback)
        exact_response = exact_search(
            store,
            repository_id,
            request,
            cancellation_callback=cancellation_callback,
        )
        if exact_response.ranked_results:
            self._select_compute_backend(request, "exact")
            return exact_response

        routed_mode = classify_non_exact_query(request.query)
        fuzzy_limit_exceeded = False
        if routed_mode == "fuzzy":
            try:
                backend = self._select_compute_backend(request, "fuzzy")
                fuzzy_response = fuzzy_search(
                    store,
                    repository_id,
                    request,
                    minimum_score=self.settings.fuzzy_threshold,
                    max_candidates=self.settings.max_fuzzy_candidates,
                    cancellation_callback=cancellation_callback,
                    backend=backend,
                    fallback_backend=(
                        self._python_backend
                        if request.backend == "auto"
                        else None
                    ),
                    minimum_accelerated_candidates=(
                        self.settings.mojo_fuzzy_min_candidates
                    ),
                )
            except SearchCandidateLimitError:
                fuzzy_response = None
                fuzzy_limit_exceeded = True
            if fuzzy_response is not None and fuzzy_response.ranked_results:
                return fuzzy_response

        semantic_response = self._semantic_search(
            store,
            repository_id,
            request,
            store.db_path,
            cancellation_callback,
        )
        if fuzzy_limit_exceeded:
            return semantic_response.model_copy(
                update={
                    "warnings": [
                        *semantic_response.warnings,
                        "Fuzzy candidate limit exceeded; used semantic search",
                    ]
                }
            )
        return semantic_response

    def _semantic_search(
        self,
        store: SQLiteIndexStore,
        repository_id: uuid.UUID,
        request: SearchRequest,
        database_path: Path,
        cancellation_callback: CancellationCallback | None,
        candidate_pool_size: int | None = None,
    ) -> SearchResponse:
        raise_if_cancelled(cancellation_callback)
        cache_key, search_index = self._load_semantic_index(
            store,
            repository_id,
            request.path,
            database_path,
            cancellation_callback,
        )
        try:
            raise_if_cancelled(cancellation_callback)
            embedder = self._get_embedder(cancellation_callback)
            raise_if_cancelled(cancellation_callback)
            backend = self._select_compute_backend(
                request,
                "semantic",
                candidate_count=len(search_index.candidates),
            )
            return semantic_search(
                store,
                repository_id,
                request,
                embedder,
                search_index=search_index,
                cancellation_callback=cancellation_callback,
                backend=backend,
                fallback_backend=(
                    self._python_backend if request.backend == "auto" else None
                ),
                score_floor=self.settings.semantic_score_floor,
                candidate_pool_size=candidate_pool_size,
            )
        finally:
            self._release_semantic_index(cache_key)

    def _hybrid_search(
        self,
        store: SQLiteIndexStore,
        repository_id: uuid.UUID,
        request: SearchRequest,
        database_path: Path,
        cancellation_callback: CancellationCallback | None,
    ) -> SearchResponse:
        """Generate and fuse bounded candidates from one index snapshot."""

        if request.request_mode == "hybrid_rrf":
            fusion_config = ReciprocalRankFusionConfig.from_settings(
                self.settings,
                final_top_k=request.top_k,
            )
        else:
            fusion_config = NormalizedWeightedFusionConfig.from_settings(
                self.settings,
                final_top_k=request.top_k,
            )

        lexical_request = request.model_copy(
            update={
                "request_mode": "lexical",
            }
        )
        lexical_started_at = time.perf_counter()
        lexical_response = lexical_search(
            store,
            repository_id,
            lexical_request,
            self._lexical_config,
            candidate_pool_size=fusion_config.lexical_pool_size,
            retrieval_config=self._retrieval_config,
            cancellation_callback=cancellation_callback,
        )
        lexical_elapsed = time.perf_counter() - lexical_started_at
        raise_if_cancelled(cancellation_callback)

        semantic_request = request.model_copy(
            update={
                "request_mode": "semantic",
            }
        )
        semantic_started_at = time.perf_counter()
        # Explicit hybrid modes require semantic retrieval. Any embedding,
        # index, or backend failure is intentionally allowed to surface.
        semantic_response = self._semantic_search(
            store,
            repository_id,
            semantic_request,
            database_path,
            cancellation_callback,
            candidate_pool_size=fusion_config.semantic_pool_size,
        )
        semantic_elapsed = time.perf_counter() - semantic_started_at
        raise_if_cancelled(cancellation_callback)

        candidates = [
            *response_candidates(repository_id, "lexical", lexical_response),
            *response_candidates(repository_id, "semantic", semantic_response),
        ]
        fusion_started_at = time.perf_counter()
        if isinstance(fusion_config, ReciprocalRankFusionConfig):
            fused_candidates = reciprocal_rank_fusion(candidates, fusion_config)
        else:
            fused_candidates = normalized_weighted_fusion(
                candidates,
                fusion_config,
            )
        fusion_elapsed = time.perf_counter() - fusion_started_at

        warnings = [
            *(f"Lexical: {warning}" for warning in lexical_response.warnings),
            *(f"Semantic: {warning}" for warning in semantic_response.warnings),
        ]
        return SearchResponse(
            original_query=request.query,
            requested_mode=request.request_mode,
            mode=request.request_mode,
            requested_backend=request.backend,
            backend=semantic_response.backend,
            elapsed_time=lexical_elapsed + semantic_elapsed + fusion_elapsed,
            ranked_results=[candidate.result for candidate in fused_candidates],
            warnings=warnings,
            retrieval_timings=[
                RetrievalTiming(
                    component="lexical",
                    elapsed_time=lexical_elapsed,
                    backend="python",
                ),
                RetrievalTiming(
                    component="semantic",
                    elapsed_time=semantic_elapsed,
                    backend=semantic_response.backend,
                ),
                RetrievalTiming(
                    component="fusion",
                    elapsed_time=fusion_elapsed,
                    backend="python",
                ),
            ],
            retrieval_config=_named_retrieval_config_identity(
                fusion_config.name,
                fusion_config.model_dump(mode="json"),
            ),
        )

    def _graph_search(
        self,
        store: SQLiteIndexStore,
        repository_id: uuid.UUID,
        request: SearchRequest,
        database_path: Path,
        cancellation_callback: CancellationCallback | None,
    ) -> SearchResponse:
        """Generate configured seeds and expand them through bounded adjacency."""

        seed_request = request.model_copy(
            update={
                "request_mode": self._graph_config.seed_mode,
                "top_k": self._graph_config.seed_count,
            }
        )
        if self._graph_config.seed_mode == "hybrid_rrf":
            seed_response = self._hybrid_search(
                store,
                repository_id,
                seed_request,
                database_path,
                cancellation_callback,
            )
        elif self._graph_config.seed_mode == "semantic":
            seed_response = self._semantic_search(
                store,
                repository_id,
                seed_request,
                database_path,
                cancellation_callback,
                candidate_pool_size=self._graph_config.seed_count,
            )
        else:
            if request.backend == "mojo":
                raise BackendUnavailableError(
                    "Graph retrieval with lexical seeds uses SQLite/Python"
                )
            seed_response = lexical_search(
                store,
                repository_id,
                seed_request,
                self._lexical_config,
                candidate_pool_size=self._graph_config.seed_count,
                retrieval_config=self._retrieval_config,
                cancellation_callback=cancellation_callback,
            )

        graph_config_values = {
            "seed_mode": self._graph_config.seed_mode,
            "seed_count": self._graph_config.seed_count,
            "max_hops": self._graph_config.max_hops,
            "maximum_neighbors_per_node": (
                self._graph_config.maximum_neighbors_per_node
            ),
            "maximum_expanded_nodes": self._graph_config.maximum_expanded_nodes,
            "allowed_edge_kinds": self._graph_config.allowed_edge_kinds,
            "directions": self._graph_config.directions,
            "minimum_edge_confidence": self._graph_config.minimum_edge_confidence,
            "hop_decay": self._graph_config.hop_decay,
            "edge_weights": self._graph_config.edge_weights,
        }
        return graph_search(
            store,
            repository_id,
            request,
            seed_response,
            self._graph_config,
            retrieval_config=_named_retrieval_config_identity(
                "graph",
                graph_config_values,
            ),
            cancellation_callback=cancellation_callback,
        )

    def _load_semantic_index(
        self,
        store: SQLiteIndexStore,
        repository_id: uuid.UUID,
        path_filter: str | None,
        database_path: Path,
        cancellation_callback: CancellationCallback | None,
    ) -> tuple[SemanticCacheKey, SemanticSearchIndex]:
        raise_if_cancelled(cancellation_callback)
        stat = database_path.stat()
        cache_key: SemanticCacheKey = (
            str(database_path),
            str(repository_id),
            path_filter,
            stat.st_ino,
            stat.st_mtime_ns,
            stat.st_size,
        )
        while True:
            raise_if_cancelled(cancellation_callback)
            with self._cache_condition:
                cached = self._semantic_cache.get(cache_key)
                if cached is not None:
                    self._semantic_cache.move_to_end(cache_key)
                    self._semantic_active_uses[cache_key] = (
                        self._semantic_active_uses.get(cache_key, 0) + 1
                    )
                    return cache_key, cached

                load_complete = self._semantic_loads.get(cache_key)
                if load_complete is None:
                    load_complete = threading.Event()
                    self._semantic_loads[cache_key] = load_complete
                    break

            while not load_complete.wait(SEMANTIC_CACHE_WAIT_SECONDS):
                raise_if_cancelled(cancellation_callback)

        reservation: SemanticCandidateSummary | None = None
        reservation_active = False
        try:
            reservation = summarize_semantic_search_index(
                store,
                repository_id,
                path_filter,
                max_candidates=self.settings.max_semantic_candidates,
                max_vector_bytes=self.settings.max_semantic_index_bytes,
                cancellation_callback=cancellation_callback,
            )
            self._reserve_semantic_capacity(
                reservation,
                cancellation_callback,
            )
            reservation_active = True
            loaded = load_semantic_search_index(
                store,
                repository_id,
                path_filter,
                max_candidates=self.settings.max_semantic_candidates,
                max_vector_bytes=self.settings.max_semantic_index_bytes,
                cancellation_callback=cancellation_callback,
                summary=reservation,
            )
            raise_if_cancelled(cancellation_callback)
            with self._cache_condition:
                self._release_semantic_capacity_unlocked(reservation)
                reservation_active = False
                repository_cache_prefix = (str(database_path), str(repository_id))
                current_database_version = (
                    stat.st_ino,
                    stat.st_mtime_ns,
                    stat.st_size,
                )
                stale_keys = [
                    key
                    for key in self._semantic_cache
                    if key[:2] == repository_cache_prefix
                    and key[3:] != current_database_version
                    and self._semantic_active_uses.get(key, 0) == 0
                ]
                for key in stale_keys:
                    del self._semantic_cache[key]
                self._semantic_cache[cache_key] = loaded
                self._semantic_active_uses[cache_key] = (
                    self._semantic_active_uses.get(cache_key, 0) + 1
                )
                self._trim_semantic_cache_unlocked()
                self._cache_condition.notify_all()
            return cache_key, loaded
        finally:
            with self._cache_condition:
                if reservation_active and reservation is not None:
                    self._release_semantic_capacity_unlocked(reservation)
                completed_load = self._semantic_loads.pop(cache_key, None)
                if completed_load is not None:
                    completed_load.set()
                self._cache_condition.notify_all()

    def _reserve_semantic_capacity(
        self,
        summary: SemanticCandidateSummary,
        cancellation_callback: CancellationCallback | None,
    ) -> None:
        """Reserve bounded cache capacity before allocating a semantic matrix."""

        while True:
            raise_if_cancelled(cancellation_callback)
            with self._cache_condition:
                while self._semantic_cache and self._reservation_exceeds_limits(
                    summary
                ):
                    if not self._evict_oldest_inactive_cache_entry_unlocked():
                        break

                if not self._reservation_exceeds_limits(summary):
                    self._semantic_inflight_vector_bytes += summary.vector_bytes
                    self._semantic_inflight_candidate_count += summary.count
                    return

                self._cache_condition.wait(SEMANTIC_CACHE_WAIT_SECONDS)

    def _reservation_exceeds_limits(
        self,
        summary: SemanticCandidateSummary,
    ) -> bool:
        vector_bytes = (
            self._semantic_cache_vector_bytes()
            + self._semantic_inflight_vector_bytes
            + summary.vector_bytes
        )
        candidate_count = (
            self._semantic_cache_candidate_count()
            + self._semantic_inflight_candidate_count
            + summary.count
        )
        return (
            vector_bytes > self.settings.max_semantic_index_bytes
            or candidate_count > self.settings.max_semantic_candidates
        )

    def _release_semantic_capacity_unlocked(
        self,
        summary: SemanticCandidateSummary,
    ) -> None:
        """Release one reservation while the cache condition is held."""

        self._semantic_inflight_vector_bytes -= summary.vector_bytes
        self._semantic_inflight_candidate_count -= summary.count

    def _release_semantic_index(self, cache_key: SemanticCacheKey) -> None:
        """Release an active-use lease after semantic ranking finishes."""

        with self._cache_condition:
            active_uses = self._semantic_active_uses.get(cache_key, 0)
            if active_uses <= 1:
                self._semantic_active_uses.pop(cache_key, None)
            else:
                self._semantic_active_uses[cache_key] = active_uses - 1
            self._trim_semantic_cache_unlocked()
            self._cache_condition.notify_all()

    def _trim_semantic_cache_unlocked(self) -> None:
        """Evict inactive LRU entries until retained cache bounds are met."""

        while (
            len(self._semantic_cache) > MAX_SEMANTIC_CACHE_ENTRIES
            or self._semantic_cache_vector_bytes()
            > self.settings.max_semantic_index_bytes
            or self._semantic_cache_candidate_count()
            > self.settings.max_semantic_candidates
        ):
            if not self._evict_oldest_inactive_cache_entry_unlocked():
                return

    def _evict_oldest_inactive_cache_entry_unlocked(self) -> bool:
        """Evict one LRU matrix that is not used by an active search."""

        for cache_key in self._semantic_cache:
            if self._semantic_active_uses.get(cache_key, 0) == 0:
                del self._semantic_cache[cache_key]
                return True
        return False

    def _semantic_cache_vector_bytes(self) -> int:
        """Return vector bytes retained by the cache while its lock is held."""

        return sum(index.matrix.nbytes for index in self._semantic_cache.values())

    def _semantic_cache_candidate_count(self) -> int:
        """Return candidate metadata retained while the cache lock is held."""

        return sum(len(index.candidates) for index in self._semantic_cache.values())

    def _get_embedder(
        self,
        cancellation_callback: CancellationCallback | None = None,
    ) -> Embedder:
        while True:
            raise_if_cancelled(cancellation_callback)
            if self._embedder_lock.acquire(timeout=SEMANTIC_CACHE_WAIT_SECONDS):
                break
        try:
            if self._embedder is None:
                self._embedder = self._embedder_factory()
            return self._embedder
        finally:
            self._embedder_lock.release()

    def _create_default_embedder(self) -> CodeRankEmbedder:
        return CodeRankEmbedder(
            model=self.settings.embedding_model,
            revision=self.settings.embedding_revision,
            batch_size=self.settings.embedding_batch_size,
            device=self.settings.embedding_device,
        )

    def _validate_runtime_limits(self, request: SearchRequest) -> None:
        if request.top_k > self.settings.max_top_k:
            raise ValueError(f"top_k must be at most {self.settings.max_top_k}")
        if request.max_snippet_chars > self.settings.max_snippet_chars:
            raise ValueError(
                "max_snippet_chars must be at most "
                f"{self.settings.max_snippet_chars}"
            )

    def _select_backend(self, request: SearchRequest) -> list[str]:
        if request.backend == "mojo" and self._mojo_backend is None:
            raise BackendUnavailableError(
                "The Mojo backend is not available: "
                f"{self._mojo_unavailable_reason or 'unknown reason'}"
            )
        if request.backend == "auto" and self._mojo_backend is None:
            return ["Mojo backend unavailable; using Python"]
        return []

    def _select_compute_backend(
        self,
        request: SearchRequest,
        capability: CapabilityName,
        *,
        candidate_count: int | None = None,
    ) -> AccelerationBackend:
        """Resolve one pure-compute operation after query routing."""

        if request.backend == "python":
            return self._python_backend

        mojo_backend = self._mojo_backend
        runtime_capability_enabled = capability in {"fuzzy", "semantic"}
        auto_size_gate_passed = (
            request.backend != "auto"
            or capability != "semantic"
            or (
                candidate_count is not None
                and candidate_count
                >= self.settings.mojo_semantic_min_candidates
            )
        )
        if (
            mojo_backend is not None
            and runtime_capability_enabled
            and auto_size_gate_passed
            and mojo_backend.supports(capability)
        ):
            return mojo_backend

        if request.backend == "mojo":
            if capability == "exact":
                reason = (
                    "Mojo exact matching is benchmark-only; indexed SQLite "
                    "remains the production exact-search backend"
                )
            else:
                reason = f"Mojo does not support {capability} search"
            raise BackendUnavailableError(reason)

        return self._python_backend

    def _bound_snippets(
        self,
        response: SearchResponse,
        per_result_limit: int,
        cancellation_callback: CancellationCallback | None,
    ) -> SearchResponse:
        remaining_characters = min(
            self.settings.max_total_snippet_chars,
            64_000,
        )
        bounded_results: list[SearchResult] = []
        omitted_result_count = 0

        for result in response.ranked_results:
            raise_if_cancelled(cancellation_callback)
            source_is_complete = not result.snippet_truncated
            source_fits_result_limit = len(result.snippet) <= per_result_limit
            source_fits_total_limit = len(result.snippet) <= remaining_characters
            if not (
                source_is_complete
                and source_fits_result_limit
                and source_fits_total_limit
            ):
                omitted_result_count += 1
                continue
            bounded_results.append(result)
            remaining_characters -= len(result.snippet)

        warnings = list(response.warnings)
        if omitted_result_count:
            warnings.append(
                f"Omitted {omitted_result_count} result(s) because complete source "
                "did not fit the configured output limits"
            )

        return response.model_copy(
            update={"ranked_results": bounded_results, "warnings": warnings}
        )


def _retrieval_config_identity(settings: Settings) -> str:
    """Return a stable, non-secret hash of effective ranking controls."""

    values = {
        "fuzzy_threshold": settings.fuzzy_threshold,
        "max_fuzzy_candidates": settings.max_fuzzy_candidates,
        "max_semantic_candidates": settings.max_semantic_candidates,
        "semantic": {
            "score_floor": settings.semantic_score_floor,
            "ranking_version": SEMANTIC_RANKING_VERSION,
        },
        "hybrid": {
            "lexical_pool_size": settings.hybrid_lexical_pool_size,
            "semantic_pool_size": settings.hybrid_semantic_pool_size,
            "rrf_k": settings.hybrid_rrf_k,
            "rrf_lexical_weight": settings.hybrid_rrf_lexical_weight,
            "rrf_semantic_weight": settings.hybrid_rrf_semantic_weight,
            "weighted_lexical_weight": settings.hybrid_weighted_lexical_weight,
            "weighted_semantic_weight": settings.hybrid_weighted_semantic_weight,
            "weighted_missing_source_value": (
                settings.hybrid_weighted_missing_source_value
            ),
            "tie_breaking_version": settings.hybrid_tie_breaking_version,
        },
        "graph": {
            "seed_mode": settings.graph_seed_mode,
            "seed_count": settings.graph_seed_count,
            "max_hops": settings.graph_max_hops,
            "maximum_neighbors_per_node": settings.graph_max_neighbors_per_node,
            "maximum_expanded_nodes": settings.graph_max_expanded_nodes,
            "allowed_edge_kinds": settings.graph_allowed_edge_kinds,
            "directions": settings.graph_directions,
            "minimum_edge_confidence": settings.graph_min_edge_confidence,
            "hop_decay": settings.graph_hop_decay,
            "edge_weights": {
                "calls": settings.graph_calls_weight,
                "imports": settings.graph_imports_weight,
                "inherits": settings.graph_inherits_weight,
                "references": settings.graph_references_weight,
                "depends_on": settings.graph_depends_on_weight,
                "tests": settings.graph_tests_weight,
            },
        },
        "lexical": {
            "exact_qualified_bonus": settings.lexical_exact_qualified_bonus,
            "exact_short_bonus": settings.lexical_exact_short_bonus,
            "path_bonus": settings.lexical_path_bonus,
            "identifier_bonus": settings.lexical_identifier_bonus,
            "bm25_bonus": settings.lexical_bm25_bonus,
            "fuzzy_bonus": settings.lexical_fuzzy_bonus,
            "exact_limit": settings.lexical_exact_candidate_limit,
            "path_limit": settings.lexical_path_candidate_limit,
            "identifier_limit": settings.lexical_identifier_candidate_limit,
            "bm25_limit": settings.lexical_bm25_candidate_limit,
            "fuzzy_limit": settings.lexical_fuzzy_candidate_limit,
            "maximum_documents_ranked": settings.max_lexical_documents_ranked,
            "bm25_weights": [
                settings.bm25_name_weight,
                settings.bm25_qualified_name_weight,
                settings.bm25_identifier_weight,
                settings.bm25_path_weight,
                settings.bm25_content_weight,
            ],
        },
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{settings.retrieval_config_name}:{digest}"


def _named_retrieval_config_identity(
    name: str,
    values: dict[str, object],
) -> str:
    """Return a stable identity for one serializable named configuration."""

    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{name}:{digest}"

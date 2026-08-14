"""Unified code-search service shared by CLI, Streamlit, and MCP."""

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.config import Settings, settings as default_settings
from app.core.coordinator import RepositoryBusyError, RepositoryCoordinator
from app.core.models import SearchRequest, SearchResponse, SearchResult
from app.core.repositories import RepositoryResolver, ResolvedRepository
from app.indexing.embedder import CodeRankEmbedder, Embedder
from app.indexing.service import INDEX_FORMAT_VERSION
from app.search.exact import exact_search
from app.search.fuzzy import fuzzy_search
from app.search.router import classify_non_exact_query
from app.search.semantic import (
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
    ) -> None:
        self.settings = settings
        self.resolver = resolver or RepositoryResolver(settings)
        self.coordinator = coordinator or RepositoryCoordinator()
        self._embedder_factory = embedder_factory or self._create_default_embedder
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
            response = exact_search(
                store,
                repository.id,
                request,
                cancellation_callback=cancellation_callback,
            )
        elif request.request_mode == "fuzzy":
            response = fuzzy_search(
                store,
                repository.id,
                request,
                minimum_score=self.settings.fuzzy_threshold,
                max_candidates=self.settings.max_fuzzy_candidates,
                cancellation_callback=cancellation_callback,
            )
        else:
            response = self._semantic_search(
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
            return exact_response

        routed_mode = classify_non_exact_query(request.query)
        fuzzy_limit_exceeded = False
        if routed_mode == "fuzzy":
            try:
                fuzzy_response = fuzzy_search(
                    store,
                    repository_id,
                    request,
                    minimum_score=self.settings.fuzzy_threshold,
                    max_candidates=self.settings.max_fuzzy_candidates,
                    cancellation_callback=cancellation_callback,
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
            return semantic_search(
                store,
                repository_id,
                request,
                embedder,
                search_index=search_index,
                cancellation_callback=cancellation_callback,
            )
        finally:
            self._release_semantic_index(cache_key)

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

    @staticmethod
    def _select_backend(request: SearchRequest) -> list[str]:
        if request.backend == "mojo":
            raise BackendUnavailableError(
                "The Mojo backend is not available in this FireLens build"
            )
        if request.backend == "auto":
            return ["Mojo backend unavailable; using Python"]
        return []

    def _bound_snippets(
        self,
        response: SearchResponse,
        per_result_limit: int,
        cancellation_callback: CancellationCallback | None,
    ) -> SearchResponse:
        remaining_characters = min(
            self.settings.max_total_snippet_chars,
            12_000,
        )
        bounded_results: list[SearchResult] = []
        any_truncated = False

        for result in response.ranked_results:
            raise_if_cancelled(cancellation_callback)
            allowed_characters = min(per_result_limit, remaining_characters)
            snippet = result.snippet[:allowed_characters]
            was_truncated = (
                result.snippet_truncated or len(snippet) < len(result.snippet)
            )
            any_truncated = any_truncated or was_truncated
            bounded_results.append(
                result.model_copy(
                    update={
                        "snippet": snippet,
                        "snippet_truncated": was_truncated,
                    }
                )
            )
            remaining_characters -= len(snippet)

        warnings = list(response.warnings)
        if any_truncated:
            warnings.append("One or more snippets were truncated to output limits")

        return response.model_copy(
            update={"ranked_results": bounded_results, "warnings": warnings}
        )

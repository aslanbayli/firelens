"""Construct and expose one shared FireLens application runtime."""

import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from app.core.cancellation import CancellationCallback
from app.core.config import Settings, settings as default_settings
from app.core.coordinator import RepositoryCoordinator
from app.core.models import (
    BackendPreference,
    IndexRepositoryResponse,
    IndexStatusResponse,
    RetrievalMode,
    SearchRequest,
    SearchResponse,
)
from app.core.repositories import RepositoryResolver
from app.indexing.embedder import CodeRankEmbedder, Embedder
from app.indexing.indexer import ProgressCallback
from app.indexing.service import AvailableIndex, IndexService
from app.search.service import SearchService


EmbedderFactory = Callable[[], Embedder]


class FireLensRuntime:
    """Application facade used by protocol and presentation interfaces."""

    def __init__(
        self,
        settings: Settings = default_settings,
        embedder_factory: EmbedderFactory | None = None,
    ) -> None:
        resolver = RepositoryResolver(settings)
        coordinator = RepositoryCoordinator()
        shared_embedder_factory = _shared_embedder_factory(
            settings,
            embedder_factory,
        )

        self.search_service = SearchService(
            settings=settings,
            resolver=resolver,
            coordinator=coordinator,
            embedder_factory=shared_embedder_factory,
        )
        self.index_service = IndexService(
            settings=settings,
            resolver=resolver,
            coordinator=coordinator,
            embedder_factory=shared_embedder_factory,
            on_index_replaced=self.search_service.invalidate_repository,
        )

    def list_available_indexes(self) -> list[AvailableIndex]:
        return self.index_service.list_available_indexes()

    def index_repository(
        self,
        repository_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> IndexRepositoryResponse:
        return self.index_service.index_repository(
            repository_path,
            progress_callback=progress_callback,
            cancellation_callback=cancellation_callback,
        )

    def get_index_status(
        self,
        repository_path: str | Path,
        cancellation_callback: CancellationCallback | None = None,
    ) -> IndexStatusResponse:
        return self.index_service.get_index_status(
            repository_path,
            cancellation_callback=cancellation_callback,
        )

    def search_code(
        self,
        repository_path: str | Path,
        query: str,
        mode: RetrievalMode = "auto",
        top_k: int = 5,
        path: str | None = None,
        backend: BackendPreference = "auto",
        max_snippet_chars: int = 2_000,
        cancellation_callback: CancellationCallback | None = None,
    ) -> SearchResponse:
        request = SearchRequest(
            query=query,
            request_mode=mode,
            top_k=top_k,
            path=path,
            backend=backend,
            max_snippet_chars=max_snippet_chars,
        )
        return self.search_service.search(
            repository_path,
            request,
            cancellation_callback=cancellation_callback,
        )


def build_runtime(
    settings: Settings = default_settings,
    embedder_factory: EmbedderFactory | None = None,
) -> FireLensRuntime:
    return FireLensRuntime(settings=settings, embedder_factory=embedder_factory)


def _shared_embedder_factory(
    settings: Settings,
    supplied_factory: EmbedderFactory | None,
) -> EmbedderFactory:
    lock = threading.Lock()
    embedder: Embedder | None = None

    def create_embedder() -> Embedder:
        nonlocal embedder
        with lock:
            if embedder is None:
                if supplied_factory is not None:
                    created_embedder = supplied_factory()
                else:
                    created_embedder = CodeRankEmbedder(
                        model=settings.embedding_model,
                        revision=settings.embedding_revision,
                        batch_size=settings.embedding_batch_size,
                        device=settings.embedding_device,
                    )
                embedder = _SynchronizedEmbedder(created_embedder)
            return embedder

    return create_embedder


class _SynchronizedEmbedder:
    """Serialize access to one model shared by indexing and search threads."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._lock = threading.Lock()

    @property
    def provider(self) -> str:
        return self._embedder.provider

    @property
    def model(self) -> str:
        return self._embedder.model

    @property
    def dimension(self) -> int:
        with self._lock:
            return self._embedder.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return self._embedder.embed(texts)

    def embed_query(self, query: str) -> list[float]:
        with self._lock:
            return self._embedder.embed_query(query)

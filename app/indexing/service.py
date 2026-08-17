"""Public indexing and freshness services shared by every interface."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.config import Settings, settings as default_settings
from app.core.coordinator import RepositoryCoordinator
from app.core.models import (
    IndexRepositoryResponse,
    IndexStatus,
    IndexStatusResponse,
    IndexingErrorResponse,
    Repository,
)
from app.core.repositories import RepositoryResolver, ResolvedRepository
from app.indexing.embedder import (
    CodeRankEmbedder,
    Embedder,
    embedding_model_identity,
)
from app.indexing.indexer import (
    ProgressCallback,
    index_to_sqlite,
    raise_if_indexing_cancelled,
)
from app.indexing.manifest import build_file_manifest, compare_file_manifests
from app.indexing.version import INDEX_FORMAT_VERSION
from app.storage.database import SQLiteIndexStore
from app.storage.locking import (
    DatabaseLockBusyError,
    database_lock_path,
    shared_database_lock,
)


MAX_STATUS_PATHS = 20
MAX_INDEX_ERRORS = 20

EmbedderFactory = Callable[[], Embedder]
IndexReplacementCallback = Callable[[Path], None]


@dataclass(frozen=True)
class AvailableIndex:
    """One repository index available through the configured runtime."""

    repository_path: str
    database_path: str
    timestamp_of_index: int
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    index_format_version: str = INDEX_FORMAT_VERSION
    status: IndexStatus = "ready"


class IndexService:
    """Resolve repositories, build indexes, and report index freshness."""

    def __init__(
        self,
        settings: Settings = default_settings,
        resolver: RepositoryResolver | None = None,
        coordinator: RepositoryCoordinator | None = None,
        embedder_factory: EmbedderFactory | None = None,
        on_index_replaced: IndexReplacementCallback | None = None,
    ) -> None:
        self.settings = settings
        self.resolver = resolver or RepositoryResolver(settings)
        self.coordinator = coordinator or RepositoryCoordinator()
        self._embedder_factory = embedder_factory or self._create_default_embedder
        self._embedder: Embedder | None = None
        self._on_index_replaced = on_index_replaced

    def list_available_indexes(self) -> list[AvailableIndex]:
        """List the latest allowed repository record from each managed index."""

        latest_by_repository: dict[str, tuple[Repository, Path]] = {}

        for database_path in sorted(self.settings.data_dir.glob("*/firelens.db")):
            store = SQLiteIndexStore(database_path)
            try:
                with shared_database_lock(database_path, blocking=False):
                    repositories = store.list_repositories()
            except DatabaseLockBusyError:
                continue

            for repository in repositories:
                try:
                    resolved = self.resolver.resolve(repository.absolute_path)
                except ValueError:
                    continue

                if database_path != resolved.database_path:
                    continue

                repository_key = str(resolved.root)
                current = latest_by_repository.get(repository_key)
                if current is not None:
                    current_repository, _current_database_path = current
                    current_sort_key = (
                        current_repository.timestamp_of_index,
                        str(current_repository.id),
                    )
                    candidate_sort_key = (
                        repository.timestamp_of_index,
                        str(repository.id),
                    )
                    if candidate_sort_key <= current_sort_key:
                        continue

                latest_by_repository[repository_key] = (repository, database_path)

        available_indexes: list[AvailableIndex] = []
        for repository_path, stored_index in latest_by_repository.items():
            repository, database_path = stored_index
            available_indexes.append(
                AvailableIndex(
                    repository_path=repository_path,
                    database_path=str(database_path),
                    timestamp_of_index=repository.timestamp_of_index,
                    embedding_provider=repository.embedding_provider,
                    embedding_model=repository.embedding_model,
                    embedding_dim=repository.embedding_dim,
                    index_format_version=repository.index_format_version,
                    status=(
                        "ready"
                        if repository.index_format_version == INDEX_FORMAT_VERSION
                        else "stale"
                    ),
                )
            )

        return sorted(
            available_indexes,
            key=lambda index: (
                Path(index.repository_path).name.casefold(),
                index.repository_path.casefold(),
            ),
        )

    def index_repository(
        self,
        repository_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> IndexRepositoryResponse:
        """Explicitly index one repository and wait for completion."""

        raise_if_indexing_cancelled(cancellation_callback)
        resolved = self.resolver.resolve(repository_path)
        started_at = time.perf_counter()
        cache_warning: str | None = None

        def check_cancellation() -> None:
            raise_if_indexing_cancelled(cancellation_callback)

        coordinator_cancellation_check: Callable[[], None] | None = None
        if cancellation_callback is not None:
            coordinator_cancellation_check = check_cancellation

        with self.coordinator.indexing(
            resolved.root,
            cancellation_check=coordinator_cancellation_check,
        ):
            raise_if_indexing_cancelled(cancellation_callback)
            embedder = self._get_embedder()
            raise_if_indexing_cancelled(cancellation_callback)
            self._validate_embedder_identity(embedder)
            raise_if_indexing_cancelled(cancellation_callback)
            report = index_to_sqlite(
                path=resolved.root,
                embedder=embedder,
                db_path=resolved.database_path,
                progress_callback=progress_callback,
                max_file_size=self.settings.max_file_size_bytes,
                max_files=self.settings.max_repository_files,
                max_entries=self.settings.max_walk_entries,
                max_chunks_per_file=self.settings.max_chunks_per_file,
                cancellation_callback=cancellation_callback,
            )

            if self._on_index_replaced is not None:
                try:
                    self._on_index_replaced(resolved.root)
                except Exception:
                    cache_warning = (
                        "Index was replaced, but the in-memory semantic cache "
                        "could not be invalidated"
                    )

        sampled_errors = report.errors[:MAX_INDEX_ERRORS]
        errors = [
            IndexingErrorResponse(
                relative_path=error.relative_path,
                stage=error.stage,
                message=error.message[:1_000],
            )
            for error in sampled_errors
        ]
        warnings: list[str] = []
        if cache_warning is not None:
            warnings.append(cache_warning)
        if any(len(error.message) > 1_000 for error in sampled_errors):
            warnings.append("One or more indexing error messages were truncated")
        if len(report.errors) > len(errors):
            warnings.append(
                f"Only the first {MAX_INDEX_ERRORS} indexing errors are included"
            )
        if len(report.changed_paths) > MAX_STATUS_PATHS:
            warnings.append(
                f"Only the first {MAX_STATUS_PATHS} changed paths are included"
            )

        return IndexRepositoryResponse(
            repository_path=str(resolved.root),
            database_path=str(resolved.database_path),
            status="stale" if report.errors else "ready",
            index_format_version=report.repository.index_format_version,
            timestamp_of_index=report.repository.timestamp_of_index,
            embedding_provider=embedder.provider,
            embedding_model=report.repository.embedding_model,
            embedding_dim=report.repository.embedding_dim,
            file_count=report.file_count,
            symbol_count=report.symbol_count,
            chunk_count=report.chunk_count,
            embedding_count=report.embedding_count,
            lexical_document_count=report.lexical_document_count,
            added_file_count=report.added_file_count,
            changed_file_count=report.changed_file_count,
            deleted_file_count=report.deleted_file_count,
            embedded_chunk_count=report.embedded_chunk_count,
            reused_embedding_count=report.reused_embedding_count,
            elapsed_time=time.perf_counter() - started_at,
            changed_paths=report.changed_paths[:MAX_STATUS_PATHS],
            error_count=len(report.errors),
            errors=errors,
            warnings=warnings,
        )

    def get_index_status(
        self,
        repository_path: str | Path,
        cancellation_callback: CancellationCallback | None = None,
    ) -> IndexStatusResponse:
        """Hash current files and compare them with the stored index."""

        raise_if_cancelled(cancellation_callback)
        resolved = self.resolver.resolve(repository_path)
        raise_if_cancelled(cancellation_callback)
        is_indexing = self.coordinator.is_indexing(resolved.root)

        if is_indexing:
            return self._missing_status(resolved, is_indexing=True)

        if not resolved.database_path.exists():
            if self._database_is_locked(
                resolved.database_path,
                cancellation_callback,
            ):
                return self._missing_status(resolved, is_indexing=True)
            return self._missing_status(resolved, is_indexing=is_indexing)

        try:
            with shared_database_lock(resolved.database_path, blocking=False):
                raise_if_cancelled(cancellation_callback)
                return self._get_index_status_unlocked(
                    resolved,
                    cancellation_callback,
                )
        except DatabaseLockBusyError:
            return self._missing_status(resolved, is_indexing=True)

    def _get_index_status_unlocked(
        self,
        resolved: ResolvedRepository,
        cancellation_callback: CancellationCallback | None,
    ) -> IndexStatusResponse:
        """Inspect one index while the caller owns its shared file lock."""

        raise_if_cancelled(cancellation_callback)
        store = SQLiteIndexStore(resolved.database_path)
        repository = store.load_latest_repository(
            absolute_path=str(resolved.root),
            index_format_version=INDEX_FORMAT_VERSION,
        )
        if repository is None:
            stale_repository = store.load_latest_repository(
                absolute_path=str(resolved.root)
            )
            if stale_repository is None:
                return self._missing_status(resolved, is_indexing=False)
            counts = _load_counts(
                store,
                stale_repository,
                cancellation_callback,
                include_lexical=False,
            )
            return self._status_response(
                resolved,
                stale_repository,
                status="stale",
                counts=counts,
                warnings=[
                    "Index format is outdated and requires a complete rebuild"
                ],
            )

        raise_if_cancelled(cancellation_callback)
        counts = _load_counts(store, repository, cancellation_callback)
        stored_files = store.load_files(repository.id)
        raise_if_cancelled(cancellation_callback)
        current_files = build_file_manifest(
            resolved.root,
            repository.id,
            max_file_size=self.settings.max_file_size_bytes,
            max_files=self.settings.max_repository_files,
            max_entries=self.settings.max_walk_entries,
            cancellation_callback=cancellation_callback,
        )
        difference = compare_file_manifests(
            current_files,
            stored_files,
            cancellation_callback=cancellation_callback,
        )
        raise_if_cancelled(cancellation_callback)
        changed_paths = difference.all_changed_paths
        warnings: list[str] = []
        expected_model = embedding_model_identity(
            self.settings.embedding_model,
            self.settings.embedding_revision,
        )
        compatible_embedding = (
            repository.embedding_provider == self.settings.embedding_provider
            and repository.embedding_model == expected_model
            and repository.embedding_dim == self.settings.embedding_dimension
        )
        if not compatible_embedding:
            warnings.append(
                "Index embedding configuration differs from the active runtime; "
                "reindex before semantic search"
            )
        if len(changed_paths) > MAX_STATUS_PATHS:
            warnings.append(
                f"Only the first {MAX_STATUS_PATHS} changed paths are included"
            )

        return self._status_response(
            resolved,
            repository,
            status=(
                "ready"
                if difference.is_current and compatible_embedding
                else "stale"
            ),
            counts=counts,
            added_file_count=len(difference.added_paths),
            changed_file_count=len(difference.changed_paths),
            deleted_file_count=len(difference.deleted_paths),
            changed_paths=changed_paths[:MAX_STATUS_PATHS],
            warnings=warnings,
        )

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        return self._embedder

    def _validate_embedder_identity(self, embedder: Embedder) -> None:
        expected_model = embedding_model_identity(
            self.settings.embedding_model,
            self.settings.embedding_revision,
        )
        if embedder.provider != self.settings.embedding_provider:
            raise ValueError(
                "Active embedding provider does not match "
                "FIRELENS_EMBEDDING_PROVIDER"
            )
        if embedder.model != expected_model:
            raise ValueError(
                "Active embedding model does not match the configured "
                "model and revision"
            )
        if embedder.dimension != self.settings.embedding_dimension:
            raise ValueError(
                "Active embedding dimension does not match "
                "FIRELENS_EMBEDDING_DIMENSION"
            )

    def _create_default_embedder(self) -> CodeRankEmbedder:
        return CodeRankEmbedder(
            model=self.settings.embedding_model,
            revision=self.settings.embedding_revision,
            batch_size=self.settings.embedding_batch_size,
            device=self.settings.embedding_device,
        )

    @staticmethod
    def _missing_status(
        resolved: ResolvedRepository,
        is_indexing: bool,
    ) -> IndexStatusResponse:
        return IndexStatusResponse(
            repository_path=str(resolved.root),
            database_path=str(resolved.database_path),
            status="indexing" if is_indexing else "missing",
        )

    def _status_response(
        self,
        resolved: ResolvedRepository,
        repository: Repository,
        status: IndexStatus,
        counts: dict[str, int],
        added_file_count: int = 0,
        changed_file_count: int = 0,
        deleted_file_count: int = 0,
        changed_paths: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> IndexStatusResponse:
        return IndexStatusResponse(
            repository_path=str(resolved.root),
            database_path=str(resolved.database_path),
            status=status,
            index_format_version=repository.index_format_version,
            timestamp_of_index=repository.timestamp_of_index,
            embedding_provider=repository.embedding_provider,
            embedding_model=repository.embedding_model,
            embedding_dim=repository.embedding_dim,
            file_count=counts["files"],
            symbol_count=counts["symbols"],
            chunk_count=counts["chunks"],
            embedding_count=counts["embeddings"],
            lexical_document_count=counts.get("lexical_documents", 0),
            added_file_count=added_file_count,
            changed_file_count=changed_file_count,
            deleted_file_count=deleted_file_count,
            changed_paths=changed_paths or [],
            warnings=warnings or [],
        )

    @staticmethod
    def _database_is_locked(
        database_path: Path,
        cancellation_callback: CancellationCallback | None = None,
    ) -> bool:
        raise_if_cancelled(cancellation_callback)
        if not database_lock_path(database_path).exists():
            return False
        try:
            with shared_database_lock(database_path, blocking=False):
                raise_if_cancelled(cancellation_callback)
                return False
        except DatabaseLockBusyError:
            return True


def _load_counts(
    store: SQLiteIndexStore,
    repository: Repository,
    cancellation_callback: CancellationCallback | None = None,
    include_lexical: bool = True,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    tables = ["files", "symbols", "chunks", "embeddings"]
    if include_lexical:
        tables.append("lexical_documents")
    for table in tables:
        raise_if_cancelled(cancellation_callback)
        counts[table] = store.count_rows(table, repository.id)
    return counts

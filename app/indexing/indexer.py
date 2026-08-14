"""Coordinate repository walking, parsing, chunking, and embedding.

The indexer is an orchestrator. It determines pipeline order and collects
results, while specialized modules retain responsibility for filesystem rules,
AST traversal, chunk boundaries, and vector generation.
"""

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from app.core.cancellation import CancellationCallback, OperationCancelledError
from app.core.models import Chunk, Repository, Symbol
from app.indexing.chunker import build_embedding_text, chunk_symbols
from app.indexing.embedder import Embedder, validate_embeddings
from app.indexing.file_io import read_regular_file
from app.indexing.manifest import build_file_manifest, compare_file_manifests
from app.indexing.parser import parse_symbols
from app.indexing.walker import walk
from app.storage.database import (
    IndexedFileRecords,
    SQLiteIndexStore,
    default_database_path,
)
from app.storage.locking import exclusive_database_lock


# Frozen prevents accidental mutation after an error has been recorded.
@dataclass(frozen=True)
class IndexingError:
    """One recoverable file-level indexing failure."""

    # Portable path relative to the indexed repository.
    relative_path: str
    # Name of the pipeline stage that failed: read, parse, or chunk.
    stage: str
    # Human-readable exception detail for diagnostics.
    message: str


@dataclass
class InMemoryIndex:
    """All generated artifacts before SQLite persistence is implemented."""

    # Metadata shared by all records produced in this run.
    repository: Repository
    # Declarations used by exact and fuzzy search.
    symbols: list[Symbol]
    # Bounded source units used by semantic search.
    chunks: list[Chunk]
    # Positional mapping: embeddings[i] belongs to chunks[i].
    embeddings: list[list[float]]
    # Recoverable failures that did not stop indexing other files.
    errors: list[IndexingError]


@dataclass
class IndexingReport:
    """User-facing summary for a completed persisted indexing run."""

    repository: Repository
    database_path: Path
    symbol_count: int
    chunk_count: int
    embedding_count: int
    file_count: int
    added_file_count: int
    changed_file_count: int
    deleted_file_count: int
    embedded_chunk_count: int
    reused_embedding_count: int
    errors: list[IndexingError]
    changed_paths: list[str]


@dataclass(frozen=True)
class IndexingProgress:
    """One progress update from a persisted indexing run."""

    stage: str
    current: int
    total: int
    message: str


ProgressCallback = Callable[[IndexingProgress], None]


class IndexingCancelledError(OperationCancelledError):
    """Raised when a caller cooperatively cancels an indexing run."""


def raise_if_indexing_cancelled(
    cancellation_callback: CancellationCallback | None,
) -> None:
    """Stop at a safe pipeline boundary when cancellation was requested."""

    if cancellation_callback is not None and cancellation_callback():
        raise IndexingCancelledError("Repository indexing was cancelled")


def _indexing_cancellation_adapter(
    cancellation_callback: CancellationCallback | None,
) -> CancellationCallback | None:
    """Preserve the indexing-specific exception in shared cancellable helpers."""

    if cancellation_callback is None:
        return None

    def check_indexing_cancellation() -> bool:
        raise_if_indexing_cancelled(cancellation_callback)
        return False

    return check_indexing_cancellation


def index(
    # Repository path from a CLI, UI, test, or Python caller.
    path: str | Path,
    # Any concrete object satisfying the Embedder protocol.
    embedder: Embedder,
    max_file_size: int = 1_000_000,
    max_files: int = 10_000,
    max_entries: int = 100_000,
    max_chunks_per_file: int = 2_048,
    cancellation_callback: CancellationCallback | None = None,
) -> InMemoryIndex:
    """Build an in-memory index for a local Python repository."""

    # Convert strings to Path, expand "~", resolve "..", and produce one
    # canonical absolute root. The walker performs existence/type validation.
    raise_if_indexing_cancelled(cancellation_callback)
    root = Path(path).expanduser().resolve()

    # Every symbol and chunk from this run refers to this repository identity.
    repository_id = uuid.uuid4()

    # Capture repository and embedding compatibility metadata before processing
    # files. A persisted index must not mix vectors from incompatible models.
    embedding_dimension = embedder.dimension
    raise_if_indexing_cancelled(cancellation_callback)
    repository = Repository(
        # Unique identity for this repository record.
        id=repository_id,
        # Serialize the canonical Path as a normal string.
        absolute_path=str(root),
        # Version the index format so future schema changes can be detected.
        index_format_version="1",
        # Store a timezone-aware current time as an integer Unix timestamp.
        timestamp_of_index=int(datetime.now(UTC).timestamp()),
        embedding_provider=embedder.provider,
        # Read provider metadata through the generic interface.
        embedding_model=embedder.model,
        # Every vector in this index must have this exact length.
        embedding_dim=embedding_dimension,
    )

    # Discover safe, supported files using walker defaults. Results are relative
    # and sorted, so processing order is deterministic.
    paths = walk(
        root,
        max_file_size=max_file_size,
        max_files=max_files,
        max_entries=max_entries,
        cancellation_callback=_indexing_cancellation_adapter(
            cancellation_callback
        ),
    )
    raise_if_indexing_cancelled(cancellation_callback)

    # Collect all successfully parsed declarations.
    symbols: list[Symbol] = []

    # Collect chunks in file/symbol/window order.
    chunks: list[Chunk] = []

    # Collect recoverable failures rather than terminating the whole repository.
    errors: list[IndexingError] = []

    # Run each discovered file through read → parse → chunk.
    for relative_path in paths:
        raise_if_indexing_cancelled(cancellation_callback)
        try:
            source = _read_source_text(root, relative_path, max_file_size)
        except (OSError, UnicodeDecodeError) as error:
            # OSError covers filesystem failures; UnicodeDecodeError indicates
            # that the bytes could not be interpreted using UTF-8.
            errors.append(
                IndexingError(
                    # POSIX separators remain stable across platforms/storage.
                    relative_path=relative_path.as_posix(),
                    # Stage names allow grouped reporting and debugging.
                    stage="read",
                    # Preserve the original exception's useful explanation.
                    message=str(error),
                )
            )

            # Parsing requires valid source text, so move to the next file.
            continue

        try:
            # Build an AST and extract classes/functions/methods. SyntaxError is
            # deliberately allowed to reach this level for file context.
            parsed_symbols = parse_symbols(source)
        except SyntaxError as error:
            # One invalid Python file should not discard healthy repository data.
            errors.append(
                IndexingError(
                    relative_path=relative_path.as_posix(),
                    stage="parse",
                    message=str(error),
                )
            )

            # Invalid syntax cannot provide trustworthy symbol line boundaries.
            continue

        # Add storage/domain identity and repository ownership to parser facts.
        file_symbols = [
            Symbol(
                # Each declaration gets an independent unique ID.
                id=uuid.uuid4(),
                # All declarations in this run belong to the same repository.
                repository_id=repository_id,
                # Copy syntax-derived values without changing their meaning.
                name=parsed.name,
                qualified_name=parsed.qualified_name,
                kind=parsed.kind,
                # Store a portable repository-relative path.
                relative_path=relative_path.as_posix(),
                # Preserve exact one-based source coordinates and source text.
                start_line=parsed.start_line,
                end_line=parsed.end_line,
                source_snippet=parsed.source_snippet,
            )
            # Create one validated Symbol for every ParsedSymbol.
            for parsed in parsed_symbols
        ]

        # Make this file's declarations available to exact/fuzzy retrieval and
        # embedding-text reconstruction.
        symbols.extend(file_symbols)

        try:
            # Split every symbol into bounded, optionally overlapping chunks.
            file_chunks = chunk_symbols(
                source,
                file_symbols,
                max_chunks=max_chunks_per_file,
            )

            # Add this file's chunks to the complete index in processing order.
            chunks.extend(file_chunks)
        except ValueError as error:
            # Current ValueErrors indicate invalid line-window configuration.
            errors.append(
                IndexingError(
                    relative_path=relative_path.as_posix(),
                    stage="chunk",
                    message=str(error),
                )
            )

        raise_if_indexing_cancelled(cancellation_callback)

    # Create O(1)-average ID lookup. Scanning all symbols for every chunk would
    # grow unnecessarily expensive as repository size increases.
    symbols_by_id = {symbol.id: symbol for symbol in symbols}

    # Reconstruct enriched embedding inputs in chunk order. This positional
    # order is the contract connecting chunks to returned vectors.
    embedding_texts = [
        _embedding_text_for_chunk(chunk, symbols_by_id) for chunk in chunks
    ]

    # Send one batch to allow real providers/models to amortize overhead.
    raise_if_indexing_cancelled(cancellation_callback)
    embeddings = embedder.embed(embedding_texts)
    raise_if_indexing_cancelled(cancellation_callback)

    # Reject missing, extra, or wrong-sized vectors before storage/search.
    validate_embeddings(
        # Establishes expected vector count.
        embedding_texts,
        # Provider output being verified.
        embeddings,
        # Establishes required length of every vector.
        expected_dimension=embedding_dimension,
    )

    # Return every artifact now so each stage can be inspected and tested before
    # persistence hides it behind repositories and SQL.
    return InMemoryIndex(
        repository=repository,
        symbols=symbols,
        chunks=chunks,
        embeddings=embeddings,
        errors=errors,
    )


def index_to_sqlite(
    # Repository path from a CLI, UI, test, or Python caller.
    path: str | Path,
    # Any concrete object satisfying the Embedder protocol.
    embedder: Embedder,
    # Optional override used by tests or callers that manage index locations.
    db_path: str | Path | None = None,
    # Optional UI/CLI hook. Callers can adapt this to tqdm, Streamlit, or logs.
    progress_callback: ProgressCallback | None = None,
    max_file_size: int = 1_000_000,
    max_files: int = 10_000,
    max_entries: int = 100_000,
    max_chunks_per_file: int = 2_048,
    cancellation_callback: CancellationCallback | None = None,
) -> IndexingReport:
    """Incrementally build an index and atomically replace the SQLite file."""

    raise_if_indexing_cancelled(cancellation_callback)
    root = Path(path).expanduser().resolve()
    database_path = (
        Path(db_path) if db_path is not None else default_database_path(root)
    )
    database_path = database_path.expanduser()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    def check_cancellation() -> None:
        raise_if_indexing_cancelled(cancellation_callback)

    lock_cancellation_check: Callable[[], None] | None = None
    if cancellation_callback is not None:
        lock_cancellation_check = check_cancellation

    with exclusive_database_lock(
        database_path,
        cancellation_check=lock_cancellation_check,
    ):
        raise_if_indexing_cancelled(cancellation_callback)
        return _index_to_sqlite_atomically(
            root=root,
            embedder=embedder,
            database_path=database_path,
            progress_callback=progress_callback,
            max_file_size=max_file_size,
            max_files=max_files,
            max_entries=max_entries,
            max_chunks_per_file=max_chunks_per_file,
            cancellation_callback=cancellation_callback,
        )


def _index_to_sqlite_atomically(
    root: Path,
    embedder: Embedder,
    database_path: Path,
    progress_callback: ProgressCallback | None,
    max_file_size: int,
    max_files: int,
    max_entries: int,
    max_chunks_per_file: int,
    cancellation_callback: CancellationCallback | None,
) -> IndexingReport:
    """Build and promote a private snapshot while owning the database lock."""

    raise_if_indexing_cancelled(cancellation_callback)
    _emit_progress(
        progress_callback,
        "model",
        0,
        1,
        f"Initializing embedding model {embedder.model}",
    )
    raise_if_indexing_cancelled(cancellation_callback)
    embedding_dimension = embedder.dimension
    raise_if_indexing_cancelled(cancellation_callback)
    _emit_progress(
        progress_callback,
        "model",
        1,
        1,
        f"Embedding model ready ({embedding_dimension} dimensions)",
    )
    raise_if_indexing_cancelled(cancellation_callback)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database_path.stem}-",
        suffix=".tmp",
        dir=database_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        raise_if_indexing_cancelled(cancellation_callback)
        if database_path.exists():
            SQLiteIndexStore(database_path).backup_to(temporary_path)
        raise_if_indexing_cancelled(cancellation_callback)

        report = _index_to_sqlite_in_place(
            root=root,
            embedder=embedder,
            database_path=temporary_path,
            reported_database_path=database_path,
            progress_callback=progress_callback,
            embedding_dimension=embedding_dimension,
            max_file_size=max_file_size,
            max_files=max_files,
            max_entries=max_entries,
            max_chunks_per_file=max_chunks_per_file,
            cancellation_callback=cancellation_callback,
        )
        raise_if_indexing_cancelled(cancellation_callback)
        _emit_progress(
            progress_callback,
            "promote",
            0,
            1,
            "Promoting staged SQLite index",
        )
        # This is the final cancellable boundary. Once os.replace starts, the
        # staged snapshot is the committed index and post-commit reporting is
        # deliberately best effort.
        raise_if_indexing_cancelled(cancellation_callback)
        os.replace(temporary_path, database_path)
        _emit_progress_best_effort(
            progress_callback,
            "promote",
            1,
            1,
            "Staged SQLite index promoted",
        )
        _emit_progress_best_effort(
            progress_callback,
            "complete",
            1,
            1,
            (
                f"Indexed {report.file_count} files, {report.chunk_count} chunks, "
                f"{report.embedding_count} embeddings"
            ),
        )
        return report
    finally:
        temporary_path.unlink(missing_ok=True)


def _index_to_sqlite_in_place(
    root: Path,
    embedder: Embedder,
    database_path: Path,
    reported_database_path: Path,
    progress_callback: ProgressCallback | None,
    embedding_dimension: int,
    max_file_size: int,
    max_files: int,
    max_entries: int,
    max_chunks_per_file: int,
    cancellation_callback: CancellationCallback | None,
) -> IndexingReport:
    """Apply one incremental indexing run to a private SQLite snapshot."""

    raise_if_indexing_cancelled(cancellation_callback)
    granular_cancellation_callback = _indexing_cancellation_adapter(
        cancellation_callback
    )
    store = SQLiteIndexStore(database_path)
    store.initialize()
    raise_if_indexing_cancelled(cancellation_callback)

    index_format_version = "1"
    previous_repository = store.load_latest_repository(
        absolute_path=str(root),
        index_format_version=index_format_version,
    )
    existing_repository = store.load_repository_by_identity(
        absolute_path=str(root),
        index_format_version=index_format_version,
        embedding_provider=embedder.provider,
        embedding_model=embedder.model,
        embedding_dim=embedding_dimension,
    )

    repository_id = existing_repository.id if existing_repository else uuid.uuid4()
    repository = Repository(
        id=repository_id,
        absolute_path=str(root),
        index_format_version=index_format_version,
        timestamp_of_index=int(datetime.now(UTC).timestamp()),
        embedding_provider=embedder.provider,
        embedding_model=embedder.model,
        embedding_dim=embedding_dimension,
    )

    _emit_progress(progress_callback, "load", 0, 1, "Loading previous index")
    raise_if_indexing_cancelled(cancellation_callback)
    previous_files = store.load_files(repository_id) if existing_repository else {}
    _emit_progress(progress_callback, "load", 1, 1, "Previous index loaded")
    raise_if_indexing_cancelled(cancellation_callback)

    _emit_progress(progress_callback, "walk", 0, 1, "Walking repository")
    raise_if_indexing_cancelled(cancellation_callback)
    current_paths = walk(
        root,
        max_file_size=max_file_size,
        max_files=max_files,
        max_entries=max_entries,
        cancellation_callback=granular_cancellation_callback,
    )
    raise_if_indexing_cancelled(cancellation_callback)
    _emit_progress(
        progress_callback,
        "walk",
        1,
        1,
        f"Found {len(current_paths)} supported source files",
    )
    _emit_progress(progress_callback, "compare", 0, len(current_paths), "Hashing files")
    raise_if_indexing_cancelled(cancellation_callback)
    current_files_by_path = build_file_manifest(
        root,
        repository_id,
        relative_paths=current_paths,
        max_file_size=max_file_size,
        max_files=max_files,
        max_entries=max_entries,
        cancellation_callback=granular_cancellation_callback,
    )
    raise_if_indexing_cancelled(cancellation_callback)
    manifest_diff = compare_file_manifests(
        current_files_by_path,
        previous_files,
        cancellation_callback=granular_cancellation_callback,
    )
    deleted_paths = manifest_diff.deleted_paths
    added_paths = manifest_diff.added_paths
    changed_paths = manifest_diff.changed_paths
    paths_to_process = added_paths + changed_paths
    _emit_progress(
        progress_callback,
        "compare",
        len(current_paths),
        len(current_paths),
        (
            f"{len(added_paths)} added, {len(changed_paths)} changed, "
            f"{len(deleted_paths)} deleted"
        ),
    )

    errors: list[IndexingError] = []
    embedded_chunk_count = 0
    reused_embedding_count = 0
    written_file_count = 0

    # The index is being built in a private SQLite snapshot, so successful
    # files can be committed one at a time without exposing a partial index.
    # Keeping only one file's symbols, chunks, and vectors in memory prevents
    # repository size from determining peak indexing memory.
    _emit_progress(
        progress_callback,
        "write",
        0,
        len(paths_to_process) + len(deleted_paths),
        "Writing changed files to staged SQLite index",
    )

    for index_number, relative_path in enumerate(paths_to_process, start=1):
        raise_if_indexing_cancelled(cancellation_callback)
        _emit_progress(
            progress_callback,
            "index",
            index_number - 1,
            len(paths_to_process),
            f"Indexing {relative_path}",
        )
        raise_if_indexing_cancelled(cancellation_callback)
        file_record = current_files_by_path[relative_path]
        _emit_progress(
            progress_callback,
            "parse",
            index_number - 1,
            len(paths_to_process),
            f"Parsing {relative_path}",
        )
        raise_if_indexing_cancelled(cancellation_callback)
        file_index = _index_single_file(
            root,
            repository_id,
            relative_path,
            max_file_size,
            max_chunks_per_file,
            expected_content_hash=file_record.content_hash,
        )
        raise_if_indexing_cancelled(cancellation_callback)
        errors.extend(file_index.errors)

        if file_index.errors:
            _emit_progress(
                progress_callback,
                "parse",
                index_number,
                len(paths_to_process),
                f"Skipped {relative_path} after parsing error",
            )
            _emit_progress(
                progress_callback,
                "index",
                index_number,
                len(paths_to_process),
                f"Skipped {relative_path} after indexing error",
            )
            continue

        _emit_progress(
            progress_callback,
            "parse",
            index_number,
            len(paths_to_process),
            f"Parsed {relative_path}",
        )
        _emit_progress(
            progress_callback,
            "embed",
            index_number - 1,
            len(paths_to_process),
            f"Embedding changed chunks from {relative_path}",
        )
        raise_if_indexing_cancelled(cancellation_callback)
        reusable_embeddings = (
            store.load_embeddings_by_content_hashes(
                repository_id,
                repository.embedding_model,
                repository.embedding_dim,
                (chunk.content_hash for chunk in file_index.chunks),
            )
            if existing_repository
            else {}
        )
        raise_if_indexing_cancelled(cancellation_callback)
        embeddings, embedded_count, reused_count = _embeddings_for_chunks(
            file_index.chunks,
            file_index.symbols,
            reusable_embeddings,
            embedder,
        )
        raise_if_indexing_cancelled(cancellation_callback)
        embedded_chunk_count += embedded_count
        reused_embedding_count += reused_count
        _emit_progress(
            progress_callback,
            "embed",
            index_number,
            len(paths_to_process),
            (
                f"Embedded {embedded_count} and reused {reused_count} chunks "
                f"from {relative_path}"
            ),
        )

        raise_if_indexing_cancelled(cancellation_callback)
        store.apply_file_updates(
            repository=repository,
            changed_files=[
                IndexedFileRecords(
                    file=file_record,
                    symbols=file_index.symbols,
                    chunks=file_index.chunks,
                    embeddings=embeddings,
                )
            ],
            deleted_relative_paths=[],
        )
        raise_if_indexing_cancelled(cancellation_callback)
        written_file_count += 1
        _emit_progress(
            progress_callback,
            "write",
            written_file_count,
            len(paths_to_process) + len(deleted_paths),
            f"Wrote {relative_path} to staged SQLite index",
        )
        _emit_progress(
            progress_callback,
            "index",
            index_number,
            len(paths_to_process),
            f"Indexed {relative_path}",
        )

    if not paths_to_process:
        _emit_progress(progress_callback, "index", 0, 0, "No file changes to index")

    incompatible_rebuild = (
        previous_repository is not None and existing_repository is None
    )
    if incompatible_rebuild and errors:
        failed_paths = ", ".join(error.relative_path for error in errors[:3])
        if len(errors) > 3:
            failed_paths = f"{failed_paths}, and {len(errors) - 3} more"
        raise RuntimeError(
            "Embedding configuration changed, but a complete rebuild failed for "
            f"{failed_paths}; the previous index was preserved"
        )

    # Finalize deletions and always upsert repository metadata, including for an
    # empty repository or a no-op reindex. Incompatible historical rows remain
    # present until the replacement has been built successfully.
    raise_if_indexing_cancelled(cancellation_callback)
    store.apply_file_updates(
        repository=repository,
        changed_files=[],
        deleted_relative_paths=deleted_paths,
    )
    raise_if_indexing_cancelled(cancellation_callback)
    if existing_repository is None:
        store.delete_other_repositories_by_path(str(root), repository.id)
    raise_if_indexing_cancelled(cancellation_callback)
    total_database_changes = written_file_count + len(deleted_paths)
    _emit_progress(
        progress_callback,
        "write",
        total_database_changes,
        total_database_changes,
        "SQLite index written",
    )

    report = IndexingReport(
        repository=repository,
        database_path=reported_database_path,
        symbol_count=store.count_rows("symbols", repository.id),
        chunk_count=store.count_rows("chunks", repository.id),
        embedding_count=store.count_rows("embeddings", repository.id),
        file_count=store.count_rows("files", repository.id),
        added_file_count=len(added_paths),
        changed_file_count=len(changed_paths),
        deleted_file_count=len(deleted_paths),
        embedded_chunk_count=embedded_chunk_count,
        reused_embedding_count=reused_embedding_count,
        errors=errors,
        changed_paths=manifest_diff.all_changed_paths,
    )

    return report


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    """Send a progress event when the caller provided a callback."""

    if callback is None:
        return

    callback(
        IndexingProgress(
            stage=stage,
            current=current,
            total=total,
            message=message,
        )
    )


def _emit_progress_best_effort(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    """Report post-commit progress without changing the committed outcome."""

    try:
        _emit_progress(callback, stage, current, total, message)
    except Exception:
        return


@dataclass
class _FileIndex:
    """Index artifacts generated from one changed file."""

    symbols: list[Symbol]
    chunks: list[Chunk]
    errors: list[IndexingError]


def _index_single_file(
    # Canonical repository root.
    root: Path,
    # Stable repository ID reused across incremental runs.
    repository_id: uuid.UUID,
    # POSIX repository-relative source path.
    relative_path: str,
    max_file_size: int,
    max_chunks: int = 2_048,
    expected_content_hash: str | None = None,
) -> _FileIndex:
    """Parse and chunk one source file."""

    try:
        source, content_hash = _read_source_text_and_hash(
            root,
            relative_path,
            max_file_size,
        )
    except (OSError, UnicodeDecodeError) as error:
        return _FileIndex(
            symbols=[],
            chunks=[],
            errors=[
                IndexingError(
                    relative_path=relative_path,
                    stage="read",
                    message=str(error),
                )
            ],
        )

    if (
        expected_content_hash is not None
        and content_hash != expected_content_hash
    ):
        return _FileIndex(
            symbols=[],
            chunks=[],
            errors=[
                IndexingError(
                    relative_path=relative_path,
                    stage="read",
                    message="Source file changed during indexing; retry",
                )
            ],
        )

    try:
        parsed_symbols = parse_symbols(source)
    except SyntaxError as error:
        return _FileIndex(
            symbols=[],
            chunks=[],
            errors=[
                IndexingError(
                    relative_path=relative_path,
                    stage="parse",
                    message=str(error),
                )
            ],
        )

    symbols = [
        Symbol(
            id=uuid.uuid4(),
            repository_id=repository_id,
            name=parsed.name,
            qualified_name=parsed.qualified_name,
            kind=parsed.kind,
            relative_path=relative_path,
            start_line=parsed.start_line,
            end_line=parsed.end_line,
            source_snippet=parsed.source_snippet,
        )
        for parsed in parsed_symbols
    ]

    try:
        chunks = chunk_symbols(source, symbols, max_chunks=max_chunks)
    except ValueError as error:
        return _FileIndex(
            symbols=symbols,
            chunks=[],
            errors=[
                IndexingError(
                    relative_path=relative_path,
                    stage="chunk",
                    message=str(error),
                )
            ],
        )

    return _FileIndex(symbols=symbols, chunks=chunks, errors=[])


def _read_source_text(
    root: Path,
    relative_path: str | Path,
    max_file_size: int,
) -> str:
    """Read one confined UTF-8 source file without following a final symlink."""

    source, _content_hash = _read_source_text_and_hash(
        root,
        relative_path,
        max_file_size,
    )
    return source


def _read_source_text_and_hash(
    root: Path,
    relative_path: str | Path,
    max_file_size: int,
) -> tuple[str, str]:
    """Read source text and return the hash of the same verified bytes."""

    absolute_path = root / relative_path
    if absolute_path.is_symlink():
        raise OSError("Refusing to read a symbolic link")
    resolved_path = absolute_path.resolve(strict=True)
    if resolved_path != root and root not in resolved_path.parents:
        raise OSError("Refusing to read a path outside the repository")

    _, source_bytes = read_regular_file(
        absolute_path,
        byte_limit=max_file_size + 1,
    )

    if len(source_bytes) > max_file_size:
        raise OSError(f"Source file exceeds the {max_file_size} byte limit")
    return source_bytes.decode("utf-8"), hashlib.sha256(source_bytes).hexdigest()


def _embeddings_for_chunks(
    # Chunks from one successfully processed file.
    chunks: list[Chunk],
    # Symbols from the same file, used to rebuild embedding text.
    symbols: list[Symbol],
    # Previously stored vectors keyed by deterministic chunk content hash.
    reusable_embeddings: dict[str, list[float]],
    # Embedding provider for chunks that cannot be reused.
    embedder: Embedder,
) -> tuple[list[list[float]], int, int]:
    """Return chunk-aligned embeddings while reusing stored vectors."""

    symbols_by_id = {symbol.id: symbol for symbol in symbols}
    embeddings: list[list[float] | None] = []
    texts_to_embed: list[str] = []
    embedding_positions: list[int] = []
    reused_count = 0

    for chunk in chunks:
        reusable_embedding = reusable_embeddings.get(chunk.content_hash)
        if reusable_embedding is not None:
            embeddings.append(reusable_embedding)
            reused_count += 1
            continue

        embeddings.append(None)
        embedding_positions.append(len(embeddings) - 1)
        texts_to_embed.append(_embedding_text_for_chunk(chunk, symbols_by_id))

    new_embeddings = embedder.embed(texts_to_embed) if texts_to_embed else []
    validate_embeddings(
        texts_to_embed,
        new_embeddings,
        expected_dimension=embedder.dimension,
    )

    for position, embedding in zip(embedding_positions, new_embeddings, strict=True):
        embeddings[position] = embedding

    final_embeddings: list[list[float]] = []
    for embedding in embeddings:
        if embedding is None:
            raise ValueError("Chunk is missing an embedding")
        final_embeddings.append(embedding)

    return final_embeddings, len(new_embeddings), reused_count


def _embedding_text_for_chunk(
    # Supplies source, path, and optional owning-symbol identity.
    chunk: Chunk,
    # Maps IDs to complete symbols without repeated list scans.
    symbols_by_id: dict[uuid.UUID, Symbol],
) -> str:
    """Reconstruct the exact enriched text represented by a chunk hash."""

    # Future module-level chunks have no symbol. Symbol chunks retrieve their
    # qualified name and kind for embedding context.
    symbol = symbols_by_id.get(chunk.symbol_id) if chunk.symbol_id else None

    # Use the shared formatting function so content hashing and vector creation
    # cannot silently diverge.
    return build_embedding_text(
        # File context is always available.
        relative_path=chunk.relative_path,
        # The source body remains unchanged.
        raw_text=chunk.raw_text,
        # Module-level chunks omit symbol-specific metadata.
        qualified_name=symbol.qualified_name if symbol else None,
        kind=symbol.kind if symbol else None,
    )

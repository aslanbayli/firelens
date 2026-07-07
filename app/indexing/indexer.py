"""Coordinate repository walking, parsing, chunking, and embedding.

The indexer is an orchestrator. It determines pipeline order and collects
results, while specialized modules retain responsibility for filesystem rules,
AST traversal, chunk boundaries, and vector generation.
"""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from app.core.models import Chunk, Repository, Symbol
from app.indexing.chunker import build_embedding_text, chunk_symbols
from app.indexing.embedder import Embedder, validate_embeddings
from app.indexing.parser import parse_symbols
from app.indexing.walker import walk
from app.storage.database import (
    IndexedFile,
    IndexedFileRecords,
    SQLiteIndexStore,
    default_database_path,
)


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


@dataclass(frozen=True)
class IndexingProgress:
    """One progress update from a persisted indexing run."""

    stage: str
    current: int
    total: int
    message: str


ProgressCallback = Callable[[IndexingProgress], None]


def index(
    # Repository path from a CLI, UI, test, or Python caller.
    path: str | Path,
    # Any concrete object satisfying the Embedder protocol.
    embedder: Embedder,
) -> InMemoryIndex:
    """Build an in-memory index for a local Python repository."""

    # Convert strings to Path, expand "~", resolve "..", and produce one
    # canonical absolute root. The walker performs existence/type validation.
    root = Path(path).expanduser().resolve()

    # Every symbol and chunk from this run refers to this repository identity.
    repository_id = uuid.uuid4()

    # Capture repository and embedding compatibility metadata before processing
    # files. A persisted index must not mix vectors from incompatible models.
    repository = Repository(
        # Unique identity for this repository record.
        id=repository_id,
        # Serialize the canonical Path as a normal string.
        absolute_path=str(root),
        # Version the index format so future schema changes can be detected.
        index_format_version="1",
        # Store a timezone-aware current time as an integer Unix timestamp.
        timestamp_of_index=int(datetime.now(UTC).timestamp()),
        # Read provider metadata through the generic interface.
        embedding_model=embedder.model,
        # Every vector in this index must have this exact length.
        embedding_dim=embedder.dimension,
    )

    # Discover safe, supported files using walker defaults. Results are relative
    # and sorted, so processing order is deterministic.
    paths = walk(root)

    # Collect all successfully parsed declarations.
    symbols: list[Symbol] = []

    # Collect chunks in file/symbol/window order.
    chunks: list[Chunk] = []

    # Collect recoverable failures rather than terminating the whole repository.
    errors: list[IndexingError] = []

    # Run each discovered file through read → parse → chunk.
    for relative_path in paths:
        # Reconstruct the source file location from the trusted root and the
        # walker-produced relative path.
        absolute_path = root / relative_path

        try:
            # Read and decode the entire Python source file as UTF-8 text.
            source = absolute_path.read_text(encoding="utf-8")
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
            file_chunks = chunk_symbols(source, file_symbols)

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

    # Create O(1)-average ID lookup. Scanning all symbols for every chunk would
    # grow unnecessarily expensive as repository size increases.
    symbols_by_id = {symbol.id: symbol for symbol in symbols}

    # Reconstruct enriched embedding inputs in chunk order. This positional
    # order is the contract connecting chunks to returned vectors.
    embedding_texts = [
        _embedding_text_for_chunk(chunk, symbols_by_id) for chunk in chunks
    ]

    # Send one batch to allow real providers/models to amortize overhead.
    embeddings = embedder.embed(embedding_texts)

    # Reject missing, extra, or wrong-sized vectors before storage/search.
    validate_embeddings(
        # Establishes expected vector count.
        embedding_texts,
        # Provider output being verified.
        embeddings,
        # Establishes required length of every vector.
        expected_dimension=embedder.dimension,
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
) -> IndexingReport:
    """Incrementally build an index and persist it to SQLite."""

    root = Path(path).expanduser().resolve()
    database_path = Path(db_path) if db_path is not None else default_database_path(root)
    store = SQLiteIndexStore(database_path)
    store.initialize()

    index_format_version = "1"
    existing_repository = store.load_repository_by_identity(
        absolute_path=str(root),
        index_format_version=index_format_version,
        embedding_model=embedder.model,
        embedding_dim=embedder.dimension,
    )

    repository_id = existing_repository.id if existing_repository else uuid.uuid4()
    repository = Repository(
        id=repository_id,
        absolute_path=str(root),
        index_format_version=index_format_version,
        timestamp_of_index=int(datetime.now(UTC).timestamp()),
        embedding_model=embedder.model,
        embedding_dim=embedder.dimension,
    )

    _emit_progress(progress_callback, "load", 0, 1, "Loading previous index")
    previous_files = store.load_files(repository_id) if existing_repository else {}
    reusable_embeddings = (
        store.load_embeddings_by_content_hash(
            repository_id,
            repository.embedding_model,
            repository.embedding_dim,
        )
        if existing_repository
        else {}
    )
    _emit_progress(progress_callback, "load", 1, 1, "Previous index loaded")

    _emit_progress(progress_callback, "walk", 0, 1, "Walking repository")
    current_paths = walk(root)
    _emit_progress(
        progress_callback,
        "walk",
        1,
        1,
        f"Found {len(current_paths)} supported source files",
    )
    _emit_progress(progress_callback, "compare", 0, len(current_paths), "Hashing files")
    current_files = _file_records_for_paths(root, repository_id, current_paths)
    current_files_by_path = {file.relative_path: file for file in current_files}

    previous_paths = set(previous_files)
    current_path_names = set(current_files_by_path)

    deleted_paths = sorted(previous_paths - current_path_names)
    added_paths = sorted(current_path_names - previous_paths)
    changed_paths = sorted(
        path_name
        for path_name in current_path_names.intersection(previous_paths)
        if current_files_by_path[path_name].content_hash
        != previous_files[path_name].content_hash
    )
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

    changed_file_indexes: list[IndexedFileRecords] = []
    errors: list[IndexingError] = []
    embedded_chunk_count = 0
    reused_embedding_count = 0

    for index_number, relative_path in enumerate(paths_to_process, start=1):
        _emit_progress(
            progress_callback,
            "index",
            index_number - 1,
            len(paths_to_process),
            f"Indexing {relative_path}",
        )
        file_record = current_files_by_path[relative_path]
        file_index = _index_single_file(root, repository_id, relative_path)
        errors.extend(file_index.errors)

        if file_index.errors:
            _emit_progress(
                progress_callback,
                "index",
                index_number,
                len(paths_to_process),
                f"Skipped {relative_path} after indexing error",
            )
            continue

        embeddings, embedded_count, reused_count = _embeddings_for_chunks(
            file_index.chunks,
            file_index.symbols,
            reusable_embeddings,
            embedder,
        )
        embedded_chunk_count += embedded_count
        reused_embedding_count += reused_count

        changed_file_indexes.append(
            IndexedFileRecords(
                file=file_record,
                symbols=file_index.symbols,
                chunks=file_index.chunks,
                embeddings=embeddings,
            )
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

    total_database_changes = len(changed_file_indexes) + len(deleted_paths)
    _emit_progress(
        progress_callback,
        "write",
        0,
        total_database_changes,
        "Writing SQLite index",
    )
    store.apply_file_updates(
        repository=repository,
        changed_files=changed_file_indexes,
        deleted_relative_paths=deleted_paths,
    )
    _emit_progress(
        progress_callback,
        "write",
        total_database_changes,
        total_database_changes,
        "SQLite index written",
    )

    report = IndexingReport(
        repository=repository,
        database_path=database_path,
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
    )

    _emit_progress(
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
) -> _FileIndex:
    """Parse and chunk one source file."""

    absolute_path = root / relative_path

    try:
        source = absolute_path.read_text(encoding="utf-8")
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
        chunks = chunk_symbols(source, symbols)
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


def _file_records_for_paths(
    # Canonical repository root used by the current index.
    root: Path,
    # Stable repository ID used for the current persisted index.
    repository_id: uuid.UUID,
    # Repository-relative paths selected by the walker.
    relative_paths: list[Path],
) -> list[IndexedFile]:
    """Build file metadata records for current source files."""

    files: list[IndexedFile] = []

    for relative_path in relative_paths:
        absolute_path = root / relative_path
        stat = absolute_path.stat()
        content_hash = hashlib.sha256(absolute_path.read_bytes()).hexdigest()

        files.append(
            IndexedFile(
                repository_id=repository_id,
                relative_path=relative_path.as_posix(),
                modified_time_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
                content_hash=content_hash,
            )
        )

    return files

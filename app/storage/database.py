"""SQLite persistence for FireLens repository indexes.

The storage layer owns SQL and vector serialization. Indexing and search code
should interact with this module through small methods instead of embedding SQL
queries directly.
"""

import hashlib
import re
import sqlite3
import uuid
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from app.core.models import Chunk, Repository, Symbol


class SearchCandidateLimitError(ValueError):
    """Raised when a retrieval mode cannot safely rank the full candidate set."""


@dataclass(frozen=True)
class IndexedFile:
    """Filesystem metadata for one file included in an index."""

    repository_id: uuid.UUID
    relative_path: str
    modified_time_ns: int
    size_bytes: int
    content_hash: str


@dataclass(frozen=True)
class IndexedFileRecords:
    """All index records generated from one source file."""

    file: IndexedFile
    symbols: list[Symbol]
    chunks: list[Chunk]
    embeddings: list[list[float]]


def default_database_path(
    repository_root: str | Path,
    data_directory: str | Path | None = None,
) -> Path:
    """Return the conventional SQLite path for a repository root."""

    root = Path(repository_root).expanduser().resolve()
    readable_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", root.name).strip("-")
    if not readable_name:
        readable_name = "repository"
    # Keep the generated directory component well below common NAME_MAX
    # limits even when the source directory itself has a 255-byte name.
    readable_name = readable_name[:64]

    path_hash = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    repository_key = f"{readable_name}-{path_hash}"

    index_root = (
        Path(data_directory)
        if data_directory is not None
        else Path("data") / "indexes"
    )
    return index_root / repository_key / "firelens.db"


def pack_vector(vector: Iterable[float]) -> bytes:
    """Serialize a vector into compact float bytes for SQLite storage."""

    return array("f", [float(value) for value in vector]).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    """Deserialize vector bytes produced by pack_vector."""

    values = array("f")
    values.frombytes(blob)
    return list(values)


@dataclass(frozen=True)
class StoredSymbolCandidate:
    """Small symbol record used while ranking fuzzy matches."""

    id: uuid.UUID
    name: str
    qualified_name: str
    relative_path: str
    start_line: int


@dataclass(frozen=True)
class StoredSemanticCandidate:
    """Chunk metadata retained in the in-memory semantic search index."""

    chunk_id: uuid.UUID
    relative_path: str
    start_line: int
    end_line: int
    qualified_symbol_name: str | None = None


@dataclass(frozen=True)
class StoredSemanticCandidateRow:
    """One streamed semantic candidate and its serialized vector."""

    candidate: StoredSemanticCandidate
    vector_blob: bytes


@dataclass(frozen=True)
class SemanticCandidateSummary:
    """Size and shape information used before allocating a search matrix."""

    count: int
    vector_bytes: int
    dimension: int | None


class SQLiteIndexStore:
    """Store and load one FireLens SQLite index database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a writable SQLite connection, creating its parent directory."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        self._configure_connection(connection)

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Open an existing SQLite database without creating or modifying it."""

        database_uri = f"{self.db_path.expanduser().resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True)
        self._configure_connection(connection)
        connection.execute("PRAGMA query_only = ON")

        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _configure_connection(connection: sqlite3.Connection) -> None:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

    def initialize(self) -> None:
        """Create the schema and indexes if they do not already exist."""

        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY,
                    absolute_path TEXT NOT NULL,
                    index_format_version TEXT NOT NULL,
                    timestamp_of_index INTEGER NOT NULL,
                    embedding_provider TEXT NOT NULL DEFAULT 'unknown',
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repository_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    modified_time_ns INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    UNIQUE(repository_id, relative_path),
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    source_snippet TEXT NOT NULL,
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    symbol_id TEXT,
                    raw_text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(symbol_id)
                        REFERENCES symbols(id)
                        ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY(chunk_id)
                        REFERENCES chunks(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_symbols_name
                    ON symbols(repository_id, name);
                CREATE INDEX IF NOT EXISTS idx_symbols_qualified_name
                    ON symbols(repository_id, qualified_name);
                CREATE INDEX IF NOT EXISTS idx_symbols_path
                    ON symbols(repository_id, relative_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_path
                    ON chunks(repository_id, relative_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_symbol_id
                    ON chunks(symbol_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_content_hash
                    ON chunks(repository_id, content_hash);
                CREATE INDEX IF NOT EXISTS idx_embeddings_repo_model
                    ON embeddings(repository_id, model);
                """
            )
            repository_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(repositories)")
            }
            if "embedding_provider" not in repository_columns:
                connection.execute(
                    "ALTER TABLE repositories "
                    "ADD COLUMN embedding_provider TEXT NOT NULL DEFAULT 'unknown'"
                )

    def backup_to(self, destination: str | Path) -> None:
        """Create a consistent SQLite snapshot at ``destination``."""

        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        with self.read_connection() as source_connection:
            destination_connection = sqlite3.connect(destination_path)
            try:
                source_connection.backup(destination_connection)
            finally:
                destination_connection.close()

    def delete_repositories_by_path(self, absolute_path: str) -> None:
        """Delete all stored index versions for one canonical repository path."""

        with self.connect() as connection:
            connection.execute(
                "DELETE FROM repositories WHERE absolute_path = ?",
                (absolute_path,),
            )

    def delete_other_repositories_by_path(
        self,
        absolute_path: str,
        repository_id_to_keep: uuid.UUID,
    ) -> None:
        """Delete incompatible index versions while preserving a staged one."""

        with self.connect() as connection:
            connection.execute(
                """
                DELETE FROM repositories
                WHERE absolute_path = ? AND id != ?
                """,
                (absolute_path, str(repository_id_to_keep)),
            )

    def replace_index(
        self,
        repository: Repository,
        files: list[IndexedFile],
        symbols: list[Symbol],
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Replace all persisted records for a repository in one transaction."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")

        for vector in embeddings:
            if len(vector) != repository.embedding_dim:
                raise ValueError("Embedding dimension does not match repository")

        repository_id = str(repository.id)

        with self.connect() as connection:
            self._upsert_repository(connection, repository)
            connection.execute(
                "DELETE FROM embeddings WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "DELETE FROM chunks WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "DELETE FROM symbols WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "DELETE FROM files WHERE repository_id = ?",
                (repository_id,),
            )
            self._insert_files(connection, files)
            self._insert_symbols(connection, symbols)
            self._insert_chunks(connection, chunks)
            self._insert_embeddings(connection, repository, chunks, embeddings)

    def apply_file_updates(
        self,
        repository: Repository,
        changed_files: list[IndexedFileRecords],
        deleted_relative_paths: list[str],
    ) -> None:
        """Apply per-file additions, updates, and deletions in one transaction."""

        for file_records in changed_files:
            if len(file_records.chunks) != len(file_records.embeddings):
                raise ValueError("Each chunk must have exactly one embedding")

            for vector in file_records.embeddings:
                if len(vector) != repository.embedding_dim:
                    raise ValueError("Embedding dimension does not match repository")

        repository_id = str(repository.id)

        with self.connect() as connection:
            self._upsert_repository(connection, repository)

            for relative_path in deleted_relative_paths:
                self._delete_file_records(connection, repository_id, relative_path)

            for file_records in changed_files:
                self._delete_file_records(
                    connection,
                    repository_id,
                    file_records.file.relative_path,
                )
                self._insert_files(connection, [file_records.file])
                self._insert_symbols(connection, file_records.symbols)
                self._insert_chunks(connection, file_records.chunks)
                self._insert_embeddings(
                    connection,
                    repository,
                    file_records.chunks,
                    file_records.embeddings,
                )

    def count_rows(self, table: str, repository_id: uuid.UUID) -> int:
        """Return a repository-scoped row count for tests and diagnostics."""

        allowed_tables = {"files", "symbols", "chunks", "embeddings"}
        if table not in allowed_tables:
            raise ValueError(f"Unsupported table: {table}")

        with self.read_connection() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE repository_id = ?",
                (str(repository_id),),
            ).fetchone()

        return int(row["count"])

    def load_repository_by_identity(
        self,
        absolute_path: str,
        index_format_version: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dim: int,
    ) -> Repository | None:
        """Load the compatible repository row for a local path, if present."""

        with self.read_connection() as connection:
            provider_select = _embedding_provider_expression(connection, select=True)
            provider_match = _embedding_provider_expression(connection)
            row = connection.execute(
                f"""
                SELECT
                    id,
                    absolute_path,
                    index_format_version,
                    timestamp_of_index,
                    {provider_select},
                    embedding_model,
                    embedding_dim
                FROM repositories
                WHERE absolute_path = ?
                    AND index_format_version = ?
                    AND {provider_match} = ?
                    AND embedding_model = ?
                    AND embedding_dim = ?
                ORDER BY timestamp_of_index DESC
                LIMIT 1
                """,
                (
                    absolute_path,
                    index_format_version,
                    embedding_provider,
                    embedding_model,
                    embedding_dim,
                ),
            ).fetchone()

        if row is None:
            return None

        return _repository_from_row(row)

    def load_repository(self, repository_id: uuid.UUID) -> Repository | None:
        """Load repository metadata by ID."""

        with self.read_connection() as connection:
            provider_expression = _embedding_provider_expression(
                connection,
                select=True,
            )
            row = connection.execute(
                f"""
                SELECT
                    id,
                    absolute_path,
                    index_format_version,
                    timestamp_of_index,
                    {provider_expression},
                    embedding_model,
                    embedding_dim
                FROM repositories
                WHERE id = ?
                """,
                (str(repository_id),),
            ).fetchone()

        if row is None:
            return None

        return _repository_from_row(row)

    def load_latest_repository(
        self,
        absolute_path: str,
        index_format_version: str | None = None,
    ) -> Repository | None:
        """Load the newest repository row without initializing an embedder."""

        parameters: list[str] = [absolute_path]
        version_clause = ""
        if index_format_version is not None:
            version_clause = "AND index_format_version = ?"
            parameters.append(index_format_version)

        with self.read_connection() as connection:
            provider_expression = _embedding_provider_expression(
                connection,
                select=True,
            )
            row = connection.execute(
                f"""
                SELECT
                    id,
                    absolute_path,
                    index_format_version,
                    timestamp_of_index,
                    {provider_expression},
                    embedding_model,
                    embedding_dim
                FROM repositories
                WHERE absolute_path = ?
                    {version_clause}
                ORDER BY timestamp_of_index DESC, id DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()

        if row is None:
            return None

        return _repository_from_row(row)

    def list_repositories(
        self,
        index_format_version: str | None = None,
    ) -> list[Repository]:
        """List repository metadata without filtering by embedding settings."""

        if not self.db_path.exists():
            return []

        parameters: list[str] = []
        version_clause = ""
        if index_format_version is not None:
            version_clause = "WHERE index_format_version = ?"
            parameters.append(index_format_version)

        with self.read_connection() as connection:
            provider_expression = _embedding_provider_expression(
                connection,
                select=True,
            )
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    absolute_path,
                    index_format_version,
                    timestamp_of_index,
                    {provider_expression},
                    embedding_model,
                    embedding_dim
                FROM repositories
                {version_clause}
                ORDER BY absolute_path, timestamp_of_index DESC, id DESC
                """,
                parameters,
            ).fetchall()

        return [_repository_from_row(row) for row in rows]

    def list_compatible_repositories(
        self,
        index_format_version: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dim: int,
    ) -> list[Repository]:
        """Load compatible repository records from this index database."""

        if not self.db_path.exists():
            return []

        with self.read_connection() as connection:
            provider_select = _embedding_provider_expression(connection, select=True)
            provider_match = _embedding_provider_expression(connection)
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    absolute_path,
                    index_format_version,
                    timestamp_of_index,
                    {provider_select},
                    embedding_model,
                    embedding_dim
                FROM repositories
                WHERE index_format_version = ?
                    AND {provider_match} = ?
                    AND embedding_model = ?
                    AND embedding_dim = ?
                ORDER BY absolute_path, timestamp_of_index DESC
                """,
                (
                    index_format_version,
                    embedding_provider,
                    embedding_model,
                    embedding_dim,
                ),
            ).fetchall()

        return [_repository_from_row(row) for row in rows]

    def load_files(self, repository_id: uuid.UUID) -> dict[str, IndexedFile]:
        """Load file metadata keyed by repository-relative path."""

        with self.read_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    repository_id,
                    relative_path,
                    modified_time_ns,
                    size_bytes,
                    content_hash
                FROM files
                WHERE repository_id = ?
                """,
                (str(repository_id),),
            ).fetchall()

        return {
            row["relative_path"]: IndexedFile(
                repository_id=uuid.UUID(row["repository_id"]),
                relative_path=row["relative_path"],
                modified_time_ns=row["modified_time_ns"],
                size_bytes=row["size_bytes"],
                content_hash=row["content_hash"],
            )
            for row in rows
        }

    def exact_search_symbols(
        self,
        repository_id: uuid.UUID,
        query: str,
        path_filter: str | None = None,
        limit: int = 10,
        max_snippet_chars: int = 4_000,
    ) -> list[Symbol]:
        """Load exact symbol matches in deterministic ranking order."""

        if limit < 1:
            raise ValueError("limit must be greater than 0")
        if max_snippet_chars < 1:
            raise ValueError("max_snippet_chars must be greater than 0")

        qualified_matches = self._load_symbols_by_column(
            repository_id=repository_id,
            column="qualified_name",
            value=query,
            path_filter=path_filter,
            limit=limit,
            max_snippet_chars=max_snippet_chars,
        )

        matches = list(qualified_matches)
        if len(matches) >= limit:
            return matches

        seen_ids = {symbol.id for symbol in matches}

        short_name_matches = self._load_symbols_by_column(
            repository_id=repository_id,
            column="name",
            value=query,
            path_filter=path_filter,
            # Fetch at most one result page. Some short-name matches can be
            # duplicates of qualified-name matches, so using ``limit`` rather
            # than only the remaining count still lets the page fill.
            limit=limit,
            max_snippet_chars=max_snippet_chars,
        )

        for symbol in short_name_matches:
            if symbol.id not in seen_ids:
                matches.append(symbol)
                seen_ids.add(symbol.id)

        return matches[:limit]

    def load_symbol_candidates(
        self,
        repository_id: uuid.UUID,
        path_filter: str | None = None,
        *,
        limit: int,
        candidate_char_limit: int,
    ) -> list[StoredSymbolCandidate]:
        """Load bounded, snippet-free symbol metadata for fuzzy ranking.

        An error is preferable to silently ranking only an arbitrary prefix of
        a repository. Callers can ask the user to narrow the path or use a
        retrieval mode designed for larger candidate sets.
        """

        if limit < 1:
            raise ValueError("limit must be greater than 0")
        if candidate_char_limit < 1:
            raise ValueError("candidate_char_limit must be greater than 0")

        parameters: list[str | int] = [
            candidate_char_limit,
            candidate_char_limit,
            str(repository_id),
        ]
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "relative_path",
                path_filter,
            )
            parameters.extend(path_parameters)
        parameters.append(limit + 1)

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    CASE WHEN length(name) <= ? THEN name ELSE '' END AS name,
                    CASE
                        WHEN length(qualified_name) <= ? THEN qualified_name
                        ELSE ''
                    END AS qualified_name,
                    relative_path,
                    start_line
                FROM symbols
                WHERE repository_id = ?
                    {path_clause}
                ORDER BY relative_path, qualified_name, start_line, id
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        if len(rows) > limit:
            raise SearchCandidateLimitError(
                "Fuzzy search candidate limit was exceeded; narrow the path "
                "filter or use semantic search"
            )

        return [
            StoredSymbolCandidate(
                id=uuid.UUID(row["id"]),
                name=row["name"],
                qualified_name=row["qualified_name"],
                relative_path=row["relative_path"],
                start_line=row["start_line"],
            )
            for row in rows
        ]

    def load_symbols_by_ids(
        self,
        symbol_ids: Iterable[uuid.UUID],
        *,
        max_snippet_chars: int,
    ) -> dict[uuid.UUID, Symbol]:
        """Load complete symbol records for a small ranked result set."""

        if max_snippet_chars < 1:
            raise ValueError("max_snippet_chars must be greater than 0")

        unique_ids = list(dict.fromkeys(symbol_ids))
        if not unique_ids:
            return {}

        placeholders = ", ".join("?" for _ in unique_ids)
        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    repository_id,
                    name,
                    qualified_name,
                    kind,
                    relative_path,
                    start_line,
                    end_line,
                    substr(source_snippet, 1, ?) AS source_snippet
                FROM symbols
                WHERE id IN ({placeholders})
                """,
                [
                    max_snippet_chars + 1,
                    *(str(symbol_id) for symbol_id in unique_ids),
                ],
            ).fetchall()

        symbols = [_symbol_from_row(row) for row in rows]
        return {symbol.id: symbol for symbol in symbols}

    def load_all_symbols(
        self, repository_id: uuid.UUID, path_filter: str | None = None
    ) -> list[Symbol]:
        """Load all symbols for a repository, optionally filtered by path."""
        parameters: list[str] = [str(repository_id)]
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "relative_path",
                path_filter,
            )
            parameters.extend(path_parameters)

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    repository_id,
                    name,
                    qualified_name,
                    kind,
                    relative_path,
                    start_line,
                    end_line,
                    source_snippet
                FROM symbols
                WHERE repository_id = ?
                    {path_clause}
                ORDER BY relative_path, qualified_name, start_line
                """,
                parameters,
            ).fetchall()

        return [_symbol_from_row(row) for row in rows]

    def semantic_candidate_summary(
        self,
        repository_id: uuid.UUID,
        path_filter: str | None = None,
        *,
        max_candidates: int,
        max_vector_bytes: int,
        cancellation_check: Callable[[], None] | None = None,
    ) -> SemanticCandidateSummary:
        """Return semantic candidate bounds without loading their contents."""

        if max_candidates < 1:
            raise ValueError("max_candidates must be greater than 0")
        if max_vector_bytes < 1:
            raise ValueError("max_vector_bytes must be greater than 0")

        parameters: list[str] = [str(repository_id)]
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "chunks.relative_path",
                path_filter,
            )
            parameters.extend(path_parameters)

        parameters.append(max_candidates + 1)

        candidate_count = 0
        vector_bytes = 0
        dimensions: set[int] = set()
        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    embeddings.dimension,
                    length(embeddings.vector) AS vector_bytes
                FROM chunks
                INNER JOIN embeddings
                    ON embeddings.chunk_id = chunks.id
                WHERE chunks.repository_id = ?
                    {path_clause}
                LIMIT ?
                """,
                parameters,
            )
            for row in rows:
                if cancellation_check is not None:
                    cancellation_check()
                candidate_count += 1
                vector_bytes += int(row["vector_bytes"])
                dimensions.add(int(row["dimension"]))
                if vector_bytes > max_vector_bytes:
                    break

        if len(dimensions) > 1:
            raise ValueError("Stored embeddings have inconsistent dimensions")

        return SemanticCandidateSummary(
            count=candidate_count,
            vector_bytes=vector_bytes,
            dimension=next(iter(dimensions), None),
        )

    def iter_semantic_candidate_rows(
        self,
        repository_id: uuid.UUID,
        path_filter: str | None = None,
    ) -> Iterator[StoredSemanticCandidateRow]:
        """Stream semantic metadata and vectors without retaining SQL rows."""

        parameters: list[str] = [str(repository_id)]
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "chunks.relative_path",
                path_filter,
            )
            parameters.extend(path_parameters)

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    chunks.id,
                    chunks.relative_path,
                    chunks.start_line,
                    chunks.end_line,
                    embeddings.vector AS embedding_vector,
                    substr(symbols.qualified_name, 1, 4096)
                        AS qualified_symbol_name
                FROM chunks
                INNER JOIN embeddings
                    ON embeddings.chunk_id = chunks.id
                LEFT JOIN symbols
                    ON symbols.id = chunks.symbol_id
                WHERE chunks.repository_id = ?
                    {path_clause}
                ORDER BY
                    chunks.relative_path,
                    chunks.start_line,
                    chunks.end_line,
                    chunks.id
                """,
                parameters,
            )
            for row in rows:
                yield StoredSemanticCandidateRow(
                    candidate=StoredSemanticCandidate(
                        chunk_id=uuid.UUID(row["id"]),
                        relative_path=row["relative_path"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        qualified_symbol_name=row["qualified_symbol_name"],
                    ),
                    vector_blob=row["embedding_vector"],
                )

    def load_chunk_texts(
        self,
        chunk_ids: Iterable[uuid.UUID],
        *,
        max_chars: int,
    ) -> dict[uuid.UUID, str]:
        """Load source text only for chunks selected by semantic ranking."""

        if max_chars < 1:
            raise ValueError("max_chars must be greater than 0")

        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}

        placeholders = ", ".join("?" for _ in unique_ids)
        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id, substr(raw_text, 1, ?) AS raw_text
                FROM chunks
                WHERE id IN ({placeholders})
                """,
                [max_chars + 1, *(str(chunk_id) for chunk_id in unique_ids)],
            ).fetchall()

        return {uuid.UUID(row["id"]): row["raw_text"] for row in rows}

    def _load_symbols_by_column(
        self,
        repository_id: uuid.UUID,
        column: str,
        value: str,
        path_filter: str | None = None,
        limit: int = 10,
        max_snippet_chars: int = 4_000,
    ) -> list[Symbol]:
        """Load symbols where an allowed text column exactly matches a value."""

        allowed_columns = {"name", "qualified_name"}
        if column not in allowed_columns:
            raise ValueError(f"Unsupported symbol lookup column: {column}")

        if limit < 1:
            raise ValueError("limit must be greater than 0")
        if max_snippet_chars < 1:
            raise ValueError("max_snippet_chars must be greater than 0")

        parameters: list[str | int] = [
            max_snippet_chars + 1,
            str(repository_id),
            value,
        ]
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "relative_path",
                path_filter,
            )
            parameters.extend(path_parameters)
        parameters.append(limit)

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    repository_id,
                    name,
                    qualified_name,
                    kind,
                    relative_path,
                    start_line,
                    end_line,
                    substr(source_snippet, 1, ?) AS source_snippet
                FROM symbols
                WHERE repository_id = ?
                    AND {column} = ?
                    {path_clause}
                ORDER BY relative_path, qualified_name, start_line
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [_symbol_from_row(row) for row in rows]

    def load_embeddings_by_content_hashes(
        self,
        repository_id: uuid.UUID,
        model: str,
        dimension: int,
        content_hashes: Iterable[str],
    ) -> dict[str, list[float]]:
        """Load reusable embeddings only for a bounded set of chunk hashes."""

        unique_hashes = list(dict.fromkeys(content_hashes))
        if not unique_hashes:
            return {}

        reusable: dict[str, list[float]] = {}
        with self.read_connection() as connection:
            # Stay below SQLite's platform-dependent bound-variable limit.
            for offset in range(0, len(unique_hashes), 500):
                hash_batch = unique_hashes[offset : offset + 500]
                placeholders = ", ".join("?" for _ in hash_batch)
                rows = connection.execute(
                    f"""
                    SELECT chunks.content_hash, embeddings.vector
                    FROM embeddings
                    INNER JOIN chunks ON chunks.id = embeddings.chunk_id
                    WHERE embeddings.repository_id = ?
                        AND embeddings.model = ?
                        AND embeddings.dimension = ?
                        AND chunks.content_hash IN ({placeholders})
                    """,
                    [str(repository_id), model, dimension, *hash_batch],
                )
                for row in rows:
                    reusable.setdefault(
                        row["content_hash"],
                        unpack_vector(row["vector"]),
                    )

        return reusable

    def load_embeddings(
        self,
        repository_id: uuid.UUID,
    ) -> list[tuple[uuid.UUID, list[float]]]:
        """Load all stored embeddings for a repository."""

        with self.read_connection() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, vector
                FROM embeddings
                WHERE repository_id = ?
                ORDER BY chunk_id
                """,
                (str(repository_id),),
            ).fetchall()

        return [
            (uuid.UUID(row["chunk_id"]), unpack_vector(row["vector"])) for row in rows
        ]

    @staticmethod
    def _upsert_repository(
        connection: sqlite3.Connection,
        repository: Repository,
    ) -> None:
        connection.execute(
            """
            INSERT INTO repositories (
                id,
                absolute_path,
                index_format_version,
                timestamp_of_index,
                embedding_provider,
                embedding_model,
                embedding_dim
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                absolute_path = excluded.absolute_path,
                index_format_version = excluded.index_format_version,
                timestamp_of_index = excluded.timestamp_of_index,
                embedding_provider = excluded.embedding_provider,
                embedding_model = excluded.embedding_model,
                embedding_dim = excluded.embedding_dim
            """,
            (
                str(repository.id),
                repository.absolute_path,
                repository.index_format_version,
                repository.timestamp_of_index,
                repository.embedding_provider,
                repository.embedding_model,
                repository.embedding_dim,
            ),
        )

    @staticmethod
    def _delete_file_records(
        connection: sqlite3.Connection,
        repository_id: str,
        relative_path: str,
    ) -> None:
        connection.execute(
            """
            DELETE FROM embeddings
            WHERE chunk_id IN (
                SELECT id FROM chunks
                WHERE repository_id = ? AND relative_path = ?
            )
            """,
            (repository_id, relative_path),
        )
        connection.execute(
            "DELETE FROM chunks WHERE repository_id = ? AND relative_path = ?",
            (repository_id, relative_path),
        )
        connection.execute(
            "DELETE FROM symbols WHERE repository_id = ? AND relative_path = ?",
            (repository_id, relative_path),
        )
        connection.execute(
            "DELETE FROM files WHERE repository_id = ? AND relative_path = ?",
            (repository_id, relative_path),
        )

    @staticmethod
    def _insert_files(
        connection: sqlite3.Connection,
        files: list[IndexedFile],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO files (
                repository_id,
                relative_path,
                modified_time_ns,
                size_bytes,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(file.repository_id),
                    file.relative_path,
                    file.modified_time_ns,
                    file.size_bytes,
                    file.content_hash,
                )
                for file in files
            ],
        )

    @staticmethod
    def _insert_symbols(
        connection: sqlite3.Connection,
        symbols: list[Symbol],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO symbols (
                id,
                repository_id,
                name,
                qualified_name,
                kind,
                relative_path,
                start_line,
                end_line,
                source_snippet
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(symbol.id),
                    str(symbol.repository_id),
                    symbol.name,
                    symbol.qualified_name,
                    symbol.kind,
                    symbol.relative_path,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.source_snippet,
                )
                for symbol in symbols
            ],
        )

    @staticmethod
    def _insert_chunks(
        connection: sqlite3.Connection,
        chunks: list[Chunk],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO chunks (
                id,
                repository_id,
                relative_path,
                start_line,
                end_line,
                symbol_id,
                raw_text,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(chunk.id),
                    str(chunk.repository_id),
                    chunk.relative_path,
                    chunk.start_line,
                    chunk.end_line,
                    str(chunk.symbol_id) if chunk.symbol_id else None,
                    chunk.raw_text,
                    chunk.content_hash,
                )
                for chunk in chunks
            ],
        )

    @staticmethod
    def _insert_embeddings(
        connection: sqlite3.Connection,
        repository: Repository,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO embeddings (
                chunk_id,
                repository_id,
                model,
                dimension,
                vector
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(chunk.id),
                    str(repository.id),
                    repository.embedding_model,
                    repository.embedding_dim,
                    pack_vector(vector),
                )
                for chunk, vector in zip(chunks, embeddings, strict=True)
            ],
        )


def _repository_from_row(row: sqlite3.Row) -> Repository:
    """Build a Repository model from a SQLite row."""

    return Repository(
        id=uuid.UUID(row["id"]),
        absolute_path=row["absolute_path"],
        index_format_version=row["index_format_version"],
        timestamp_of_index=row["timestamp_of_index"],
        embedding_provider=row["embedding_provider"],
        embedding_model=row["embedding_model"],
        embedding_dim=row["embedding_dim"],
    )


def _embedding_provider_expression(
    connection: sqlite3.Connection,
    *,
    select: bool = False,
) -> str:
    """Select a stored provider or a legacy-compatible unknown value."""

    repository_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(repositories)")
    }
    if "embedding_provider" in repository_columns:
        return "embedding_provider"
    if select:
        return "'unknown' AS embedding_provider"
    return "'unknown'"


def _path_filter_clause(column: str, path_filter: str) -> tuple[str, list[str]]:
    """Return a safe SQL clause for one file or directory-prefix filter."""

    allowed_columns = {"relative_path", "chunks.relative_path"}
    if column not in allowed_columns:
        raise ValueError(f"Unsupported path filter column: {column}")

    escaped_prefix = (
        path_filter.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    clause = f"AND ({column} = ? OR {column} LIKE ? ESCAPE '\\')"
    return clause, [path_filter, f"{escaped_prefix}/%"]


def _symbol_from_row(row: sqlite3.Row) -> Symbol:
    """Build a Symbol model from a SQLite row."""

    return Symbol(
        id=uuid.UUID(row["id"]),
        repository_id=uuid.UUID(row["repository_id"]),
        name=row["name"],
        qualified_name=row["qualified_name"],
        kind=row["kind"],
        relative_path=row["relative_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        source_snippet=row["source_snippet"],
    )

"""SQLite persistence for FireLens repository indexes.

The storage layer owns SQL and vector serialization. Indexing and search code
should interact with this module through small methods instead of embedding SQL
queries directly.
"""

import hashlib
import math
import re
import sqlite3
import uuid
from array import array
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from app.core.models import Chunk, GraphEdge, GraphFact, GraphNode, Repository, Symbol


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
    language: str = "python"


@dataclass(frozen=True)
class LexicalDocument:
    """A denormalized retrievable record owned by the storage layer."""

    document_id: str
    repository_id: uuid.UUID
    record_id: uuid.UUID
    result_type: str
    semantic_unit_kind: str | None
    language: str
    relative_path: str
    name: str | None
    qualified_name: str | None
    identifier_terms: str
    content: str
    start_line: int
    end_line: int
    snippet: str


@dataclass(frozen=True)
class IndexedFileRecords:
    """All index records generated from one source file."""

    file: IndexedFile
    symbols: list[Symbol]
    chunks: list[Chunk]
    embeddings: list[list[float]]
    lexical_documents: list[LexicalDocument] = field(default_factory=list)
    graph_nodes: list[GraphNode] = field(default_factory=list)
    graph_facts: list[GraphFact] = field(default_factory=list)


@dataclass(frozen=True)
class StoredGraphNeighbor:
    """One bounded adjacency row oriented from the traversal node."""

    edge_id: uuid.UUID
    current_node_id: uuid.UUID
    neighbor_node_id: uuid.UUID
    kind: str
    direction: str
    confidence: float


@dataclass(frozen=True)
class StoredGraphResult:
    """A graph node mapped to one retrievable symbol or chunk record."""

    node_id: uuid.UUID
    record_id: uuid.UUID
    result_type: str
    semantic_unit_kind: str | None
    language: str
    relative_path: str
    name: str | None
    qualified_name: str | None
    start_line: int
    end_line: int
    snippet: str
    symbol_id: uuid.UUID | None


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
    symbol_id: uuid.UUID | None = None
    qualified_symbol_name: str | None = None
    language: str = "python"
    semantic_unit_kind: str = "symbol"


@dataclass(frozen=True)
class StoredLexicalCandidate:
    """A lexical result reconstructed entirely by the storage layer."""

    record_id: uuid.UUID
    result_type: str
    semantic_unit_kind: str | None
    language: str
    relative_path: str
    name: str | None
    qualified_name: str | None
    start_line: int
    end_line: int
    snippet: str
    raw_bm25_rank: float | None = None


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
                    language TEXT NOT NULL DEFAULT 'python',
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
                    language TEXT NOT NULL DEFAULT 'python',
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
                    language TEXT NOT NULL DEFAULT 'python',
                    semantic_unit_kind TEXT NOT NULL DEFAULT 'symbol',
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

                CREATE TABLE IF NOT EXISTS lexical_documents (
                    document_id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    semantic_unit_kind TEXT,
                    language TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    name TEXT,
                    qualified_name TEXT,
                    identifier_terms TEXT NOT NULL,
                    content TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    snippet TEXT NOT NULL,
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    relative_path TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    symbol_id TEXT,
                    language TEXT,
                    UNIQUE(repository_id, kind, qualified_name, relative_path),
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(symbol_id)
                        REFERENCES symbols(id)
                        ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS graph_facts (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source_reference TEXT NOT NULL,
                    target_reference TEXT NOT NULL,
                    source_scope TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    extraction_adapter TEXT NOT NULL,
                    adapter_version TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_qualified_hint TEXT,
                    hint_resolution_method TEXT,
                    evidence_text TEXT,
                    resolution_status TEXT NOT NULL DEFAULT 'unresolved',
                    resolution_method TEXT,
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(source_node_id)
                        REFERENCES graph_nodes(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    extraction_adapter TEXT NOT NULL,
                    adapter_version TEXT NOT NULL,
                    resolution_method TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_text TEXT,
                    FOREIGN KEY(repository_id)
                        REFERENCES repositories(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(source_node_id)
                        REFERENCES graph_nodes(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(target_node_id)
                        REFERENCES graph_nodes(id)
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
                CREATE INDEX IF NOT EXISTS idx_lexical_documents_repository
                    ON lexical_documents(repository_id);
                CREATE INDEX IF NOT EXISTS idx_lexical_documents_names
                    ON lexical_documents(repository_id, qualified_name, name);
                CREATE INDEX IF NOT EXISTS idx_lexical_documents_path
                    ON lexical_documents(repository_id, relative_path);
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_repository_kind
                    ON graph_nodes(repository_id, kind);
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_path
                    ON graph_nodes(repository_id, relative_path);
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_qualified_name
                    ON graph_nodes(repository_id, qualified_name);
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_symbol_id
                    ON graph_nodes(repository_id, symbol_id);
                CREATE INDEX IF NOT EXISTS idx_graph_facts_repository_file
                    ON graph_facts(repository_id, source_file);
                CREATE INDEX IF NOT EXISTS idx_graph_facts_status
                    ON graph_facts(repository_id, resolution_status);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_source
                    ON graph_edges(repository_id, source_node_id, kind);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_target
                    ON graph_edges(repository_id, target_node_id, kind);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_file
                    ON graph_edges(repository_id, source_file);
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
            self._ensure_column(
                connection,
                "files",
                "language",
                "TEXT NOT NULL DEFAULT 'python'",
            )
            self._ensure_column(
                connection,
                "symbols",
                "language",
                "TEXT NOT NULL DEFAULT 'python'",
            )
            self._ensure_column(
                connection,
                "chunks",
                "language",
                "TEXT NOT NULL DEFAULT 'python'",
            )
            self._ensure_column(
                connection,
                "chunks",
                "semantic_unit_kind",
                "TEXT NOT NULL DEFAULT 'symbol'",
            )
            try:
                connection.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS lexical_documents_fts
                    USING fts5(
                        name,
                        qualified_name,
                        identifier_terms,
                        relative_path,
                        content,
                        tokenize = 'unicode61'
                    );

                    CREATE TRIGGER IF NOT EXISTS lexical_documents_after_insert
                    AFTER INSERT ON lexical_documents BEGIN
                        INSERT INTO lexical_documents_fts (
                            rowid,
                            name,
                            qualified_name,
                            identifier_terms,
                            relative_path,
                            content
                        ) VALUES (
                            new.rowid,
                            new.name,
                            new.qualified_name,
                            new.identifier_terms,
                            new.relative_path,
                            new.content
                        );
                    END;

                    CREATE TRIGGER IF NOT EXISTS lexical_documents_after_delete
                    AFTER DELETE ON lexical_documents BEGIN
                        DELETE FROM lexical_documents_fts WHERE rowid = old.rowid;
                    END;

                    CREATE TRIGGER IF NOT EXISTS lexical_documents_after_update
                    AFTER UPDATE ON lexical_documents BEGIN
                        DELETE FROM lexical_documents_fts WHERE rowid = old.rowid;
                        INSERT INTO lexical_documents_fts (
                            rowid,
                            name,
                            qualified_name,
                            identifier_terms,
                            relative_path,
                            content
                        ) VALUES (
                            new.rowid,
                            new.name,
                            new.qualified_name,
                            new.identifier_terms,
                            new.relative_path,
                            new.content
                        );
                    END;
                    """
                )
                connection.execute(
                    "SELECT rowid FROM lexical_documents_fts LIMIT 1"
                ).fetchone()
            except sqlite3.OperationalError as error:
                raise RuntimeError(
                    "FireLens requires a Python SQLite build with FTS5 support"
                ) from error

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
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
        lexical_documents: list[LexicalDocument] | None = None,
        graph_nodes: list[GraphNode] | None = None,
        graph_facts: list[GraphFact] | None = None,
        graph_edges: list[GraphEdge] | None = None,
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
                "DELETE FROM graph_edges WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "DELETE FROM graph_facts WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "DELETE FROM graph_nodes WHERE repository_id = ?",
                (repository_id,),
            )
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
            connection.execute(
                "DELETE FROM lexical_documents WHERE repository_id = ?",
                (repository_id,),
            )
            self._insert_files(connection, files)
            self._insert_symbols(connection, symbols)
            self._insert_chunks(connection, chunks)
            self._insert_embeddings(connection, repository, chunks, embeddings)
            self._insert_lexical_documents(connection, lexical_documents or [])
            self._insert_graph_nodes(connection, graph_nodes or [])
            self._insert_graph_facts(connection, graph_facts or [])
            self._insert_graph_edges(connection, graph_edges or [])

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
                self._insert_lexical_documents(
                    connection,
                    file_records.lexical_documents,
                )
                self._insert_graph_nodes(connection, file_records.graph_nodes)
                self._insert_graph_facts(connection, file_records.graph_facts)

    def count_rows(self, table: str, repository_id: uuid.UUID) -> int:
        """Return a repository-scoped row count for tests and diagnostics."""

        allowed_tables = {
            "files",
            "symbols",
            "chunks",
            "embeddings",
            "lexical_documents",
            "graph_nodes",
            "graph_facts",
            "graph_edges",
        }
        if table not in allowed_tables:
            raise ValueError(f"Unsupported table: {table}")

        with self.read_connection() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE repository_id = ?",
                (str(repository_id),),
            ).fetchone()

        return int(row["count"])

    def count_graph_facts_by_status(
        self,
        repository_id: uuid.UUID,
        status: str,
    ) -> int:
        """Return a bounded graph-resolution diagnostic count."""

        if status not in {"resolved", "unresolved", "ambiguous"}:
            raise ValueError(f"Unsupported graph fact status: {status}")
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM graph_facts "
                "WHERE repository_id = ? AND resolution_status = ?",
                (str(repository_id), status),
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
                    content_hash,
                    language
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
                language=row["language"],
            )
            for row in rows
        }

    def load_graph_nodes(self, repository_id: uuid.UUID) -> list[GraphNode]:
        """Load all graph nodes for deterministic repository-wide resolution."""

        with self.read_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    repository_id,
                    kind,
                    qualified_name,
                    name,
                    relative_path,
                    start_line,
                    end_line,
                    symbol_id,
                    language
                FROM graph_nodes
                WHERE repository_id = ?
                ORDER BY relative_path, qualified_name, kind, id
                """,
                (str(repository_id),),
            ).fetchall()
        return [_graph_node_from_row(row) for row in rows]

    def load_graph_facts(self, repository_id: uuid.UUID) -> list[GraphFact]:
        """Load adapter facts without exposing graph SQL to the resolver."""

        with self.read_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    repository_id,
                    source_node_id,
                    kind,
                    source_reference,
                    target_reference,
                    source_scope,
                    source_file,
                    start_line,
                    end_line,
                    extraction_adapter,
                    adapter_version,
                    confidence,
                    target_kind,
                    target_qualified_hint,
                    hint_resolution_method,
                    evidence_text
                FROM graph_facts
                WHERE repository_id = ?
                ORDER BY source_file, start_line, end_line, kind, id
                """,
                (str(repository_id),),
            ).fetchall()
        return [_graph_fact_from_row(row) for row in rows]

    def load_graph_edges(
        self,
        repository_id: uuid.UUID,
        edge_kind: str | None = None,
    ) -> list[GraphEdge]:
        """Load resolved edges for diagnostics and focused graph tests."""

        kind_clause = ""
        parameters = [str(repository_id)]
        if edge_kind is not None:
            kind_clause = "AND kind = ?"
            parameters.append(edge_kind)
        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    repository_id,
                    source_node_id,
                    target_node_id,
                    kind,
                    source_file,
                    start_line,
                    end_line,
                    extraction_adapter,
                    adapter_version,
                    resolution_method,
                    confidence,
                    evidence_text
                FROM graph_edges
                WHERE repository_id = ? {kind_clause}
                ORDER BY source_node_id, kind, target_node_id, source_file, start_line, id
                """,
                parameters,
            ).fetchall()
        return [_graph_edge_from_row(row) for row in rows]

    def replace_graph_resolution(
        self,
        repository_node: GraphNode,
        edges: list[GraphEdge],
        fact_resolutions: dict[uuid.UUID, tuple[str, str | None]],
    ) -> None:
        """Atomically replace resolved edges and fact diagnostics."""

        repository_id = str(repository_node.repository_id)
        with self.connect() as connection:
            self._insert_graph_nodes(connection, [repository_node])
            connection.execute(
                "DELETE FROM graph_edges WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "UPDATE graph_facts SET resolution_status = 'unresolved', "
                "resolution_method = NULL WHERE repository_id = ?",
                (repository_id,),
            )
            for fact_id, (status, method) in fact_resolutions.items():
                connection.execute(
                    "UPDATE graph_facts SET resolution_status = ?, "
                    "resolution_method = ? WHERE repository_id = ? AND id = ?",
                    (status, method, repository_id, str(fact_id)),
                )
            self._insert_graph_edges(connection, edges)

    def load_graph_nodes_for_results(
        self,
        repository_id: uuid.UUID,
        symbol_ids: Iterable[uuid.UUID],
        relative_paths: Iterable[str],
    ) -> list[GraphNode]:
        """Map a bounded result set to symbol, module, and file graph nodes."""

        unique_symbol_ids = list(dict.fromkeys(symbol_ids))
        unique_paths = list(dict.fromkeys(relative_paths))
        clauses: list[str] = []
        parameters: list[str] = [str(repository_id)]
        if unique_symbol_ids:
            placeholders = ", ".join("?" for _ in unique_symbol_ids)
            clauses.append(f"symbol_id IN ({placeholders})")
            parameters.extend(str(symbol_id) for symbol_id in unique_symbol_ids)
        if unique_paths:
            placeholders = ", ".join("?" for _ in unique_paths)
            clauses.append(
                f"(relative_path IN ({placeholders}) "
                "AND kind IN ('file', 'test_file', 'module'))"
            )
            parameters.extend(unique_paths)
        if not clauses:
            return []

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    repository_id,
                    kind,
                    qualified_name,
                    name,
                    relative_path,
                    start_line,
                    end_line,
                    symbol_id,
                    language
                FROM graph_nodes
                WHERE repository_id = ? AND ({' OR '.join(clauses)})
                ORDER BY relative_path, kind, qualified_name, id
                """,
                parameters,
            ).fetchall()
        return [_graph_node_from_row(row) for row in rows]

    def load_graph_adjacency(
        self,
        repository_id: uuid.UUID,
        node_ids: Iterable[uuid.UUID],
        *,
        edge_kinds: Iterable[str],
        directions: Iterable[str],
        minimum_confidence: float,
        maximum_neighbors_per_node: int,
    ) -> list[StoredGraphNeighbor]:
        """Load bounded incoming and outgoing adjacency in stable order."""

        unique_node_ids = list(dict.fromkeys(node_ids))
        unique_edge_kinds = list(dict.fromkeys(edge_kinds))
        unique_directions = list(dict.fromkeys(directions))
        if not unique_node_ids or not unique_edge_kinds:
            return []
        if any(direction not in {"incoming", "outgoing"} for direction in unique_directions):
            raise ValueError("Graph direction must be incoming or outgoing")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if maximum_neighbors_per_node < 1:
            raise ValueError("maximum_neighbors_per_node must be greater than 0")

        node_placeholders = ", ".join("?" for _ in unique_node_ids)
        kind_placeholders = ", ".join("?" for _ in unique_edge_kinds)
        neighbors: list[StoredGraphNeighbor] = []
        counts: dict[str, int] = {}
        with self.read_connection() as connection:
            for direction in unique_directions:
                current_column = (
                    "source_node_id" if direction == "outgoing" else "target_node_id"
                )
                neighbor_column = (
                    "target_node_id" if direction == "outgoing" else "source_node_id"
                )
                rows = connection.execute(
                    f"""
                    WITH ranked_neighbors AS (
                        SELECT
                            id,
                            {current_column} AS current_node_id,
                            {neighbor_column} AS neighbor_node_id,
                            kind,
                            confidence,
                            ROW_NUMBER() OVER (
                                PARTITION BY {current_column}
                                ORDER BY
                                    confidence DESC,
                                    kind,
                                    {neighbor_column},
                                    id
                            ) AS neighbor_rank
                        FROM graph_edges
                        WHERE repository_id = ?
                            AND {current_column} IN ({node_placeholders})
                            AND kind IN ({kind_placeholders})
                            AND confidence >= ?
                    )
                    SELECT
                        id,
                        current_node_id,
                        neighbor_node_id,
                        kind,
                        confidence
                    FROM ranked_neighbors
                    WHERE neighbor_rank <= ?
                    ORDER BY
                        current_node_id,
                        confidence DESC,
                        kind,
                        neighbor_node_id,
                        id
                    """,
                    [
                        str(repository_id),
                        *(str(node_id) for node_id in unique_node_ids),
                        *unique_edge_kinds,
                        minimum_confidence,
                        maximum_neighbors_per_node,
                    ],
                ).fetchall()
                for row in rows:
                    current_id = row["current_node_id"]
                    count = counts.get(current_id, 0)
                    if count >= maximum_neighbors_per_node:
                        continue
                    counts[current_id] = count + 1
                    neighbors.append(
                        StoredGraphNeighbor(
                            edge_id=uuid.UUID(row["id"]),
                            current_node_id=uuid.UUID(current_id),
                            neighbor_node_id=uuid.UUID(row["neighbor_node_id"]),
                            kind=row["kind"],
                            direction=direction,
                            confidence=float(row["confidence"]),
                        )
                    )
        return neighbors

    def load_graph_results(
        self,
        repository_id: uuid.UUID,
        node_ids: Iterable[uuid.UUID],
        *,
        max_snippet_chars: int,
        path_filter: str | None = None,
    ) -> dict[uuid.UUID, StoredGraphResult]:
        """Map expanded graph nodes to bounded retrievable records."""

        unique_node_ids = list(dict.fromkeys(node_ids))
        if not unique_node_ids:
            return {}
        if max_snippet_chars < 1:
            raise ValueError("max_snippet_chars must be greater than 0")
        placeholders = ", ".join("?" for _ in unique_node_ids)
        path_clause = ""
        path_parameters: list[str] = []
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "n.relative_path",
                path_filter,
            )
        common_parameters: list[str | int] = [
            max_snippet_chars + 1,
            str(repository_id),
            *(str(node_id) for node_id in unique_node_ids),
            *path_parameters,
        ]

        results: dict[uuid.UUID, StoredGraphResult] = {}
        with self.read_connection() as connection:
            symbol_rows = connection.execute(
                f"""
                SELECT
                    n.id AS node_id,
                    s.id AS record_id,
                    'symbol' AS result_type,
                    NULL AS semantic_unit_kind,
                    s.language,
                    s.relative_path,
                    s.name,
                    s.qualified_name,
                    s.start_line,
                    s.end_line,
                    substr(s.source_snippet, 1, ?) AS snippet,
                    s.id AS symbol_id
                FROM graph_nodes AS n
                JOIN symbols AS s ON s.id = n.symbol_id
                WHERE n.repository_id = ?
                    AND n.id IN ({placeholders})
                    {path_clause}
                """,
                common_parameters,
            ).fetchall()
            for row in symbol_rows:
                result = _graph_result_from_row(row)
                results[result.node_id] = result

            chunk_rows = connection.execute(
                f"""
                SELECT
                    n.id AS node_id,
                    c.id AS record_id,
                    'chunk' AS result_type,
                    c.semantic_unit_kind,
                    c.language,
                    c.relative_path,
                    s.name,
                    s.qualified_name,
                    c.start_line,
                    c.end_line,
                    substr(c.raw_text, 1, ?) AS snippet,
                    c.symbol_id
                FROM graph_nodes AS n
                JOIN chunks AS c ON c.id = (
                    SELECT candidate.id
                    FROM chunks AS candidate
                    WHERE candidate.repository_id = n.repository_id
                        AND candidate.relative_path = n.relative_path
                    ORDER BY
                        CASE candidate.semantic_unit_kind
                            WHEN 'imports' THEN 0
                            WHEN 'module_code' THEN 1
                            ELSE 2
                        END,
                        candidate.start_line,
                        candidate.end_line,
                        candidate.id
                    LIMIT 1
                )
                LEFT JOIN symbols AS s ON s.id = c.symbol_id
                WHERE n.repository_id = ?
                    AND n.id IN ({placeholders})
                    AND n.symbol_id IS NULL
                    AND n.kind IN ('file', 'test_file', 'module')
                    {path_clause}
                """,
                common_parameters,
            ).fetchall()
            for row in chunk_rows:
                result = _graph_result_from_row(row)
                results[result.node_id] = result
        return results

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

    def exact_lexical_candidates(
        self,
        repository_id: uuid.UUID,
        query: str,
        path_filter: str | None,
        *,
        limit: int,
        max_snippet_chars: int,
    ) -> list[StoredLexicalCandidate]:
        """Load exact qualified and short symbol-name candidates."""

        _validate_lexical_limits(limit, max_snippet_chars)

        path_parameters: list[str] = []
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "lexical_documents.relative_path",
                path_filter,
            )

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    record_id,
                    result_type,
                    semantic_unit_kind,
                    language,
                    relative_path,
                    name,
                    qualified_name,
                    start_line,
                    end_line,
                    substr(snippet, 1, ?) AS snippet
                FROM lexical_documents
                WHERE repository_id = ?
                    AND result_type = 'symbol'
                    AND (qualified_name = ? OR name = ?)
                    {path_clause}
                ORDER BY
                    CASE WHEN qualified_name = ? THEN 0 ELSE 1 END,
                    relative_path,
                    qualified_name,
                    start_line,
                    record_id
                LIMIT ?
                """,
                [
                    max_snippet_chars + 1,
                    str(repository_id),
                    query,
                    query,
                    *path_parameters,
                    query,
                    limit,
                ],
            ).fetchall()
        return [_lexical_candidate_from_row(row) for row in rows]

    def path_lexical_candidates(
        self,
        repository_id: uuid.UUID,
        query_path: str,
        path_filter: str | None,
        *,
        limit: int,
        max_snippet_chars: int,
    ) -> list[StoredLexicalCandidate]:
        """Load exact or prefix path candidates in deterministic order."""

        _validate_lexical_limits(limit, max_snippet_chars)

        escaped_path = _escape_like(query_path)
        parameters: list[str | int] = [
            max_snippet_chars + 1,
            str(repository_id),
            query_path,
            f"{escaped_path}%",
        ]
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "lexical_documents.relative_path",
                path_filter,
            )
            parameters.extend(path_parameters)
        parameters.extend([query_path, limit])

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    record_id,
                    result_type,
                    semantic_unit_kind,
                    language,
                    relative_path,
                    name,
                    qualified_name,
                    start_line,
                    end_line,
                    substr(snippet, 1, ?) AS snippet
                FROM lexical_documents
                WHERE repository_id = ?
                    AND (
                        relative_path = ?
                        OR relative_path LIKE ? ESCAPE '\\'
                    )
                    {path_clause}
                ORDER BY
                    CASE WHEN relative_path = ? THEN 0 ELSE 1 END,
                    length(relative_path),
                    relative_path,
                    start_line,
                    record_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [_lexical_candidate_from_row(row) for row in rows]

    def fts_lexical_candidates(
        self,
        repository_id: uuid.UUID,
        fts_query: str,
        path_filter: str | None,
        *,
        limit: int,
        max_snippet_chars: int,
        field_weights: tuple[float, float, float, float, float],
    ) -> list[StoredLexicalCandidate]:
        """Rank a safely constructed FTS5 expression with BM25."""

        if not fts_query:
            return []
        _validate_lexical_limits(limit, max_snippet_chars)
        if len(field_weights) != 5 or any(
            not math.isfinite(weight) or weight < 0.0 for weight in field_weights
        ):
            raise ValueError(
                "BM25 field weights must be five finite nonnegative values"
            )
        path_parameters: list[str] = []
        path_clause = ""
        if path_filter is not None:
            path_clause, path_parameters = _path_filter_clause(
                "lexical_documents.relative_path",
                path_filter,
            )

        with self.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    lexical_documents.record_id,
                    lexical_documents.result_type,
                    lexical_documents.semantic_unit_kind,
                    lexical_documents.language,
                    lexical_documents.relative_path,
                    lexical_documents.name,
                    lexical_documents.qualified_name,
                    lexical_documents.start_line,
                    lexical_documents.end_line,
                    substr(lexical_documents.snippet, 1, ?) AS snippet,
                    bm25(
                        lexical_documents_fts,
                        ?, ?, ?, ?, ?
                    ) AS raw_bm25_rank
                FROM lexical_documents_fts
                INNER JOIN lexical_documents
                    ON lexical_documents.rowid = lexical_documents_fts.rowid
                WHERE lexical_documents_fts MATCH ?
                    AND lexical_documents.repository_id = ?
                    {path_clause}
                ORDER BY
                    raw_bm25_rank,
                    lexical_documents.relative_path,
                    lexical_documents.start_line,
                    lexical_documents.end_line,
                    lexical_documents.record_id
                LIMIT ?
                """,
                [
                    max_snippet_chars + 1,
                    *field_weights,
                    fts_query,
                    str(repository_id),
                    *path_parameters,
                    limit,
                ],
            ).fetchall()
        return [_lexical_candidate_from_row(row) for row in rows]

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
                    substr(source_snippet, 1, ?) AS source_snippet,
                    language
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
                    source_snippet,
                    language
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
                    chunks.symbol_id,
                    chunks.relative_path,
                    chunks.start_line,
                    chunks.end_line,
                    chunks.language,
                    chunks.semantic_unit_kind,
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
                        symbol_id=(
                            uuid.UUID(row["symbol_id"])
                            if row["symbol_id"] is not None
                            else None
                        ),
                        relative_path=row["relative_path"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        qualified_symbol_name=row["qualified_symbol_name"],
                        language=row["language"],
                        semantic_unit_kind=row["semantic_unit_kind"],
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
                    substr(source_snippet, 1, ?) AS source_snippet,
                    language
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
            "DELETE FROM graph_facts "
            "WHERE repository_id = ? AND source_file = ?",
            (repository_id, relative_path),
        )
        connection.execute(
            "DELETE FROM graph_nodes "
            "WHERE repository_id = ? AND relative_path = ?",
            (repository_id, relative_path),
        )
        connection.execute(
            "DELETE FROM lexical_documents "
            "WHERE repository_id = ? AND relative_path = ?",
            (repository_id, relative_path),
        )
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
                content_hash,
                language
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(file.repository_id),
                    file.relative_path,
                    file.modified_time_ns,
                    file.size_bytes,
                    file.content_hash,
                    file.language,
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
                source_snippet,
                language
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    symbol.language,
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
                content_hash,
                language,
                semantic_unit_kind
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    chunk.language,
                    chunk.semantic_unit_kind,
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

    @staticmethod
    def _insert_lexical_documents(
        connection: sqlite3.Connection,
        documents: list[LexicalDocument],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO lexical_documents (
                document_id,
                repository_id,
                record_id,
                result_type,
                semantic_unit_kind,
                language,
                relative_path,
                name,
                qualified_name,
                identifier_terms,
                content,
                start_line,
                end_line,
                snippet
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    document.document_id,
                    str(document.repository_id),
                    str(document.record_id),
                    document.result_type,
                    document.semantic_unit_kind,
                    document.language,
                    document.relative_path,
                    document.name,
                    document.qualified_name,
                    document.identifier_terms,
                    document.content,
                    document.start_line,
                    document.end_line,
                    document.snippet,
                )
                for document in documents
            ],
        )

    @staticmethod
    def _insert_graph_nodes(
        connection: sqlite3.Connection,
        nodes: list[GraphNode],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO graph_nodes (
                id,
                repository_id,
                kind,
                qualified_name,
                name,
                relative_path,
                start_line,
                end_line,
                symbol_id,
                language
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                qualified_name = excluded.qualified_name,
                name = excluded.name,
                relative_path = excluded.relative_path,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                symbol_id = excluded.symbol_id,
                language = excluded.language
            """,
            [
                (
                    str(node.id),
                    str(node.repository_id),
                    node.kind,
                    node.qualified_name,
                    node.name,
                    node.relative_path,
                    node.start_line,
                    node.end_line,
                    str(node.symbol_id) if node.symbol_id is not None else None,
                    node.language,
                )
                for node in nodes
            ],
        )

    @staticmethod
    def _insert_graph_facts(
        connection: sqlite3.Connection,
        facts: list[GraphFact],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO graph_facts (
                id,
                repository_id,
                source_node_id,
                kind,
                source_reference,
                target_reference,
                source_scope,
                source_file,
                start_line,
                end_line,
                extraction_adapter,
                adapter_version,
                confidence,
                target_kind,
                target_qualified_hint,
                hint_resolution_method,
                evidence_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(fact.id),
                    str(fact.repository_id),
                    str(fact.source_node_id),
                    fact.kind,
                    fact.source_reference,
                    fact.target_reference,
                    fact.source_scope,
                    fact.source_file,
                    fact.start_line,
                    fact.end_line,
                    fact.extraction_adapter,
                    fact.adapter_version,
                    fact.confidence,
                    fact.target_kind,
                    fact.target_qualified_hint,
                    fact.hint_resolution_method,
                    fact.evidence_text,
                )
                for fact in facts
            ],
        )

    @staticmethod
    def _insert_graph_edges(
        connection: sqlite3.Connection,
        edges: list[GraphEdge],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO graph_edges (
                id,
                repository_id,
                source_node_id,
                target_node_id,
                kind,
                source_file,
                start_line,
                end_line,
                extraction_adapter,
                adapter_version,
                resolution_method,
                confidence,
                evidence_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(edge.id),
                    str(edge.repository_id),
                    str(edge.source_node_id),
                    str(edge.target_node_id),
                    edge.kind,
                    edge.source_file,
                    edge.start_line,
                    edge.end_line,
                    edge.extraction_adapter,
                    edge.adapter_version,
                    edge.resolution_method,
                    edge.confidence,
                    edge.evidence_text,
                )
                for edge in edges
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


def _graph_node_from_row(row: sqlite3.Row) -> GraphNode:
    return GraphNode(
        id=uuid.UUID(row["id"]),
        repository_id=uuid.UUID(row["repository_id"]),
        kind=row["kind"],
        qualified_name=row["qualified_name"],
        name=row["name"],
        relative_path=row["relative_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        symbol_id=(uuid.UUID(row["symbol_id"]) if row["symbol_id"] else None),
        language=row["language"],
    )


def _graph_fact_from_row(row: sqlite3.Row) -> GraphFact:
    return GraphFact(
        id=uuid.UUID(row["id"]),
        repository_id=uuid.UUID(row["repository_id"]),
        source_node_id=uuid.UUID(row["source_node_id"]),
        kind=row["kind"],
        source_reference=row["source_reference"],
        target_reference=row["target_reference"],
        source_scope=row["source_scope"],
        source_file=row["source_file"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        extraction_adapter=row["extraction_adapter"],
        adapter_version=row["adapter_version"],
        confidence=float(row["confidence"]),
        target_kind=row["target_kind"],
        target_qualified_hint=row["target_qualified_hint"],
        hint_resolution_method=row["hint_resolution_method"],
        evidence_text=row["evidence_text"],
    )


def _graph_edge_from_row(row: sqlite3.Row) -> GraphEdge:
    return GraphEdge(
        id=uuid.UUID(row["id"]),
        repository_id=uuid.UUID(row["repository_id"]),
        source_node_id=uuid.UUID(row["source_node_id"]),
        target_node_id=uuid.UUID(row["target_node_id"]),
        kind=row["kind"],
        source_file=row["source_file"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        extraction_adapter=row["extraction_adapter"],
        adapter_version=row["adapter_version"],
        resolution_method=row["resolution_method"],
        confidence=float(row["confidence"]),
        evidence_text=row["evidence_text"],
    )


def _graph_result_from_row(row: sqlite3.Row) -> StoredGraphResult:
    return StoredGraphResult(
        node_id=uuid.UUID(row["node_id"]),
        record_id=uuid.UUID(row["record_id"]),
        result_type=row["result_type"],
        semantic_unit_kind=row["semantic_unit_kind"],
        language=row["language"],
        relative_path=row["relative_path"],
        name=row["qualified_name"] or row["name"],
        qualified_name=row["qualified_name"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        snippet=row["snippet"],
        symbol_id=(uuid.UUID(row["symbol_id"]) if row["symbol_id"] else None),
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

    allowed_columns = {
        "relative_path",
        "chunks.relative_path",
        "lexical_documents.relative_path",
        "n.relative_path",
    }
    if column not in allowed_columns:
        raise ValueError(f"Unsupported path filter column: {column}")

    escaped_prefix = _escape_like(path_filter)
    clause = f"AND ({column} = ? OR {column} LIKE ? ESCAPE '\\')"
    return clause, [path_filter, f"{escaped_prefix}/%"]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _validate_lexical_limits(limit: int, max_snippet_chars: int) -> None:
    if limit < 1:
        raise ValueError("limit must be greater than 0")
    if max_snippet_chars < 1:
        raise ValueError("max_snippet_chars must be greater than 0")


def _lexical_candidate_from_row(row: sqlite3.Row) -> StoredLexicalCandidate:
    keys = set(row.keys())
    return StoredLexicalCandidate(
        record_id=uuid.UUID(row["record_id"]),
        result_type=row["result_type"],
        semantic_unit_kind=row["semantic_unit_kind"],
        language=row["language"],
        relative_path=row["relative_path"],
        name=row["name"],
        qualified_name=row["qualified_name"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        snippet=row["snippet"],
        raw_bm25_rank=(
            float(row["raw_bm25_rank"])
            if "raw_bm25_rank" in keys
            else None
        ),
    )


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
        language=row["language"],
    )

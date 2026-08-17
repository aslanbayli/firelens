"""Shared data contracts used by indexing, storage, and search.

These models define the shape of data exchanged between subsystems. Keeping
the contracts in one module prevents the parser, indexer, storage layer, and
search layer from inventing slightly different representations of the same
concept.
"""

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

SymbolKind = Annotated[
    str,
    Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$"),
]
SemanticUnitKind = Annotated[
    str,
    Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$"),
]

# ``RetrievalKind`` is the mode that actually produced a result. ``auto`` is
# only a caller preference, so it belongs to ``RetrievalMode`` instead.
RetrievalKind = Literal[
    "exact",
    "fuzzy",
    "lexical",
    "semantic",
    "hybrid_rrf",
    "hybrid_weighted",
]
RetrievalMode = Literal[
    "exact",
    "fuzzy",
    "lexical",
    "semantic",
    "hybrid_rrf",
    "hybrid_weighted",
    "auto",
]
RETRIEVAL_MODE_OPTIONS = (
    "auto",
    "exact",
    "fuzzy",
    "lexical",
    "semantic",
    "hybrid_rrf",
    "hybrid_weighted",
)
FusionMethod = Literal["rrf", "normalized_weighted"]
BackendKind = Literal["python", "mojo"]
BackendPreference = Literal["auto", "python", "mojo"]
IndexStatus = Literal["missing", "ready", "stale", "indexing"]

TopK = Annotated[int, Field(ge=1, le=20)]
SnippetCharacterLimit = Annotated[int, Field(ge=1, le=4_000)]
QueryText = Annotated[str, Field(min_length=1, max_length=2_000)]
BoundedPathSamples = Annotated[list[str], Field(max_length=20)]
BoundedIndexingErrors = Annotated[
    list["IndexingErrorResponse"],
    Field(max_length=20),
]


class Repository(BaseModel):
    """Metadata describing one indexed repository."""

    # Unique identity used to connect symbols and chunks to this repository.
    id: uuid.UUID
    # Canonical absolute path used when opening source files on this machine.
    absolute_path: str
    # Version of FireLens's persisted index format, used for migrations later.
    index_format_version: str
    # UTC Unix timestamp representing when this index was created or refreshed.
    timestamp_of_index: int
    # Stable implementation name used to produce embedding vectors.
    embedding_provider: str
    # Name of the embedding model used to create vectors for this index.
    embedding_model: str
    # Number of floating-point values in every vector from the embedding model.
    embedding_dim: int


class Symbol(BaseModel):
    """A searchable function, class, or method extracted from source code."""

    # Unique symbol identity; chunks can refer back to this value.
    id: uuid.UUID
    # Foreign-key-style reference to the repository containing this symbol.
    repository_id: uuid.UUID
    # Short source name, for example "authenticate".
    name: str
    # Scope-aware name, for example "UserService.authenticate".
    qualified_name: str
    # Validated category describing what kind of Python declaration this is.
    kind: SymbolKind
    # Portable path relative to the indexed repository root.
    relative_path: str
    # One-based line where the declaration begins, including decorators.
    start_line: int
    # One-based line where the complete declaration ends.
    end_line: int
    # Exact source slice covering the complete declaration.
    source_snippet: str
    # String-valued adapter language ID, such as ``python``.
    language: str = "python"


class Chunk(BaseModel):
    """A bounded unit of source text used for semantic retrieval."""

    # Unique identity used to associate this chunk with an embedding vector.
    id: uuid.UUID
    # Repository that owns the file from which this chunk was created.
    repository_id: uuid.UUID
    # Portable source-file location relative to the repository root.
    relative_path: str
    # One-based first source line included in this chunk.
    start_line: int
    # One-based final source line included in this chunk.
    end_line: int
    # Symbol that owns this text. None is allowed because future module-level
    # chunks for imports or constants will not belong to a function or class.
    symbol_id: uuid.UUID | None = None
    # Original source code displayed to users in semantic-search results.
    raw_text: str
    # SHA-256 of the final text sent to the embedder. Matching hashes allow a
    # future incremental indexer to reuse an existing embedding.
    content_hash: str
    # Language adapter that produced this semantic unit.
    language: str = "python"
    # Extensible semantic role such as symbol, imports, or documentation.
    semantic_unit_kind: SemanticUnitKind = "symbol"


class RetrievalEvidence(BaseModel):
    """Public, normalized evidence contributed by one retrieval channel."""

    channel: Annotated[str, Field(min_length=1, max_length=64)]
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    raw_score: float | None = Field(default=None, allow_inf_nan=False)
    backend: BackendKind | None = None


class RetrievalTiming(BaseModel):
    """Bounded response-level timing for one retrieval component."""

    component: Literal["lexical", "semantic", "fusion"]
    elapsed_time: float = Field(ge=0.0, allow_inf_nan=False)
    backend: BackendKind


class SearchRequest(BaseModel):
    """Validated input supplied to the future unified search service."""

    # User text, symbol name, typo, or natural-language description to retrieve.
    query: QueryText
    # Explicit retrieval strategy selected by the caller.
    request_mode: RetrievalMode = "auto"
    # Maximum number of ranked results the caller wants returned.
    top_k: TopK = 5
    # Optional repository-relative path used to narrow the search space.
    path: Annotated[str, Field(max_length=4_096)] | None = None
    # Compute implementation requested for supported hot loops.
    backend: BackendPreference = "auto"
    # Maximum source characters included in each returned result.
    max_snippet_chars: SnippetCharacterLimit = 2_000

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        """Reject queries that contain no non-whitespace characters."""

        if query.strip() == "":
            raise ValueError("query must not be empty")
        return query


class SearchResult(BaseModel):
    """One ranked symbol or chunk returned by retrieval."""

    # Identity of the matched Symbol or Chunk record.
    id: uuid.UUID
    # Indicates which record type should be loaded or interpreted.
    result_type: Literal["symbol", "chunk"]
    # Repository-relative source file containing the match.
    file_path: Annotated[str, Field(max_length=4_096)]
    # Inclusive one-based source range displayed to the caller.
    start_line: int
    end_line: int
    # Available for symbol-owned results; absent for module-level chunks.
    symbol_name: Annotated[str, Field(max_length=4_096)] | None = None
    # Stable owning symbol identity used to collapse symbol/chunk overlap.
    symbol_id: uuid.UUID | None = None
    # Source language used by renderers and coding agents.
    language: str = "python"
    # Present for semantic-unit results and absent for standalone symbols.
    semantic_unit_kind: SemanticUnitKind | None = None
    # Bounded source text included with the result.
    snippet: Annotated[str, Field(max_length=4_000)]
    # True when the source text was shortened to satisfy an output limit.
    snippet_truncated: bool = False
    # Mode-specific relevance normalized into a comparable output range.
    score: float = Field(ge=0.0, le=1.0)
    # Retrieval strategy that produced this result.
    mode: RetrievalKind
    # Present only when independent retrieval channels were fused.
    fusion_method: FusionMethod | None = None
    # Compute implementation that actually performed the relevant operation.
    backend: BackendKind
    # Ordered names of all channels that contributed to the public score.
    retrieval_channels: list[str] = Field(default_factory=list, max_length=16)
    # Per-channel rank and normalized score evidence.
    retrieval_evidence: list[RetrievalEvidence] = Field(
        default_factory=list,
        max_length=16,
    )


class SearchResponse(BaseModel):
    """Complete structured response returned by CLI, Streamlit, or MCP."""

    # Preserve caller input for logging and structured tool responses.
    original_query: QueryText
    # Caller preference and the mode selected by the router.
    requested_mode: RetrievalMode = "auto"
    mode: RetrievalKind
    # Caller preference and backend actually used after availability checks.
    requested_backend: BackendPreference = "auto"
    backend: BackendKind
    # End-to-end retrieval duration measured in seconds.
    elapsed_time: float
    # Ordered best-first results.
    ranked_results: list[SearchResult] = Field(max_length=20)
    # Non-fatal details such as falling back from Mojo to Python.
    warnings: list[str] = Field(default_factory=list)
    # Candidate-generation and fusion timings, omitted for single-source modes.
    retrieval_timings: list[RetrievalTiming] = Field(
        default_factory=list,
        max_length=3,
    )
    # Stable name and hash for the effective retrieval calibration settings.
    retrieval_config: Annotated[str, Field(min_length=1, max_length=128)] = "default"


class IndexingErrorResponse(BaseModel):
    """One bounded, structured error reported by an indexing operation."""

    relative_path: str
    stage: str
    message: str


class IndexRepositoryResponse(BaseModel):
    """Structured result returned after an explicit repository index run."""

    repository_path: str
    database_path: str
    status: Literal["ready", "stale"]
    index_format_version: str
    timestamp_of_index: int
    embedding_provider: str
    embedding_model: str
    embedding_dim: int = Field(ge=1)
    file_count: int = Field(default=0, ge=0)
    symbol_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    embedding_count: int = Field(default=0, ge=0)
    lexical_document_count: int = Field(default=0, ge=0)
    added_file_count: int = Field(default=0, ge=0)
    changed_file_count: int = Field(default=0, ge=0)
    deleted_file_count: int = Field(default=0, ge=0)
    embedded_chunk_count: int = Field(default=0, ge=0)
    reused_embedding_count: int = Field(default=0, ge=0)
    elapsed_time: float = Field(default=0.0, ge=0.0)
    changed_paths: BoundedPathSamples = Field(default_factory=list)
    error_count: int = Field(default=0, ge=0)
    errors: BoundedIndexingErrors = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IndexStatusResponse(BaseModel):
    """Current index presence and freshness for one repository."""

    repository_path: str
    database_path: str
    status: IndexStatus
    index_format_version: str | None = None
    timestamp_of_index: int | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = Field(default=None, ge=1)
    file_count: int = Field(default=0, ge=0)
    symbol_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    embedding_count: int = Field(default=0, ge=0)
    lexical_document_count: int = Field(default=0, ge=0)
    added_file_count: int = Field(default=0, ge=0)
    changed_file_count: int = Field(default=0, ge=0)
    deleted_file_count: int = Field(default=0, ge=0)
    changed_paths: BoundedPathSamples = Field(default_factory=list)
    error_count: int = Field(default=0, ge=0)
    errors: BoundedIndexingErrors = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

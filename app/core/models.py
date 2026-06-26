"""Shared data contracts used by indexing, storage, and search.

These models define the shape of data exchanged between subsystems. Keeping
the contracts in one module prevents the parser, indexer, storage layer, and
search layer from inventing slightly different representations of the same
concept.
"""

import uuid
from typing import Literal

from pydantic import BaseModel

SymbolKind = Literal[
    # A regular function declared outside a class.
    "function",
    # An async function declared outside a class.
    "async_function",
    # A Python class declaration.
    "class",
    # A regular function whose immediate lexical parent is a class.
    "method",
    # An async function whose immediate lexical parent is a class.
    "async_method",
]

RetrievalKind = Literal["exact", "fuzzy", "semantic"]


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


class SearchRequest(BaseModel):
    """Validated input supplied to the future unified search service."""

    # User text, symbol name, typo, or natural-language description to retrieve.
    query: str
    # Explicit retrieval strategy selected by the caller.
    request_mode: RetrievalKind
    # Maximum number of ranked results the caller wants returned.
    top_k: int
    # Optional repository-relative path used to narrow the search space.
    path: str | None = None
    # Compute implementation requested for supported hot loops.
    backend: Literal["python", "mojo"]


class SearchResult(BaseModel):
    """One ranked symbol or chunk returned by retrieval."""

    # Identity of the matched Symbol or Chunk record.
    id: uuid.UUID
    # Indicates which record type should be loaded or interpreted.
    result_type: Literal["symbol", "chunk"]
    # Repository-relative source file containing the match.
    file_path: str
    # Inclusive one-based source range displayed to the caller.
    start_line: int
    end_line: int
    # Available for symbol-owned results; absent for module-level chunks.
    symbol_name: str | None = None
    # Bounded source text included with the result.
    snippet: str
    # Mode-specific relevance normalized into a comparable output range.
    score: float
    # Retrieval strategy that produced this result.
    mode: RetrievalKind
    # Compute implementation that actually performed the relevant operation.
    backend: Literal["python", "mojo"]


class SearchResponse(BaseModel):
    """Complete structured response returned by CLI, Streamlit, or MCP."""

    # Preserve caller input for logging and structured tool responses.
    original_query: str
    # Mode actually used; future auto-routing may differ from requested mode.
    mode: RetrievalKind
    # Backend actually used after availability checks and fallback.
    backend: Literal["python", "mojo"]
    # End-to-end retrieval duration measured in seconds.
    elapsed_time: float
    # Ordered best-first results.
    ranked_results: list[SearchResult]
    # Non-fatal details such as falling back from Mojo to Python.
    warnings: list[str]

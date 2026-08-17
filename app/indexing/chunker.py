"""Split extracted symbols into bounded semantic-search chunks.

A chunk is the unit retrieved by semantic search. Whole files are often too
broad, while individual lines lack context. Symbol boundaries provide a useful
starting unit, and long symbols are divided into overlapping windows so each
piece remains small enough to embed and retrieve precisely.
"""

import hashlib
import uuid

from app.core.models import Chunk, Symbol
from app.indexing.analysis import (
    SemanticUnit,
    SourceFile,
)


def build_embedding_text(
    # Relative paths tell the embedding model where code lives without leaking
    # machine-specific absolute paths.
    relative_path: str,
    # This is the exact source fragment represented by the chunk.
    raw_text: str,
    # Symbol context helps distinguish identical code in different scopes.
    qualified_name: str | None = None,
    # Kind gives additional semantic context such as class versus method.
    kind: str | None = None,
    language: str = "python",
) -> str:
    """Construct the exact text that will be converted into an embedding."""

    source_file = SourceFile(
        relative_path=relative_path,
        language=language,
        text=raw_text,
    )
    metadata = [
        f"Language: {source_file.language}",
        f"Path: {source_file.relative_path}",
        f"Kind: {kind or 'symbol'}",
    ]
    if qualified_name is not None:
        metadata.append(f"Symbol: {qualified_name}")
    return "\n".join([*metadata, raw_text])


def calculate_content_hash(text: str) -> str:
    """Return a stable SHA-256 identifier for embedding input text."""

    # Hash functions consume bytes, so encode the Unicode string as UTF-8.
    encoded_text = text.encode("utf-8")

    # hexdigest returns a database-friendly lowercase hexadecimal string rather
    # than raw binary digest bytes.
    return hashlib.sha256(encoded_text).hexdigest()


def _line_windows(
    # One-based first line belonging to the symbol.
    start_line: int,
    # One-based final line belonging to the symbol.
    end_line: int,
    # Maximum number of source lines in any generated chunk.
    max_lines: int,
    # Number of trailing lines repeated at the start of the next chunk.
    overlap: int,
) -> list[tuple[int, int]]:
    """Calculate inclusive line ranges for bounded overlapping chunks."""

    if max_lines <= 0:
        raise ValueError("max_lines must be positive")

    # Negative overlap is meaningless. Overlap equal to or larger than the
    # window would make the step zero or negative and cause an infinite loop.
    if overlap < 0 or overlap >= max_lines:
        raise ValueError("overlap must be between 0 and max_lines - 1")

    # Each tuple in this list is an inclusive `(start_line, end_line)` range.
    windows: list[tuple[int, int]] = []

    current_start = start_line

    # Moving by less than max_lines causes neighboring windows to overlap.
    # Example: max=100, overlap=20 produces a step of 80.
    step = max_lines - overlap

    while current_start <= end_line:
        # Limit the proposed end to the symbol boundary. Subtract one because
        # both line values are inclusive.
        current_end = min(current_start + max_lines - 1, end_line)

        windows.append((current_start, current_end))

        if current_end == end_line:
            break

        current_start += step

    return windows


def subtract_owned_spans(
    start_line: int,
    end_line: int,
    owned_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return inclusive gaps after subtracting merged owned source spans."""

    if start_line < 1 or end_line < start_line:
        raise ValueError("source range must be one-based and inclusive")

    clipped = sorted(
        (max(start_line, start), min(end_line, end))
        for start, end in owned_spans
        if end >= start_line and start <= end_line and end >= start
    )
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    gaps: list[tuple[int, int]] = []
    cursor = start_line
    for start, end in merged:
        if cursor < start:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= end_line:
        gaps.append((cursor, end_line))
    return gaps


def group_spans_with_windows(
    spans: list[tuple[int, int]],
    max_lines: int = 100,
    overlap: int = 20,
) -> list[tuple[int, int]]:
    """Merge adjacent spans and split them into deterministic line windows."""

    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [
        window
        for start, end in merged
        for window in _line_windows(start, end, max_lines, overlap)
    ]


def chunk_semantic_units(
    source_file: SourceFile,
    semantic_units: tuple[SemanticUnit, ...] | list[SemanticUnit],
    repository_id: uuid.UUID,
    symbols_by_qualified_name: dict[str, Symbol],
    max_lines: int = 100,
    overlap: int = 20,
    max_chunks: int = 2_048,
) -> list[Chunk]:
    """Convert neutral semantic units into persisted, bounded chunks."""

    if max_chunks < 1:
        raise ValueError("max_chunks must be positive")
    source_lines = source_file.text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    seen: set[tuple[str, int, int, uuid.UUID | None, str]] = set()

    for unit in semantic_units:
        unit_lines = unit.text.splitlines(keepends=True)
        unit_line_count = unit.end_line - unit.start_line + 1
        persisted_symbol = (
            symbols_by_qualified_name.get(unit.symbol.qualified_name)
            if unit.symbol is not None
            else None
        )
        for start_line, end_line in _line_windows(
            unit.start_line,
            unit.end_line,
            max_lines,
            overlap,
        ):
            if len(unit_lines) == unit_line_count:
                relative_start = start_line - unit.start_line
                relative_end = end_line - unit.start_line + 1
                raw_text = "".join(unit_lines[relative_start:relative_end])
            else:
                raw_text = "".join(source_lines[start_line - 1 : end_line])
            if not raw_text.strip():
                continue
            key = (
                unit.kind,
                start_line,
                end_line,
                persisted_symbol.id if persisted_symbol is not None else None,
                raw_text,
            )
            if key in seen:
                continue
            seen.add(key)
            if len(chunks) >= max_chunks:
                raise ValueError(
                    f"File exceeds the {max_chunks} semantic chunk limit"
                )

            embedding_text = build_embedding_text(
                relative_path=source_file.relative_path,
                raw_text=raw_text,
                qualified_name=(
                    persisted_symbol.qualified_name
                    if persisted_symbol is not None
                    else None
                ),
                kind=unit.kind,
                language=source_file.language,
            )
            chunks.append(
                Chunk(
                    id=uuid.uuid5(
                        repository_id,
                        f"{unit.stable_id_input}:{start_line}:{end_line}",
                    ),
                    repository_id=repository_id,
                    relative_path=source_file.relative_path,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_id=(
                        persisted_symbol.id if persisted_symbol is not None else None
                    ),
                    raw_text=raw_text,
                    content_hash=calculate_content_hash(embedding_text),
                    language=source_file.language,
                    semantic_unit_kind=unit.kind,
                )
            )
    return chunks


def chunk_symbols(
    # Complete file source is needed because symbol lines refer to file-level
    # coordinates rather than offsets inside each snippet.
    source: str,
    # Symbols should all belong to this source file.
    symbols: list[Symbol],
    # A simple line limit is used first; token-aware limits can be added later.
    max_lines: int = 100,
    # Overlap preserves context where a long function is split.
    overlap: int = 20,
    # Bound vectors and embedding inputs produced by one adversarial file.
    max_chunks: int = 2_048,
) -> list[Chunk]:
    """Create one or more semantic-search chunks for each symbol."""

    if max_chunks < 1:
        raise ValueError("max_chunks must be positive")

    # Keep line endings so joining slices reconstructs exact source formatting.
    source_lines = source.splitlines(keepends=True)

    chunks: list[Chunk] = []

    # Symbol boundaries are the primary semantic boundaries in this first
    # implementation.
    for symbol in symbols:
        # A short symbol creates one range; a long symbol creates several
        # overlapping ranges.
        for start_line, end_line in _line_windows(
            symbol.start_line,
            symbol.end_line,
            max_lines,
            overlap,
        ):
            if len(chunks) >= max_chunks:
                raise ValueError(
                    f"File exceeds the {max_chunks} semantic chunk limit"
                )
            # Translate one-based inclusive source lines into a zero-based
            # Python slice with an exclusive end.
            raw_text = "".join(source_lines[start_line - 1 : end_line])

            # Add file and symbol context before embedding. The user-facing
            # raw_text remains clean source code.
            embedding_text = build_embedding_text(
                relative_path=symbol.relative_path,
                raw_text=raw_text,
                qualified_name=symbol.qualified_name,
                kind=symbol.kind,
                language=symbol.language,
            )

            # Construct a validated Chunk model and append it in one operation.
            chunks.append(
                Chunk(
                    # UUIDs make chunks independently addressable in storage.
                    id=uuid.uuid4(),
                    # Preserve repository ownership from the source symbol.
                    repository_id=symbol.repository_id,
                    # Preserve the portable file location.
                    relative_path=symbol.relative_path,
                    # Record exact source coordinates for UI and MCP output.
                    start_line=start_line,
                    end_line=end_line,
                    # Link the chunk to the symbol that supplied its context.
                    symbol_id=symbol.id,
                    # Store only original code for result display.
                    raw_text=raw_text,
                    # Hash the enriched embedding input, not only raw code.
                    content_hash=calculate_content_hash(embedding_text),
                    language=symbol.language,
                    semantic_unit_kind="symbol",
                )
            )

    # TODO: Consider token-based limits after line-based chunking is tested.

    return chunks

"""Split extracted symbols into bounded semantic-search chunks.

A chunk is the unit retrieved by semantic search. Whole files are often too
broad, while individual lines lack context. Symbol boundaries provide a useful
starting unit, and long symbols are divided into overlapping windows so each
piece remains small enough to embed and retrieve precisely.
"""

import hashlib
import uuid

from app.core.models import Chunk, Symbol


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
) -> str:
    """Construct the exact text that will be converted into an embedding."""

    metadata = [f"File: {relative_path}"]

    # Module-level chunks will have no symbol name, so append this conditionally.
    if qualified_name is not None:
        metadata.append(f"Symbol: {qualified_name}")

    # Kind is meaningful only for symbol-owned chunks.
    if kind is not None:
        metadata.append(f"Kind: {kind}")

    # Join metadata with single newlines, add a blank separator, then preserve
    # the original source unchanged. This format must remain deterministic
    # because the same text is later hashed for embedding reuse.
    return "\n".join(metadata) + "\n\n" + raw_text


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
) -> list[Chunk]:
    """Create one or more semantic-search chunks for each symbol."""

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
                )
            )

    # TODO: Add chunks for imports, constants, module docstrings, and
    # executable module-level statements outside top-level symbol ranges.

    # TODO: Consider token-based limits after line-based chunking is tested.

    return chunks

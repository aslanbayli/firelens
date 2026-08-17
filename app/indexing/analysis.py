"""Language-neutral source analysis records.

Adapters translate language-specific parser output into these immutable records.
Nothing in this module exposes a parser implementation such as ``ast``.
"""

from dataclasses import dataclass


def _validate_label(value: str, field_name: str) -> None:
    if not value or not value.replace("-", "_").isidentifier():
        raise ValueError(f"{field_name} must be a non-empty identifier")


@dataclass(frozen=True)
class SourceFile:
    """Decoded text and neutral metadata for one repository file."""

    relative_path: str
    language: str
    text: str

    def __post_init__(self) -> None:
        if not self.relative_path:
            raise ValueError("relative_path must not be empty")
        _validate_label(self.language, "language")


@dataclass(frozen=True)
class ParsedSymbol:
    """Parser-independent declaration extracted from source."""

    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    source_snippet: str

    def __post_init__(self) -> None:
        if not self.name or not self.qualified_name:
            raise ValueError("symbol names must not be empty")
        _validate_label(self.kind, "symbol kind")
        _validate_source_range(self.start_line, self.end_line)


@dataclass(frozen=True)
class SemanticUnit:
    """One parser-neutral unit that can be chunked and embedded."""

    stable_id_input: str
    kind: str
    start_line: int
    end_line: int
    text: str
    embedding_text: str
    symbol: ParsedSymbol | None = None

    def __post_init__(self) -> None:
        if not self.stable_id_input:
            raise ValueError("stable_id_input must not be empty")
        _validate_label(self.kind, "semantic unit kind")
        _validate_source_range(self.start_line, self.end_line)
        if not self.text.strip():
            raise ValueError("semantic unit text must not be blank")


@dataclass(frozen=True)
class GraphNodeDefinition:
    """A language-neutral graph node emitted by a source adapter."""

    node_kind: str
    qualified_name: str
    name: str
    relative_path: str
    start_line: int
    end_line: int
    symbol_qualified_name: str | None = None

    def __post_init__(self) -> None:
        _validate_label(self.node_kind, "graph node kind")
        if not self.qualified_name or not self.name or not self.relative_path:
            raise ValueError("graph node identity fields must not be empty")
        _validate_source_range(self.start_line, self.end_line)


@dataclass(frozen=True)
class UnresolvedGraphFact:
    """A parser-neutral relationship awaiting repository-wide resolution."""

    edge_kind: str
    source_reference: str
    target_reference: str
    source_scope: str
    source_file: str
    start_line: int
    end_line: int
    extraction_adapter: str
    adapter_version: str
    confidence: float
    target_kind: str = "any"
    target_qualified_hint: str | None = None
    hint_resolution_method: str | None = None
    evidence_text: str | None = None

    def __post_init__(self) -> None:
        _validate_label(self.edge_kind, "edge kind")
        _validate_label(self.extraction_adapter, "extraction adapter")
        _validate_label(self.target_kind, "graph target kind")
        if not self.source_reference or not self.target_reference:
            raise ValueError("graph references must not be empty")
        if not self.source_scope or not self.source_file:
            raise ValueError("graph source context must not be empty")
        if not self.adapter_version:
            raise ValueError("adapter_version must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("graph fact confidence must be between 0 and 1")
        if self.evidence_text is not None and len(self.evidence_text) > 256:
            raise ValueError("graph evidence text must be at most 256 characters")
        _validate_source_range(self.start_line, self.end_line)


@dataclass(frozen=True)
class AnalysisDiagnostic:
    """A bounded parser or extraction diagnostic for one source file."""

    stage: str
    message: str
    line: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """All neutral analysis output for one source file."""

    source_file: SourceFile
    symbols: tuple[ParsedSymbol, ...] = ()
    semantic_units: tuple[SemanticUnit, ...] = ()
    graph_nodes: tuple[GraphNodeDefinition, ...] = ()
    graph_facts: tuple[UnresolvedGraphFact, ...] = ()
    diagnostics: tuple[AnalysisDiagnostic, ...] = ()


def build_semantic_embedding_text(
    source_file: SourceFile,
    kind: str,
    text: str,
    symbol: ParsedSymbol | None = None,
) -> str:
    """Build the exact, metadata-rich text supplied to an embedder."""

    metadata = [
        f"Language: {source_file.language}",
        f"Path: {source_file.relative_path}",
        f"Kind: {kind}",
    ]
    if symbol is not None:
        metadata.append(f"Symbol: {symbol.qualified_name}")
    return "\n".join([*metadata, text])


def semantic_unit(
    source_file: SourceFile,
    kind: str,
    start_line: int,
    end_line: int,
    text: str,
    symbol: ParsedSymbol | None = None,
) -> SemanticUnit:
    """Create a validated unit with deterministic identity and embedding text."""

    symbol_name = symbol.qualified_name if symbol is not None else ""
    stable_id_input = (
        f"{source_file.language}:{source_file.relative_path}:{kind}:"
        f"{start_line}:{end_line}:{symbol_name}"
    )
    return SemanticUnit(
        stable_id_input=stable_id_input,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
        text=text,
        embedding_text=build_semantic_embedding_text(
            source_file,
            kind,
            text,
            symbol,
        ),
        symbol=symbol,
    )


def _validate_source_range(start_line: int, end_line: int) -> None:
    if start_line < 1 or end_line < start_line:
        raise ValueError("source range must be one-based and inclusive")

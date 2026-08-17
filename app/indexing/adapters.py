"""Static registry and protocol for language analysis adapters."""

from pathlib import Path
from typing import Protocol

from app.indexing.analysis import ParsedDocument, SourceFile


class LanguageAdapter(Protocol):
    """The language-specific boundary used by indexing and lexical search."""

    language: str
    extensions: frozenset[str]

    def analyze(self, source_file: SourceFile) -> ParsedDocument:
        """Parse a decoded source file into neutral analysis records."""

    def identifier_terms(self, identifier: str) -> tuple[str, ...]:
        """Return deterministic searchable terms while preserving the input."""


class LanguageAdapterRegistry:
    """Map file extensions to adapters without runtime discovery."""

    def __init__(self) -> None:
        self._adapters_by_extension: dict[str, LanguageAdapter] = {}

    def register(self, adapter: LanguageAdapter) -> None:
        extensions = {_normalize_extension(value) for value in adapter.extensions}
        if not extensions:
            raise ValueError("An adapter must support at least one extension")

        ambiguous = sorted(
            extension
            for extension in extensions
            if extension in self._adapters_by_extension
            and self._adapters_by_extension[extension] is not adapter
        )
        if ambiguous:
            joined = ", ".join(ambiguous)
            raise ValueError(f"Ambiguous language adapter registration: {joined}")

        for extension in extensions:
            self._adapters_by_extension[extension] = adapter

    def adapter_for_path(self, path: str | Path) -> LanguageAdapter | None:
        return self._adapters_by_extension.get(Path(path).suffix.casefold())

    def require_adapter(self, path: str | Path) -> LanguageAdapter:
        adapter = self.adapter_for_path(path)
        if adapter is None:
            raise ValueError(f"Unsupported source file: {path}")
        return adapter

    def supports(self, path: str | Path) -> bool:
        return self.adapter_for_path(path) is not None

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._adapters_by_extension)

    def identifier_terms(self, value: str) -> tuple[str, ...]:
        """Combine neutral query terms from registered adapter conventions."""

        terms: list[str] = []
        seen: set[str] = set()
        seen_adapter_ids: set[int] = set()
        for adapter in self._adapters_by_extension.values():
            if id(adapter) in seen_adapter_ids:
                continue
            seen_adapter_ids.add(id(adapter))
            for term in adapter.identifier_terms(value):
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
        return tuple(terms)


def _normalize_extension(extension: str) -> str:
    normalized = extension.strip().casefold()
    if not normalized.startswith(".") or len(normalized) < 2:
        raise ValueError(f"Invalid source extension: {extension!r}")
    return normalized


def create_default_registry() -> LanguageAdapterRegistry:
    """Build the explicit built-in adapter registry."""

    from app.indexing.documentation_adapter import DocumentationAdapter
    from app.indexing.python_adapter import PythonAdapter

    registry = LanguageAdapterRegistry()
    registry.register(PythonAdapter())
    registry.register(
        DocumentationAdapter(
            language="markdown",
            extensions=frozenset({".md", ".markdown"}),
        )
    )
    registry.register(
        DocumentationAdapter(
            language="restructuredtext",
            extensions=frozenset({".rst"}),
        )
    )
    return registry


DEFAULT_ADAPTER_REGISTRY = create_default_registry()

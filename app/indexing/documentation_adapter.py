"""Language-neutral adapter for repository documentation files."""

import re

from app.indexing.analysis import ParsedDocument, SourceFile, semantic_unit
from app.indexing.python_adapter import PythonAdapter


_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+")
_RST_UNDERLINE = re.compile(r"^[=\-~^\"`:+*#<>_]{3,}\s*$")


class DocumentationAdapter:
    """Extract bounded sections from Markdown and reStructuredText."""

    def __init__(
        self,
        language: str = "documentation",
        extensions: frozenset[str] = frozenset({".md", ".markdown", ".rst"}),
    ) -> None:
        self.language = language
        self.extensions = extensions

    def analyze(self, source_file: SourceFile) -> ParsedDocument:
        lines = source_file.text.splitlines(keepends=True)
        if not lines or not source_file.text.strip():
            return ParsedDocument(source_file=source_file)

        section_starts = [1]
        for line_number, line in enumerate(lines, start=1):
            if _MARKDOWN_HEADING.match(line):
                section_starts.append(line_number)
            elif line_number > 1 and _RST_UNDERLINE.match(line):
                section_starts.append(line_number - 1)
        starts = sorted(set(section_starts))

        units = []
        for index, start_line in enumerate(starts):
            end_line = (
                starts[index + 1] - 1 if index + 1 < len(starts) else len(lines)
            )
            text = "".join(lines[start_line - 1 : end_line])
            if text.strip():
                units.append(
                    semantic_unit(
                        source_file,
                        "documentation",
                        start_line,
                        end_line,
                        text,
                    )
                )
        return ParsedDocument(source_file=source_file, semantic_units=tuple(units))

    def identifier_terms(self, identifier: str) -> tuple[str, ...]:
        return PythonAdapter().identifier_terms(identifier)

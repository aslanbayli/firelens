"""Python analysis implemented behind the language-adapter boundary."""

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from typing import Literal

from app.indexing.analysis import (
    AnalysisDiagnostic,
    ParsedDocument,
    ParsedSymbol,
    SemanticUnit,
    SourceFile,
    UnresolvedGraphFact,
    semantic_unit,
)


_IDENTIFIER_PARTS = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]+|\d+"
)
_IDENTIFIER_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class _VisitedSymbol:
    symbol: ParsedSymbol
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.lines = source.splitlines(keepends=True)
        self.scope: list[tuple[str, Literal["class", "function"]]] = []
        self.visited_symbols: list[_VisitedSymbol] = []

    @property
    def symbols(self) -> list[ParsedSymbol]:
        return [visited.symbol for visited in self.visited_symbols]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, "class")
        self.scope.append((node.name, "class"))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self._inside_class() else "function"
        self._record(node, kind)
        self.scope.append((node.name, "function"))
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "async_method" if self._inside_class() else "async_function"
        self._record(node, kind)
        self.scope.append((node.name, "function"))
        self.generic_visit(node)
        self.scope.pop()

    def _inside_class(self) -> bool:
        return bool(self.scope and self.scope[-1][1] == "class")

    def _record(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
    ) -> None:
        start_line = _node_start_line(node)
        end_line = node.end_lineno or node.lineno
        qualified_name = ".".join([*(name for name, _ in self.scope), node.name])
        parsed = ParsedSymbol(
            name=node.name,
            qualified_name=qualified_name,
            kind=kind,
            start_line=start_line,
            end_line=end_line,
            source_snippet="".join(self.lines[start_line - 1 : end_line]),
        )
        self.visited_symbols.append(_VisitedSymbol(parsed, node))


class PythonAdapter:
    """Analyze Python with the standard-library AST and tokenizer."""

    language = "python"
    extensions = frozenset({".py"})

    def analyze(self, source_file: SourceFile) -> ParsedDocument:
        try:
            tree = ast.parse(source_file.text)
        except SyntaxError as error:
            return ParsedDocument(
                source_file=source_file,
                diagnostics=(
                    AnalysisDiagnostic(
                        stage="parse",
                        message=str(error),
                        line=error.lineno,
                    ),
                ),
            )

        visitor = _SymbolVisitor(source_file.text)
        visitor.visit(tree)
        units = self._semantic_units(source_file, tree, visitor)
        facts = self._graph_facts(source_file, tree)
        return ParsedDocument(
            source_file=source_file,
            symbols=tuple(visitor.symbols),
            semantic_units=tuple(units),
            graph_facts=tuple(facts),
        )

    def identifier_terms(self, identifier: str) -> tuple[str, ...]:
        normalized_original = identifier.strip().casefold()
        if not normalized_original:
            return ()

        terms: list[str] = [normalized_original]
        for fragment in _IDENTIFIER_SEPARATORS.split(identifier):
            if not fragment:
                continue
            terms.append(fragment.casefold())
            terms.extend(
                part.casefold() for part in _IDENTIFIER_PARTS.findall(fragment)
            )
        return tuple(dict.fromkeys(term for term in terms if term))

    def _semantic_units(
        self,
        source_file: SourceFile,
        tree: ast.Module,
        visitor: _SymbolVisitor,
    ) -> list[SemanticUnit]:
        lines = source_file.text.splitlines(keepends=True)
        units: list[SemanticUnit] = []

        for visited in visitor.visited_symbols:
            symbol = visited.symbol
            units.append(
                semantic_unit(
                    source_file,
                    "symbol",
                    symbol.start_line,
                    symbol.end_line,
                    symbol.source_snippet,
                    symbol,
                )
            )
            docstring_node = _docstring_node(visited.node)
            if docstring_node is not None:
                units.append(
                    _unit_from_node(
                        source_file,
                        "symbol_docstring",
                        docstring_node,
                        lines,
                        symbol,
                    )
                )

        module_docstring = _docstring_node(tree)
        if module_docstring is not None:
            units.append(
                _unit_from_node(
                    source_file,
                    "module_docstring",
                    module_docstring,
                    lines,
                )
            )

        top_level_nodes = list(tree.body)
        docstring_ids = (
            {id(module_docstring)} if module_docstring is not None else set()
        )
        grouped_nodes: list[tuple[str, list[ast.stmt]]] = []
        for node in top_level_nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if id(node) in docstring_ids:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                kind = "imports"
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                kind = "assignment"
            else:
                kind = "module_code"

            previous_end = None
            if grouped_nodes:
                previous_node = grouped_nodes[-1][1][-1]
                previous_end = previous_node.end_lineno or previous_node.lineno
            if (
                grouped_nodes
                and grouped_nodes[-1][0] == kind
                and previous_end is not None
                and node.lineno <= previous_end + 2
            ):
                grouped_nodes[-1][1].append(node)
            else:
                grouped_nodes.append((kind, [node]))

        for kind, nodes in grouped_nodes:
            units.extend(_grouped_node_units(source_file, kind, nodes, lines))
        units.extend(self._comment_units(source_file, visitor.symbols))
        return _deduplicate_units(units)

    def _comment_units(
        self,
        source_file: SourceFile,
        symbols: list[ParsedSymbol],
    ) -> list[SemanticUnit]:
        lines = source_file.text.splitlines(keepends=True)
        comments_by_line: dict[int, str] = {}
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source_file.text).readline)
            comments_by_line = {
                token.start[0]: token.string + "\n"
                for token in tokens
                if token.type == tokenize.COMMENT
            }
        except (IndentationError, tokenize.TokenError):
            return []

        groups = _group_consecutive_lines(list(comments_by_line))
        units: list[SemanticUnit] = []
        symbols_by_start = sorted(symbols, key=lambda symbol: symbol.start_line)
        for start_line, end_line in groups:
            text = "".join(
                comments_by_line[line]
                for line in range(start_line, end_line + 1)
            )
            associated_symbol = next(
                (
                    symbol
                    for symbol in symbols_by_start
                    if 0 < symbol.start_line - end_line <= 2
                    and all(
                        not line.strip()
                        for line in lines[end_line : symbol.start_line - 1]
                    )
                ),
                None,
            )
            if associated_symbol is None:
                containing_symbols = [
                    symbol
                    for symbol in symbols
                    if symbol.start_line <= start_line <= symbol.end_line
                ]
                associated_symbol = min(
                    containing_symbols,
                    key=lambda symbol: symbol.end_line - symbol.start_line,
                    default=None,
                )
            kind = (
                "symbol_comment"
                if associated_symbol is not None
                else "module_comment"
            )
            units.append(
                semantic_unit(
                    source_file,
                    kind,
                    start_line,
                    end_line,
                    text,
                    associated_symbol,
                )
            )
        return units

    def _graph_facts(
        self,
        source_file: SourceFile,
        tree: ast.Module,
    ) -> list[UnresolvedGraphFact]:
        facts: list[UnresolvedGraphFact] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [node.module or "." * node.level]
            else:
                continue
            facts.extend(
                UnresolvedGraphFact(
                    edge_kind="imports",
                    source_reference=source_file.relative_path,
                    target_reference=target,
                )
                for target in targets
            )
        return facts


def parse_python_tree(code: str) -> ast.Module:
    """Compatibility entry point whose implementation remains adapter-owned."""

    return ast.parse(code)


def parse_python_symbols(code: str) -> list[ParsedSymbol]:
    """Compatibility parser preserving historical symbol output and errors."""

    tree = parse_python_tree(code)
    visitor = _SymbolVisitor(code)
    visitor.visit(tree)
    return visitor.symbols


def _node_start_line(node: ast.AST) -> int:
    decorator_list = getattr(node, "decorator_list", ())
    return min((decorator.lineno for decorator in decorator_list), default=node.lineno)


def _docstring_node(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Expr | None:
    if not node.body:
        return None
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


def _unit_from_node(
    source_file: SourceFile,
    kind: str,
    node: ast.AST,
    lines: list[str],
    symbol: ParsedSymbol | None = None,
) -> SemanticUnit:
    start_line = node.lineno
    end_line = node.end_lineno or start_line
    return semantic_unit(
        source_file,
        kind,
        start_line,
        end_line,
        "".join(lines[start_line - 1 : end_line]),
        symbol,
    )


def _grouped_node_units(
    source_file: SourceFile,
    kind: str,
    nodes: list[ast.stmt],
    lines: list[str],
) -> list[SemanticUnit]:
    if not nodes:
        return []
    ordered = sorted(nodes, key=lambda node: node.lineno)
    groups: list[list[ast.stmt]] = [[ordered[0]]]
    for node in ordered[1:]:
        previous_end = groups[-1][-1].end_lineno or groups[-1][-1].lineno
        if node.lineno <= previous_end + 2:
            groups[-1].append(node)
        else:
            groups.append([node])

    units: list[SemanticUnit] = []
    for group in groups:
        start_line = group[0].lineno
        end_line = group[-1].end_lineno or group[-1].lineno
        text = "".join(lines[start_line - 1 : end_line])
        if text.strip():
            units.append(
                semantic_unit(source_file, kind, start_line, end_line, text)
            )
    return units


def _group_consecutive_lines(lines: list[int]) -> list[tuple[int, int]]:
    if not lines:
        return []
    ordered = sorted(set(lines))
    groups = [(ordered[0], ordered[0])]
    for line in ordered[1:]:
        start, end = groups[-1]
        if line == end + 1:
            groups[-1] = (start, line)
        else:
            groups.append((line, line))
    return groups


def _deduplicate_units(units: list[SemanticUnit]) -> list[SemanticUnit]:
    deduplicated: list[SemanticUnit] = []
    seen: set[tuple[str, int, int, str]] = set()
    for unit in units:
        key = (unit.kind, unit.start_line, unit.end_line, unit.text)
        if key in seen or not unit.text.strip():
            continue
        seen.add(key)
        deduplicated.append(unit)
    return sorted(
        deduplicated,
        key=lambda unit: (unit.start_line, unit.end_line, unit.kind),
    )

"""Python analysis implemented behind the language-adapter boundary."""

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from app.indexing.analysis import (
    AnalysisDiagnostic,
    GraphNodeDefinition,
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
        graph_nodes, facts = self._graph_analysis(source_file, tree, visitor)
        return ParsedDocument(
            source_file=source_file,
            symbols=tuple(visitor.symbols),
            semantic_units=tuple(units),
            graph_nodes=tuple(graph_nodes),
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

    def _graph_analysis(
        self,
        source_file: SourceFile,
        tree: ast.Module,
        symbol_visitor: _SymbolVisitor,
    ) -> tuple[list[GraphNodeDefinition], list[UnresolvedGraphFact]]:
        module_name = _python_module_name(source_file.relative_path)
        line_count = max(1, len(source_file.text.splitlines()))
        is_test_file = _is_test_file(source_file.relative_path)
        file_kind = "test_file" if is_test_file else "file"
        path_name = PurePosixPath(source_file.relative_path).name
        graph_nodes = [
            GraphNodeDefinition(
                node_kind=file_kind,
                qualified_name=source_file.relative_path,
                name=path_name,
                relative_path=source_file.relative_path,
                start_line=1,
                end_line=line_count,
            ),
            GraphNodeDefinition(
                node_kind="module",
                qualified_name=module_name,
                name=module_name.rsplit(".", 1)[-1],
                relative_path=source_file.relative_path,
                start_line=1,
                end_line=line_count,
            ),
        ]
        symbol_references: dict[int, str] = {}
        for visited in symbol_visitor.visited_symbols:
            qualified_name = f"{module_name}.{visited.symbol.qualified_name}"
            symbol_references[id(visited.node)] = qualified_name
            graph_nodes.append(
                GraphNodeDefinition(
                    node_kind="test_symbol" if is_test_file else "symbol",
                    qualified_name=qualified_name,
                    name=visited.symbol.name,
                    relative_path=source_file.relative_path,
                    start_line=visited.symbol.start_line,
                    end_line=visited.symbol.end_line,
                    symbol_qualified_name=visited.symbol.qualified_name,
                )
            )

        graph_visitor = _GraphVisitor(
            source_file=source_file,
            module_name=module_name,
            package_name=_python_package_name(source_file.relative_path, module_name),
            symbol_references=symbol_references,
            is_test_file=is_test_file,
        )
        graph_visitor.visit(tree)
        return graph_nodes, graph_visitor.facts


@dataclass(frozen=True)
class _ImportBinding:
    qualified_name: str
    target_kind: str
    resolution_method: str


class _GraphVisitor(ast.NodeVisitor):
    """Translate Python AST relationships into neutral unresolved facts."""

    def __init__(
        self,
        *,
        source_file: SourceFile,
        module_name: str,
        package_name: str,
        symbol_references: dict[int, str],
        is_test_file: bool,
    ) -> None:
        self.source_file = source_file
        self.module_name = module_name
        self.package_name = package_name
        self.symbol_references = symbol_references
        self.is_test_file = is_test_file
        self.scope: list[str] = [module_name]
        self.scope_kinds: list[str] = ["module"]
        self.import_bindings: list[dict[str, _ImportBinding]] = [{}]
        self.facts: list[UnresolvedGraphFact] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_reference = self.symbol_references[id(node)]
        for base in node.bases:
            reference = _expression_reference(base)
            if reference is not None:
                self._record_fact(
                    "inherits",
                    reference,
                    base,
                    source_reference=class_reference,
                    source_scope=class_reference,
                    confidence=0.95,
                    target_kind="symbol",
                )

        self._push_scope(class_reference, "class")
        for statement in node.body:
            self.visit(statement)
        self._pop_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        function_reference = self.symbol_references[id(node)]
        self._push_scope(function_reference, "function")

        if self.is_test_file and node.name.startswith("test_"):
            subject = node.name.removeprefix("test_").strip("_")
            if subject:
                self._record_fact(
                    "tests",
                    subject,
                    node,
                    confidence=0.65,
                    target_kind="symbol",
                    evidence_text=node.name,
                )
            fixture_arguments = [*node.args.posonlyargs, *node.args.args]
            for argument in fixture_arguments:
                if argument.arg not in {"self", "cls"}:
                    self._record_fact(
                        "tests",
                        argument.arg,
                        argument,
                        confidence=0.55,
                        target_kind="symbol",
                        evidence_text=argument.arg,
                    )

        for statement in node.body:
            self.visit(statement)
        self._pop_scope()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_fact(
                "imports",
                alias.name,
                node,
                confidence=1.0,
                target_kind="module",
                target_qualified_hint=alias.name,
                hint_resolution_method="explicitly_imported_module",
            )
            local_name = alias.asname or alias.name.split(".", 1)[0]
            bound_target = alias.name if alias.asname else local_name
            self.import_bindings[-1][local_name] = _ImportBinding(
                qualified_name=bound_target,
                target_kind="module",
                resolution_method="explicitly_imported_module",
            )
            self._record_test_relationship(alias.name, node, "module", 0.7)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_reference = _absolute_import_reference(
            self.package_name,
            node.module,
            node.level,
        )
        if module_reference:
            self._record_fact(
                "imports",
                module_reference,
                node,
                confidence=1.0,
                target_kind="module",
                target_qualified_hint=module_reference,
                hint_resolution_method="explicitly_imported_module",
            )
            self._record_test_relationship(module_reference, node, "module", 0.7)

        for alias in node.names:
            if alias.name == "*" or not module_reference:
                continue
            qualified_name = f"{module_reference}.{alias.name}"
            self._record_fact(
                "imports",
                alias.name,
                node,
                confidence=1.0,
                # ``from package import name`` can bind either a declaration
                # or a submodule. The repository resolver accepts only one
                # exact qualified match and leaves true ambiguity unresolved.
                target_kind="any",
                target_qualified_hint=qualified_name,
                hint_resolution_method="explicitly_imported_symbol",
            )
            local_name = alias.asname or alias.name
            self.import_bindings[-1][local_name] = _ImportBinding(
                qualified_name=qualified_name,
                target_kind="any",
                resolution_method="explicitly_imported_symbol",
            )
            self._record_test_relationship(alias.name, node, "any", 0.75)

    def visit_Call(self, node: ast.Call) -> None:
        reference = _expression_reference(node.func)
        if reference is not None:
            self._record_fact(
                "calls",
                reference,
                node.func,
                confidence=0.9,
                target_kind="symbol",
            )
            self._record_test_relationship(reference, node.func, "symbol", 0.85)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            reference = _expression_reference(node)
            if reference is not None:
                self._record_fact(
                    "references",
                    reference,
                    node,
                    confidence=0.7,
                    target_kind="any",
                )
        # The complete qualified expression is more useful than duplicate
        # facts for each Name nested beneath it.

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._record_fact(
                "references",
                node.id,
                node,
                confidence=0.6,
                target_kind="any",
            )

    def _record_test_relationship(
        self,
        target_reference: str,
        node: ast.AST,
        target_kind: str,
        confidence: float,
    ) -> None:
        if not self.is_test_file:
            return
        self._record_fact(
            "tests",
            target_reference,
            node,
            confidence=confidence,
            target_kind=target_kind,
        )

    def _record_fact(
        self,
        edge_kind: str,
        target_reference: str,
        node: ast.AST,
        *,
        source_reference: str | None = None,
        source_scope: str | None = None,
        confidence: float,
        target_kind: str,
        target_qualified_hint: str | None = None,
        hint_resolution_method: str | None = None,
        evidence_text: str | None = None,
    ) -> None:
        inferred_hint, inferred_method, inferred_kind = self._resolution_hint(
            target_reference
        )
        if target_qualified_hint is None:
            target_qualified_hint = inferred_hint
        if hint_resolution_method is None:
            hint_resolution_method = inferred_method
        if target_kind == "any" and inferred_kind is not None:
            target_kind = inferred_kind

        evidence = evidence_text or ast.get_source_segment(
            self.source_file.text,
            node,
        )
        if evidence is not None:
            evidence = " ".join(evidence.split())[:256]
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line) or start_line
        self.facts.append(
            UnresolvedGraphFact(
                edge_kind=edge_kind,
                source_reference=source_reference or self.scope[-1],
                target_reference=target_reference,
                source_scope=source_scope or self.scope[-1],
                source_file=self.source_file.relative_path,
                start_line=start_line,
                end_line=end_line,
                extraction_adapter="python_ast",
                adapter_version="1",
                confidence=confidence,
                target_kind=target_kind,
                target_qualified_hint=target_qualified_hint,
                hint_resolution_method=hint_resolution_method,
                evidence_text=evidence,
            )
        )

    def _resolution_hint(
        self,
        reference: str,
    ) -> tuple[str | None, str | None, str | None]:
        parts = reference.split(".")
        if parts[0] in {"self", "cls"}:
            class_reference = self._nearest_class_reference()
            if class_reference is not None and len(parts) > 1:
                return (
                    ".".join([class_reference, *parts[1:]]),
                    "same_scope_qualified",
                    "symbol",
                )

        binding = self._lookup_binding(parts[0])
        if binding is None:
            return None, None, None
        suffix = parts[1:]
        qualified_name = ".".join([binding.qualified_name, *suffix])
        method = binding.resolution_method
        target_kind = binding.target_kind
        if binding.target_kind == "module" and suffix:
            method = "explicitly_imported_module_member"
            target_kind = "symbol"
        return qualified_name, method, target_kind

    def _lookup_binding(self, name: str) -> _ImportBinding | None:
        for bindings in reversed(self.import_bindings):
            binding = bindings.get(name)
            if binding is not None:
                return binding
        return None

    def _nearest_class_reference(self) -> str | None:
        for reference, kind in zip(
            reversed(self.scope),
            reversed(self.scope_kinds),
            strict=True,
        ):
            if kind == "class":
                return reference
        return None

    def _push_scope(self, reference: str, kind: str) -> None:
        self.scope.append(reference)
        self.scope_kinds.append(kind)
        self.import_bindings.append({})

    def _pop_scope(self) -> None:
        self.scope.pop()
        self.scope_kinds.pop()
        self.import_bindings.pop()


def _python_module_name(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "__root__"


def _python_package_name(relative_path: str, module_name: str) -> str:
    if PurePosixPath(relative_path).name == "__init__.py":
        return module_name
    return module_name.rpartition(".")[0]


def _absolute_import_reference(
    package_name: str,
    imported_module: str | None,
    level: int,
) -> str:
    if level == 0:
        return imported_module or ""
    package_parts = [part for part in package_name.split(".") if part]
    parents_to_remove = max(0, level - 1)
    if parents_to_remove:
        package_parts = package_parts[:-parents_to_remove]
    if imported_module:
        package_parts.extend(imported_module.split("."))
    return ".".join(package_parts)


def _expression_reference(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expression_reference(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _is_test_file(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    lowered_parts = {part.casefold() for part in path.parts[:-1]}
    filename = path.name.casefold()
    return (
        "tests" in lowered_parts
        or "test" in lowered_parts
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    )


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

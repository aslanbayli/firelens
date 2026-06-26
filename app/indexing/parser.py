"""Parse Python source and extract searchable symbols.

Python's AST converts source text into a tree of syntax nodes. Walking that
tree is more reliable than searching text with regular expressions because the
AST understands nesting, decorators, async declarations, and exact line ranges.
"""

import ast
from dataclasses import dataclass
from typing import Literal

from app.core.models import SymbolKind


# Frozen makes ParsedSymbol immutable after creation. Parser output represents
# facts discovered in source and should not be mutated by later pipeline stages.
@dataclass(frozen=True)
class ParsedSymbol:
    """A symbol before repository and storage identifiers are assigned."""

    # Unqualified declaration name exactly as represented by the AST.
    name: str
    # Dot-separated nesting path, such as `Service.fetch`.
    qualified_name: str
    # Validated conceptual category shared with the persisted Symbol model.
    kind: SymbolKind
    # One-based first line, moved upward to include decorators when present.
    start_line: int
    # One-based final line reported by the AST.
    end_line: int
    # Original source text from start_line through end_line, inclusive.
    source_snippet: str


# NodeVisitor implements the visitor pattern for AST trees. Python dispatches a
# node named `FunctionDef` to a method named `visit_FunctionDef`.
class SymbolVisitor(ast.NodeVisitor):
    """Walk one Python AST while tracking lexical class/function scopes."""

    def __init__(self, source: str) -> None:
        # Preserve newline characters so joining a line slice reconstructs the
        # original source rather than collapsing lines together.
        self.lines = source.splitlines(keepends=True)

        # Each tuple stores `(scope_name, scope_category)`. A stack is needed
        # because AST nodes are nested and qualified names depend on every
        # containing declaration.
        self.scope: list[tuple[str, Literal["class", "function"]]] = []

        # Extracted symbols are appended in AST traversal order, which generally
        # follows source order and is deterministic for the same file.
        self.symbols: list[ParsedSymbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record a class, then visit declarations nested inside that class."""

        # Record before pushing the class so its name is not duplicated in its
        # own qualified name.
        self._record(node, "class")

        # Enter the class scope so child methods become `ClassName.method`.
        self.scope.append((node.name, "class"))

        # Continue normal traversal into the class body. Without this call,
        # methods and nested classes would never be visited.
        self.generic_visit(node)

        # Leave the class after all descendants have been processed. Balanced
        # push/pop operations are essential for correct sibling names.
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record a regular function or method and inspect nested definitions."""

        # The immediate parent determines whether Python treats this declaration
        # as a class method-like member or as a standalone/local function.
        kind: SymbolKind = "method" if self._inside_class() else "function"

        # Capture the declaration before entering its own function scope.
        self._record(node, kind)

        # Entering function scope allows a nested declaration to receive a name
        # such as `outer.inner`.
        self.scope.append((node.name, "function"))

        # Visit nested functions, nested classes, and other descendants.
        self.generic_visit(node)

        # Restore the containing scope before the next sibling is visited.
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record an async function or async method."""

        # Async declarations use distinct kinds so search results can retain
        # the exact source-level distinction.
        kind: SymbolKind = "async_method" if self._inside_class() else "async_function"

        # The remaining traversal behavior mirrors a regular FunctionDef.
        self._record(node, kind)
        self.scope.append((node.name, "function"))
        self.generic_visit(node)
        self.scope.pop()

    def _inside_class(self) -> bool:
        """Return whether the immediate lexical parent is a class."""

        # `self.scope` must be non-empty before reading its final element. Only
        # the immediate parent is checked: a function nested inside a method is
        # a local function, even though a class exists farther down the stack.
        return bool(self.scope and self.scope[-1][1] == "class")

    def _record(
        # `self` gives this helper access to lines, scope, and output storage.
        self,
        # All supported declaration nodes expose name, lineno, end_lineno, and
        # decorator_list, which are the properties this helper needs.
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        # The visitor decides the kind because context determines method status.
        kind: SymbolKind,
    ) -> None:
        """Convert one AST declaration node into a ParsedSymbol."""

        # Decorators can begin before `class` or `def`, so calculate the true
        # first source line through a dedicated helper.
        start_line = self._start_line(node)

        # Modern Python AST nodes normally include end_lineno. Falling back to
        # lineno keeps the parser defensive if an unusual node lacks it.
        end_line = node.end_lineno or node.lineno

        # Copy all active scope names, ignoring their internal categories.
        qualified_parts = [name for name, _ in self.scope]

        # Add this declaration's own name to finish the qualified path.
        qualified_parts.append(node.name)

        # Source line numbers are one-based, while list indexes are zero-based.
        # The slice end is exclusive, so using `end_line` includes that line.
        snippet = "".join(self.lines[start_line - 1 : end_line])

        # Append one immutable parser result. UUIDs and repository paths are
        # intentionally assigned later by the indexer, not by syntax parsing.
        self.symbols.append(
            ParsedSymbol(
                name=node.name,
                qualified_name=".".join(qualified_parts),
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                source_snippet=snippet,
            )
        )

    @staticmethod
    def _start_line(
        # The helper does not use instance state, so it is a static method.
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        """Return the first decorator line or the declaration line."""

        # A declaration may have multiple decorators. Collect all their
        # one-based starting lines so the earliest one can be selected.
        decorator_lines = [decorator.lineno for decorator in node.decorator_list]

        # `default=node.lineno` handles declarations without decorators.
        return min(decorator_lines, default=node.lineno)


def parse(code: str) -> ast.Module:
    """Convert Python source text into its complete AST.

    SyntaxError is intentionally not caught here. The indexer knows the file
    path and can therefore produce a better error while continuing other files.
    """

    # ast.parse validates Python syntax and returns the module root node.
    return ast.parse(code)


def parse_symbols(code: str) -> list[ParsedSymbol]:
    """Extract all supported declarations from one Python source string."""

    # First build the syntax tree. Invalid Python raises SyntaxError here.
    tree = parse(code)

    # Give the visitor the original text so it can recover exact snippets.
    visitor = SymbolVisitor(code)

    # Start recursive dispatch from the module root.
    visitor.visit(tree)

    # TODO: Decide whether nested local functions should remain searchable.
    # They are currently included with names such as `outer.inner`.

    # Return only parsed symbol facts; storage-specific data is added later.
    return visitor.symbols

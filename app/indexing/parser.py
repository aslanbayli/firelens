"""Compatibility wrappers for the Python language adapter."""

from app.indexing.analysis import ParsedSymbol
from app.indexing.python_adapter import parse_python_symbols, parse_python_tree


def parse(code: str):
    return parse_python_tree(code)


def parse_symbols(code: str) -> list[ParsedSymbol]:
    return parse_python_symbols(code)

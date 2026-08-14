"""Shared bounds for safe search computation and result construction."""


MAX_FUZZY_QUERY_CHARS = 256
MAX_FUZZY_CANDIDATE_CHARS = 512
MAX_RESULT_SYMBOL_NAME_CHARS = 4_096


def bounded_symbol_name(symbol_name: str | None) -> str | None:
    """Return a symbol name that fits the public SearchResult contract."""

    if symbol_name is None:
        return None
    return symbol_name[:MAX_RESULT_SYMBOL_NAME_CHARS]

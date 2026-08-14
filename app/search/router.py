"""Named, testable heuristics for automatic retrieval routing."""

import re

from app.core.models import RetrievalKind
from app.search.limits import MAX_FUZZY_QUERY_CHARS


_IDENTIFIER_QUERY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def classify_non_exact_query(query: str) -> RetrievalKind:
    """Choose fuzzy for identifier-like text and semantic otherwise."""

    return "fuzzy" if is_identifier_like(query) else "semantic"


def is_identifier_like(query: str) -> bool:
    """Return whether a query resembles one code identifier or qualified name."""

    normalized_query = query.strip()
    if not normalized_query:
        return False
    if len(normalized_query) > MAX_FUZZY_QUERY_CHARS:
        return False
    return _IDENTIFIER_QUERY.fullmatch(normalized_query) is not None

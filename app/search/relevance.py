"""Query-aware result eligibility rules shared by retrieval modes."""

import re


_QUERY_TERM = re.compile(r"[^\W_]+", re.UNICODE)
_IMPORT_TERMS = {
    "import",
    "imported",
    "importing",
    "imports",
}
_COMMENT_TERMS = {
    "comment",
    "commentary",
    "commented",
    "commenting",
    "comments",
}
_REQUIRED_INTENT_TERMS = {
    "imports": _IMPORT_TERMS,
    "module_comment": _COMMENT_TERMS,
    "symbol_comment": _COMMENT_TERMS,
}


def is_result_kind_requested(query: str, semantic_unit_kind: str | None) -> bool:
    """Return whether a low-signal source fragment was explicitly requested."""

    required_terms = _REQUIRED_INTENT_TERMS.get(semantic_unit_kind)
    if required_terms is None:
        return True

    query_terms = {
        match.group(0).casefold() for match in _QUERY_TERM.finditer(query)
    }
    return not required_terms.isdisjoint(query_terms)

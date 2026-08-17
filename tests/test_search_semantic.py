import math
import unittest
import uuid

import numpy as np

from app.acceleration.python_backend import PythonBackend
from app.core.models import Repository, SearchRequest, Symbol
from app.search.semantic import SemanticSearchIndex, semantic_search
from app.storage.database import StoredFileSource, StoredSemanticCandidate


REPOSITORY_ID = uuid.UUID("00000000-0000-0000-0000-000000000100")


class ControlledEmbedder:
    provider = "controlled-test"
    model = "controlled-semantic"
    dimension = 2

    def embed_query(self, query: str) -> list[float]:
        del query
        return [1.0, 0.0]


class RecordingBackend(PythonBackend):
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def semantic_top_k(self, matrix, query, top_k):
        self.requested_top_k = top_k
        return super().semantic_top_k(matrix, query, top_k)


class SemanticStore:
    def __init__(
        self,
        symbols: list[Symbol],
        file_texts: dict[str, str],
    ) -> None:
        self.repository = Repository(
            id=REPOSITORY_ID,
            absolute_path="/repository",
            index_format_version="4",
            timestamp_of_index=1,
            embedding_provider=ControlledEmbedder.provider,
            embedding_model=ControlledEmbedder.model,
            embedding_dim=ControlledEmbedder.dimension,
        )
        self.symbols = {symbol.id: symbol for symbol in symbols}
        self.file_texts = file_texts

    def load_repository(self, repository_id: uuid.UUID) -> Repository | None:
        return self.repository if repository_id == self.repository.id else None

    def load_symbols_by_ids(
        self,
        symbol_ids,
        *,
        max_snippet_chars: int,
    ) -> dict[uuid.UUID, Symbol]:
        return {
            symbol_id: self.symbols[symbol_id].model_copy(
                update={
                    "source_snippet": self.symbols[symbol_id].source_snippet[
                        : max_snippet_chars + 1
                    ]
                }
            )
            for symbol_id in dict.fromkeys(symbol_ids)
        }

    def load_file_sources_by_paths(
        self,
        repository_id: uuid.UUID,
        relative_paths,
        *,
        max_snippet_chars: int,
    ) -> dict[str, StoredFileSource]:
        if repository_id != self.repository.id:
            return {}
        return {
            relative_path: StoredFileSource(
                relative_path=relative_path,
                language="python",
                line_count=max(1, len(self.file_texts[relative_path].splitlines())),
                source_text=self.file_texts[relative_path][
                    : max_snippet_chars + 1
                ],
            )
            for relative_path in dict.fromkeys(relative_paths)
        }


class SemanticResultContextTests(unittest.TestCase):
    def test_symbol_matches_are_expanded_and_deduplicated(self) -> None:
        first_symbol = _symbol(
            1,
            "authenticate",
            10,
            14,
            "def authenticate(user):\n"
            "    # Check the user credential.\n"
            "    return bool(user)\n",
        )
        second_symbol = _symbol(
            2,
            "authorize",
            20,
            22,
            "def authorize(user):\n    return user.is_admin\n",
        )
        candidates = [
            _candidate(
                100 + index,
                kind="symbol_comment",
                start_line=11 + index,
                symbol=first_symbol,
            )
            for index in range(10)
        ]
        candidates.append(
            _candidate(
                200,
                kind="symbol",
                start_line=second_symbol.start_line,
                symbol=second_symbol,
            )
        )
        similarities = [0.99 - index * 0.01 for index in range(10)] + [0.80]
        backend = RecordingBackend()

        response = semantic_search(
            SemanticStore([first_symbol, second_symbol], {}),
            REPOSITORY_ID,
            SearchRequest(query="access checks", request_mode="semantic", top_k=2),
            ControlledEmbedder(),
            search_index=_search_index(candidates, similarities),
            backend=backend,
        )

        self.assertEqual(backend.requested_top_k, len(candidates))
        self.assertEqual(
            [result.symbol_name for result in response.ranked_results],
            ["authenticate", "authorize"],
        )
        self.assertTrue(
            all(result.result_type == "symbol" for result in response.ranked_results)
        )
        self.assertEqual(
            response.ranked_results[0].snippet,
            first_symbol.source_snippet,
        )
        self.assertEqual(response.ranked_results[0].start_line, 10)
        self.assertEqual(response.ranked_results[0].end_line, 14)
        self.assertIsNone(response.ranked_results[0].semantic_unit_kind)

    def test_low_signal_fragments_require_explicit_query_intent(self) -> None:
        candidates = [
            _candidate(1, kind="imports", start_line=1, relative_path="imports.py"),
            _candidate(
                2,
                kind="module_comment",
                start_line=5,
                relative_path="comments.py",
            ),
            _candidate(
                3,
                kind="module_code",
                start_line=10,
                relative_path="module.py",
            ),
            _candidate(
                4,
                kind="documentation",
                start_line=20,
                relative_path="README.md",
            ),
        ]
        similarities = [0.90, 0.89, 0.86, 0.85]
        file_texts = {
            candidate.relative_path: f"{candidate.semantic_unit_kind} content\n"
            for candidate in candidates
        }

        response = semantic_search(
            SemanticStore([], file_texts),
            REPOSITORY_ID,
            SearchRequest(query="implementation logic", request_mode="semantic", top_k=2),
            ControlledEmbedder(),
            search_index=_search_index(candidates, similarities),
            backend=PythonBackend(),
        )

        self.assertEqual(
            [result.semantic_unit_kind for result in response.ranked_results],
            ["module_code", "documentation"],
        )
        self.assertTrue(
            all(result.result_type == "file" for result in response.ranked_results)
        )
        self.assertEqual(
            response.ranked_results[0].snippet,
            file_texts["module.py"],
        )

    def test_import_and_comment_queries_can_return_matching_fragments(self) -> None:
        candidates = [
            _candidate(1, kind="imports", start_line=1),
            _candidate(2, kind="module_comment", start_line=5),
        ]
        file_texts = {
            candidate.relative_path: f"{candidate.semantic_unit_kind} content\n"
            for candidate in candidates
        }
        search_index = _search_index(candidates, [0.90, 0.89])
        store = SemanticStore([], file_texts)

        imports_response = semantic_search(
            store,
            REPOSITORY_ID,
            SearchRequest(query="where is this imported", request_mode="semantic"),
            ControlledEmbedder(),
            search_index=search_index,
        )
        comments_response = semantic_search(
            store,
            REPOSITORY_ID,
            SearchRequest(query="find explanatory comments", request_mode="semantic"),
            ControlledEmbedder(),
            search_index=search_index,
        )

        self.assertEqual(
            [result.semantic_unit_kind for result in imports_response.ranked_results],
            ["imports"],
        )
        self.assertEqual(
            [result.semantic_unit_kind for result in comments_response.ranked_results],
            ["module_comment"],
        )

    def test_intent_terms_are_matched_as_words(self) -> None:
        candidate = _candidate(1, kind="imports", start_line=1)

        response = semantic_search(
            SemanticStore([], {candidate.relative_path: "import package\n"}),
            REPOSITORY_ID,
            SearchRequest(query="important behavior", request_mode="semantic"),
            ControlledEmbedder(),
            search_index=_search_index([candidate], [0.99]),
        )

        self.assertFalse(response.ranked_results)


def _symbol(
    identifier: int,
    name: str,
    start_line: int,
    end_line: int,
    source_snippet: str,
) -> Symbol:
    return Symbol(
        id=uuid.UUID(int=identifier),
        repository_id=REPOSITORY_ID,
        name=name,
        qualified_name=name,
        kind="function",
        relative_path="service.py",
        start_line=start_line,
        end_line=end_line,
        source_snippet=source_snippet,
        language="python",
    )


def _candidate(
    identifier: int,
    *,
    kind: str,
    start_line: int,
    symbol: Symbol | None = None,
    relative_path: str = "module.py",
) -> StoredSemanticCandidate:
    return StoredSemanticCandidate(
        chunk_id=uuid.UUID(int=1_000 + identifier),
        relative_path=symbol.relative_path if symbol is not None else relative_path,
        start_line=start_line,
        end_line=start_line,
        symbol_id=symbol.id if symbol is not None else None,
        qualified_symbol_name=(symbol.qualified_name if symbol is not None else None),
        language="python",
        semantic_unit_kind=kind,
    )


def _search_index(
    candidates: list[StoredSemanticCandidate],
    similarities: list[float],
) -> SemanticSearchIndex:
    matrix = np.asarray(
        [
            [similarity, math.sqrt(1.0 - similarity * similarity)]
            for similarity in similarities
        ],
        dtype=np.float32,
    )
    return SemanticSearchIndex(candidates=candidates, matrix=matrix)


if __name__ == "__main__":
    unittest.main()

import math
import unittest
import uuid

import numpy as np

from app.acceleration.python_backend import PythonBackend
from app.core.models import Repository, SearchRequest, Symbol
from app.search.semantic import SemanticSearchIndex, semantic_search
from app.storage.database import StoredSemanticCandidate


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
        chunk_texts: dict[uuid.UUID, str],
    ) -> None:
        self.repository = Repository(
            id=REPOSITORY_ID,
            absolute_path="/repository",
            index_format_version="3",
            timestamp_of_index=1,
            embedding_provider=ControlledEmbedder.provider,
            embedding_model=ControlledEmbedder.model,
            embedding_dim=ControlledEmbedder.dimension,
        )
        self.symbols = {symbol.id: symbol for symbol in symbols}
        self.chunk_texts = chunk_texts

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

    def load_chunk_texts(
        self,
        chunk_ids,
        *,
        max_chars: int,
    ) -> dict[uuid.UUID, str]:
        return {
            chunk_id: self.chunk_texts[chunk_id][: max_chars + 1]
            for chunk_id in dict.fromkeys(chunk_ids)
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

    def test_low_context_module_fragments_are_demoted(self) -> None:
        candidates = [
            _candidate(1, kind="imports", start_line=1),
            _candidate(2, kind="module_comment", start_line=5),
            _candidate(3, kind="module_code", start_line=10),
            _candidate(4, kind="documentation", start_line=20),
        ]
        similarities = [0.90, 0.89, 0.86, 0.85]
        chunk_texts = {
            candidate.chunk_id: f"{candidate.semantic_unit_kind} content\n"
            for candidate in candidates
        }

        response = semantic_search(
            SemanticStore([], chunk_texts),
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
        self.assertGreater(
            response.ranked_results[0].score,
            response.ranked_results[1].score,
        )


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
) -> StoredSemanticCandidate:
    return StoredSemanticCandidate(
        chunk_id=uuid.UUID(int=1_000 + identifier),
        relative_path=symbol.relative_path if symbol is not None else "module.py",
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

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from mcp.client import Client

from app.core.config import Settings
from app.core.models import IndexRepositoryResponse
from app.core.repositories import (
    REPOSITORY_UNAVAILABLE_MESSAGE,
    RepositoryResolver,
)
from app.core.runtime import FireLensRuntime
from app.indexing.embedder import FakeEmbedder
from app.indexing.indexer import IndexingProgress, ProgressCallback
from app.interfaces.mcp_server import create_mcp_server
from app.search.fuzzy import fuzzy_score, levenshtein_distance
from app.search.limits import (
    MAX_FUZZY_CANDIDATE_CHARS,
    MAX_FUZZY_QUERY_CHARS,
    MAX_RESULT_SYMBOL_NAME_CHARS,
)
from app.search.router import classify_non_exact_query


class FuzzySecurityTests(unittest.TestCase):
    def test_banded_levenshtein_returns_one_past_the_cutoff(self) -> None:
        self.assertEqual(
            levenshtein_distance("kitten", "sitting", max_distance=3),
            3,
        )
        self.assertEqual(
            levenshtein_distance("kitten", "sitting", max_distance=2),
            3,
        )
        self.assertEqual(
            levenshtein_distance("short", "much-longer", max_distance=2),
            3,
        )

    def test_fuzzy_score_rejects_overlong_inputs_before_edit_distance(self) -> None:
        with patch(
            "app.search.fuzzy.levenshtein_distance",
            side_effect=AssertionError("edit distance should not run"),
        ):
            self.assertEqual(
                fuzzy_score("q" * (MAX_FUZZY_QUERY_CHARS + 1), "candidate"),
                0.0,
            )
            self.assertEqual(
                fuzzy_score(
                    "query",
                    "c" * (MAX_FUZZY_CANDIDATE_CHARS + 1),
                ),
                0.0,
            )

    def test_auto_routing_sends_long_identifiers_to_semantic_search(self) -> None:
        self.assertEqual(
            classify_non_exact_query("a" * MAX_FUZZY_QUERY_CHARS),
            "fuzzy",
        )
        self.assertEqual(
            classify_non_exact_query("a" * (MAX_FUZZY_QUERY_CHARS + 1)),
            "semantic",
        )


class RepositoryResolverSecurityTests(unittest.TestCase):
    def test_repository_failures_use_one_public_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed_root = root / "allowed"
            outside_root = root / "outside"
            allowed_root.mkdir()
            outside_root.mkdir()
            file_path = allowed_root / "file.py"
            file_path.write_text("pass\n", encoding="utf-8")
            symlink_path = allowed_root / "outside-link"
            symlink_path.symlink_to(outside_root, target_is_directory=True)
            resolver = RepositoryResolver(
                Settings(
                    _env_file=None,
                    data_dir=root / "indexes",
                    allowed_roots=[allowed_root],
                )
            )

            invalid_paths = [
                outside_root,
                file_path,
                allowed_root / "missing",
                symlink_path,
                "x" * 4_097,
            ]
            errors = []
            for invalid_path in invalid_paths:
                with self.subTest(path=invalid_path), self.assertRaises(
                    ValueError
                ) as raised:
                    resolver.resolve(invalid_path)
                errors.append(str(raised.exception))

        self.assertEqual(errors, [REPOSITORY_UNAVAILABLE_MESSAGE] * len(errors))

    def test_lexically_outside_path_is_rejected_before_type_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed_root = root / "allowed"
            outside_root = root / "outside"
            allowed_root.mkdir()
            outside_root.mkdir()
            resolver = RepositoryResolver(
                Settings(
                    _env_file=None,
                    data_dir=root / "indexes",
                    allowed_roots=[allowed_root],
                )
            )

            with patch.object(
                Path,
                "is_dir",
                side_effect=AssertionError("outside path was type-probed"),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    f"^{REPOSITORY_UNAVAILABLE_MESSAGE}$",
                ):
                    resolver.resolve(outside_root)


class SearchResultSecurityTests(unittest.TestCase):
    def test_all_search_modes_bound_long_qualified_symbol_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            repository.mkdir()
            outer_name = "Outer" + ("a" * 2_100)
            inner_name = "Inner" + ("b" * 2_100)
            (repository / "long_name.py").write_text(
                f"class {outer_name}:\n"
                f"    class {inner_name}:\n"
                "        def target(self):\n"
                "            return 1\n",
                encoding="utf-8",
            )
            settings = Settings(
                _env_file=None,
                data_dir=root / "indexes",
                allowed_roots=[repository],
                embedding_provider=FakeEmbedder.provider,
                embedding_model=FakeEmbedder.model,
                embedding_revision=None,
                embedding_dimension=8,
            )
            runtime = FireLensRuntime(
                settings,
                embedder_factory=lambda: FakeEmbedder(dimension=8),
            )
            runtime.index_repository(repository)

            exact = runtime.search_code(
                repository,
                "target",
                mode="exact",
                backend="python",
            )
            fuzzy = runtime.search_code(
                repository,
                "targat",
                mode="fuzzy",
                backend="python",
            )
            semantic = runtime.search_code(
                repository,
                "find the nested target",
                mode="semantic",
                top_k=20,
                backend="python",
            )

        self.assertEqual(
            len(exact.ranked_results[0].symbol_name or ""),
            MAX_RESULT_SYMBOL_NAME_CHARS,
        )
        self.assertEqual(
            len(fuzzy.ranked_results[0].symbol_name or ""),
            MAX_RESULT_SYMBOL_NAME_CHARS,
        )
        semantic_name_lengths = [
            len(result.symbol_name)
            for result in semantic.ranked_results
            if result.symbol_name is not None
        ]
        self.assertIn(MAX_RESULT_SYMBOL_NAME_CHARS, semantic_name_lengths)
        self.assertTrue(
            all(
                length <= MAX_RESULT_SYMBOL_NAME_CHARS
                for length in semantic_name_lengths
            )
        )

    def test_explicit_fuzzy_search_rejects_overlong_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            repository.mkdir()
            (repository / "example.py").write_text(
                "def example():\n    return 1\n",
                encoding="utf-8",
            )
            settings = Settings(
                _env_file=None,
                data_dir=root / "indexes",
                allowed_roots=[repository],
                embedding_provider=FakeEmbedder.provider,
                embedding_model=FakeEmbedder.model,
                embedding_revision=None,
                embedding_dimension=8,
            )
            runtime = FireLensRuntime(
                settings,
                embedder_factory=lambda: FakeEmbedder(dimension=8),
            )
            runtime.index_repository(repository)

            with self.assertRaisesRegex(ValueError, "at most 256 characters"):
                runtime.search_code(
                    repository,
                    "q" * (MAX_FUZZY_QUERY_CHARS + 1),
                    mode="fuzzy",
                    backend="python",
                )

            automatic = runtime.search_code(
                repository,
                "q" * (MAX_FUZZY_QUERY_CHARS + 1),
                mode="auto",
                backend="python",
            )

        self.assertEqual(automatic.mode, "semantic")


class _HighVolumeProgressRuntime:
    def index_repository(
        self,
        repository_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        cancellation_callback=None,
    ) -> IndexRepositoryResponse:
        del cancellation_callback
        total = 10_000
        if progress_callback is not None:
            for current in range(total + 1):
                for stage in ("index", "parse", "embed"):
                    progress_callback(
                        IndexingProgress(
                            stage=stage,
                            current=current,
                            total=total,
                            message=f"{stage} item {current}",
                        )
                    )

        return IndexRepositoryResponse(
            repository_path=str(repository_path),
            database_path="/indexes/example/firelens.db",
            status="ready",
            index_format_version="1",
            timestamp_of_index=1,
            embedding_provider="test",
            embedding_model="sha256-fake",
            embedding_dim=8,
            file_count=total,
        )


class McpProgressSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_interleaved_file_progress_is_throttled_per_stage(self) -> None:
        progress_events: list[tuple[float, float | None, str | None]] = []

        async def record_progress(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            progress_events.append((progress, total, message))

        server = create_mcp_server(_HighVolumeProgressRuntime())
        async with Client(server) as client:
            result = await client.call_tool(
                "index_repository",
                {"repository_path": "/repo"},
                progress_callback=record_progress,
            )

        self.assertFalse(result.is_error)
        self.assertEqual(len(progress_events), 33)
        self.assertEqual(
            [event[0] for event in progress_events],
            list(range(1, 34)),
        )
        stages = Counter(
            (event[2] or "").split(":", maxsplit=1)[0]
            for event in progress_events
        )
        self.assertEqual(stages, {"index": 11, "parse": 11, "embed": 11})
        for stage in stages:
            self.assertTrue(
                any(
                    (event[2] or "").startswith(f"{stage}: 10000/10000")
                    for event in progress_events
                )
            )


if __name__ == "__main__":
    unittest.main()

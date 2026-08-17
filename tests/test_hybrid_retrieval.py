import tempfile
import unittest
import uuid
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings
from app.core.models import SearchResponse, SearchResult
from app.core.runtime import FireLensRuntime
from app.indexing.embedder import FakeEmbedder
from app.search.hybrid import (
    NormalizedWeightedFusionConfig,
    ReciprocalRankFusionConfig,
    deduplicate_candidates,
    normalize_scores,
    normalized_weighted_fusion,
    reciprocal_rank_fusion,
    response_candidates,
)


REPOSITORY_ID = uuid.UUID("00000000-0000-0000-0000-000000000100")


def _result(
    record_id: int,
    path: str,
    score: float,
    *,
    mode: str,
    start_line: int = 1,
    end_line: int = 2,
    result_type: str = "symbol",
    symbol_id: uuid.UUID | None = None,
    symbol_name: str | None = None,
    snippet: str = "def value():\n    pass",
) -> SearchResult:
    identifier = uuid.UUID(int=record_id)
    return SearchResult(
        id=identifier,
        result_type=result_type,
        file_path=path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name or f"value_{record_id}",
        symbol_id=symbol_id,
        snippet=snippet,
        score=score,
        mode=mode,
        backend="python",
        retrieval_channels=[mode],
        retrieval_evidence=[
            {"channel": mode, "score": score, "rank": 1, "backend": "python"}
        ],
    )


def _response(mode: str, results: list[SearchResult]) -> SearchResponse:
    return SearchResponse(
        original_query="value",
        requested_mode=mode,
        mode=mode,
        requested_backend="python",
        backend="python",
        elapsed_time=0.001,
        ranked_results=results,
    )


class FusionCalculationTests(unittest.TestCase):
    def test_rrf_golden_scores_include_missing_sources(self) -> None:
        lexical = response_candidates(
            REPOSITORY_ID,
            "lexical",
            _response(
                "lexical",
                [
                    _result(1, "a.py", 0.9, mode="lexical"),
                    _result(2, "b.py", 0.8, mode="lexical"),
                ],
            ),
        )
        semantic = response_candidates(
            REPOSITORY_ID,
            "semantic",
            _response(
                "semantic",
                [
                    _result(3, "c.py", 0.9, mode="semantic"),
                    _result(1, "a.py", 0.8, mode="semantic"),
                ],
            ),
        )
        config = ReciprocalRankFusionConfig(
            lexical_pool_size=20,
            semantic_pool_size=20,
            rrf_k=60,
            lexical_weight=1,
            semantic_weight=1,
            final_top_k=3,
            tie_breaking_version="source-location-v1",
        )

        fused = reciprocal_rank_fusion([*lexical, *semantic], config)

        self.assertEqual([item.result.file_path for item in fused], ["a.py", "c.py", "b.py"])
        self.assertAlmostEqual(
            fused[0].raw_fusion_score,
            0.5 / 61 + 0.5 / 62,
        )
        self.assertAlmostEqual(fused[1].raw_fusion_score, 0.5 / 61)
        self.assertAlmostEqual(fused[2].raw_fusion_score, 0.5 / 62)
        self.assertEqual(fused[0].result.score, 1.0)

    def test_rrf_ties_use_the_documented_source_location_order(self) -> None:
        lexical = response_candidates(
            REPOSITORY_ID,
            "lexical",
            _response("lexical", [_result(2, "b.py", 0.5, mode="lexical")]),
        )
        semantic = response_candidates(
            REPOSITORY_ID,
            "semantic",
            _response("semantic", [_result(1, "a.py", 0.5, mode="semantic")]),
        )
        config = ReciprocalRankFusionConfig(
            lexical_pool_size=1,
            semantic_pool_size=1,
            rrf_k=60,
            lexical_weight=1,
            semantic_weight=1,
            final_top_k=2,
            tie_breaking_version="source-location-v1",
        )

        first = reciprocal_rank_fusion([*lexical, *semantic], config)
        second = reciprocal_rank_fusion([*semantic, *lexical], config)

        self.assertEqual([item.result.file_path for item in first], ["a.py", "b.py"])
        self.assertEqual(
            [item.result.id for item in first],
            [item.result.id for item in second],
        )

    def test_weighted_fusion_handles_constant_ranges_and_missing_sources(self) -> None:
        lexical = response_candidates(
            REPOSITORY_ID,
            "lexical",
            _response(
                "lexical",
                [
                    _result(1, "a.py", 0.4, mode="lexical"),
                    _result(2, "b.py", 0.4, mode="lexical"),
                ],
            ),
        )
        semantic = response_candidates(
            REPOSITORY_ID,
            "semantic",
            _response("semantic", [_result(2, "b.py", 0.2, mode="semantic")]),
        )
        config = NormalizedWeightedFusionConfig(
            lexical_pool_size=2,
            semantic_pool_size=1,
            lexical_weight=1,
            semantic_weight=1,
            missing_source_value=0,
            final_top_k=2,
            tie_breaking_version="source-location-v1",
        )

        fused = normalized_weighted_fusion([*lexical, *semantic], config)

        self.assertEqual(normalize_scores([0.4, 0.4]), [1.0, 1.0])
        self.assertEqual([item.result.file_path for item in fused], ["b.py", "a.py"])
        self.assertEqual([item.result.score for item in fused], [1.0, 0.5])

    def test_symbol_and_chunk_overlap_uses_one_slot_and_richest_snippet(self) -> None:
        symbol_id = uuid.UUID(int=7)
        lexical_result = _result(
            7,
            "service.py",
            0.9,
            mode="lexical",
            start_line=4,
            end_line=8,
            symbol_id=symbol_id,
            symbol_name="Service.run",
            snippet="def run(): pass",
        )
        semantic_result = _result(
            8,
            "service.py",
            0.8,
            mode="semantic",
            start_line=4,
            end_line=8,
            result_type="chunk",
            symbol_id=symbol_id,
            symbol_name="Service.run",
            snippet="def run():\n    value = prepare()\n    return value",
        )
        candidates = [
            *response_candidates(
                REPOSITORY_ID,
                "lexical",
                _response("lexical", [lexical_result]),
            ),
            *response_candidates(
                REPOSITORY_ID,
                "semantic",
                _response("semantic", [semantic_result]),
            ),
        ]
        config = ReciprocalRankFusionConfig(
            lexical_pool_size=1,
            semantic_pool_size=1,
            rrf_k=60,
            lexical_weight=1,
            semantic_weight=1,
            final_top_k=2,
            tie_breaking_version="source-location-v1",
        )

        fused = reciprocal_rank_fusion(candidates, config)

        self.assertEqual(len(deduplicate_candidates(candidates)), 1)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].result.snippet, semantic_result.snippet)
        component_evidence = {
            item.channel: item for item in fused[0].result.retrieval_evidence
        }
        self.assertEqual(component_evidence["lexical"].rank, 1)
        self.assertEqual(component_evidence["semantic"].rank, 1)

    def test_fusion_weights_are_validated_and_normalized(self) -> None:
        config = NormalizedWeightedFusionConfig(
            lexical_pool_size=2,
            semantic_pool_size=2,
            lexical_weight=3,
            semantic_weight=1,
            final_top_k=2,
            tie_breaking_version="source-location-v1",
        )
        self.assertEqual(config.lexical_weight, 0.75)
        self.assertEqual(config.semantic_weight, 0.25)

        with self.assertRaises(ValidationError):
            NormalizedWeightedFusionConfig(
                lexical_pool_size=2,
                semantic_pool_size=2,
                lexical_weight=0,
                semantic_weight=0,
                final_top_k=2,
                tie_breaking_version="source-location-v1",
            )


class FailingQueryEmbedder(FakeEmbedder):
    def embed_query(self, query: str) -> list[float]:
        raise RuntimeError("semantic backend failed")


class HybridServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        (self.repository / "service.py").write_text(
            "def authenticate(user):\n"
            "    return user is not None\n\n"
            "def authorize(user):\n"
            "    return user is not None\n",
            encoding="utf-8",
        )
        self.settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=8,
        )
        self.runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )
        self.runtime.index_repository(self.repository)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_named_hybrid_modes_return_provenance_timings_and_stable_results(self) -> None:
        for mode, method in (
            ("hybrid_rrf", "rrf"),
            ("hybrid_weighted", "normalized_weighted"),
        ):
            with self.subTest(mode=mode):
                first = self.runtime.search_code(
                    self.repository,
                    "authentication user check",
                    mode=mode,
                    top_k=2,
                    backend="python",
                )
                second = self.runtime.search_code(
                    self.repository,
                    "authentication user check",
                    mode=mode,
                    top_k=2,
                    backend="python",
                )
                reopened_runtime = FireLensRuntime(
                    self.settings,
                    embedder_factory=lambda: FakeEmbedder(dimension=8),
                )
                reopened = reopened_runtime.search_code(
                    self.repository,
                    "authentication user check",
                    mode=mode,
                    top_k=2,
                    backend="python",
                )

                self.assertEqual(first.mode, mode)
                self.assertEqual(first.retrieval_config.split(":", 1)[0], mode)
                self.assertEqual(
                    [result.id for result in first.ranked_results],
                    [result.id for result in second.ranked_results],
                )
                self.assertEqual(
                    [result.id for result in first.ranked_results],
                    [result.id for result in reopened.ranked_results],
                )
                self.assertEqual(
                    [timing.component for timing in first.retrieval_timings],
                    ["lexical", "semantic", "fusion"],
                )
                locations = [
                    (result.file_path, result.start_line, result.end_line)
                    for result in first.ranked_results
                ]
                self.assertEqual(len(locations), len(set(locations)))
                for result in first.ranked_results:
                    self.assertEqual(result.fusion_method, method)
                    self.assertTrue(
                        {"lexical", "semantic"}
                        & set(result.retrieval_channels)
                    )

    def test_path_filter_is_applied_before_both_sources_are_fused(self) -> None:
        package = self.repository / "package"
        package.mkdir()
        (package / "auth.py").write_text(
            "def authenticate_request(request):\n"
            "    return request is not None\n",
            encoding="utf-8",
        )
        self.runtime.index_repository(self.repository)

        response = self.runtime.search_code(
            self.repository,
            "authenticate request",
            mode="hybrid_rrf",
            path="package/",
            backend="python",
        )

        self.assertTrue(response.ranked_results)
        self.assertEqual(
            {result.file_path for result in response.ranked_results},
            {"package/auth.py"},
        )

    def test_explicit_hybrid_mode_surfaces_semantic_backend_failure(self) -> None:
        failing_runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FailingQueryEmbedder(dimension=8),
        )

        with self.assertRaisesRegex(RuntimeError, "semantic backend failed"):
            failing_runtime.search_code(
                self.repository,
                "authentication user check",
                mode="hybrid_rrf",
                backend="python",
            )

    def test_hybrid_final_results_obey_count_and_total_context_limits(self) -> None:
        bounded_runtime = FireLensRuntime(
            self.settings.model_copy(update={"max_total_snippet_chars": 30}),
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )

        response = bounded_runtime.search_code(
            self.repository,
            "user authentication authorization",
            mode="hybrid_weighted",
            top_k=1,
            backend="python",
            max_snippet_chars=20,
        )

        self.assertLessEqual(len(response.ranked_results), 1)
        self.assertLessEqual(
            sum(len(result.snippet) for result in response.ranked_results),
            30,
        )
        self.assertTrue(
            all(len(result.snippet) <= 20 for result in response.ranked_results)
        )


if __name__ == "__main__":
    unittest.main()

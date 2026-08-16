import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The project package configuration intentionally includes only ``app*``.
# Make the repository-local benchmark package importable when this file is run
# directly as well as through unittest discovery.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.datasets import (
    make_exact_dataset,
    make_fuzzy_dataset,
    make_semantic_dataset,
)
from benchmarks.runner import (
    BenchmarkConfig,
    format_markdown_table,
    measure_operation,
    run_benchmarks,
)


@dataclass(frozen=True)
class RankedScores:
    indices: np.ndarray
    scores: np.ndarray


class DeterministicBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def semantic_top_k(
        self,
        matrix: np.ndarray,
        query: np.ndarray,
        top_k: int,
    ) -> RankedScores:
        scores = np.clip(matrix @ query, -1.0, 1.0)
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return RankedScores(
            indices=np.asarray(indices, dtype=np.int64),
            scores=np.asarray(scores[indices], dtype=np.float32),
        )

    def fuzzy_scores(
        self,
        query: str,
        candidates: tuple[tuple[str, str], ...],
        minimum_score: float,
    ) -> np.ndarray:
        del minimum_score
        return np.asarray(
            [
                1.0 if query in (short_name, qualified_name) else 0.0
                for short_name, qualified_name in candidates
            ],
            dtype=np.float64,
        )

    def exact_match_indices(
        self,
        query: str,
        candidates: tuple[tuple[str, str], ...],
    ) -> np.ndarray:
        return np.asarray(
            [
                index
                for index, (short_name, qualified_name) in enumerate(candidates)
                if query in (short_name, qualified_name)
            ],
            dtype=np.int64,
        )


class BenchmarkTests(unittest.TestCase):
    def test_datasets_are_deterministic(self) -> None:
        first_semantic = make_semantic_dataset(8, 4, seed=7)
        second_semantic = make_semantic_dataset(8, 4, seed=7)
        np.testing.assert_array_equal(
            first_semantic.matrix,
            second_semantic.matrix,
        )
        np.testing.assert_array_equal(
            first_semantic.query,
            second_semantic.query,
        )

        first_fuzzy = make_fuzzy_dataset(16, seed=7)
        second_fuzzy = make_fuzzy_dataset(16, seed=7)
        self.assertEqual(first_fuzzy, second_fuzzy)

        first_exact = make_exact_dataset(16, seed=7)
        second_exact = make_exact_dataset(16, seed=7)
        self.assertEqual(first_exact, second_exact)

    def test_measure_operation_records_requested_iterations(self) -> None:
        calls = 0

        def operation() -> int:
            nonlocal calls
            calls += 1
            return calls

        timing, output = measure_operation(operation, warmups=2, runs=3)

        self.assertEqual(calls, 5)
        self.assertEqual(output, 5)
        self.assertEqual(timing.warmups, 2)
        self.assertEqual(timing.runs, 3)
        self.assertEqual(len(timing.samples_ms), 3)
        self.assertLessEqual(timing.min_ms, timing.median_ms)
        self.assertLessEqual(timing.median_ms, timing.p95_ms)

    def test_full_profile_matches_release_gate_sizes(self) -> None:
        config = BenchmarkConfig.full()

        self.assertEqual(config.semantic_sizes, (1_000, 10_000, 50_000))
        self.assertEqual(config.semantic_dimension, 768)
        self.assertEqual(config.fuzzy_sizes, (128, 512))
        self.assertEqual(config.exact_sizes, (10_000, 50_000, 100_000))
        self.assertEqual(config.warmups, 5)
        self.assertEqual(config.runs, 30)

    def test_smoke_report_is_json_serializable_and_passes_parity(self) -> None:
        config = BenchmarkConfig(
            semantic_sizes=(8,),
            semantic_dimension=4,
            fuzzy_sizes=(8,),
            exact_sizes=(16,),
            top_k=3,
            warmups=0,
            runs=2,
            seed=7,
        )
        report = run_benchmarks(
            config=config,
            python_backend=DeterministicBackend("python-test"),
            comparison_backend=DeterministicBackend("mojo-test"),
        )

        encoded_report = json.dumps(report)
        self.assertTrue(encoded_report)
        self.assertEqual(
            report["summary"],
            {
                "case_count": 3,
                "compared_case_count": 3,
                "parity_passed_count": 3,
                "all_compared_cases_passed": True,
                "median_speedup": report["summary"]["median_speedup"],
            },
        )
        self.assertTrue(
            all(case["parity"]["passed"] for case in report["cases"])
        )
        self.assertTrue(
            all(
                case["reference"]["throughput"]["candidates_per_second"] > 0
                for case in report["cases"]
            )
        )
        self.assertEqual(len(report["comparison_table"]), 3)
        markdown_table = format_markdown_table(report)
        self.assertIn("| semantic | 8 |", markdown_table)
        self.assertIn("| fuzzy | 8 |", markdown_table)
        self.assertIn("| exact | 16 |", markdown_table)

    def test_report_records_missing_comparison_backend(self) -> None:
        report = run_benchmarks(
            config=BenchmarkConfig(
                semantic_sizes=(4,),
                semantic_dimension=2,
                fuzzy_sizes=(),
                exact_sizes=(),
                warmups=0,
                runs=1,
                operations=("semantic",),
            ),
            python_backend=DeterministicBackend("python-test"),
            comparison_skip_reason="Mojo library was not built",
        )

        self.assertEqual(report["summary"]["compared_case_count"], 0)
        self.assertEqual(
            report["cases"][0]["parity"],
            {
                "status": "not_run",
                "passed": None,
                "details": "Mojo library was not built",
            },
        )


if __name__ == "__main__":
    unittest.main()

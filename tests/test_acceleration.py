import unittest

import numpy as np

from app.acceleration.protocol import AccelerationBackend, RankedScores
from app.acceleration.python_backend import (
    PythonBackend,
    fuzzy_score,
    levenshtein_distance,
    normalize_identifier,
)


class PythonBackendRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = PythonBackend()

    def test_backend_satisfies_acceleration_protocol(self) -> None:
        self.assertIsInstance(self.backend, AccelerationBackend)
        self.assertTrue(self.backend.supports("semantic"))
        self.assertTrue(self.backend.supports("top_k"))
        self.assertTrue(self.backend.supports("fuzzy"))
        self.assertTrue(self.backend.supports("exact"))

    def test_top_k_orders_descending_and_preserves_stable_ties(self) -> None:
        scores = np.asarray([0.25, 0.75, 0.75, -0.5], dtype=np.float32)

        ranked = self.backend.top_k(scores, top_k=3)

        np.testing.assert_array_equal(ranked.indices, [1, 2, 0])
        np.testing.assert_array_equal(
            ranked.scores,
            np.asarray([0.75, 0.75, 0.25], dtype=np.float32),
        )
        self.assertEqual(ranked.indices.dtype, np.dtype(np.int64))
        self.assertEqual(ranked.scores.dtype, np.dtype(np.float32))

    def test_top_k_returns_every_available_score_without_clipping(self) -> None:
        scores = np.asarray([2.0, -3.0], dtype=np.float32)

        ranked = self.backend.top_k(scores, top_k=10)

        np.testing.assert_array_equal(ranked.indices, [0, 1])
        np.testing.assert_array_equal(ranked.scores, scores)

    def test_semantic_top_k_clips_before_stable_ranking(self) -> None:
        matrix = np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        query = np.asarray([1.0, 0.0], dtype=np.float32)

        ranked = self.backend.semantic_top_k(matrix, query, top_k=3)

        np.testing.assert_array_equal(ranked.indices, [0, 1, 3])
        np.testing.assert_array_equal(
            ranked.scores,
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_semantic_top_k_supports_an_empty_candidate_matrix(self) -> None:
        matrix = np.empty((0, 3), dtype=np.float32)
        query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        ranked = self.backend.semantic_top_k(matrix, query, top_k=5)

        self.assertEqual(ranked.indices.shape, (0,))
        self.assertEqual(ranked.scores.shape, (0,))

    def test_numeric_operations_validate_boundary_inputs(self) -> None:
        valid_scores = np.asarray([0.1], dtype=np.float32)
        invalid_inputs = [
            lambda: self.backend.top_k(valid_scores.astype(np.float64), 1),
            lambda: self.backend.top_k(valid_scores.reshape(1, 1), 1),
            lambda: self.backend.top_k(valid_scores, 0),
            lambda: self.backend.top_k(valid_scores, True),
            lambda: self.backend.top_k(
                np.asarray([np.nan], dtype=np.float32),
                1,
            ),
            lambda: self.backend.semantic_top_k(
                np.empty((2, 3), dtype=np.float32),
                np.empty(2, dtype=np.float32),
                1,
            ),
            lambda: self.backend.semantic_top_k(
                np.empty((2, 3), dtype=np.float32)[:, ::2],
                np.empty(2, dtype=np.float32),
                1,
            ),
        ]

        for operation in invalid_inputs:
            with self.subTest(operation=operation), self.assertRaises(
                (TypeError, ValueError)
            ):
                operation()

    def test_ranked_scores_rejects_misaligned_arrays(self) -> None:
        with self.assertRaises(ValueError):
            RankedScores(
                indices=np.asarray([0, 1], dtype=np.int64),
                scores=np.asarray([1.0], dtype=np.float32),
            )


class PythonBackendStringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = PythonBackend()

    def test_fuzzy_scores_compare_short_and_qualified_names(self) -> None:
        candidates = [
            ("loadHTTPServer", "network.loadHTTPServer"),
            ("other", "load_http_server.helper"),
            ("unrelated", "package.unrelated"),
            ("buscarValor", "módulo.buscarValor"),
        ]

        scores = self.backend.fuzzy_scores(
            "load-http-server",
            candidates,
            minimum_score=0.55,
        )

        np.testing.assert_array_equal(
            scores,
            np.asarray([1.0, 0.95, 0.0, 0.0], dtype=np.float64),
        )
        self.assertEqual(scores.dtype, np.dtype(np.float64))

    def test_fuzzy_reference_preserves_existing_normalization_and_cutoff(self) -> None:
        self.assertEqual(
            normalize_identifier("  HTTPServer.run_task  "),
            "http server run task",
        )
        self.assertEqual(fuzzy_score("run-task", "runTask"), 1.0)
        self.assertEqual(fuzzy_score("load", "loadValue"), 0.95)
        self.assertEqual(fuzzy_score("value", "loadValue"), 0.85)
        self.assertEqual(fuzzy_score("kitten", "sitting", 0.7), 0.0)
        self.assertEqual(levenshtein_distance("kitten", "sitting", 2), 3)
        self.assertEqual(fuzzy_score("Δelta", "Δelta"), 1.0)

    def test_exact_matches_are_qualified_first_and_deduplicated(self) -> None:
        candidates = [
            ("run", "package.first"),
            ("other", "run"),
            ("run", "run"),
            ("run", "package.last"),
        ]

        indexes = self.backend.exact_match_indices("  run  ", candidates)

        np.testing.assert_array_equal(indexes, [1, 2, 0, 3])
        self.assertEqual(indexes.dtype, np.dtype(np.int64))

    def test_empty_exact_query_returns_no_matches(self) -> None:
        indexes = self.backend.exact_match_indices("   ", [("", "")])

        self.assertEqual(indexes.shape, (0,))

    def test_string_operations_validate_inputs(self) -> None:
        invalid_operations = [
            lambda: self.backend.fuzzy_scores(1, [], 0.5),
            lambda: self.backend.fuzzy_scores("query", [("name", 2)], 0.5),
            lambda: self.backend.fuzzy_scores("query", [], float("nan")),
            lambda: self.backend.exact_match_indices("query", ["not-a-pair"]),
        ]

        for operation in invalid_operations:
            with self.subTest(operation=operation), self.assertRaises(
                (TypeError, ValueError)
            ):
                operation()


if __name__ == "__main__":
    unittest.main()

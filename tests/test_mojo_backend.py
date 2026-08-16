import random
import string
import unittest

import numpy as np

from app.acceleration.mojo_backend import MojoBackend
from app.acceleration.python_backend import PythonBackend


class MojoBackendParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mojo_backend, reason = MojoBackend.try_create()
        if cls.mojo_backend is None:
            raise unittest.SkipTest(reason or "Mojo backend is unavailable")
        cls.python_backend = PythonBackend()

    def test_semantic_top_k_matches_python_with_stable_ties(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        expected = self.python_backend.semantic_top_k(matrix, query, 3)
        actual = self.mojo_backend.semantic_top_k(matrix, query, 3)

        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_allclose(actual.scores, expected.scores, atol=1e-5)

    def test_randomized_semantic_results_match_python(self) -> None:
        random_generator = np.random.default_rng(20260816)
        for case_number in range(25):
            row_count = int(random_generator.integers(1, 129))
            column_count = int(random_generator.integers(1, 65))
            matrix = np.ascontiguousarray(
                random_generator.normal(size=(row_count, column_count)),
                dtype=np.float32,
            )
            query = np.ascontiguousarray(
                random_generator.normal(size=column_count),
                dtype=np.float32,
            )
            top_k = min(20, row_count)

            expected = self.python_backend.semantic_top_k(matrix, query, top_k)
            actual = self.mojo_backend.semantic_top_k(matrix, query, top_k)

            with self.subTest(case_number=case_number):
                np.testing.assert_array_equal(actual.indices, expected.indices)
                np.testing.assert_allclose(
                    actual.scores,
                    expected.scores,
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_top_k_matches_python(self) -> None:
        scores = np.asarray([0.5, 0.9, 0.9, -0.2], dtype=np.float32)

        expected = self.python_backend.top_k(scores, 3)
        actual = self.mojo_backend.top_k(scores, 3)

        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.scores, expected.scores)

    def test_fuzzy_scores_match_python_for_unicode_and_identifier_forms(self) -> None:
        candidates = [
            ("loadHTTPServer", "network.loadHTTPServer"),
            ("authenticate", "Service.authenticate"),
            ("buscarValor", "módulo.buscarValor"),
            ("Δelta", "package.Δelta"),
        ]

        for query in ("load-http-server", "authnticate", "módulo", "Δelta"):
            with self.subTest(query=query):
                expected = self.python_backend.fuzzy_scores(query, candidates, 0.55)
                actual = self.mojo_backend.fuzzy_scores(query, candidates, 0.55)
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_randomized_unicode_fuzzy_scores_match_python(self) -> None:
        random_generator = random.Random(20260816)
        alphabet = string.ascii_letters + string.digits + "_.- éΔ字İ"
        for case_number in range(20):
            query = "".join(
                random_generator.choice(alphabet)
                for _ in range(random_generator.randrange(25))
            )
            candidates = [
                (
                    "".join(
                        random_generator.choice(alphabet)
                        for _ in range(random_generator.randrange(35))
                    ),
                    "".join(
                        random_generator.choice(alphabet)
                        for _ in range(random_generator.randrange(45))
                    ),
                )
                for _ in range(32)
            ]
            minimum_score = random_generator.choice((0.0, 0.25, 0.55, 0.8, 1.0))

            expected = self.python_backend.fuzzy_scores(
                query,
                candidates,
                minimum_score,
            )
            actual = self.mojo_backend.fuzzy_scores(
                query,
                candidates,
                minimum_score,
            )

            with self.subTest(case_number=case_number):
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_exact_matches_match_python(self) -> None:
        candidates = [
            ("run", "package.first"),
            ("other", "run"),
            ("run", "run"),
            ("run", "package.last"),
        ]

        expected = self.python_backend.exact_match_indices(" run ", candidates)
        actual = self.mojo_backend.exact_match_indices(" run ", candidates)

        np.testing.assert_array_equal(actual, expected)

    def test_loaded_library_reports_the_expected_abi(self) -> None:
        self.assertEqual(self.mojo_backend.abi_version, 1)


if __name__ == "__main__":
    unittest.main()

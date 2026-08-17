import tempfile
import threading
import time
import unittest
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from app.acceleration.protocol import AccelerationError
from app.acceleration.python_backend import PythonBackend
from app.core.cancellation import OperationCancelledError
from app.core.config import Settings
from app.core.coordinator import RepositoryBusyError
from app.core.runtime import FireLensRuntime
from app.indexing.embedder import FakeEmbedder
from app.search.exact import exact_search as exact_search_implementation
from app.search.service import BackendUnavailableError
from app.search.semantic import semantic_search as semantic_search_implementation
from app.search.semantic import load_semantic_search_index
from app.storage.database import SQLiteIndexStore


class RecordingFakeEmbedder(FakeEmbedder):
    """Deterministic embedder that records indexing and query work."""

    def __init__(self, dimension: int = 8) -> None:
        super().__init__(dimension=dimension)
        self.embed_call_count = 0
        self.embedded_text_count = 0
        self.query_call_count = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_call_count += 1
        self.embedded_text_count += len(texts)
        return super().embed(texts)

    def embed_query(self, query: str) -> list[float]:
        self.query_call_count += 1
        return super().embed_query(query)


class FailingFakeEmbedder(FakeEmbedder):
    """Compatible embedder that fails only when new vectors are generated."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding failed")


class AlternateFakeEmbedder(FakeEmbedder):
    model = "alternate-fake"


class RecordingMojoBackend(PythonBackend):
    """Reference-compatible backend that records accelerated operations."""

    name = "mojo"

    def __init__(self) -> None:
        self.fuzzy_call_count = 0
        self.semantic_call_count = 0

    def fuzzy_scores(self, query, candidates, minimum_score):
        self.fuzzy_call_count += 1
        return super().fuzzy_scores(query, candidates, minimum_score)

    def semantic_top_k(self, matrix, query, top_k):
        self.semantic_call_count += 1
        return super().semantic_top_k(matrix, query, top_k)


class FailingMojoBackend(RecordingMojoBackend):
    def fuzzy_scores(self, query, candidates, minimum_score):
        raise AccelerationError("simulated Mojo failure")

    def semantic_top_k(self, matrix, query, top_k):
        raise AccelerationError("simulated Mojo failure")


class ServiceRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=8,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_status_transitions_from_missing_to_ready_to_stale_after_indexing(
        self,
    ) -> None:
        source_file = self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        embedder = RecordingFakeEmbedder()
        factory_call_count = 0

        def create_embedder() -> RecordingFakeEmbedder:
            nonlocal factory_call_count
            factory_call_count += 1
            return embedder

        runtime = FireLensRuntime(self.settings, embedder_factory=create_embedder)

        missing = runtime.get_index_status(self.repository)
        self.assertEqual(missing.status, "missing")
        self.assertEqual(factory_call_count, 0)

        progress_events = []
        indexed = runtime.index_repository(
            self.repository,
            progress_callback=progress_events.append,
        )
        self.assertEqual(indexed.status, "ready")
        self.assertEqual(indexed.embedding_provider, "test")
        self.assertEqual(indexed.file_count, 1)
        self.assertEqual(indexed.symbol_count, 1)
        self.assertEqual(indexed.chunk_count, 1)
        self.assertEqual(indexed.embedding_count, 1)
        self.assertEqual(factory_call_count, 1)
        self.assertEqual(embedder.embedded_text_count, 1)
        self.assertEqual(progress_events[-1].stage, "complete")

        ready = runtime.get_index_status(self.repository)
        self.assertEqual(ready.status, "ready")
        self.assertEqual(ready.changed_paths, [])
        self.assertEqual(factory_call_count, 1)

        source_file.write_text(
            "def authenticate(user):\n"
            "    return bool(user and user.is_active)\n",
            encoding="utf-8",
        )
        stale = runtime.get_index_status(self.repository)
        self.assertEqual(stale.status, "stale")
        self.assertEqual(stale.changed_file_count, 1)
        self.assertEqual(stale.changed_paths, ["service.py"])
        self.assertEqual(factory_call_count, 1)

    def test_auto_search_uses_exact_first_without_loading_an_embedder(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        factory_call_count = 0

        def create_embedder() -> FakeEmbedder:
            nonlocal factory_call_count
            factory_call_count += 1
            raise AssertionError("exact search must not create an embedder")

        runtime = FireLensRuntime(self.settings, embedder_factory=create_embedder)

        status = runtime.get_index_status(self.repository)
        response = runtime.search_code(
            self.repository,
            "authenticate",
            mode="auto",
            backend="python",
        )

        self.assertEqual(status.status, "ready")
        self.assertEqual(response.requested_mode, "auto")
        self.assertEqual(response.mode, "exact")
        self.assertEqual(
            [result.symbol_name for result in response.ranked_results],
            ["authenticate"],
        )
        self.assertEqual(factory_call_count, 0)

    def test_auto_search_routes_identifier_typo_to_fuzzy(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        factory_call_count = 0

        def create_embedder() -> FakeEmbedder:
            nonlocal factory_call_count
            factory_call_count += 1
            raise AssertionError("fuzzy search must not create an embedder")

        runtime = FireLensRuntime(self.settings, embedder_factory=create_embedder)

        response = runtime.search_code(
            self.repository,
            "authentcate",
            mode="auto",
            backend="python",
        )

        self.assertEqual(response.mode, "fuzzy")
        self.assertEqual(response.ranked_results[0].symbol_name, "authenticate")
        self.assertGreaterEqual(response.ranked_results[0].score, 0.55)
        self.assertEqual(factory_call_count, 0)

    def test_auto_search_falls_back_when_fuzzy_candidate_limit_is_exceeded(
        self,
    ) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n\n"
            "def authorize(user):\n"
            "    return user is not None\n",
        )
        limited_settings = self.settings.model_copy(
            update={"max_fuzzy_candidates": 1}
        )
        runtime = FireLensRuntime(
            limited_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )
        runtime.index_repository(self.repository)

        response = runtime.search_code(
            self.repository,
            "authentcate",
            mode="auto",
            backend="python",
        )

        self.assertEqual(response.mode, "semantic")
        self.assertIn(
            "Fuzzy candidate limit exceeded; used semantic search",
            response.warnings,
        )

    def test_auto_search_routes_natural_language_to_semantic(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        embedder = RecordingFakeEmbedder()
        factory_call_count = 0

        def create_embedder() -> RecordingFakeEmbedder:
            nonlocal factory_call_count
            factory_call_count += 1
            return embedder

        runtime = FireLensRuntime(self.settings, embedder_factory=create_embedder)

        response = runtime.search_code(
            self.repository,
            "where is authentication checked",
            mode="auto",
            backend="python",
        )

        self.assertEqual(response.mode, "semantic")
        self.assertEqual(len(response.ranked_results), 1)
        self.assertEqual(response.ranked_results[0].file_path, "service.py")
        self.assertEqual(factory_call_count, 1)
        self.assertEqual(embedder.query_call_count, 1)

    def test_search_omits_source_units_that_do_not_fit_output_limits(self) -> None:
        for file_name in ("a.py", "b.py", "c.py"):
            self._write_source(
                file_name,
                "def shared():\n"
                "    payload = 'abcdefghijklmnopqrstuvwxyz0123456789'\n"
                "    return payload\n",
            )

        bounded_settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=8,
            max_total_snippet_chars=50,
        )
        runtime = FireLensRuntime(
            bounded_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )
        runtime.index_repository(self.repository)

        response = runtime.search_code(
            self.repository,
            "shared",
            mode="exact",
            top_k=3,
            backend="python",
            max_snippet_chars=30,
        )

        self.assertFalse(response.ranked_results)
        self.assertIn(
            "Omitted 3 result(s) because complete source did not fit the "
            "configured output limits",
            response.warnings,
        )

    def test_search_path_filter_matches_a_file_or_directory_prefix(self) -> None:
        self._write_source(
            "package/inside.py",
            "def locate():\n"
            "    return 'inside'\n",
        )
        self._write_source(
            "package_extra.py",
            "def locate():\n"
            "    return 'similarly named file'\n",
        )
        self._write_source(
            "outside.py",
            "def locate():\n"
            "    return 'outside'\n",
        )
        runtime = self._seed_index()

        directory_response = runtime.search_code(
            self.repository,
            "locate",
            mode="exact",
            path="package/",
            backend="python",
        )
        file_response = runtime.search_code(
            self.repository,
            "locate",
            mode="exact",
            path="outside.py",
            backend="python",
        )

        self.assertEqual(
            [result.file_path for result in directory_response.ranked_results],
            ["package/inside.py"],
        )
        self.assertEqual(
            [result.file_path for result in file_response.ranked_results],
            ["outside.py"],
        )

    def test_explicit_mojo_backend_is_rejected_when_unavailable(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        unavailable_settings = self.settings.model_copy(
            update={"mojo_library_path": self.root / "missing-mojo-library"}
        )
        runtime = FireLensRuntime(
            unavailable_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )

        with self.assertRaisesRegex(
            BackendUnavailableError,
            "Mojo backend is not available",
        ):
            runtime.search_code(
                self.repository,
                "authenticate",
                mode="exact",
                backend="mojo",
            )

    def test_explicit_mojo_backend_executes_fuzzy_and_semantic_kernels(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        backend = RecordingMojoBackend()
        runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
            mojo_backend=backend,
        )

        fuzzy_response = runtime.search_code(
            self.repository,
            "authnticate",
            mode="fuzzy",
            backend="mojo",
        )
        semantic_response = runtime.search_code(
            self.repository,
            "where is authentication checked",
            mode="semantic",
            backend="mojo",
        )

        self.assertEqual(fuzzy_response.backend, "mojo")
        self.assertEqual(semantic_response.backend, "mojo")
        self.assertGreater(backend.fuzzy_call_count, 0)
        self.assertGreater(backend.semantic_call_count, 0)

    def test_automatic_mojo_failure_falls_back_to_python(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        runtime = FireLensRuntime(
            self.settings.model_copy(update={"mojo_fuzzy_min_candidates": 1}),
            embedder_factory=lambda: FakeEmbedder(dimension=8),
            mojo_backend=FailingMojoBackend(),
        )

        response = runtime.search_code(
            self.repository,
            "authnticate",
            mode="fuzzy",
            backend="auto",
        )

        self.assertEqual(response.backend, "python")
        self.assertIn(
            "Mojo fuzzy acceleration failed; using Python",
            response.warnings,
        )

    def test_automatic_semantic_mojo_failure_falls_back_to_python(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        accelerated_settings = self.settings.model_copy(
            update={"mojo_semantic_min_candidates": 1}
        )
        runtime = FireLensRuntime(
            accelerated_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
            mojo_backend=FailingMojoBackend(),
        )

        response = runtime.search_code(
            self.repository,
            "where is authentication checked",
            mode="semantic",
            backend="auto",
        )

        self.assertEqual(response.backend, "python")
        self.assertIn(
            "Mojo semantic acceleration failed; using Python",
            response.warnings,
        )

    def test_automatic_backend_keeps_small_workloads_in_python(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        backend = RecordingMojoBackend()
        runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
            mojo_backend=backend,
        )

        fuzzy_response = runtime.search_code(
            self.repository,
            "authnticate",
            mode="fuzzy",
            backend="auto",
        )
        semantic_response = runtime.search_code(
            self.repository,
            "where is authentication checked",
            mode="semantic",
            backend="auto",
        )

        self.assertEqual(fuzzy_response.backend, "python")
        self.assertEqual(semantic_response.backend, "python")
        self.assertEqual(backend.fuzzy_call_count, 0)
        self.assertEqual(backend.semantic_call_count, 0)

    def test_automatic_backend_uses_configured_crossover_thresholds(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        backend = RecordingMojoBackend()
        accelerated_settings = self.settings.model_copy(
            update={
                "mojo_fuzzy_min_candidates": 1,
                "mojo_semantic_min_candidates": 1,
            }
        )
        runtime = FireLensRuntime(
            accelerated_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
            mojo_backend=backend,
        )

        fuzzy_response = runtime.search_code(
            self.repository,
            "authnticate",
            mode="fuzzy",
            backend="auto",
        )
        semantic_response = runtime.search_code(
            self.repository,
            "where is authentication checked",
            mode="semantic",
            backend="auto",
        )

        self.assertEqual(fuzzy_response.backend, "mojo")
        self.assertEqual(semantic_response.backend, "mojo")
        self.assertEqual(backend.fuzzy_call_count, 1)
        self.assertEqual(backend.semantic_call_count, 1)

    def test_explicit_mojo_exact_search_reports_benchmark_only_capability(
        self,
    ) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        self._seed_index()
        runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
            mojo_backend=RecordingMojoBackend(),
        )

        with self.assertRaisesRegex(BackendUnavailableError, "benchmark-only"):
            runtime.search_code(
                self.repository,
                "authenticate",
                mode="exact",
                backend="mojo",
            )

    def test_status_reports_indexing_and_search_is_rejected_while_busy(self) -> None:
        self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        runtime = self._seed_index()
        coordinator = runtime.index_service.coordinator
        self.assertIs(coordinator, runtime.search_service.coordinator)

        with coordinator.indexing(self.repository):
            status = runtime.get_index_status(self.repository)
            self.assertEqual(status.status, "indexing")

            with self.assertRaisesRegex(
                RepositoryBusyError,
                "currently being indexed",
            ):
                runtime.search_code(
                    self.repository,
                    "authenticate",
                    mode="exact",
                    backend="python",
                )

    def test_embedding_failure_preserves_the_previous_database(self) -> None:
        source_file = self._write_source(
            "service.py",
            "def stable():\n"
            "    return 'old index'\n",
        )
        initial_runtime = self._seed_index()
        initial_status = initial_runtime.get_index_status(self.repository)
        database_path = Path(initial_status.database_path)
        database_before_failure = database_path.read_bytes()

        source_file.write_text(
            "def replacement():\n"
            "    return 'new source'\n",
            encoding="utf-8",
        )
        failing_runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FailingFakeEmbedder(dimension=8),
        )

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            failing_runtime.index_repository(self.repository)

        self.assertEqual(database_path.read_bytes(), database_before_failure)

        preserved_runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )
        old_result = preserved_runtime.search_code(
            self.repository,
            "stable",
            mode="exact",
            backend="python",
        )
        new_result = preserved_runtime.search_code(
            self.repository,
            "replacement",
            mode="exact",
            backend="python",
        )
        stale_status = preserved_runtime.get_index_status(self.repository)

        self.assertEqual(len(old_result.ranked_results), 1)
        self.assertEqual(new_result.ranked_results, [])
        self.assertEqual(stale_status.status, "stale")
        self.assertEqual(stale_status.changed_paths, ["service.py"])

    def test_embedding_configuration_change_rebuilds_one_index_version(self) -> None:
        self._write_source(
            "service.py",
            "def stable():\n"
            "    return 'indexed'\n",
        )
        initial_runtime = self._seed_index()
        database_path = Path(
            initial_runtime.get_index_status(self.repository).database_path
        )

        alternate_settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=AlternateFakeEmbedder.provider,
            embedding_model=AlternateFakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=12,
        )
        def fail_if_embedder_is_loaded() -> FakeEmbedder:
            raise AssertionError("status must not load an embedder")

        status_runtime = FireLensRuntime(
            alternate_settings,
            embedder_factory=fail_if_embedder_is_loaded,
        )
        stale_status = status_runtime.get_index_status(self.repository)
        self.assertEqual(stale_status.status, "stale")
        self.assertTrue(
            any(
                "embedding configuration" in warning
                for warning in stale_status.warnings
            )
        )

        alternate_runtime = FireLensRuntime(
            alternate_settings,
            embedder_factory=lambda: AlternateFakeEmbedder(dimension=12),
        )
        report = alternate_runtime.index_repository(self.repository)
        repositories = SQLiteIndexStore(database_path).list_repositories()

        self.assertEqual(report.embedding_model, "alternate-fake")
        self.assertEqual(report.embedding_dim, 12)
        self.assertEqual(report.embedded_chunk_count, 1)
        self.assertEqual(len(repositories), 1)
        self.assertEqual(repositories[0].embedding_model, "alternate-fake")
        self.assertEqual(repositories[0].embedding_dim, 12)

    def test_embedding_dimension_change_marks_status_stale_without_model_load(
        self,
    ) -> None:
        self._write_source(
            "service.py",
            "def stable():\n    return 'indexed'\n",
        )
        self._seed_index()
        changed_dimension_settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=12,
        )

        def fail_if_embedder_is_loaded() -> FakeEmbedder:
            raise AssertionError("status must not load an embedder")

        runtime = FireLensRuntime(
            changed_dimension_settings,
            embedder_factory=fail_if_embedder_is_loaded,
        )

        status = runtime.get_index_status(self.repository)

        self.assertEqual(status.status, "stale")
        self.assertEqual(status.embedding_dim, 8)
        self.assertTrue(
            any("embedding configuration" in warning for warning in status.warnings)
        )

    def test_index_rejects_an_embedder_with_the_wrong_configured_dimension(
        self,
    ) -> None:
        self._write_source(
            "service.py",
            "def stable():\n    return 'indexed'\n",
        )
        mismatched_settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=12,
        )
        runtime = FireLensRuntime(
            mismatched_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )

        with self.assertRaisesRegex(ValueError, "FIRELENS_EMBEDDING_DIMENSION"):
            runtime.index_repository(self.repository)

        self.assertEqual(runtime.get_index_status(self.repository).status, "missing")

    def test_configured_repository_file_limit_is_enforced_by_service(self) -> None:
        self._write_source("one.py", "def one():\n    return 1\n")
        self._write_source("two.py", "def two():\n    return 2\n")
        limited_settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=8,
            max_repository_files=1,
        )
        runtime = FireLensRuntime(
            limited_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )

        with self.assertRaisesRegex(ValueError, "file limit"):
            runtime.index_repository(self.repository)

    def test_index_waits_for_same_repository_search_to_finish(self) -> None:
        self._write_source("service.py", "def stable():\n    return 1\n")
        runtime = self._seed_index()
        search_started = threading.Event()
        release_search = threading.Event()

        def blocked_exact_search(*args, **kwargs):
            search_started.set()
            if not release_search.wait(timeout=2):
                raise TimeoutError("search was not released")
            return exact_search_implementation(*args, **kwargs)

        with patch("app.search.service.exact_search", side_effect=blocked_exact_search):
            with ThreadPoolExecutor(max_workers=2) as executor:
                search_future = executor.submit(
                    runtime.search_code,
                    self.repository,
                    "stable",
                    "exact",
                    5,
                    None,
                    "python",
                )
                self.assertTrue(search_started.wait(timeout=1))
                index_future = executor.submit(
                    runtime.index_repository,
                    self.repository,
                )

                deadline = time.monotonic() + 1
                while (
                    not runtime.index_service.coordinator.is_indexing(self.repository)
                    and time.monotonic() < deadline
                ):
                    threading.Event().wait(0.01)

                self.assertTrue(
                    runtime.index_service.coordinator.is_indexing(self.repository)
                )
                self.assertFalse(index_future.done())
                release_search.set()
                self.assertEqual(search_future.result(timeout=2).mode, "exact")
                self.assertEqual(index_future.result(timeout=2).status, "ready")

    def test_index_of_unrelated_repository_runs_during_search(self) -> None:
        self._write_source("service.py", "def stable():\n    return 1\n")
        second_repository = self.root / "second"
        second_repository.mkdir()
        (second_repository / "other.py").write_text(
            "def other():\n    return 2\n",
            encoding="utf-8",
        )
        runtime = self._seed_index()
        search_started = threading.Event()
        release_search = threading.Event()

        def blocked_exact_search(*args, **kwargs):
            search_started.set()
            if not release_search.wait(timeout=2):
                raise TimeoutError("search was not released")
            return exact_search_implementation(*args, **kwargs)

        try:
            with patch(
                "app.search.service.exact_search",
                side_effect=blocked_exact_search,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    search_future = executor.submit(
                        runtime.search_code,
                        self.repository,
                        "stable",
                        "exact",
                        5,
                        None,
                        "python",
                    )
                    self.assertTrue(search_started.wait(timeout=1))
                    index_future = executor.submit(
                        runtime.index_repository,
                        second_repository,
                    )
                    self.assertEqual(index_future.result(timeout=2).status, "ready")
                    self.assertFalse(search_future.done())
                    release_search.set()
                    self.assertEqual(search_future.result(timeout=2).mode, "exact")
        finally:
            release_search.set()

    def test_unrelated_semantic_searches_are_not_globally_serialized(self) -> None:
        self._write_source("service.py", "def stable():\n    return 1\n")
        second_repository = self.root / "second"
        second_repository.mkdir()
        (second_repository / "other.py").write_text(
            "def other():\n    return 2\n",
            encoding="utf-8",
        )
        runtime = self._seed_index()
        runtime.index_repository(second_repository)
        both_searches = threading.Barrier(2)

        def synchronized_semantic_search(*args, **kwargs):
            both_searches.wait(timeout=2)
            return semantic_search_implementation(*args, **kwargs)

        with patch(
            "app.search.service.semantic_search",
            side_effect=synchronized_semantic_search,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(
                    runtime.search_code,
                    self.repository,
                    "find stable behavior",
                    "semantic",
                    5,
                    None,
                    "python",
                )
                second = executor.submit(
                    runtime.search_code,
                    second_repository,
                    "find other behavior",
                    "semantic",
                    5,
                    None,
                    "python",
                )
                self.assertEqual(first.result(timeout=3).mode, "semantic")
                self.assertEqual(second.result(timeout=3).mode, "semantic")

    def test_same_key_semantic_cache_misses_share_one_matrix_load(self) -> None:
        self._write_source("service.py", "def stable():\n    return 1\n")
        runtime = self._seed_index()
        load_started = threading.Event()
        release_load = threading.Event()
        call_count_lock = threading.Lock()
        load_call_count = 0

        def blocked_load(*args, **kwargs):
            nonlocal load_call_count
            with call_count_lock:
                load_call_count += 1
            load_started.set()
            if not release_load.wait(timeout=2):
                raise TimeoutError("semantic matrix load was not released")
            return load_semantic_search_index(*args, **kwargs)

        try:
            with patch(
                "app.search.service.load_semantic_search_index",
                side_effect=blocked_load,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(
                        runtime.search_code,
                        self.repository,
                        "find stable behavior",
                        "semantic",
                        5,
                        None,
                        "python",
                    )
                    self.assertTrue(load_started.wait(timeout=1))
                    second = executor.submit(
                        runtime.search_code,
                        self.repository,
                        "locate stable behavior",
                        "semantic",
                        5,
                        None,
                        "python",
                    )
                    time.sleep(0.05)
                    with call_count_lock:
                        self.assertEqual(load_call_count, 1)

                    release_load.set()
                    self.assertEqual(first.result(timeout=2).mode, "semantic")
                    self.assertEqual(second.result(timeout=2).mode, "semantic")
        finally:
            release_load.set()

        self.assertEqual(load_call_count, 1)

    def test_semantic_cache_bounds_candidate_metadata_across_path_keys(self) -> None:
        for file_name in ("one.py", "two.py", "three.py"):
            self._write_source(
                file_name,
                f"def {file_name.removesuffix('.py')}():\n    return 1\n",
            )
        limited_settings = self.settings.model_copy(
            update={"max_semantic_candidates": 2}
        )
        runtime = FireLensRuntime(
            limited_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )
        runtime.index_repository(self.repository)

        for file_name in ("one.py", "two.py", "three.py"):
            response = runtime.search_code(
                self.repository,
                f"find behavior in {file_name}",
                mode="semantic",
                path=file_name,
                backend="python",
            )
            self.assertEqual(response.mode, "semantic")

        cached_indexes = runtime.search_service._semantic_cache.values()
        self.assertLessEqual(
            sum(len(index.candidates) for index in cached_indexes),
            limited_settings.max_semantic_candidates,
        )

    def test_active_semantic_matrix_blocks_and_cancels_conflicting_load(self) -> None:
        self._write_source("one.py", "def one():\n    return 1\n")
        self._write_source("two.py", "def two():\n    return 2\n")
        limited_settings = self.settings.model_copy(
            update={"max_semantic_candidates": 1}
        )
        runtime = FireLensRuntime(
            limited_settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )
        runtime.index_repository(self.repository)
        first_search_started = threading.Event()
        release_first_search = threading.Event()
        second_search_reached_ranking = threading.Event()
        cancel_second_search = threading.Event()

        def blocked_semantic_search(*args, **kwargs):
            request = args[2]
            if request.path == "one.py":
                first_search_started.set()
                if not release_first_search.wait(timeout=2):
                    raise TimeoutError("first semantic search was not released")
            else:
                second_search_reached_ranking.set()
            return semantic_search_implementation(*args, **kwargs)

        try:
            with patch(
                "app.search.service.semantic_search",
                side_effect=blocked_semantic_search,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(
                        runtime.search_code,
                        self.repository,
                        "find one behavior",
                        "semantic",
                        5,
                        "one.py",
                        "python",
                    )
                    self.assertTrue(first_search_started.wait(timeout=1))
                    second = executor.submit(
                        runtime.search_code,
                        self.repository,
                        "find two behavior",
                        "semantic",
                        5,
                        "two.py",
                        "python",
                        2_000,
                        cancel_second_search.is_set,
                    )
                    time.sleep(0.1)
                    self.assertFalse(second.done())
                    self.assertFalse(second_search_reached_ranking.is_set())

                    cancel_second_search.set()
                    with self.assertRaises(OperationCancelledError):
                        second.result(timeout=1)

                    release_first_search.set()
                    self.assertEqual(first.result(timeout=2).mode, "semantic")
        finally:
            release_first_search.set()

    def test_semantic_matrix_is_cached_and_invalidated_after_indexing(self) -> None:
        source_file = self._write_source(
            "service.py",
            "def authenticate(user):\n"
            "    return user is not None\n",
        )
        runtime = self._seed_index()

        with patch(
            "app.search.service.load_semantic_search_index",
            wraps=load_semantic_search_index,
        ) as load_index:
            runtime.search_code(
                self.repository,
                "where is authentication checked",
                mode="semantic",
                backend="python",
            )
            runtime.search_code(
                self.repository,
                "how is user access checked",
                mode="semantic",
                backend="python",
            )
            self.assertEqual(load_index.call_count, 1)

            source_file.write_text(
                "def authenticate(user):\n"
                "    return bool(user and user.is_active)\n",
                encoding="utf-8",
            )
            runtime.index_repository(self.repository)
            runtime.search_code(
                self.repository,
                "where is authentication checked",
                mode="semantic",
                backend="python",
            )

        self.assertEqual(load_index.call_count, 2)

    def _seed_index(self) -> FireLensRuntime:
        runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )
        runtime.index_repository(self.repository)
        return runtime

    def _write_source(self, relative_path: str, source: str) -> Path:
        destination = self.repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
        return destination


if __name__ == "__main__":
    unittest.main()

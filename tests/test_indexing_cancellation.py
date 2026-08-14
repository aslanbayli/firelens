import asyncio
import concurrent.futures
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from app.core.cancellation import OperationCancelledError
from app.core.config import Settings
from app.core.coordinator import RepositoryCoordinator
from app.core.runtime import FireLensRuntime
from app.indexing.embedder import FakeEmbedder
from app.indexing.indexer import (
    CancellationCallback,
    IndexingCancelledError,
    IndexingProgress,
    ProgressCallback,
    index_to_sqlite,
    raise_if_indexing_cancelled,
)
from app.interfaces.mcp_server import (
    _get_index_status_with_cancellation,
    _index_repository_with_cancellation,
    _search_code_with_cancellation,
    _wait_for_progress_delivery,
)
from app.storage.database import SQLiteIndexStore
from app.storage.locking import exclusive_database_lock, shared_database_lock


class IndexingCancellationTests(unittest.TestCase):
    def test_cancellation_before_promotion_preserves_previous_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repository = temp_path / "repository"
            repository.mkdir()
            source_file = repository / "service.py"
            source_file.write_text(
                "def original():\n    return 1\n",
                encoding="utf-8",
            )
            database_path = temp_path / "indexes" / "firelens.db"
            first_report = index_to_sqlite(
                repository,
                FakeEmbedder(dimension=8),
                database_path,
            )
            original_database = database_path.read_bytes()
            source_file.write_text(
                "def replacement():\n    return 2\n",
                encoding="utf-8",
            )

            cancellation_requested = threading.Event()

            def cancel_at_promotion(event: IndexingProgress) -> None:
                if event.stage == "promote" and event.current == 0:
                    cancellation_requested.set()

            with self.assertRaisesRegex(
                IndexingCancelledError,
                "indexing was cancelled",
            ):
                index_to_sqlite(
                    repository,
                    FakeEmbedder(dimension=8),
                    database_path,
                    progress_callback=cancel_at_promotion,
                    cancellation_callback=cancellation_requested.is_set,
                )

            store = SQLiteIndexStore(database_path)
            symbols = store.load_all_symbols(first_report.repository.id)
            staged_files = list(database_path.parent.glob(".firelens-*.tmp"))
            preserved_database = database_path.read_bytes()

        self.assertEqual(preserved_database, original_database)
        self.assertEqual(
            [symbol.qualified_name for symbol in symbols],
            ["original"],
        )
        self.assertEqual(staged_files, [])

    def test_cancellation_interrupts_an_in_process_coordinator_wait(self) -> None:
        coordinator = RepositoryCoordinator()
        cancellation_requested = threading.Event()

        def wait_for_indexing_lease() -> None:
            def check_cancellation() -> None:
                raise_if_indexing_cancelled(cancellation_requested.is_set)

            with coordinator.indexing(
                "/repository",
                cancellation_check=check_cancellation,
            ):
                self.fail("cancelled indexing unexpectedly acquired the lease")

        with coordinator.searching("/repository"):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(wait_for_indexing_lease)
                deadline = time.monotonic() + 1.0
                while not coordinator.is_indexing("/repository"):
                    if time.monotonic() >= deadline:
                        self.fail("indexing did not begin waiting for the lease")
                    time.sleep(0.001)

                cancellation_requested.set()
                with self.assertRaises(IndexingCancelledError):
                    future.result(timeout=1.0)

        self.assertFalse(coordinator.is_indexing("/repository"))
        with coordinator.indexing("/repository"):
            pass

    def test_cancellation_interrupts_a_cross_process_lock_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repository = temp_path / "repository"
            repository.mkdir()
            (repository / "service.py").write_text(
                "def service():\n    return 1\n",
                encoding="utf-8",
            )
            database_path = temp_path / "indexes" / "firelens.db"
            cancellation_requested = threading.Event()
            lock_wait_started = threading.Event()
            check_count = 0
            check_count_lock = threading.Lock()

            def cancellation_callback() -> bool:
                nonlocal check_count
                with check_count_lock:
                    check_count += 1
                    if check_count >= 3:
                        lock_wait_started.set()
                return cancellation_requested.is_set()

            with shared_database_lock(database_path):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        index_to_sqlite,
                        repository,
                        FakeEmbedder(dimension=8),
                        database_path,
                        cancellation_callback=cancellation_callback,
                    )
                    self.assertTrue(lock_wait_started.wait(timeout=1.0))
                    cancellation_requested.set()
                    with self.assertRaises(IndexingCancelledError):
                        future.result(timeout=1.0)

            self.assertFalse(database_path.exists())


class _BlockingRuntime:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cleaned_up = threading.Event()

    def index_repository(
        self,
        repository_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> None:
        del repository_path, progress_callback
        if cancellation_callback is None:
            raise AssertionError("MCP did not supply a cancellation callback")

        self.started.set()
        deadline = time.monotonic() + 1.0
        while not cancellation_callback():
            if time.monotonic() >= deadline:
                raise AssertionError("MCP did not signal worker cancellation")
            time.sleep(0.001)

        self.cleaned_up.set()
        raise IndexingCancelledError("Repository indexing was cancelled")


class _UnusedProgressContext:
    async def report_progress(
        self,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        del progress, total, message


class _BlockingReadRuntime:
    def __init__(self) -> None:
        self.status_started = threading.Event()
        self.status_cleaned_up = threading.Event()
        self.search_started = threading.Event()
        self.search_cleaned_up = threading.Event()

    @staticmethod
    def _block_until_cancelled(
        cancellation_callback: CancellationCallback | None,
        started: threading.Event,
        cleaned_up: threading.Event,
    ) -> None:
        if cancellation_callback is None:
            raise AssertionError("MCP did not supply a cancellation callback")
        started.set()
        deadline = time.monotonic() + 1.0
        while not cancellation_callback():
            if time.monotonic() >= deadline:
                raise AssertionError("MCP did not signal worker cancellation")
            time.sleep(0.001)
        cleaned_up.set()
        raise OperationCancelledError("FireLens operation was cancelled")

    def get_index_status(
        self,
        repository_path: str | Path,
        cancellation_callback: CancellationCallback | None = None,
    ) -> None:
        del repository_path
        self._block_until_cancelled(
            cancellation_callback,
            self.status_started,
            self.status_cleaned_up,
        )

    def search_code(
        self,
        repository_path: str | Path,
        query: str,
        mode: str = "auto",
        top_k: int = 5,
        path: str | None = None,
        backend: str = "auto",
        max_snippet_chars: int = 2_000,
        cancellation_callback: CancellationCallback | None = None,
    ) -> None:
        del repository_path, query, mode, top_k, path, backend, max_snippet_chars
        self._block_until_cancelled(
            cancellation_callback,
            self.search_started,
            self.search_cleaned_up,
        )


class McpIndexingCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_cancellation_signals_and_awaits_worker_cleanup(self) -> None:
        runtime = _BlockingRuntime()
        task = asyncio.create_task(
            _index_repository_with_cancellation(
                runtime,  # type: ignore[arg-type]
                "/repository",
                _UnusedProgressContext(),  # type: ignore[arg-type]
            )
        )
        self.assertTrue(await asyncio.to_thread(runtime.started.wait, 1.0))

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            async with asyncio.timeout(2.0):
                await task

        self.assertTrue(runtime.cleaned_up.is_set())

    async def test_mcp_status_cancellation_awaits_worker_cleanup(self) -> None:
        runtime = _BlockingReadRuntime()
        task = asyncio.create_task(
            _get_index_status_with_cancellation(
                runtime,  # type: ignore[arg-type]
                "/repository",
            )
        )
        self.assertTrue(await asyncio.to_thread(runtime.status_started.wait, 1.0))

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            async with asyncio.timeout(2.0):
                await task

        self.assertTrue(runtime.status_cleaned_up.is_set())

    async def test_mcp_search_cancellation_awaits_worker_cleanup(self) -> None:
        runtime = _BlockingReadRuntime()
        task = asyncio.create_task(
            _search_code_with_cancellation(
                runtime,  # type: ignore[arg-type]
                repository_path="/repository",
                query="service",
                mode="exact",
                top_k=5,
                path=None,
                backend="python",
                max_snippet_chars=2_000,
            )
        )
        self.assertTrue(await asyncio.to_thread(runtime.search_started.wait, 1.0))

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            async with asyncio.timeout(2.0):
                await task

        self.assertTrue(runtime.search_cleaned_up.is_set())

    async def test_cancelled_real_search_releases_repository_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            repository.mkdir()
            (repository / "service.py").write_text(
                "def service():\n    return 1\n",
                encoding="utf-8",
            )
            runtime = self._build_runtime(root)
            runtime.index_repository(repository)
            search_started = threading.Event()

            def blocked_exact_search(
                *args,
                cancellation_callback: CancellationCallback | None = None,
                **kwargs,
            ):
                del args, kwargs
                if cancellation_callback is None:
                    raise AssertionError("search did not receive cancellation")
                search_started.set()
                while not cancellation_callback():
                    time.sleep(0.001)
                raise OperationCancelledError("FireLens operation was cancelled")

            with patch(
                "app.search.service.exact_search",
                side_effect=blocked_exact_search,
            ):
                task = asyncio.create_task(
                    _search_code_with_cancellation(
                        runtime,
                        repository_path=str(repository),
                        query="service",
                        mode="exact",
                        top_k=5,
                        path=None,
                        backend="python",
                        max_snippet_chars=2_000,
                    )
                )
                self.assertTrue(
                    await asyncio.to_thread(search_started.wait, 1.0)
                )
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    async with asyncio.timeout(2.0):
                        await task

            with runtime.search_service.coordinator.indexing(repository):
                pass

    async def test_cancelled_real_status_releases_database_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repository"
            repository.mkdir()
            (repository / "service.py").write_text(
                "def service():\n    return 1\n",
                encoding="utf-8",
            )
            runtime = self._build_runtime(root)
            indexed = runtime.index_repository(repository)
            status_started = threading.Event()

            def blocked_manifest(
                *args,
                cancellation_callback: CancellationCallback | None = None,
                **kwargs,
            ):
                del args, kwargs
                if cancellation_callback is None:
                    raise AssertionError("status did not receive cancellation")
                status_started.set()
                while not cancellation_callback():
                    time.sleep(0.001)
                raise OperationCancelledError("FireLens operation was cancelled")

            with patch(
                "app.indexing.service.build_file_manifest",
                side_effect=blocked_manifest,
            ):
                task = asyncio.create_task(
                    _get_index_status_with_cancellation(
                        runtime,
                        str(repository),
                    )
                )
                self.assertTrue(
                    await asyncio.to_thread(status_started.wait, 1.0)
                )
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    async with asyncio.timeout(2.0):
                        await task

            with exclusive_database_lock(
                indexed.database_path,
                blocking=False,
            ):
                pass

    async def test_cancelled_progress_delivery_does_not_stall_cleanup(self) -> None:
        progress_future: concurrent.futures.Future[object] = (
            concurrent.futures.Future()
        )
        cancellation_requested = threading.Event()
        cancellation_requested.set()

        with self.assertRaises(IndexingCancelledError):
            _wait_for_progress_delivery(
                progress_future,
                cancellation_requested.is_set,
            )

        self.assertTrue(progress_future.cancelled())

    @staticmethod
    def _build_runtime(root: Path) -> FireLensRuntime:
        settings = Settings(
            _env_file=None,
            data_dir=root / "indexes",
            allowed_roots=[root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=8,
        )
        return FireLensRuntime(
            settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )

    async def test_stalled_progress_delivery_has_a_bounded_timeout(self) -> None:
        progress_future: concurrent.futures.Future[object] = (
            concurrent.futures.Future()
        )

        with (
            patch(
                "app.interfaces.mcp_server.PROGRESS_REPORT_TIMEOUT_SECONDS",
                0.01,
            ),
            patch(
                "app.interfaces.mcp_server.PROGRESS_WAIT_INTERVAL_SECONDS",
                0.001,
            ),
            self.assertRaisesRegex(TimeoutError, "reporting indexing progress"),
        ):
            _wait_for_progress_delivery(progress_future, lambda: False)

        self.assertTrue(progress_future.cancelled())


if __name__ == "__main__":
    unittest.main()

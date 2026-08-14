import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from app.core.models import Repository
from app.storage import locking
from app.storage.database import SQLiteIndexStore
from app.storage.locking import (
    DatabaseLockBusyError,
    database_lock_path,
    database_writer_intent_path,
    exclusive_database_lock,
    shared_database_lock,
)


class DatabaseLockTests(unittest.TestCase):
    def test_lock_file_is_adjacent_to_database(self) -> None:
        database_path = Path("indexes") / "repository" / "firelens.db"

        self.assertEqual(
            database_lock_path(database_path),
            Path("indexes") / "repository" / "firelens.db.lock",
        )
        self.assertEqual(
            database_writer_intent_path(database_path),
            Path("indexes") / "repository" / "firelens.db.lock.intent",
        )

    def test_nonblocking_shared_lock_reports_cross_process_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            child = _start_exclusive_lock_holder(database_path)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")

                with self.assertRaisesRegex(DatabaseLockBusyError, "Database is busy"):
                    with shared_database_lock(database_path, blocking=False):
                        self.fail("shared lock unexpectedly acquired")
            finally:
                child.communicate("release\n", timeout=10)

            with shared_database_lock(database_path, blocking=False):
                pass
            self.assertEqual(database_lock_path(database_path).read_bytes(), b"")
            self.assertEqual(
                database_writer_intent_path(database_path).read_bytes(),
                b"",
            )

    @unittest.skipUnless(os.name == "posix", "POSIX flock supports shared readers")
    def test_posix_shared_locks_can_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            child = _start_shared_lock_holder(database_path)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with shared_database_lock(database_path, blocking=False):
                    pass
            finally:
                child.communicate("release\n", timeout=10)

    def test_nonblocking_exclusive_lock_reports_an_active_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"

            with shared_database_lock(database_path):
                with self.assertRaises(DatabaseLockBusyError):
                    with exclusive_database_lock(database_path, blocking=False):
                        self.fail("exclusive lock unexpectedly acquired")

    @unittest.skipUnless(os.name == "posix", "requires concurrent shared readers")
    def test_queued_writer_prevents_later_readers_from_barging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            child = None
            try:
                with shared_database_lock(database_path):
                    child = _start_queued_exclusive_lock_holder(database_path)
                    self.assertEqual(child.stdout.readline().strip(), "waiting")
                    _wait_until_reader_is_blocked(database_path, child)

                    for _ in range(5):
                        with self.assertRaises(DatabaseLockBusyError):
                            with shared_database_lock(database_path, blocking=False):
                                self.fail("reader barged ahead of queued writer")

                stdout, stderr = child.communicate("release\n", timeout=10)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertEqual(stdout.strip(), "locked")
            finally:
                _terminate_child(child)

    def test_process_exit_releases_writer_intent_and_database_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            child = _start_exclusive_lock_holder(database_path)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                child.kill()
                child.communicate(timeout=10)

                with shared_database_lock(database_path, blocking=False):
                    pass
            finally:
                _terminate_child(child)

    @unittest.skipUnless(os.name == "posix", "requires concurrent shared readers")
    def test_killing_a_queued_writer_reopens_reader_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            child = None
            try:
                with shared_database_lock(database_path):
                    child = _start_queued_exclusive_lock_holder(database_path)
                    self.assertEqual(child.stdout.readline().strip(), "waiting")
                    _wait_until_reader_is_blocked(database_path, child)

                    child.kill()
                    child.communicate(timeout=10)

                    with shared_database_lock(database_path, blocking=False):
                        pass
            finally:
                _terminate_child(child)

    def test_lock_files_do_not_follow_a_symbolic_link(self) -> None:
        lock_path_functions = (database_lock_path, database_writer_intent_path)

        for lock_path_function in lock_path_functions:
            with self.subTest(lock_path_function=lock_path_function.__name__):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    database_path = root / "firelens.db"
                    external_file = root / "external"
                    external_file.write_bytes(b"")
                    try:
                        lock_path_function(database_path).symlink_to(external_file)
                    except OSError as error:
                        self.skipTest(f"Symbolic links are unavailable: {error}")

                    with self.assertRaises(OSError):
                        with exclusive_database_lock(database_path):
                            self.fail("symbolic-link lock unexpectedly acquired")

                    self.assertEqual(external_file.read_bytes(), b"")

    def test_verified_open_fallback_rejects_a_symbolic_link(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external_file = root / "external"
            lock_path = root / "firelens.db.lock"
            external_file.write_bytes(b"")
            try:
                lock_path.symlink_to(external_file)
            except OSError as error:
                self.skipTest(f"Symbolic links are unavailable: {error}")

            with self.assertRaises(OSError):
                locking._open_verified_lock_file(lock_path)

            self.assertEqual(external_file.read_bytes(), b"")

    def test_lock_path_must_be_a_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            database_lock_path(database_path).mkdir()

            with self.assertRaises(OSError):
                with exclusive_database_lock(database_path):
                    self.fail("directory lock path unexpectedly acquired")


class ReadOnlySQLiteTests(unittest.TestCase):
    def test_missing_database_read_does_not_create_database_or_parents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "missing" / "firelens.db"
            store = SQLiteIndexStore(database_path)

            with self.assertRaises(sqlite3.OperationalError):
                store.load_repository(uuid.uuid4())

            self.assertFalse(database_path.parent.exists())
            self.assertFalse(database_path.exists())

    def test_read_connection_rejects_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            store = SQLiteIndexStore(database_path)
            store.initialize()

            with store.read_connection() as connection:
                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "readonly|read-only",
                ):
                    connection.execute("DELETE FROM repositories")

    def test_legacy_repository_provider_is_readable_and_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            repository_id = uuid.uuid4()
            _create_legacy_repository_database(database_path, repository_id)
            store = SQLiteIndexStore(database_path)

            legacy_repository = store.load_repository(repository_id)
            self.assertIsNotNone(legacy_repository)
            self.assertEqual(legacy_repository.embedding_provider, "unknown")
            self.assertIsNotNone(
                store.load_repository_by_identity(
                    absolute_path="/tmp/legacy",
                    index_format_version="1",
                    embedding_provider="unknown",
                    embedding_model="legacy-model",
                    embedding_dim=8,
                )
            )
            self.assertIsNone(
                store.load_repository_by_identity(
                    absolute_path="/tmp/legacy",
                    index_format_version="1",
                    embedding_provider="sentence-transformers",
                    embedding_model="legacy-model",
                    embedding_dim=8,
                )
            )
            self.assertEqual(
                store.list_compatible_repositories(
                    index_format_version="1",
                    embedding_provider="unknown",
                    embedding_model="legacy-model",
                    embedding_dim=8,
                ),
                [legacy_repository],
            )

            store.initialize()

            with store.read_connection() as connection:
                columns = {
                    row["name"]: row
                    for row in connection.execute("PRAGMA table_info(repositories)")
                }
            self.assertIn("embedding_provider", columns)
            self.assertEqual(columns["embedding_provider"]["notnull"], 1)
            self.assertEqual(columns["embedding_provider"]["dflt_value"], "'unknown'")

    def test_repository_provider_is_persisted_and_used_for_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "firelens.db"
            store = SQLiteIndexStore(database_path)
            repository = Repository(
                id=uuid.uuid4(),
                absolute_path="/tmp/current",
                index_format_version="1",
                timestamp_of_index=1,
                embedding_provider="sentence-transformers",
                embedding_model="model",
                embedding_dim=8,
            )
            store.initialize()
            store.replace_index(repository, [], [], [], [])

            loaded = store.load_repository(repository.id)
            compatible = store.load_repository_by_identity(
                absolute_path=repository.absolute_path,
                index_format_version=repository.index_format_version,
                embedding_provider=repository.embedding_provider,
                embedding_model=repository.embedding_model,
                embedding_dim=repository.embedding_dim,
            )
            incompatible = store.load_repository_by_identity(
                absolute_path=repository.absolute_path,
                index_format_version=repository.index_format_version,
                embedding_provider="different-provider",
                embedding_model=repository.embedding_model,
                embedding_dim=repository.embedding_dim,
            )

            self.assertEqual(loaded, repository)
            self.assertEqual(compatible, repository)
            self.assertIsNone(incompatible)


def _start_exclusive_lock_holder(database_path: Path) -> subprocess.Popen[str]:
    script = """
import sys
from app.storage.locking import exclusive_database_lock

with exclusive_database_lock(sys.argv[1]):
    print("locked", flush=True)
    input()
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(database_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _start_shared_lock_holder(database_path: Path) -> subprocess.Popen[str]:
    script = """
import sys
from app.storage.locking import shared_database_lock

with shared_database_lock(sys.argv[1]):
    print("locked", flush=True)
    input()
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(database_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _start_queued_exclusive_lock_holder(
    database_path: Path,
) -> subprocess.Popen[str]:
    script = """
import sys
from app.storage.locking import exclusive_database_lock

print("waiting", flush=True)
with exclusive_database_lock(sys.argv[1]):
    print("locked", flush=True)
    input()
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(database_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_until_reader_is_blocked(
    database_path: Path,
    writer: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if writer.poll() is not None:
            _, stderr = writer.communicate()
            raise AssertionError(f"queued writer exited unexpectedly: {stderr}")
        try:
            with shared_database_lock(database_path, blocking=False):
                pass
        except DatabaseLockBusyError:
            return
        time.sleep(0.01)
    raise AssertionError("queued writer did not close reader admission")


def _terminate_child(child: subprocess.Popen[str] | None) -> None:
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate(timeout=10)


def _create_legacy_repository_database(
    database_path: Path,
    repository_id: uuid.UUID,
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            CREATE TABLE repositories (
                id TEXT PRIMARY KEY,
                absolute_path TEXT NOT NULL,
                index_format_version TEXT NOT NULL,
                timestamp_of_index INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO repositories (
                id,
                absolute_path,
                index_format_version,
                timestamp_of_index,
                embedding_model,
                embedding_dim
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(repository_id),
                "/tmp/legacy",
                "1",
                1,
                "legacy-model",
                8,
            ),
        )
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()

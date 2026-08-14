import tempfile
import unittest
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.models import Repository
from app.core.runtime import FireLensRuntime
from app.storage.database import SQLiteIndexStore, default_database_path


class AvailableIndexTests(unittest.TestCase):
    def test_discovery_returns_latest_allowed_indexes_without_loading_embedder(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed_repository = root / "allowed" / "project"
            disallowed_repository = root / "outside" / "project"
            data_dir = root / "indexes"
            allowed_repository.mkdir(parents=True)
            disallowed_repository.mkdir(parents=True)

            allowed_database = default_database_path(allowed_repository, data_dir)
            self._store_repository(
                allowed_database,
                allowed_repository,
                timestamp=10,
                model="older-model",
            )
            self._store_repository(
                allowed_database,
                allowed_repository,
                timestamp=20,
                model="latest-model",
            )
            self._store_repository(
                default_database_path(disallowed_repository, data_dir),
                disallowed_repository,
                timestamp=30,
                model="outside-model",
            )

            settings = Settings(
                _env_file=None,
                data_dir=data_dir,
                allowed_roots=[allowed_repository.parent],
            )

            def fail_if_embedder_is_created():
                raise AssertionError("index discovery must not load an embedder")

            runtime = FireLensRuntime(
                settings,
                embedder_factory=fail_if_embedder_is_created,
            )
            available_indexes = runtime.list_available_indexes()

        self.assertEqual(len(available_indexes), 1)
        available_index = available_indexes[0]
        self.assertEqual(
            available_index.repository_path,
            str(allowed_repository.resolve()),
        )
        self.assertEqual(
            available_index.database_path,
            str(allowed_database.resolve()),
        )
        self.assertEqual(available_index.timestamp_of_index, 20)
        self.assertEqual(available_index.embedding_provider, "test")
        self.assertEqual(available_index.embedding_model, "latest-model")
        self.assertEqual(available_index.embedding_dim, 8)

    @staticmethod
    def _store_repository(
        database_path: Path,
        repository_path: Path,
        timestamp: int,
        model: str,
    ) -> None:
        store = SQLiteIndexStore(database_path)
        store.initialize()
        store.replace_index(
            repository=Repository(
                id=uuid.uuid4(),
                absolute_path=str(repository_path.resolve()),
                index_format_version="1",
                timestamp_of_index=timestamp,
                embedding_provider="test",
                embedding_model=model,
                embedding_dim=8,
            ),
            files=[],
            symbols=[],
            chunks=[],
            embeddings=[],
        )


if __name__ == "__main__":
    unittest.main()

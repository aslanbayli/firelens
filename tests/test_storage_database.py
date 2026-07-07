import tempfile
import unittest
import uuid
from pathlib import Path

from app.core.models import Chunk, Repository, Symbol
from app.storage.database import (
    IndexedFile,
    SQLiteIndexStore,
    pack_vector,
    unpack_vector,
)


class SQLiteIndexStoreTests(unittest.TestCase):
    def test_pack_vector_round_trips_float_values(self) -> None:
        vector = [0.25, -0.5, 1.0]

        restored = unpack_vector(pack_vector(vector))

        self.assertEqual(len(restored), len(vector))
        for actual, expected in zip(restored, vector, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_replace_index_stores_repository_records_and_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "firelens.db"
            store = SQLiteIndexStore(db_path)
            repository, files, symbols, chunks, embeddings = _sample_index()

            store.initialize()
            store.replace_index(
                repository=repository,
                files=files,
                symbols=symbols,
                chunks=chunks,
                embeddings=embeddings,
            )

            loaded_repository = store.load_repository(repository.id)
            loaded_embeddings = store.load_embeddings(repository.id)
            file_count = store.count_rows("files", repository.id)
            symbol_count = store.count_rows("symbols", repository.id)
            chunk_count = store.count_rows("chunks", repository.id)
            embedding_count = store.count_rows("embeddings", repository.id)

        self.assertEqual(loaded_repository, repository)
        self.assertEqual(len(loaded_embeddings), 1)
        self.assertEqual(loaded_embeddings[0][0], chunks[0].id)
        self.assertEqual(len(loaded_embeddings[0][1]), repository.embedding_dim)
        self.assertEqual(file_count, 1)
        self.assertEqual(symbol_count, 1)
        self.assertEqual(chunk_count, 1)
        self.assertEqual(embedding_count, 1)

    def test_replace_index_rejects_embedding_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteIndexStore(Path(temp_dir) / "firelens.db")
            repository, files, symbols, chunks, _embeddings = _sample_index()

            store.initialize()

            with self.assertRaises(ValueError):
                store.replace_index(
                    repository=repository,
                    files=files,
                    symbols=symbols,
                    chunks=chunks,
                    embeddings=[],
                )


def _sample_index() -> tuple[
    Repository,
    list[IndexedFile],
    list[Symbol],
    list[Chunk],
    list[list[float]],
]:
    repository_id = uuid.uuid4()
    symbol_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    repository = Repository(
        id=repository_id,
        absolute_path="/tmp/example",
        index_format_version="1",
        timestamp_of_index=1,
        embedding_model="test-model",
        embedding_dim=3,
    )
    files = [
        IndexedFile(
            repository_id=repository_id,
            relative_path="example.py",
            modified_time_ns=10,
            size_bytes=20,
            content_hash="abc",
        )
    ]
    symbols = [
        Symbol(
            id=symbol_id,
            repository_id=repository_id,
            name="hello",
            qualified_name="hello",
            kind="function",
            relative_path="example.py",
            start_line=1,
            end_line=2,
            source_snippet="def hello():\n    return 'world'\n",
        )
    ]
    chunks = [
        Chunk(
            id=chunk_id,
            repository_id=repository_id,
            relative_path="example.py",
            start_line=1,
            end_line=2,
            symbol_id=symbol_id,
            raw_text="def hello():\n    return 'world'\n",
            content_hash="def",
        )
    ]
    embeddings = [[0.25, -0.5, 1.0]]

    return repository, files, symbols, chunks, embeddings


if __name__ == "__main__":
    unittest.main()

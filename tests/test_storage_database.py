import tempfile
import unittest
import uuid
from pathlib import Path

from app.core.models import Chunk, Repository, Symbol
from app.search.semantic import load_semantic_search_index
from app.storage.database import (
    IndexedFile,
    SQLiteIndexStore,
    default_database_path,
    pack_vector,
    unpack_vector,
)


class SQLiteIndexStoreTests(unittest.TestCase):
    def test_default_database_path_bounds_a_long_repository_name(self) -> None:
        repository_root = Path("/") / ("a" * 255)

        database_path = default_database_path(
            repository_root,
            data_directory="indexes",
        )

        self.assertLessEqual(len(database_path.parent.name.encode("utf-8")), 255)
        readable_name, path_hash = database_path.parent.name.rsplit("-", 1)
        self.assertEqual(len(readable_name), 64)
        self.assertEqual(len(path_hash), 12)
        self.assertTrue(all(character in "0123456789abcdef" for character in path_hash))

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

    def test_exact_search_symbols_orders_and_filters_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteIndexStore(Path(temp_dir) / "firelens.db")
            repository, files, symbols, chunks, embeddings = _sample_index()
            symbols.extend(
                [
                    Symbol(
                        id=uuid.uuid4(),
                        repository_id=repository.id,
                        name="run",
                        qualified_name="Service.run",
                        kind="method",
                        relative_path="service.py",
                        start_line=3,
                        end_line=4,
                        source_snippet="    def run(self):\n        pass\n",
                    ),
                    Symbol(
                        id=uuid.uuid4(),
                        repository_id=repository.id,
                        name="run",
                        qualified_name="run",
                        kind="function",
                        relative_path="main.py",
                        start_line=1,
                        end_line=2,
                        source_snippet="def run():\n    pass\n",
                    ),
                    Symbol(
                        id=uuid.uuid4(),
                        repository_id=repository.id,
                        name="run",
                        qualified_name="Worker.run",
                        kind="method",
                        relative_path="worker.py",
                        start_line=8,
                        end_line=9,
                        source_snippet="    def run(self):\n        pass\n",
                    ),
                ]
            )

            store.initialize()
            store.replace_index(
                repository=repository,
                files=files,
                symbols=symbols,
                chunks=chunks,
                embeddings=embeddings,
            )

            matches = store.exact_search_symbols(repository.id, "run")
            filtered_matches = store.exact_search_symbols(
                repository.id,
                "run",
                path_filter="worker.py",
            )

        self.assertEqual(
            [symbol.qualified_name for symbol in matches],
            ["run", "Service.run", "Worker.run"],
        )
        self.assertEqual(
            [symbol.qualified_name for symbol in filtered_matches],
            ["Worker.run"],
        )

    def test_search_loaders_keep_candidates_bounded_and_fetch_results_lazily(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteIndexStore(Path(temp_dir) / "firelens.db")
            repository, files, symbols, chunks, embeddings = _sample_index()
            symbols.append(
                Symbol(
                    id=uuid.uuid4(),
                    repository_id=repository.id,
                    name="helper",
                    qualified_name="Service.helper",
                    kind="method",
                    relative_path="example.py",
                    start_line=4,
                    end_line=5,
                    source_snippet="    def helper(self):\n        pass\n",
                )
            )
            store.initialize()
            store.replace_index(repository, files, symbols, chunks, embeddings)

            with self.assertRaisesRegex(ValueError, "candidate limit"):
                store.load_symbol_candidates(
                    repository.id,
                    limit=1,
                    candidate_char_limit=512,
                )

            candidates = store.load_symbol_candidates(
                repository.id,
                limit=2,
                candidate_char_limit=512,
            )
            loaded_symbols = store.load_symbols_by_ids(
                (candidate.id for candidate in candidates),
                max_snippet_chars=4_000,
            )
            summary = store.semantic_candidate_summary(
                repository.id,
                max_candidates=10,
                max_vector_bytes=1_024,
            )
            semantic_rows = list(store.iter_semantic_candidate_rows(repository.id))
            chunk_texts = store.load_chunk_texts(
                [chunks[0].id],
                max_chars=4_000,
            )
            file_sources = store.load_file_sources_by_paths(
                repository.id,
                ["example.py"],
                max_snippet_chars=4_000,
            )
            semantic_index = load_semantic_search_index(
                store,
                repository.id,
                max_candidates=10,
                max_vector_bytes=1_024,
            )
            with self.assertRaisesRegex(ValueError, "vector memory limit"):
                load_semantic_search_index(
                    store,
                    repository.id,
                    max_candidates=10,
                    max_vector_bytes=1,
                )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            set(loaded_symbols),
            {candidate.id for candidate in candidates},
        )
        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.dimension, repository.embedding_dim)
        self.assertEqual(summary.vector_bytes, repository.embedding_dim * 4)
        self.assertEqual(semantic_rows[0].candidate.chunk_id, chunks[0].id)
        self.assertEqual(chunk_texts[chunks[0].id], chunks[0].raw_text)
        self.assertEqual(
            file_sources["example.py"].source_text,
            files[0].source_text,
        )
        self.assertEqual(file_sources["example.py"].line_count, 2)
        self.assertEqual(semantic_index.matrix.shape, (1, repository.embedding_dim))


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
        embedding_provider="test",
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
            source_text="def hello():\n    return 'world'\n",
            line_count=2,
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

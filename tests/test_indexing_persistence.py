import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from app.indexing.embedder import FakeEmbedder
from app.indexing.indexer import IndexingProgress, index_to_sqlite
from app.storage.database import SQLiteIndexStore


class CountingEmbedder(FakeEmbedder):
    """Fake embedder that records how many texts were embedded."""

    def __init__(self, dimension: int = 8) -> None:
        super().__init__(dimension=dimension)
        self.call_count = 0
        self.text_count = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.call_count += 1
        self.text_count += len(texts)
        return super().embed(texts)


class IndexingPersistenceTests(unittest.TestCase):
    def test_index_to_sqlite_persists_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            (repo / "service.py").write_text(
                "class Service:\n"
                "    def method(self):\n"
                "        return 'ok'\n\n"
                "def helper():\n"
                "    return Service()\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"

            report = index_to_sqlite(repo, FakeEmbedder(dimension=8), db_path)
            store = SQLiteIndexStore(db_path)

            repository = store.load_repository(report.repository.id)
            embeddings = store.load_embeddings(report.repository.id)

        self.assertEqual(report.errors, [])
        self.assertEqual(report.database_path, db_path)
        self.assertEqual(report.symbol_count, 3)
        self.assertEqual(report.chunk_count, 3)
        self.assertEqual(report.embedding_count, 3)
        self.assertEqual(report.file_count, 1)
        self.assertEqual(repository, report.repository)
        self.assertEqual(len(embeddings), 3)
        self.assertTrue(all(len(vector) == 8 for _chunk_id, vector in embeddings))

    def test_reindex_without_changes_skips_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            (repo / "service.py").write_text(
                "def helper():\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"
            embedder = CountingEmbedder(dimension=8)

            first_report = index_to_sqlite(repo, embedder, db_path)
            first_text_count = embedder.text_count

            second_report = index_to_sqlite(repo, embedder, db_path)

        self.assertEqual(first_report.embedded_chunk_count, 1)
        self.assertEqual(first_text_count, 1)
        self.assertEqual(second_report.added_file_count, 0)
        self.assertEqual(second_report.changed_file_count, 0)
        self.assertEqual(second_report.deleted_file_count, 0)
        self.assertEqual(second_report.embedded_chunk_count, 0)
        self.assertEqual(embedder.text_count, first_text_count)

    def test_reindex_only_embeds_added_and_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            (repo / "changed.py").write_text(
                "def changed():\n"
                "    return 'before'\n",
                encoding="utf-8",
            )
            (repo / "same.py").write_text(
                "def same():\n"
                "    return 'same'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"
            embedder = CountingEmbedder(dimension=8)

            first_report = index_to_sqlite(repo, embedder, db_path)
            first_text_count = embedder.text_count

            (repo / "changed.py").write_text(
                "def changed():\n"
                "    return 'after'\n",
                encoding="utf-8",
            )
            (repo / "added.py").write_text(
                "def added():\n"
                "    return 'new'\n",
                encoding="utf-8",
            )

            second_report = index_to_sqlite(repo, embedder, db_path)

        self.assertEqual(first_report.embedded_chunk_count, 2)
        self.assertEqual(first_text_count, 2)
        self.assertEqual(second_report.added_file_count, 1)
        self.assertEqual(second_report.changed_file_count, 1)
        self.assertEqual(second_report.deleted_file_count, 0)
        self.assertEqual(second_report.embedded_chunk_count, 2)
        self.assertEqual(second_report.symbol_count, 3)
        self.assertEqual(embedder.text_count, 4)

    def test_reindex_removes_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            deleted_file = repo / "deleted.py"
            deleted_file.write_text(
                "def removed():\n"
                "    return 'gone'\n",
                encoding="utf-8",
            )
            (repo / "kept.py").write_text(
                "def kept():\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"

            first_report = index_to_sqlite(repo, FakeEmbedder(dimension=8), db_path)
            deleted_file.unlink()
            second_report = index_to_sqlite(repo, FakeEmbedder(dimension=8), db_path)

        self.assertEqual(first_report.symbol_count, 2)
        self.assertEqual(second_report.deleted_file_count, 1)
        self.assertEqual(second_report.file_count, 1)
        self.assertEqual(second_report.symbol_count, 1)
        self.assertEqual(second_report.chunk_count, 1)
        self.assertEqual(second_report.embedding_count, 1)

    def test_reindex_reuses_unchanged_chunk_embeddings_in_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            source_file = repo / "service.py"
            source_file.write_text(
                "def stable():\n"
                "    return 'same'\n\n"
                "def edited():\n"
                "    return 'before'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"
            embedder = CountingEmbedder(dimension=8)

            index_to_sqlite(repo, embedder, db_path)
            first_text_count = embedder.text_count

            source_file.write_text(
                "def stable():\n"
                "    return 'same'\n\n"
                "def edited():\n"
                "    return 'after'\n",
                encoding="utf-8",
            )
            second_report = index_to_sqlite(repo, embedder, db_path)

        self.assertEqual(first_text_count, 2)
        self.assertEqual(second_report.changed_file_count, 1)
        self.assertEqual(second_report.embedded_chunk_count, 1)
        self.assertEqual(second_report.reused_embedding_count, 1)
        self.assertEqual(embedder.text_count, 3)

    def test_changed_file_parse_error_preserves_previous_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            source_file = repo / "service.py"
            source_file.write_text(
                "def valid():\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"

            first_report = index_to_sqlite(repo, FakeEmbedder(dimension=8), db_path)

            source_file.write_text(
                "def valid(:\n"
                "    return 'broken'\n",
                encoding="utf-8",
            )
            second_report = index_to_sqlite(repo, FakeEmbedder(dimension=8), db_path)

        self.assertEqual(first_report.symbol_count, 1)
        self.assertEqual(second_report.changed_file_count, 1)
        self.assertEqual(len(second_report.errors), 1)
        self.assertEqual(second_report.errors[0].stage, "parse")
        self.assertEqual(second_report.symbol_count, 1)
        self.assertEqual(second_report.chunk_count, 1)
        self.assertEqual(second_report.embedding_count, 1)

    def test_index_to_sqlite_emits_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            (repo / "service.py").write_text(
                "def helper():\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"
            events: list[IndexingProgress] = []

            index_to_sqlite(
                repo,
                FakeEmbedder(dimension=8),
                db_path,
                progress_callback=events.append,
            )

        stages = [event.stage for event in events]
        self.assertIn("load", stages)
        self.assertIn("walk", stages)
        self.assertIn("compare", stages)
        self.assertIn("index", stages)
        self.assertIn("write", stages)
        self.assertEqual(events[-1].stage, "complete")

    def test_unchanged_reindex_emits_no_changes_progress_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            (repo / "service.py").write_text(
                "def helper():\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"

            index_to_sqlite(repo, FakeEmbedder(dimension=8), db_path)

            events: list[IndexingProgress] = []
            index_to_sqlite(
                repo,
                FakeEmbedder(dimension=8),
                db_path,
                progress_callback=events.append,
            )

        self.assertIn(
            "No file changes to index",
            [event.message for event in events],
        )


if __name__ == "__main__":
    unittest.main()

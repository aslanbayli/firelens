import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

from app.core.models import Repository
from app.indexing.embedder import FakeEmbedder
from app.indexing.indexer import IndexingProgress, index_to_sqlite
from app.storage.database import IndexedFileRecords, SQLiteIndexStore


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


class AlternateEmbedder(FakeEmbedder):
    provider = "alternate-test"
    model = "alternate-fake"


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

    def test_changed_files_are_written_to_the_staged_database_one_at_a_time(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            for file_number in range(3):
                (repo / f"module_{file_number}.py").write_text(
                    f"def function_{file_number}():\n    return {file_number}\n",
                    encoding="utf-8",
                )
            db_path = temp_path / "index" / "firelens.db"
            changed_file_batch_sizes: list[int] = []
            original_apply_file_updates = SQLiteIndexStore.apply_file_updates

            def record_batch_size(
                store: SQLiteIndexStore,
                repository: Repository,
                changed_files: list[IndexedFileRecords],
                deleted_relative_paths: list[str],
            ) -> None:
                changed_file_batch_sizes.append(len(changed_files))
                original_apply_file_updates(
                    store,
                    repository,
                    changed_files,
                    deleted_relative_paths,
                )

            with patch.object(
                SQLiteIndexStore,
                "apply_file_updates",
                new=record_batch_size,
            ):
                report = index_to_sqlite(
                    repo,
                    FakeEmbedder(dimension=8),
                    db_path,
                )

        self.assertEqual(report.file_count, 3)
        self.assertEqual(sum(changed_file_batch_sizes), 3)
        self.assertLessEqual(max(changed_file_batch_sizes), 1)

    def test_per_file_chunk_limit_prevents_unbounded_embedding_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            (repo / "large.py").write_text(
                "def first():\n    return 1\n\n"
                "def second():\n    return 2\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"
            embedder = CountingEmbedder(dimension=8)

            report = index_to_sqlite(
                repo,
                embedder,
                db_path,
                max_chunks_per_file=1,
            )

        self.assertEqual(report.file_count, 0)
        self.assertEqual(report.embedding_count, 0)
        self.assertEqual(embedder.text_count, 0)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0].stage, "chunk")
        self.assertIn("semantic chunk limit", report.errors[0].message)

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

    def test_file_changed_after_manifest_hash_preserves_previous_records(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            source_file = repo / "service.py"
            source_file.write_text(
                "def original():\n    return 1\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"
            first_report = index_to_sqlite(
                repo,
                FakeEmbedder(dimension=8),
                db_path,
            )
            source_file.write_text(
                "def intermediate():\n    return 2\n",
                encoding="utf-8",
            )

            from app.indexing import indexer

            original_build_manifest = indexer.build_file_manifest

            def mutate_after_manifest(*args: Any, **kwargs: Any) -> Any:
                manifest = original_build_manifest(*args, **kwargs)
                source_file.write_text(
                    "def final_version():\n    return 3\n",
                    encoding="utf-8",
                )
                return manifest

            with patch.object(
                indexer,
                "build_file_manifest",
                side_effect=mutate_after_manifest,
            ):
                second_report = index_to_sqlite(
                    repo,
                    FakeEmbedder(dimension=8),
                    db_path,
                )

            stored_symbols = SQLiteIndexStore(db_path).load_all_symbols(
                first_report.repository.id
            )

        self.assertEqual(len(second_report.errors), 1)
        self.assertEqual(second_report.errors[0].stage, "read")
        self.assertIn("changed during indexing", second_report.errors[0].message)
        self.assertEqual(
            [symbol.qualified_name for symbol in stored_symbols],
            ["original"],
        )

    def test_failed_incompatible_rebuild_preserves_previous_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo = temp_path / "repo"
            repo.mkdir()
            source_file = repo / "service.py"
            source_file.write_text(
                "def stable():\n"
                "    return 'old index'\n",
                encoding="utf-8",
            )
            db_path = temp_path / "index" / "firelens.db"
            first_report = index_to_sqlite(
                repo,
                FakeEmbedder(dimension=8),
                db_path,
            )
            database_before_failure = db_path.read_bytes()

            source_file.write_text(
                "def broken(:\n"
                "    return 'new source'\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "previous index was preserved"):
                index_to_sqlite(
                    repo,
                    AlternateEmbedder(dimension=12),
                    db_path,
                )

            store = SQLiteIndexStore(db_path)
            preserved_repository = store.load_repository(first_report.repository.id)
            preserved_symbols = store.load_all_symbols(first_report.repository.id)
            database_after_failure = db_path.read_bytes()

        self.assertEqual(database_after_failure, database_before_failure)
        self.assertEqual(preserved_repository, first_report.repository)
        self.assertEqual(
            [symbol.qualified_name for symbol in preserved_symbols],
            ["stable"],
        )

    def test_post_promotion_progress_failures_do_not_fail_indexing(self) -> None:
        for failing_stage in ("promote", "complete"):
            with self.subTest(failing_stage=failing_stage):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    repo = temp_path / "repo"
                    repo.mkdir()
                    source_file = repo / "service.py"
                    source_file.write_text(
                        "def old():\n    return 1\n",
                        encoding="utf-8",
                    )
                    db_path = temp_path / "index" / "firelens.db"
                    index_to_sqlite(repo, FakeEmbedder(dimension=8), db_path)
                    source_file.write_text(
                        "def replacement():\n    return 2\n",
                        encoding="utf-8",
                    )

                    def fail_after_promotion(event: IndexingProgress) -> None:
                        should_fail = event.stage == failing_stage
                        if failing_stage == "promote":
                            should_fail = should_fail and event.current == 1
                        if should_fail:
                            raise RuntimeError("progress transport failed")

                    report = index_to_sqlite(
                        repo,
                        FakeEmbedder(dimension=8),
                        db_path,
                        progress_callback=fail_after_promotion,
                    )
                    store = SQLiteIndexStore(db_path)
                    symbols = store.load_all_symbols(report.repository.id)

                self.assertEqual(report.errors, [])
                self.assertEqual(
                    [symbol.qualified_name for symbol in symbols],
                    ["replacement"],
                )

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
            database_exists_at_event: list[bool] = []

            def record_progress(event: IndexingProgress) -> None:
                events.append(event)
                database_exists_at_event.append(db_path.exists())

            index_to_sqlite(
                repo,
                FakeEmbedder(dimension=8),
                db_path,
                progress_callback=record_progress,
            )

        stages = [event.stage for event in events]
        self.assertEqual(stages[0], "model")
        self.assertIn("model", stages)
        self.assertIn("load", stages)
        self.assertIn("walk", stages)
        self.assertIn("compare", stages)
        self.assertIn("index", stages)
        self.assertIn("write", stages)
        self.assertIn("promote", stages)
        self.assertEqual(events[-1].stage, "complete")
        self.assertTrue(database_exists_at_event[-1])

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

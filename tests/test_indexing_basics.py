import math
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import Symbol
from app.indexing.chunker import (
    build_embedding_text,
    calculate_content_hash,
    chunk_symbols,
)
from app.indexing.embedder import FakeEmbedder
from app.indexing.file_io import read_regular_file
from app.indexing.indexer import index
from app.indexing.parser import parse_symbols
from app.indexing.walker import walk


class ParserTests(unittest.TestCase):
    def test_parse_symbols_extracts_nested_and_async_symbols(self) -> None:
        source = """
class Service:
    def method(self):
        def local():
            pass

async def fetch_data():
    pass
"""

        symbols = parse_symbols(source)

        self.assertEqual(
            [(symbol.qualified_name, symbol.kind) for symbol in symbols],
            [
                ("Service", "class"),
                ("Service.method", "method"),
                ("Service.method.local", "function"),
                ("fetch_data", "async_function"),
            ],
        )
        self.assertEqual(symbols[0].start_line, 2)
        self.assertEqual(symbols[0].end_line, 5)
        self.assertIn("class Service:", symbols[0].source_snippet)


class ChunkerTests(unittest.TestCase):
    def test_chunk_symbols_creates_chunk_with_source_and_hash(self) -> None:
        repository_id = uuid.uuid4()
        symbol = Symbol(
            id=uuid.uuid4(),
            repository_id=repository_id,
            name="hello",
            qualified_name="hello",
            kind="function",
            relative_path="example.py",
            start_line=1,
            end_line=2,
            source_snippet="def hello():\n    return 'world'\n",
        )
        source = "def hello():\n    return 'world'\n"

        chunks = chunk_symbols(source, [symbol], max_lines=10, overlap=2)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.repository_id, repository_id)
        self.assertEqual(chunk.relative_path, "example.py")
        self.assertEqual(chunk.start_line, 1)
        self.assertEqual(chunk.end_line, 2)
        self.assertEqual(chunk.symbol_id, symbol.id)
        self.assertEqual(chunk.raw_text, source)

        embedding_text = build_embedding_text(
            relative_path="example.py",
            raw_text=source,
            qualified_name="hello",
            kind="function",
        )
        self.assertEqual(chunk.content_hash, calculate_content_hash(embedding_text))

    def test_chunk_symbols_splits_long_symbol_with_overlap(self) -> None:
        repository_id = uuid.uuid4()
        source = "\n".join(
            [
                "def many_lines():",
                "    line_1 = 1",
                "    line_2 = 2",
                "    line_3 = 3",
                "    line_4 = 4",
            ]
        )
        symbol = Symbol(
            id=uuid.uuid4(),
            repository_id=repository_id,
            name="many_lines",
            qualified_name="many_lines",
            kind="function",
            relative_path="example.py",
            start_line=1,
            end_line=5,
            source_snippet=source,
        )

        chunks = chunk_symbols(source, [symbol], max_lines=3, overlap=1)

        self.assertEqual(
            [(chunk.start_line, chunk.end_line) for chunk in chunks],
            [(1, 3), (3, 5)],
        )


class FakeEmbedderTests(unittest.TestCase):
    def test_fake_embedder_returns_deterministic_normalized_vectors(self) -> None:
        embedder = FakeEmbedder(dimension=16)

        first_run = embedder.embed(["hello", "world"])
        second_run = embedder.embed(["hello", "world"])

        self.assertEqual(first_run, second_run)
        self.assertEqual(len(first_run), 2)
        self.assertEqual(len(first_run[0]), 16)
        self.assertEqual(len(first_run[1]), 16)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in first_run[0])),
            1.0,
        )


class FileIOTests(unittest.TestCase):
    def test_regular_file_reads_are_bounded_and_do_not_follow_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.py"
            source.write_bytes(b"abcdef")

            file_status, contents = read_regular_file(source, byte_limit=3)
            self.assertEqual(file_status.st_size, 6)
            self.assertEqual(contents, b"abc")

            linked_source = root / "linked.py"
            try:
                linked_source.symlink_to(source)
            except OSError as error:
                self.skipTest(f"Symbolic links are unavailable: {error}")

            with self.assertRaises(OSError):
                read_regular_file(linked_source, byte_limit=3)


class WalkerTests(unittest.TestCase):
    def test_walk_enforces_entry_limit_while_scandir_is_streaming(self) -> None:
        class FakeEntry:
            def __init__(self, name: str) -> None:
                self.name = name

            def stat(self, *, follow_symlinks: bool):
                self._assert_no_follow(follow_symlinks)
                return SimpleNamespace(st_mode=0, st_size=0, st_file_attributes=0)

            def is_dir(self, *, follow_symlinks: bool) -> bool:
                self._assert_no_follow(follow_symlinks)
                return False

            def is_file(self, *, follow_symlinks: bool) -> bool:
                self._assert_no_follow(follow_symlinks)
                return True

            @staticmethod
            def _assert_no_follow(follow_symlinks: bool) -> None:
                if follow_symlinks:
                    raise AssertionError("walker followed a symbolic link")

        class StreamingEntries:
            def __init__(self) -> None:
                self.yield_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                del args

            def __iter__(self):
                return self

            def __next__(self) -> FakeEntry:
                self.yield_count += 1
                if self.yield_count > 3:
                    raise AssertionError("walker consumed beyond the entry bound")
                return FakeEntry(f"ignored-{self.yield_count}.txt")

        with tempfile.TemporaryDirectory() as temp_dir:
            entries = StreamingEntries()
            with (
                patch("app.indexing.walker.os.scandir", return_value=entries),
                self.assertRaisesRegex(ValueError, "entry scan limit"),
            ):
                walk(Path(temp_dir), max_entries=2)

        self.assertEqual(entries.yield_count, 3)

    def test_walk_rejects_excessive_gitignore_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = Path(temp_dir)
            (repository / ".gitignore").write_text(
                "".join(f"ignored-{index_number}\n" for index_number in range(513)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "512 rule limit"):
                walk(repository)

    def test_walk_skips_the_firelens_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            data_directory = repo / "data"
            data_directory.mkdir()
            (data_directory / "generated.py").write_text(
                "def generated():\n    pass\n",
                encoding="utf-8",
            )
            (repo / "included.py").write_text(
                "def included():\n    pass\n",
                encoding="utf-8",
            )

            paths = walk(repo)

        self.assertEqual(paths, [Path("included.py")])

    def test_walk_keeps_nested_source_packages_named_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            package = repo / "app" / "data"
            package.mkdir(parents=True)
            (package / "models.py").write_text(
                "def load_model():\n    pass\n",
                encoding="utf-8",
            )

            paths = walk(repo)

        self.assertEqual(paths, [Path("app/data/models.py")])

    def test_walk_respects_root_gitignore_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / ".gitignore").write_text(
                "ignored_dir/\n"
                "/root_ignored_dir/\n"
                "/root_ignored.py\n"
                "*.generated.py\n"
                "!keep.generated.py\n",
                encoding="utf-8",
            )
            (repo / "included.py").write_text("def included():\n    pass\n")
            (repo / "root_ignored.py").write_text("def root_ignored():\n    pass\n")
            (repo / "skip.generated.py").write_text("def generated():\n    pass\n")
            (repo / "keep.generated.py").write_text("def keep():\n    pass\n")
            ignored_dir = repo / "ignored_dir"
            ignored_dir.mkdir()
            (ignored_dir / "hidden.py").write_text("def hidden():\n    pass\n")
            root_ignored_dir = repo / "root_ignored_dir"
            root_ignored_dir.mkdir()
            (root_ignored_dir / "hidden.py").write_text("def hidden():\n    pass\n")

            paths = walk(repo)

        self.assertEqual(
            [path.as_posix() for path in paths],
            ["included.py", "keep.generated.py"],
        )

    def test_walk_rejects_file_symlinks_that_escape_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            repo.mkdir()
            outside_source = root / "outside.py"
            outside_source.write_text("def secret():\n    return 1\n", encoding="utf-8")
            try:
                (repo / "linked.py").symlink_to(outside_source)
            except OSError as error:
                self.skipTest(f"Symbolic links are unavailable: {error}")

            paths = walk(repo)

        self.assertEqual(paths, [])

    def test_walk_prunes_ignored_directories_before_applying_entry_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ignored = repo / ".git"
            ignored.mkdir()
            for index_number in range(20):
                (ignored / f"object-{index_number}").write_text("ignored")
            (repo / "included.py").write_text("def included():\n    pass\n")

            paths = walk(repo, max_entries=2)

        self.assertEqual(paths, [Path("included.py")])

    def test_walk_bounds_all_visited_entries_and_accepted_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            for file_name in ("one.txt", "two.txt", "three.py"):
                (repo / file_name).write_text("value", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "entry scan limit"):
                walk(repo, max_entries=2)

            with self.assertRaisesRegex(ValueError, "file limit"):
                walk(repo, max_files=0)

    def test_walk_skips_files_over_the_configured_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "small.py").write_text("x = 1\n", encoding="utf-8")
            (repo / "large.py").write_text("x = 'too large'\n", encoding="utf-8")

            paths = walk(repo, max_file_size=8)

        self.assertEqual(paths, [Path("small.py")])


class IndexerTests(unittest.TestCase):
    def test_index_builds_symbols_chunks_and_embeddings_for_local_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "service.py").write_text(
                "class Service:\n"
                "    def method(self):\n"
                "        return 'ok'\n\n"
                "def helper():\n"
                "    return Service()\n",
                encoding="utf-8",
            )

            result = index(repo, FakeEmbedder(dimension=8))

        self.assertEqual(result.errors, [])
        self.assertEqual(
            [symbol.qualified_name for symbol in result.symbols],
            ["Service", "Service.method", "helper"],
        )
        self.assertEqual(len(result.chunks), 3)
        self.assertEqual(len(result.embeddings), len(result.chunks))
        self.assertTrue(all(len(vector) == 8 for vector in result.embeddings))


if __name__ == "__main__":
    unittest.main()

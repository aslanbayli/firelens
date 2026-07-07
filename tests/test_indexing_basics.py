import math
import tempfile
import unittest
import uuid
from pathlib import Path

from app.core.models import Symbol
from app.indexing.chunker import build_embedding_text, calculate_content_hash, chunk_symbols
from app.indexing.embedder import FakeEmbedder
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


class WalkerTests(unittest.TestCase):
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

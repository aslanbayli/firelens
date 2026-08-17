import tempfile
import unittest
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.models import Repository
from app.core.runtime import FireLensRuntime
from app.indexing.adapters import LanguageAdapterRegistry
from app.indexing.analysis import SourceFile
from app.indexing.chunker import group_spans_with_windows, subtract_owned_spans
from app.indexing.documentation_adapter import DocumentationAdapter
from app.indexing.embedder import FakeEmbedder
from app.indexing.python_adapter import PythonAdapter
from app.indexing.version import INDEX_FORMAT_VERSION
from app.interfaces.cli import build_parser
from app.search.lexical import build_safe_fts_query, safe_fts_terms
from app.storage.database import SQLiteIndexStore


class AdapterContractTests(unittest.TestCase):
    def test_registry_selects_extensions_and_rejects_ambiguity(self) -> None:
        registry = LanguageAdapterRegistry()
        registry.register(PythonAdapter())

        self.assertEqual(registry.require_adapter("source.PY").language, "python")
        self.assertFalse(registry.supports("source.go"))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            registry.register(PythonAdapter())

    def test_python_analysis_preserves_symbols_and_adds_semantic_units(self) -> None:
        source = (
            '"""Module documentation."""\n'
            "import os\n"
            "SETTING = 1\n\n"
            "# Authenticate a caller.\n"
            "def authenticate_user(user):\n"
            '    """Check a user credential."""\n'
            "    return bool(user)\n\n"
            "print(SETTING)\n"
        )
        adapter = PythonAdapter()
        parsed = adapter.analyze(SourceFile("service.py", "python", source))

        self.assertFalse(parsed.diagnostics)
        self.assertEqual(
            [(symbol.qualified_name, symbol.kind) for symbol in parsed.symbols],
            [("authenticate_user", "function")],
        )
        kinds = {unit.kind for unit in parsed.semantic_units}
        self.assertTrue(
            {
                "symbol",
                "module_docstring",
                "symbol_docstring",
                "imports",
                "assignment",
                "module_code",
                "symbol_comment",
            }.issubset(kinds)
        )
        self.assertTrue(
            all(unit.embedding_text.startswith("Language: python\nPath: service.py\n")
                for unit in parsed.semantic_units)
        )

    def test_identifier_normalization_splits_common_python_names(self) -> None:
        adapter = PythonAdapter()
        terms = adapter.identifier_terms("HTTPClient.fetch_userID")

        self.assertIn("httpclient.fetch_userid", terms)
        self.assertIn("http", terms)
        self.assertIn("client", terms)
        self.assertIn("fetch", terms)
        self.assertIn("user", terms)
        self.assertIn("id", terms)

    def test_documentation_adapter_extracts_sections(self) -> None:
        source = "# Install\nRun the setup.\n\n# Search\nFind code quickly.\n"
        parsed = DocumentationAdapter().analyze(
            SourceFile("README.md", "documentation", source)
        )

        self.assertEqual(len(parsed.semantic_units), 2)
        self.assertTrue(
            all(unit.kind == "documentation" for unit in parsed.semantic_units)
        )

    def test_span_subtraction_and_windows_are_deterministic(self) -> None:
        self.assertEqual(
            subtract_owned_spans(1, 12, [(3, 5), (8, 10)]),
            [(1, 2), (6, 7), (11, 12)],
        )
        self.assertEqual(
            group_spans_with_windows([(1, 3), (4, 7)], max_lines=4, overlap=1),
            [(1, 4), (4, 7)],
        )


class LexicalQueryTests(unittest.TestCase):
    def test_fts_query_quotes_only_tokenized_terms(self) -> None:
        query = build_safe_fts_query('foo" OR * bar_baz', "content")

        self.assertNotIn("*", query)
        self.assertIn('content : "foo"', query)
        self.assertIn('content : "bar"', query)
        self.assertEqual(safe_fts_terms("HTTPClient"), ("httpclient", "http", "client"))

    def test_cli_accepts_lexical_hybrid_and_graph_modes(self) -> None:
        for mode in ("lexical", "hybrid_rrf", "hybrid_weighted", "graph"):
            with self.subTest(mode=mode):
                parsed = build_parser().parse_args(
                    ["search", ".", "query", "--mode", mode]
                )
                self.assertEqual(parsed.mode, mode)


class RetrievalCoreIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.source = self.repository / "service.py"
        self.source.write_text(
            "import secrets\n\n"
            "TOKEN_TTL = 60\n\n"
            "def authenticateUser(user):\n"
            "    # Check whether credentials are valid.\n"
            "    return bool(user)\n\n"
            "print('service ready')\n",
            encoding="utf-8",
        )
        (self.repository / "README.md").write_text(
            "# Authentication\nCredentials are checked by the local service.\n",
            encoding="utf-8",
        )
        self.settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=8,
        )
        self.runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_index_and_retrieve_symbol_module_code_and_documentation(self) -> None:
        report = self.runtime.index_repository(self.repository)

        self.assertEqual(report.index_format_version, INDEX_FORMAT_VERSION)
        self.assertEqual(report.file_count, 2)
        self.assertGreater(report.lexical_document_count, report.symbol_count)

        exact = self.runtime.search_code(
            self.repository, "authenticateUser", mode="lexical"
        )
        self.assertEqual(exact.mode, "lexical")
        self.assertEqual(exact.ranked_results[0].result_type, "symbol")
        self.assertIn("exact_qualified", exact.ranked_results[0].retrieval_channels)
        self.assertTrue(exact.retrieval_config.startswith("default:"))

        natural = self.runtime.search_code(
            self.repository,
            "credentials checked by local service",
            mode="lexical",
            top_k=10,
        )
        self.assertTrue(natural.ranked_results)
        self.assertTrue(
            any(result.file_path == "README.md" for result in natural.ranked_results)
        )
        self.assertTrue(
            all(result.language for result in natural.ranked_results)
        )
        self.assertTrue(
            all(result.retrieval_evidence for result in natural.ranked_results)
        )

        reopened_runtime = FireLensRuntime(
            self.settings,
            embedder_factory=lambda: self.fail(
                "lexical search must not initialize the embedder"
            ),
        )
        reopened = reopened_runtime.search_code(
            self.repository,
            "credentials checked by local service",
            mode="lexical",
            top_k=10,
        )
        self.assertEqual(
            [
                (result.id, result.score, result.retrieval_channels)
                for result in natural.ranked_results
            ],
            [
                (result.id, result.score, result.retrieval_channels)
                for result in reopened.ranked_results
            ],
        )

        module_code = self.runtime.search_code(
            self.repository, "service ready", mode="semantic", top_k=10
        )
        self.assertTrue(
            any(
                result.semantic_unit_kind == "module_code"
                for result in module_code.ranked_results
            )
        )

    def test_path_filter_applies_to_every_lexical_channel_and_deletion(self) -> None:
        self.runtime.index_repository(self.repository)

        filtered = self.runtime.search_code(
            self.repository,
            "credentials",
            mode="lexical",
            path="service.py",
            top_k=20,
        )
        self.assertTrue(filtered.ranked_results)
        self.assertEqual(
            {result.file_path for result in filtered.ranked_results},
            {"service.py"},
        )

        self.source.write_text(
            "def replacement():\n"
            "    return 'new module behavior'\n",
            encoding="utf-8",
        )
        changed_report = self.runtime.index_repository(self.repository)
        self.assertEqual(changed_report.changed_file_count, 1)
        stale_source = self.runtime.search_code(
            self.repository,
            "credentials",
            mode="lexical",
            path="service.py",
            top_k=20,
        )
        self.assertFalse(stale_source.ranked_results)

        (self.repository / "README.md").unlink()
        report = self.runtime.index_repository(self.repository)
        self.assertEqual(report.deleted_file_count, 1)
        after_delete = self.runtime.search_code(
            self.repository, "Credentials checked", mode="lexical", top_k=20
        )
        self.assertNotIn(
            "README.md",
            {result.file_path for result in after_delete.ranked_results},
        )

        store = SQLiteIndexStore(Path(report.database_path))
        repository = store.load_latest_repository(str(self.repository.resolve()))
        self.assertIsNotNone(repository)
        self.assertEqual(
            store.count_rows("lexical_documents", repository.id),
            report.lexical_document_count,
        )

    def test_old_format_is_reported_stale_and_rebuilt(self) -> None:
        database_path = self.runtime.index_service.resolver.resolve(
            self.repository
        ).database_path
        old_repository = Repository(
            id=uuid.uuid4(),
            absolute_path=str(self.repository.resolve()),
            index_format_version="1",
            timestamp_of_index=1,
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_dim=8,
        )
        store = SQLiteIndexStore(database_path)
        store.initialize()
        store.replace_index(old_repository, [], [], [], [])

        stale = self.runtime.get_index_status(self.repository)
        self.assertEqual(stale.status, "stale")
        self.assertIn("complete rebuild", stale.warnings[0])

        rebuilt = self.runtime.index_repository(self.repository)
        self.assertEqual(rebuilt.index_format_version, INDEX_FORMAT_VERSION)
        repositories = store.list_repositories()
        self.assertEqual(len(repositories), 1)
        self.assertEqual(repositories[0].index_format_version, INDEX_FORMAT_VERSION)


if __name__ == "__main__":
    unittest.main()

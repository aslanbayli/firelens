import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.cancellation import OperationCancelledError
from app.core.config import Settings
from app.core.runtime import FireLensRuntime
from app.indexing.analysis import SourceFile
from app.indexing.embedder import FakeEmbedder
from app.indexing.indexer import IndexingCancelledError
from app.indexing.python_adapter import PythonAdapter
from app.storage.database import SQLiteIndexStore


class PythonGraphExtractionTests(unittest.TestCase):
    def test_adapter_emits_neutral_nodes_and_every_python_fact_category(self) -> None:
        source = SourceFile(
            relative_path="tests/test_service.py",
            language="python",
            text=(
                "from ..app.base import Base as Parent\n"
                "from ..app.util import helper as run_helper\n"
                "import app.dependency as dependency\n\n"
                "class TestService(Parent):\n"
                "    def test_run(self, service_fixture):\n"
                "        dependency.load()\n"
                "        return run_helper()\n"
            ),
        )

        document = PythonAdapter().analyze(source)

        self.assertFalse(document.diagnostics)
        self.assertEqual(
            {node.node_kind for node in document.graph_nodes},
            {"module", "test_file", "test_symbol"},
        )
        fact_kinds = {fact.edge_kind for fact in document.graph_facts}
        self.assertTrue(
            {"imports", "calls", "inherits", "references", "tests"}
            <= fact_kinds
        )
        imported_call = next(
            fact
            for fact in document.graph_facts
            if fact.edge_kind == "calls" and fact.target_reference == "run_helper"
        )
        self.assertEqual(imported_call.target_qualified_hint, "app.util.helper")
        self.assertEqual(
            imported_call.hint_resolution_method,
            "explicitly_imported_symbol",
        )
        nested_test_fact = next(
            fact
            for fact in document.graph_facts
            if fact.edge_kind == "tests" and fact.target_reference == "run"
        )
        self.assertEqual(
            nested_test_fact.source_reference,
            "tests.test_service.TestService.test_run",
        )


class GraphIndexingAndRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.settings = Settings(
            _env_file=None,
            data_dir=self.root / "indexes",
            allowed_roots=[self.root],
            embedding_provider=FakeEmbedder.provider,
            embedding_model=FakeEmbedder.model,
            embedding_revision=None,
            embedding_dimension=8,
            graph_seed_mode="lexical",
            graph_seed_count=1,
            graph_max_hops=2,
            graph_max_neighbors_per_node=20,
            graph_max_expanded_nodes=20,
            graph_allowed_edge_kinds=[
                "calls",
                "imports",
                "inherits",
                "references",
                "depends_on",
                "tests",
            ],
            graph_directions=["outgoing"],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_index_resolves_all_relationships_with_provenance_and_diagnostics(
        self,
    ) -> None:
        self._write_relationship_fixture()
        runtime = self._runtime()

        report = runtime.index_repository(self.repository)
        store = SQLiteIndexStore(Path(report.database_path))
        repository = store.load_latest_repository(str(self.repository.resolve()))
        assert repository is not None
        nodes = store.load_graph_nodes(repository.id)
        edges = store.load_graph_edges(repository.id)

        self.assertEqual(
            {"imports", "calls", "inherits", "references", "depends_on", "tests"},
            {edge.kind for edge in edges},
        )
        self.assertGreater(report.graph_node_count, report.symbol_count)
        self.assertEqual(report.graph_edge_count, len(edges))
        self.assertGreater(report.unresolved_graph_fact_count, 0)
        self.assertTrue(
            all(
                edge.extraction_adapter == "python_ast"
                and edge.adapter_version == "1"
                and 0.0 <= edge.confidence <= 1.0
                and edge.resolution_method
                for edge in edges
            )
        )
        qualified_by_id = {node.id: node.qualified_name for node in nodes}
        inherited = next(edge for edge in edges if edge.kind == "inherits")
        self.assertEqual(qualified_by_id[inherited.target_node_id], "app.base.Base")
        imported_call = next(
            edge
            for edge in edges
            if edge.kind == "calls" and edge.source_file == "app/service.py"
        )
        self.assertEqual(
            imported_call.resolution_method,
            "explicitly_imported_symbol",
        )

    def test_ambiguous_short_names_remain_visible_in_index_diagnostics(self) -> None:
        self._write("a.py", "def helper():\n    return 1\n")
        self._write("b.py", "def helper():\n    return 2\n")
        self._write("caller.py", "def caller():\n    return helper()\n")

        report = self._runtime().index_repository(self.repository)
        store = SQLiteIndexStore(Path(report.database_path))
        repository = store.load_latest_repository(str(self.repository.resolve()))
        assert repository is not None

        self.assertGreaterEqual(report.ambiguous_graph_fact_count, 1)
        self.assertEqual(
            report.ambiguous_graph_fact_count,
            store.count_graph_facts_by_status(repository.id, "ambiguous"),
        )
        caller_edges = [
            edge
            for edge in store.load_graph_edges(repository.id, "calls")
            if edge.source_file == "caller.py"
        ]
        self.assertEqual(caller_edges, [])

    def test_changed_and_deleted_files_replace_graph_records(self) -> None:
        self._write_relationship_fixture()
        runtime = self._runtime()
        first = runtime.index_repository(self.repository)
        first_store = SQLiteIndexStore(Path(first.database_path))
        first_repository = first_store.load_latest_repository(
            str(self.repository.resolve())
        )
        assert first_repository is not None
        self.assertTrue(
            any(
                fact.target_reference == "run_helper"
                for fact in first_store.load_graph_facts(first_repository.id)
            )
        )

        (self.repository / "app" / "util.py").unlink()
        self._write(
            "app/service.py",
            "from .base import Base\n\n"
            "class Service(Base):\n"
            "    def run(self):\n"
            "        return 1\n",
        )
        second = runtime.index_repository(self.repository)
        second_store = SQLiteIndexStore(Path(second.database_path))
        second_repository = second_store.load_latest_repository(
            str(self.repository.resolve())
        )
        assert second_repository is not None
        nodes = second_store.load_graph_nodes(second_repository.id)
        facts = second_store.load_graph_facts(second_repository.id)

        self.assertEqual(second.deleted_file_count, 1)
        self.assertEqual(second.changed_file_count, 1)
        self.assertFalse(
            any(node.relative_path == "app/util.py" for node in nodes)
        )
        self.assertFalse(
            any(fact.target_reference == "run_helper" for fact in facts)
        )

    def test_graph_mode_bounds_hops_cycles_neighbors_and_explains_results(
        self,
    ) -> None:
        self._write(
            "chain.py",
            "def first():\n"
            "    second()\n"
            "    return second()\n\n"
            "def second():\n"
            "    first()\n"
            "    return third()\n\n"
            "def third():\n"
            "    return 3\n",
        )
        runtime = self._runtime()
        runtime.index_repository(self.repository)

        two_hop = runtime.search_code(
            self.repository,
            "first",
            mode="graph",
            top_k=3,
            backend="python",
        )
        names = [result.symbol_name for result in two_hop.ranked_results]
        self.assertEqual(names, ["first", "second", "third"])
        self.assertEqual(two_hop.mode, "graph")
        self.assertEqual(two_hop.retrieval_config.split(":", 1)[0], "graph")
        expanded = [result for result in two_hop.ranked_results if result.graph_evidence]
        self.assertEqual(len(expanded), 2)
        self.assertEqual([item.graph_evidence[0].hop_count for item in expanded], [1, 2])
        self.assertTrue(
            all(len(item.graph_evidence) == 1 for item in expanded)
        )
        self.assertLess(expanded[1].score, expanded[0].score)
        self.assertEqual(
            set(expanded[0].graph_evidence[0].model_dump()),
            {
                "originating_seed_id",
                "originating_seed_path",
                "edge_kind",
                "direction",
                "hop_count",
                "edge_confidence",
                "graph_contribution",
            },
        )

        one_hop_settings = self.settings.model_copy(
            update={"graph_max_hops": 1}
        )
        one_hop = self._runtime(one_hop_settings).search_code(
            self.repository,
            "first",
            mode="graph",
            top_k=3,
            backend="python",
        )
        self.assertEqual(
            [result.symbol_name for result in one_hop.ranked_results],
            ["first", "second"],
        )

        bounded_settings = self.settings.model_copy(
            update={
                "graph_max_hops": 1,
                "graph_max_neighbors_per_node": 1,
                "graph_max_expanded_nodes": 1,
            }
        )
        bounded = self._runtime(bounded_settings).search_code(
            self.repository,
            "first",
            mode="graph",
            top_k=3,
            backend="python",
        )
        self.assertEqual(len(bounded.ranked_results), 2)

        hybrid_seed_settings = self.settings.model_copy(
            update={"graph_seed_mode": "hybrid_rrf", "graph_max_hops": 1}
        )
        hybrid_seed_response = self._runtime(hybrid_seed_settings).search_code(
            self.repository,
            "first",
            mode="graph",
            top_k=2,
            backend="python",
        )
        self.assertEqual(
            [timing.component for timing in hybrid_seed_response.retrieval_timings],
            ["lexical", "semantic", "fusion", "graph"],
        )

    def test_incoming_direction_and_cancellation_during_expansion(self) -> None:
        self._write(
            "chain.py",
            "def first():\n    return second()\n\n"
            "def second():\n    return third()\n\n"
            "def third():\n    return 3\n",
        )
        runtime = self._runtime()
        runtime.index_repository(self.repository)
        incoming_settings = self.settings.model_copy(
            update={"graph_max_hops": 1, "graph_directions": ["incoming"]}
        )
        incoming_runtime = self._runtime(incoming_settings)
        incoming = incoming_runtime.search_code(
            self.repository,
            "third",
            mode="graph",
            top_k=2,
            backend="python",
        )
        self.assertEqual(
            [result.symbol_name for result in incoming.ranked_results],
            ["third", "second"],
        )
        self.assertEqual(
            incoming.ranked_results[1].graph_evidence[0].direction,
            "incoming",
        )

        expansion_started = False
        original = SQLiteIndexStore.load_graph_adjacency

        def load_and_cancel(store, *args, **kwargs):
            nonlocal expansion_started
            result = original(store, *args, **kwargs)
            expansion_started = True
            return result

        with patch.object(SQLiteIndexStore, "load_graph_adjacency", load_and_cancel):
            with self.assertRaises(OperationCancelledError):
                runtime.search_code(
                    self.repository,
                    "first",
                    mode="graph",
                    top_k=3,
                    backend="python",
                    cancellation_callback=lambda: expansion_started,
                )

    def test_cancellation_during_resolution_preserves_previous_graph(self) -> None:
        self._write("stable.py", "def stable():\n    return 1\n")
        runtime = self._runtime()
        first = runtime.index_repository(self.repository)
        initial_edge_count = first.graph_edge_count
        self._write(
            "new_calls.py",
            "from stable import stable\n\n"
            "def added():\n"
            "    return stable()\n",
        )
        resolving_graph = False

        def track_progress(event) -> None:
            nonlocal resolving_graph
            if event.stage == "graph" and event.current == 0:
                resolving_graph = True

        with self.assertRaises(IndexingCancelledError):
            runtime.index_repository(
                self.repository,
                progress_callback=track_progress,
                cancellation_callback=lambda: resolving_graph,
            )

        store = SQLiteIndexStore(Path(first.database_path))
        repository = store.load_latest_repository(str(self.repository.resolve()))
        assert repository is not None
        self.assertEqual(
            store.count_rows("graph_edges", repository.id),
            initial_edge_count,
        )
        self.assertFalse(
            any(
                node.relative_path == "new_calls.py"
                for node in store.load_graph_nodes(repository.id)
            )
        )

    def _runtime(self, settings: Settings | None = None) -> FireLensRuntime:
        return FireLensRuntime(
            settings or self.settings,
            embedder_factory=lambda: FakeEmbedder(dimension=8),
        )

    def _write_relationship_fixture(self) -> None:
        self._write("app/__init__.py", "\n")
        self._write("app/base.py", "class Base:\n    pass\n")
        self._write("app/util.py", "def helper():\n    return 1\n")
        self._write(
            "app/service.py",
            "from .base import Base\n"
            "from .util import helper as run_helper\n\n"
            "class Service(Base):\n"
            "    def run(self):\n"
            "        return run_helper()\n",
        )
        self._write(
            "tests/test_service.py",
            "from app.service import Service\n\n"
            "def test_run(service_fixture):\n"
            "    return Service().run()\n",
        )

    def _write(self, relative_path: str, source: str) -> None:
        path = self.repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

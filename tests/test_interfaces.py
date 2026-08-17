import asyncio
import io
import json
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters, stdio_client
from mcp.client import Client

from app.core.models import (
    IndexRepositoryResponse,
    IndexStatusResponse,
    SearchResponse,
    SearchResult,
)
from app.indexing.indexer import (
    CancellationCallback,
    IndexingProgress,
    ProgressCallback,
)
from app.interfaces.cli import run
from app.interfaces.mcp_server import create_mcp_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeRuntime:
    """Small runtime double that keeps interface tests deterministic."""

    def __init__(self) -> None:
        self.index_calls: list[str] = []
        self.status_calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []

    def index_repository(
        self,
        repository_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> IndexRepositoryResponse:
        del cancellation_callback
        repository = str(repository_path)
        self.index_calls.append(repository)
        if progress_callback is not None:
            progress_callback(
                IndexingProgress(
                    stage="discovery",
                    current=1,
                    total=2,
                    message="Discovered source files",
                )
            )
            progress_callback(
                IndexingProgress(
                    stage="persistence",
                    current=2,
                    total=2,
                    message="Promoted staged index",
                )
            )

        return IndexRepositoryResponse(
            repository_path=repository,
            database_path="/indexes/example/firelens.db",
            status="ready",
            index_format_version="1",
            timestamp_of_index=1_754_435_200,
            embedding_provider="test",
            embedding_model="sha256-fake",
            embedding_dim=8,
            file_count=1,
            symbol_count=1,
            chunk_count=1,
            embedding_count=1,
            added_file_count=1,
            embedded_chunk_count=1,
            changed_paths=["app/service.py"],
            elapsed_time=0.01,
        )

    def get_index_status(
        self,
        repository_path: str | Path,
        cancellation_callback: CancellationCallback | None = None,
    ) -> IndexStatusResponse:
        del cancellation_callback
        repository = str(repository_path)
        self.status_calls.append(repository)
        return IndexStatusResponse(
            repository_path=repository,
            database_path="/indexes/example/firelens.db",
            status="ready",
            index_format_version="1",
            timestamp_of_index=1_754_435_200,
            embedding_provider="test",
            embedding_model="sha256-fake",
            embedding_dim=8,
            file_count=1,
            symbol_count=1,
            chunk_count=1,
            embedding_count=1,
        )

    def search_code(
        self,
        repository_path: str | Path,
        query: str,
        mode: str = "auto",
        top_k: int = 5,
        path: str | None = None,
        backend: str = "auto",
        max_snippet_chars: int = 2_000,
        cancellation_callback: CancellationCallback | None = None,
    ) -> SearchResponse:
        del cancellation_callback
        arguments = {
            "repository_path": str(repository_path),
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "path": path,
            "backend": backend,
            "max_snippet_chars": max_snippet_chars,
        }
        self.search_calls.append(arguments)
        actual_mode = "semantic" if mode == "auto" else mode
        requested_backend = backend
        actual_backend = "python" if backend == "auto" else backend
        return SearchResponse(
            original_query=query,
            requested_mode=mode,
            mode=actual_mode,
            requested_backend=requested_backend,
            backend=actual_backend,
            elapsed_time=0.002,
            ranked_results=[
                SearchResult(
                    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    result_type="symbol",
                    file_path="app/service.py",
                    start_line=4,
                    end_line=6,
                    symbol_name="authenticate",
                    snippet="def authenticate(user):\n    return user.is_valid",
                    score=0.98,
                    mode=actual_mode,
                    backend=actual_backend,
                )
            ][:top_k],
        )


class _FailingRuntime(_FakeRuntime):
    def get_index_status(
        self,
        repository_path: str | Path,
    ) -> IndexStatusResponse:
        raise ValueError("Repository path is outside FIRELENS_ALLOWED_ROOTS")


class CliContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = _FakeRuntime()

    def test_index_writes_structured_json_and_progress_to_stderr(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = run(["index", "/repo"], runtime=self.runtime)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["embedding_model"], "sha256-fake")
        self.assertEqual(self.runtime.index_calls, ["/repo"])
        self.assertIn("[discovery] 1/2 Discovered source files", stderr.getvalue())
        self.assertIn("[persistence] 2/2 Promoted staged index", stderr.getvalue())
        self.assertNotIn("discovery", stdout.getvalue())

    def test_search_forwards_options_and_writes_structured_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = run(
                [
                    "search",
                    "/repo",
                    "authentication",
                    "--mode",
                    "semantic",
                    "--top-k",
                    "3",
                    "--path",
                    "app",
                    "--backend",
                    "python",
                    "--max-snippet-chars",
                    "512",
                ],
                runtime=self.runtime,
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["original_query"], "authentication")
        self.assertEqual(payload["requested_mode"], "semantic")
        self.assertEqual(payload["ranked_results"][0]["file_path"], "app/service.py")
        self.assertEqual(
            self.runtime.search_calls,
            [
                {
                    "repository_path": "/repo",
                    "query": "authentication",
                    "mode": "semantic",
                    "top_k": 3,
                    "path": "app",
                    "backend": "python",
                    "max_snippet_chars": 512,
                }
            ],
        )

    def test_runtime_errors_use_stderr_and_a_nonzero_exit_code(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = run(
                ["status", "/outside"],
                runtime=_FailingRuntime(),
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            "firelens: Repository path is outside FIRELENS_ALLOWED_ROOTS\n",
        )


class McpInMemoryContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.runtime = _FakeRuntime()
        self.server = create_mcp_server(self.runtime)

    async def test_tool_listing_includes_annotations_and_output_schemas(self) -> None:
        async with Client(self.server) as client:
            tools_result = await client.list_tools(cache_mode="bypass")

            self.assertTrue(
                client.instructions.startswith("Use get_index_status first.")
            )

        tools = {tool.name: tool for tool in tools_result.tools}
        self.assertEqual(
            set(tools),
            {"index_repository", "get_index_status", "search_code"},
        )

        index_annotations = tools["index_repository"].annotations
        self.assertIsNotNone(index_annotations)
        self.assertFalse(index_annotations.read_only_hint)
        self.assertTrue(index_annotations.destructive_hint)
        self.assertFalse(index_annotations.idempotent_hint)
        self.assertTrue(index_annotations.open_world_hint)

        status_annotations = tools["get_index_status"].annotations
        self.assertIsNotNone(status_annotations)
        self.assertTrue(status_annotations.read_only_hint)
        self.assertFalse(status_annotations.open_world_hint)

        search_annotations = tools["search_code"].annotations
        self.assertIsNotNone(search_annotations)
        self.assertTrue(search_annotations.read_only_hint)
        self.assertFalse(search_annotations.open_world_hint)

        expected_output_properties = {
            "index_repository": {"status", "embedding_model", "changed_paths"},
            "get_index_status": {"status", "file_count", "changed_paths"},
            "search_code": {"requested_mode", "mode", "ranked_results"},
        }
        for tool_name, property_names in expected_output_properties.items():
            schema = tools[tool_name].output_schema
            self.assertIsNotNone(schema)
            self.assertEqual(schema["type"], "object")
            self.assertTrue(property_names <= set(schema["properties"]))

        search_input = tools["search_code"].input_schema["properties"]
        self.assertEqual(search_input["mode"]["default"], "auto")
        self.assertTrue(
            {"hybrid_rrf", "hybrid_weighted", "graph"}
            <= set(search_input["mode"]["enum"])
        )
        self.assertEqual(search_input["query"]["maxLength"], 2_000)
        self.assertEqual(search_input["top_k"]["maximum"], 20)
        self.assertEqual(search_input["max_snippet_chars"]["maximum"], 4_000)

    async def test_status_and_search_return_structured_content(self) -> None:
        async with Client(self.server) as client:
            status_result = await client.call_tool(
                "get_index_status",
                {"repository_path": "/repo"},
            )
            search_result = await client.call_tool(
                "search_code",
                {
                    "repository_path": "/repo",
                    "query": "authentication",
                },
            )

        self.assertFalse(status_result.is_error)
        self.assertEqual(status_result.structured_content["status"], "ready")
        self.assertEqual(status_result.structured_content["file_count"], 1)

        self.assertFalse(search_result.is_error)
        self.assertEqual(
            search_result.structured_content["original_query"],
            "authentication",
        )
        self.assertEqual(search_result.structured_content["requested_mode"], "auto")
        self.assertEqual(search_result.structured_content["mode"], "semantic")
        self.assertEqual(
            search_result.structured_content["ranked_results"][0]["file_path"],
            "app/service.py",
        )
        self.assertEqual(self.runtime.status_calls, ["/repo"])
        self.assertEqual(self.runtime.search_calls[0]["top_k"], 5)
        self.assertEqual(self.runtime.search_calls[0]["max_snippet_chars"], 2_000)

    async def test_index_returns_structured_content_and_reports_progress(self) -> None:
        progress_events: list[tuple[float, float | None, str | None]] = []

        async def record_progress(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            progress_events.append((progress, total, message))

        async with Client(self.server) as client:
            result = await client.call_tool(
                "index_repository",
                {"repository_path": "/repo"},
                progress_callback=record_progress,
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "ready")
        self.assertEqual(result.structured_content["embedded_chunk_count"], 1)
        self.assertEqual(self.runtime.index_calls, ["/repo"])
        self.assertEqual([event[0] for event in progress_events], [1.0, 2.0])
        self.assertIsNone(progress_events[0][1])
        self.assertIn("discovery: 1/2", progress_events[0][2])
        self.assertIn("persistence: 2/2", progress_events[1][2])


class McpStdioContractTests(unittest.TestCase):
    def test_stdio_server_lifecycle_with_real_services_and_fake_embedder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_root = Path(temp_dir)
            repository = temporary_root / "repository"
            data_dir = temporary_root / "indexes"
            repository.mkdir()
            (repository / "example.py").write_text(
                "def example():\n    return 1\n",
                encoding="utf-8",
            )

            server_stderr = asyncio.run(
                self._exercise_stdio_lifecycle(repository, data_dir)
            )

        self.assertNotIn("Traceback", server_stderr)

    async def _exercise_stdio_lifecycle(
        self,
        repository: Path,
        data_dir: Path,
    ) -> str:
        environment = {
            "FIRELENS_ALLOWED_ROOTS": str(repository),
            "FIRELENS_DATA_DIR": str(data_dir),
        }
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(PROJECT_ROOT / "tests" / "stdio_fake_server.py")],
            env=environment,
            cwd=PROJECT_ROOT,
        )
        progress_events: list[tuple[float, float | None, str | None]] = []

        async def record_progress(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            progress_events.append((progress, total, message))

        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as server_stderr:
            async with Client(
                stdio_client(parameters, errlog=server_stderr),
                read_timeout_seconds=20,
            ) as client:
                self.assertIsNotNone(client.protocol_version)
                self.assertTrue(
                    client.instructions.startswith("Use get_index_status first.")
                )

                tools_result = await client.list_tools(cache_mode="bypass")
                self.assertEqual(
                    {tool.name for tool in tools_result.tools},
                    {"index_repository", "get_index_status", "search_code"},
                )

                missing_result = await client.call_tool(
                    "get_index_status",
                    {"repository_path": str(repository)},
                )
                self.assertFalse(missing_result.is_error)
                self.assertEqual(missing_result.structured_content["status"], "missing")
                self.assertEqual(
                    missing_result.structured_content["repository_path"],
                    str(repository.resolve()),
                )
                self.assertFalse(data_dir.exists())

                index_result = await client.call_tool(
                    "index_repository",
                    {"repository_path": str(repository)},
                    progress_callback=record_progress,
                )
                self.assertFalse(index_result.is_error)
                self.assertEqual(index_result.structured_content["status"], "ready")
                self.assertEqual(index_result.structured_content["file_count"], 1)
                self.assertEqual(index_result.structured_content["embedding_count"], 1)

                ready_result = await client.call_tool(
                    "get_index_status",
                    {"repository_path": str(repository)},
                )
                self.assertFalse(ready_result.is_error)
                self.assertEqual(ready_result.structured_content["status"], "ready")

                search_result = await client.call_tool(
                    "search_code",
                    {
                        "repository_path": str(repository),
                        "query": "example",
                        "backend": "python",
                    },
                )
                self.assertFalse(search_result.is_error)
                self.assertEqual(search_result.structured_content["mode"], "exact")
                self.assertEqual(
                    search_result.structured_content["ranked_results"][0]["file_path"],
                    "example.py",
                )

                hybrid_result = await client.call_tool(
                    "search_code",
                    {
                        "repository_path": str(repository),
                        "query": "function returning a value",
                        "mode": "hybrid_rrf",
                        "backend": "python",
                    },
                )
                self.assertFalse(hybrid_result.is_error)
                self.assertEqual(
                    hybrid_result.structured_content["mode"],
                    "hybrid_rrf",
                )
                self.assertTrue(
                    hybrid_result.structured_content["retrieval_config"].startswith(
                        "hybrid_rrf:"
                    )
                )
                self.assertEqual(
                    [
                        timing["component"]
                        for timing in hybrid_result.structured_content[
                            "retrieval_timings"
                        ]
                    ],
                    ["lexical", "semantic", "fusion"],
                )

                invalid_result = await client.call_tool(
                    "search_code",
                    {
                        "repository_path": str(repository),
                        "query": "example",
                        "top_k": 0,
                    },
                )
                self.assertTrue(invalid_result.is_error)

            self.assertGreater(len(progress_events), 0)
            progress_values = [event[0] for event in progress_events]
            self.assertEqual(progress_values, sorted(progress_values))
            progress_messages = [event[2] or "" for event in progress_events]
            self.assertTrue(
                any(message.startswith("walk:") for message in progress_messages)
            )
            self.assertTrue(
                any(message.startswith("embed:") for message in progress_messages)
            )
            self.assertTrue(
                any(message.startswith("write:") for message in progress_messages)
            )

            server_stderr.seek(0)
            stderr_text = server_stderr.read()

        return stderr_text


if __name__ == "__main__":
    unittest.main()

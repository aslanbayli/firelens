"""Local STDIO MCP server for coding-agent repository retrieval."""

import asyncio
import concurrent.futures
import threading
import time
from typing import Annotated, Callable, Literal, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from app.core.cancellation import CancellationCallback
from app.core.models import (
    IndexRepositoryResponse,
    IndexStatusResponse,
    SearchResponse,
)
from app.core.runtime import FireLensRuntime, build_runtime
from app.indexing.indexer import (
    IndexingProgress,
    raise_if_indexing_cancelled,
)


PROGRESS_PERCENT_STEP = 10
PROGRESS_REPORT_TIMEOUT_SECONDS = 30.0
PROGRESS_WAIT_INTERVAL_SECONDS = 0.1
WorkerResult = TypeVar("WorkerResult")

SERVER_INSTRUCTIONS = """Use get_index_status first. If the index is missing or
stale, call index_repository and wait for completion. Then call search_code for
bounded repository context. Repository paths must be inside the configured
FIRELENS_ALLOWED_ROOTS. Search never refreshes an index implicitly."""


class _ProgressThrottle:
    """Limit each indexing stage to percentage milestone notifications."""

    def __init__(self) -> None:
        self._last_bucket_by_stage: dict[str, int] = {}

    def should_report(self, event: IndexingProgress) -> bool:
        """Return whether an event starts a stage or crosses a 10% boundary."""

        if event.total <= 0:
            bucket = 0
        else:
            bounded_current = min(max(event.current, 0), event.total)
            percentage = (bounded_current * 100) // event.total
            bucket = percentage // PROGRESS_PERCENT_STEP

        previous_bucket = self._last_bucket_by_stage.get(event.stage)
        if previous_bucket is not None and bucket <= previous_bucket:
            return False

        self._last_bucket_by_stage[event.stage] = bucket
        return True


def _wait_for_progress_delivery(
    future: concurrent.futures.Future[object],
    cancellation_callback: Callable[[], bool],
) -> None:
    """Wait boundedly for one progress notification from a worker thread."""

    deadline = time.monotonic() + PROGRESS_REPORT_TIMEOUT_SECONDS
    while True:
        if cancellation_callback():
            future.cancel()
            raise_if_indexing_cancelled(cancellation_callback)

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            future.cancel()
            raise TimeoutError("Timed out while reporting indexing progress")

        try:
            future.result(
                timeout=min(PROGRESS_WAIT_INTERVAL_SECONDS, remaining_seconds)
            )
            break
        except concurrent.futures.TimeoutError:
            if future.done():
                # Propagate a TimeoutError raised by the notification itself;
                # only an unfinished future represents a polling timeout.
                future.result()
            continue

    raise_if_indexing_cancelled(cancellation_callback)


async def _await_worker_cleanup(
    worker: asyncio.Task[WorkerResult],
) -> None:
    """Wait for a cancelled worker thread to release operation resources."""

    while not worker.done():
        try:
            await asyncio.wait({worker})
        except asyncio.CancelledError:
            # A client may repeat its cancellation while cleanup is underway.
            # The worker still needs to reach a safe cooperative boundary.
            continue
        except Exception:
            return

    if worker.cancelled():
        return

    try:
        worker.result()
    except Exception:
        # The original asyncio cancellation remains protocol-visible after
        # the synchronous operation acknowledges cancellation and exits.
        return


async def _run_cancellable_worker(
    operation: Callable[[CancellationCallback], WorkerResult],
) -> WorkerResult:
    """Run synchronous work and drain its thread before surfacing cancellation."""

    cancellation_requested = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(operation, cancellation_requested.is_set)
    )
    try:
        # asyncio.wait does not cancel the worker when this request task is
        # cancelled, so the thread remains awaitable during cleanup.
        await asyncio.wait({worker})
        return worker.result()
    except asyncio.CancelledError:
        cancellation_requested.set()
        await _await_worker_cleanup(worker)
        raise


async def _index_repository_with_cancellation(
    services: FireLensRuntime,
    repository_path: str,
    ctx: Context,
) -> IndexRepositoryResponse:
    """Run synchronous indexing and safely drain its worker on cancellation."""

    event_loop = asyncio.get_running_loop()
    progress_sequence = 0
    progress_throttle = _ProgressThrottle()
    progress_delivery_failed = threading.Event()

    def run_indexing(
        cancellation_callback: CancellationCallback,
    ) -> IndexRepositoryResponse:
        def report_progress(event: IndexingProgress) -> None:
            nonlocal progress_sequence
            raise_if_indexing_cancelled(cancellation_callback)
            if progress_delivery_failed.is_set():
                raise RuntimeError("Indexing progress delivery previously failed")
            if not progress_throttle.should_report(event):
                return
            progress_sequence += 1
            message = f"{event.stage}: {event.current}/{event.total} {event.message}"
            future = asyncio.run_coroutine_threadsafe(
                ctx.report_progress(float(progress_sequence), message=message),
                event_loop,
            )
            try:
                _wait_for_progress_delivery(future, cancellation_callback)
            except Exception:
                progress_delivery_failed.set()
                raise

        return services.index_repository(
            repository_path,
            progress_callback=report_progress,
            cancellation_callback=cancellation_callback,
        )

    return await _run_cancellable_worker(run_indexing)


async def _get_index_status_with_cancellation(
    services: FireLensRuntime,
    repository_path: str,
) -> IndexStatusResponse:
    """Run status inspection without abandoning its filesystem worker."""

    return await _run_cancellable_worker(
        lambda cancellation_callback: services.get_index_status(
            repository_path,
            cancellation_callback=cancellation_callback,
        )
    )


async def _search_code_with_cancellation(
    services: FireLensRuntime,
    *,
    repository_path: str,
    query: str,
    mode: Literal[
        "auto",
        "exact",
        "fuzzy",
        "lexical",
        "semantic",
        "hybrid_rrf",
        "hybrid_weighted",
    ],
    top_k: int,
    path: str | None,
    backend: Literal["auto", "python", "mojo"],
    max_snippet_chars: int,
) -> SearchResponse:
    """Run code search without abandoning its database/model worker."""

    def run_search(
        cancellation_callback: CancellationCallback,
    ) -> SearchResponse:
        return services.search_code(
            repository_path=repository_path,
            query=query,
            mode=mode,
            top_k=top_k,
            path=path,
            backend=backend,
            max_snippet_chars=max_snippet_chars,
            cancellation_callback=cancellation_callback,
        )

    return await _run_cancellable_worker(run_search)


def create_mcp_server(runtime: FireLensRuntime | None = None) -> MCPServer:
    """Create a configured server, optionally using a test runtime."""

    services = runtime or build_runtime()
    server = MCPServer(
        "firelens",
        version="0.1.0",
        instructions=SERVER_INSTRUCTIONS,
    )

    @server.tool(
        title="Index repository",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True,
        ),
        structured_output=True,
    )
    async def index_repository(
        repository_path: Annotated[
            str,
            Field(
                min_length=1,
                max_length=4_096,
                description="Local repository path inside an allowed root.",
            ),
        ],
        ctx: Context,
    ) -> IndexRepositoryResponse:
        """Create or incrementally update a repository's hybrid search index."""

        return await _index_repository_with_cancellation(
            services,
            repository_path,
            ctx,
        )

    @server.tool(
        title="Get index status",
        annotations=ToolAnnotations(
            read_only_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def get_index_status(
        repository_path: Annotated[
            str,
            Field(
                min_length=1,
                max_length=4_096,
                description="Local repository path to inspect.",
            ),
        ],
    ) -> IndexStatusResponse:
        """Check whether a repository index is missing, ready, stale, or indexing."""

        return await _get_index_status_with_cancellation(
            services,
            repository_path,
        )

    @server.tool(
        title="Search code",
        annotations=ToolAnnotations(
            read_only_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def search_code(
        repository_path: Annotated[
            str,
            Field(
                min_length=1,
                max_length=4_096,
                description="Indexed local repository path.",
            ),
        ],
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=2_000,
                description=(
                    "Symbol, typo, or code question. Explicit fuzzy queries "
                    "are limited to 256 characters."
                ),
            ),
        ],
        mode: Literal[
            "auto",
            "exact",
            "fuzzy",
            "lexical",
            "semantic",
            "hybrid_rrf",
            "hybrid_weighted",
        ] = "auto",
        top_k: Annotated[int, Field(ge=1, le=20)] = 5,
        path: Annotated[str, Field(max_length=4_096)] | None = None,
        backend: Literal["auto", "python", "mojo"] = "auto",
        max_snippet_chars: Annotated[int, Field(ge=1, le=4_000)] = 2_000,
    ) -> SearchResponse:
        """Search an existing index without implicitly refreshing it."""

        return await _search_code_with_cancellation(
            services,
            repository_path=repository_path,
            query=query,
            mode=mode,
            top_k=top_k,
            path=path,
            backend=backend,
            max_snippet_chars=max_snippet_chars,
        )

    return server


mcp = create_mcp_server()


def main() -> None:
    """Run the FireLens MCP server over standard input/output."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

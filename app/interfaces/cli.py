"""Small JSON CLI over the shared FireLens application runtime."""

import argparse
import sys
from collections.abc import Sequence

from app.core.models import RETRIEVAL_MODE_OPTIONS
from app.core.runtime import FireLensRuntime, build_runtime
from app.indexing.indexer import IndexingProgress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firelens",
        description="Index and search local code repositories.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="Index a repository")
    index_parser.add_argument("repository")

    status_parser = commands.add_parser("status", help="Check index freshness")
    status_parser.add_argument("repository")

    search_parser = commands.add_parser("search", help="Search an existing index")
    search_parser.add_argument("repository")
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODE_OPTIONS,
        default="auto",
    )
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--path")
    search_parser.add_argument(
        "--backend",
        choices=("auto", "python", "mojo"),
        default="auto",
    )
    search_parser.add_argument("--max-snippet-chars", type=int, default=2_000)

    return parser


def run(
    arguments: Sequence[str] | None = None,
    runtime: FireLensRuntime | None = None,
) -> int:
    """Execute one CLI command and return its process exit code."""

    parsed = build_parser().parse_args(arguments)
    services = runtime or build_runtime()

    try:
        if parsed.command == "index":
            response = services.index_repository(
                parsed.repository,
                progress_callback=_print_progress,
            )
        elif parsed.command == "status":
            response = services.get_index_status(parsed.repository)
        else:
            response = services.search_code(
                repository_path=parsed.repository,
                query=parsed.query,
                mode=parsed.mode,
                top_k=parsed.top_k,
                path=parsed.path,
                backend=parsed.backend,
                max_snippet_chars=parsed.max_snippet_chars,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"firelens: {error}", file=sys.stderr)
        return 2

    print(response.model_dump_json(indent=2))
    return 0


def main() -> None:
    raise SystemExit(run())


def _print_progress(event: IndexingProgress) -> None:
    print(
        f"[{event.stage}] {event.current}/{event.total} {event.message}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

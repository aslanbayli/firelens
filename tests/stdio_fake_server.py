"""Run the production FireLens MCP server with a deterministic test embedder."""

from app.core.config import Settings
from app.core.runtime import FireLensRuntime
from app.indexing.embedder import FakeEmbedder
from app.interfaces.mcp_server import create_mcp_server


def main() -> None:
    """Serve FireLens over STDIO without loading or downloading a real model."""

    settings = Settings(
        _env_file=None,
        embedding_provider=FakeEmbedder.provider,
        embedding_model=FakeEmbedder.model,
        embedding_revision=None,
        embedding_dimension=8,
    )
    runtime = FireLensRuntime(
        settings=settings,
        embedder_factory=lambda: FakeEmbedder(dimension=8),
    )
    create_mcp_server(runtime).run(transport="stdio")


if __name__ == "__main__":
    main()

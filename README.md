<div align="center">

# 🔥 FireLens

**Local-first code search for coding agents.**

Give MCP-compatible agents small, relevant, and explainable slices of a
repository—without turning the repository into a chat service.

[![Tests](https://img.shields.io/github/actions/workflow/status/aslanbayli/firelens/tests.yml?branch=main&label=tests)](https://github.com/aslanbayli/firelens/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/aslanbayli/firelens?sort=semver)](https://github.com/aslanbayli/firelens/releases)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-STDIO-6f42c1)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

[Quick start](#quick-start) · [Agent setup guide](#let-your-coding-agent-handle-setup) · [MCP tools](#mcp-tools) · [Contributing](CONTRIBUTING.md)

</div>

FireLens builds a durable SQLite index from local source code and repository
documentation. It combines symbol lookup, full-text search, semantic
similarity, and statically derived code relationships so an agent can retrieve
the context it needs instead of loading whole files or guessing from filenames.

> [!IMPORTANT]
> FireLens is a retrieval engine, not a chatbot. It does not generate answers or
> modify source files. Indexing and search run locally; the embedding model is
> downloaded once and then runs on the local machine.

## Why FireLens?

Coding agents are only as good as the context they receive. Plain text search
misses conceptual matches, vector search can miss exact identifiers, and either
one alone ignores how code is connected. FireLens keeps those signals separate,
fuses them explicitly, and returns bounded source snippets with provenance.

| Capability | What it gives an agent |
| --- | --- |
| Exact + fuzzy symbols | Fast lookup for known or misspelled Python identifiers |
| SQLite FTS5 lexical search | Names, paths, identifiers, content, and typo recovery |
| Local semantic search | Natural-language retrieval with CodeRankEmbed |
| Hybrid ranking | Deterministic lexical + semantic fusion with score evidence |
| Repository graph | Bounded expansion across calls, imports, inheritance, references, dependencies, and tests |
| Incremental indexing | Re-parse and re-embed only changed files; reuse unchanged embeddings |
| Agent-safe MCP | Explicit freshness checks, repository allowlists, cancellation, and bounded output |
| Optional Mojo kernels | Accelerated fuzzy scoring and large vector rankings with Python fallback |

FireLens structurally parses Python and semantically indexes Python, Markdown,
reStructuredText, and repository documentation sections.

## Quick start

### Requirements

- Python `>=3.14,<3.15`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git

Clone FireLens into a stable location and install its locked dependencies:

```bash
git clone https://github.com/aslanbayli/firelens.git /absolute/path/to/firelens
cd /absolute/path/to/firelens
uv sync --frozen
```

Then register the local STDIO server with your MCP client. For Codex:

```bash
codex mcp add firelens \
  --env FIRELENS_ALLOWED_ROOTS=/absolute/path/to/repositories \
  --env FIRELENS_DATA_DIR=/absolute/path/to/firelens-data \
  -- uv run --project /absolute/path/to/firelens firelens-mcp
```

Restart the client, then ask your agent:

```text
Use FireLens to check the index for this repository, refresh it if needed,
and find the code responsible for authentication.
```

The first semantic index may download
[`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed). Set a
generous MCP tool timeout and provide `HF_TOKEN` when the model requires it.

## Let your coding agent handle setup

<details>
<summary><strong>Copy a prompt to install, update, remove, or use FireLens</strong></summary>

These prompts are intentionally client-agnostic. A capable coding agent should
detect its own MCP configuration format and preserve unrelated settings.

### Install

```text
Install FireLens from https://github.com/aslanbayli/firelens as a local STDIO
MCP server for this coding agent and the repository I currently have open.

Requirements:
1. Verify that Git, uv, and Python 3.14 are available. If a system-wide install
   is needed, explain it and ask before making that change.
2. Clone FireLens into a stable user-level tools directory. If it is already
   present, do not overwrite it.
3. Run `uv sync --frozen` in the FireLens checkout.
4. Add an MCP server named `firelens` using:
   `uv run --project /absolute/path/to/firelens firelens-mcp`
5. Set `FIRELENS_ALLOWED_ROOTS` to the smallest parent directory containing the
   repository I have open. Set `FIRELENS_DATA_DIR` to a persistent directory
   outside all indexed repositories. Use absolute paths.
6. Preserve every unrelated MCP setting. Add `HF_TOKEN` only if it is already
   available; never print or copy the token value.
7. Use a 30-second startup timeout and a 600-second tool timeout when the client
   supports them.
8. Verify that `get_index_status`, `index_repository`, and `search_code` are
   visible. Tell me which files you changed and whether I need to restart the
   client. Do not index anything until I ask.
```

### Update

```text
Update my existing FireLens MCP installation.

Locate the checkout from the configured `firelens` MCP command. Confirm that it
is the official https://github.com/aslanbayli/firelens repository and that the
checkout has no uncommitted changes. Stop and explain if either check fails.
Otherwise, fetch the latest tags, fast-forward the current branch only, run
`uv sync --frozen`, and verify `uv run firelens --help`. Preserve my allowed
roots, data directory, tokens, and all unrelated MCP settings. Tell me whether
the MCP client must be restarted, then verify the three MCP tools after restart
and summarize the installed version.
```

### Remove

```text
Remove FireLens from this coding agent safely.

First locate and remove only the MCP registration named `firelens`, preserving
all unrelated settings, then verify that the client no longer starts the
server. Show me the FireLens checkout path, index data path, and any model cache
that would remain. Ask for confirmation before deleting any of those files;
indexes can contain source snippets and their deletion is not recoverable unless
backed up. Do not remove Git, uv, Python, shared model caches, or dependencies
used by other projects.
```

### Use

```text
Use FireLens for repository discovery before broad file reads or searches.

Call `get_index_status` with the repository's canonical absolute path. If the
status is `missing` or `stale`, call `index_repository` and wait for it to
finish. Then call `search_code` with a focused query, `top_k` between 3 and 8,
and a path filter when I name a subsystem. Use `auto` for identifiers or simple
questions, `hybrid_rrf` for broader concept discovery, and `graph` when callers,
dependencies, inheritance, or tests matter. Cite returned file paths and line
numbers in your answer. Never assume search refreshes the index automatically.
```

</details>

## MCP tools

FireLens exposes three structured tools over local STDIO:

| Tool | Purpose | Mutates the index? |
| --- | --- | --- |
| `get_index_status` | Report `missing`, `ready`, `stale`, or `indexing`, including changed paths | No |
| `index_repository` | Create or incrementally refresh a repository index | Yes |
| `search_code` | Search an existing index without silently refreshing it | No |

The intended workflow is always:

```text
get_index_status → index_repository when missing/stale → search_code
```

`FIRELENS_ALLOWED_ROOTS` limits which local paths the MCP server can access.
Separate multiple roots with `:` on macOS/Linux or `;` on Windows. Keep
`FIRELENS_DATA_DIR` outside those roots because indexes contain source snippets,
metadata, and embeddings.

<details>
<summary><strong>Codex configuration</strong></summary>

Add this to `~/.codex/config.toml`, using absolute paths:

```toml
[mcp_servers.firelens]
command = "uv"
args = [
  "run",
  "--project",
  "/absolute/path/to/firelens",
  "firelens-mcp",
]
startup_timeout_sec = 30
tool_timeout_sec = 600
env_vars = ["HF_TOKEN"]

[mcp_servers.firelens.env]
FIRELENS_ALLOWED_ROOTS = "/absolute/repo/one:/absolute/repo/two"
FIRELENS_DATA_DIR = "/absolute/path/to/firelens-data"
```

Use `codex mcp list` to confirm registration. Restart Codex after changing MCP
configuration.

</details>

<details>
<summary><strong>Claude Code configuration</strong></summary>

```bash
claude mcp add --transport stdio \
  --env FIRELENS_ALLOWED_ROOTS=/absolute/path/to/repositories \
  --env FIRELENS_DATA_DIR=/absolute/path/to/firelens-data \
  firelens -- \
  uv run --project /absolute/path/to/firelens firelens-mcp
```

Check the connection with `claude mcp get firelens` or `/mcp` inside Claude
Code. Restart or reload the client after changing the configuration.

</details>

<details>
<summary><strong>Generic STDIO client configuration</strong></summary>

Clients that accept the common `mcpServers` JSON shape can use:

```json
{
  "mcpServers": {
    "firelens": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/absolute/path/to/firelens",
        "firelens-mcp"
      ],
      "env": {
        "FIRELENS_ALLOWED_ROOTS": "/absolute/path/to/repositories",
        "FIRELENS_DATA_DIR": "/absolute/path/to/firelens-data"
      }
    }
  }
}
```

Configuration locations and timeout keys vary by client. The command must stay
attached to an MCP client; running `firelens-mcp` in a terminal and typing into
it will not work because its standard input/output is the protocol stream.

</details>

## Search modes

Choose a mode explicitly when the retrieval strategy matters:

| Mode | Best for |
| --- | --- |
| `auto` | Exact identifiers, typos, and simple natural-language queries |
| `exact` | Known Python symbol names |
| `fuzzy` | Partial or misspelled Python symbol names |
| `lexical` | Text, identifiers, paths, and documentation keywords |
| `semantic` | Conceptual natural-language questions |
| `hybrid_rrf` | Robust discovery across lexical and semantic results |
| `hybrid_weighted` | Experiments with explicitly normalized source weights |
| `graph` | Related callers, imports, references, inheritance, dependencies, and tests |

Hybrid results include contributing channels, per-channel ranks and scores,
backend details, and component timings. Graph results add the originating seed,
edge kind and direction, hop count, confidence, and bounded contribution.

## CLI

The same runtime is available as a JSON CLI:

```bash
uv run firelens status /absolute/path/to/repository
uv run firelens index /absolute/path/to/repository
uv run firelens search /absolute/path/to/repository "SQLiteIndexStore" --mode auto
uv run firelens search /absolute/path/to/repository \
  "where are repository permissions checked?" \
  --mode hybrid_rrf --top-k 5 --path app
uv run firelens search /absolute/path/to/repository \
  "what calls the indexing service?" \
  --mode graph --top-k 8
```

Commands write structured JSON to stdout. Indexing progress and errors go to
stderr, so output remains safe to pipe into another program.

## Streamlit interface

Launch the optional local UI:

```bash
uv run streamlit run app/client/streamlit_app.py --server.fileWatcherType none
```

Use the sidebar to select or index a repository, then compare search modes,
backends, snippets, scores, and graph evidence.

![FireLens Streamlit search interface](static/streamlit-ss.png)

## How it works

```text
MCP · CLI · Streamlit
          │
          ▼
   shared Python runtime
     ├── index service ── parsers ── chunks + graph facts
     └── search service ─ exact · fuzzy · lexical · semantic · graph
          │
          ▼
    repository/store boundary
          │
          ▼
  SQLite metadata · FTS5 · vectors · graph edges
```

- Indexes are staged and promoted atomically, so cancellation or failure does
  not replace the last valid database.
- Cross-process locks coordinate CLI, Streamlit, and MCP readers and writers.
- Re-indexing hashes files, parses only changes, removes deleted records, and
  reuses matching embeddings.
- Results are deterministically ordered, deduplicated, and bounded by candidate,
  memory, result-count, per-snippet, and total-context limits.
- A lightweight root `.gitignore` implementation handles common ignore,
  negation, anchored, directory, and glob rules.

Source checkouts default to `data/indexes`. Installed packages use the current
user's platform data directory. Set `FIRELENS_DATA_DIR` explicitly when multiple
clients should share an index.

## Configuration

All runtime settings use the `FIRELENS_` prefix. The most useful settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FIRELENS_ALLOWED_ROOTS` | Current directory | MCP filesystem allowlist |
| `FIRELENS_DATA_DIR` | Platform/source-checkout default | Persistent index parent directory |
| `FIRELENS_EMBEDDING_MODEL` | `nomic-ai/CodeRankEmbed` | Local semantic embedding model |
| `FIRELENS_EMBEDDING_DEVICE` | Auto | PyTorch device such as `cpu`, `mps`, or `cuda` |
| `FIRELENS_MAX_REPOSITORY_FILES` | `10000` | Accepted source/documentation file cap |
| `FIRELENS_MAX_FILE_SIZE_BYTES` | `1000000` | Per-file indexing cap |
| `FIRELENS_MAX_TOTAL_SNIPPET_CHARS` | `12000` | Total returned context cap |
| `FIRELENS_GRAPH_MAX_HOPS` | `1` | Graph expansion depth, hard maximum `2` |
| `FIRELENS_MOJO_LIBRARY_PATH` | Auto-discovered | Optional native library override |

FireLens loads `.env` only from its own source checkout. A repository being
indexed cannot override the server's configuration with its own `.env`.

<details>
<summary><strong>Optional Mojo acceleration</strong></summary>

The default installation is Python-only. To install the pinned Mojo compiler
and build the optional CPU shared library:

```bash
uv sync --frozen --extra mojo
uv run --extra mojo python scripts/build_mojo.py
```

The build is written to `build/mojo/libfirelens_mojo.dylib` on macOS or
`build/mojo/libfirelens_mojo.so` on Linux. In `auto` mode, FireLens uses Mojo
only above measured crossover sizes and falls back to the Python/NumPy backend
if a native operation fails. `--backend mojo` requires the library.

Benchmark your machine before changing thresholds:

```bash
uv run python -m benchmarks --profile full --comparison-backend mojo \
  --output build/benchmarks/full.json \
  --table-output build/benchmarks/full.md
```

See the [benchmark guide](benchmarks/README.md), [CPU baseline](benchmarks/CPU_BASELINE.md),
and [GPU gate](benchmarks/GPU_GATE.md).

</details>

## Current limitations

- Python is the only language with structural symbol and graph extraction.
- Markdown, `.markdown`, and reStructuredText are indexed as documentation
  sections rather than program structure.
- `.gitignore` handling is intentionally limited to the repository root and
  common rule forms.
- Semantic search quality and score thresholds still need evaluation on broader
  public code-search datasets.
- GPU acceleration remains gated on toolchain compatibility and measured wins.

## Development

Install locked dependencies and run the complete test suite:

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), follow the
[Code of Conduct](CODE_OF_CONDUCT.md), and report vulnerabilities through the
private process in [SECURITY.md](SECURITY.md). Release history is recorded in
[CHANGELOG.md](CHANGELOG.md).

## License

FireLens is open source under the [MIT License](LICENSE.md).

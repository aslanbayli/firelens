# Contributing to FireLens

Thank you for helping improve FireLens. Contributions are welcome as bug
reports, design discussions, documentation, tests, benchmarks, and code.

## Before you start

- Search the existing issues and pull requests before opening a duplicate.
- Open an issue before a large change so the approach can be discussed early.
- Report security vulnerabilities privately according to
  [SECURITY.md](SECURITY.md).
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

FireLens requires Python 3.14 and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

1. Fork the repository and clone your fork.
2. Create a focused branch from `main`.
3. Install the locked dependencies.

```bash
git clone https://github.com/YOUR-USERNAME/firelens.git
cd firelens
git switch -c feat/short-description
uv sync --frozen
```

The default environment is Python-only. To work on the optional Mojo kernels,
install the pinned toolchain and build the shared library:

```bash
uv sync --frozen --extra mojo
uv run --extra mojo python scripts/build_mojo.py
```

## Project boundaries

FireLens is a local-first retrieval engine, not a chatbot.

- Python owns ingestion, parsing, persistence, query routing, formatting,
  Streamlit, and MCP.
- Mojo is reserved for pure compute kernels such as fuzzy scoring, vector dot
  products, and top-k ranking.
- Keep storage access behind repository or store classes. Search and indexing
  orchestration must not contain raw SQL.
- Do not add new behavior through `app/legacy`.
- Prefer the standard library and straightforward code. Add dependencies only
  when they provide a clear correctness, performance, or maintenance benefit.
- Keep retrieval deterministic, bounded, and explicit about provenance.

## Test your change

Run the full test suite before opening a pull request:

```bash
uv run python -m unittest discover -s tests -v
```

When changing an MCP or CLI contract, add or update interface tests. When
changing ranking or acceleration, include deterministic parity tests and a
benchmark when performance is part of the claim. Mojo changes should also pass:

```bash
uv run --extra mojo python scripts/build_mojo.py
uv run python -m unittest tests.test_mojo_backend tests.test_acceleration -v
```

## Commit and pull request guidance

Keep commits focused and use a concise Conventional Commits-style subject, for
example:

```text
feat: add bounded TypeScript symbol extraction
fix: preserve the active index after cancellation
docs: clarify MCP client configuration
```

In your pull request:

- explain the problem and the chosen solution;
- link related issues;
- describe how you tested the change;
- call out compatibility, storage-format, security, or performance effects;
- include screenshots for visible Streamlit changes; and
- update public documentation when behavior changes.

By contributing, you agree that your contribution is licensed under the
[MIT License](LICENSE.md).

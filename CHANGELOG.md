# Changelog

All notable changes to FireLens are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-08-16

FireLens 1.0.0 is the first stable open-source release.

### Added

- Local STDIO MCP tools for repository indexing, freshness checks, and bounded
  code search.
- Exact, fuzzy, lexical, semantic, hybrid RRF, normalized weighted, and
  graph-aware retrieval modes.
- Incremental and atomic SQLite indexing for Python source, Markdown, and
  reStructuredText documentation.
- Statically derived calls, imports, inheritance, references, file dependency,
  and test-to-implementation graph edges.
- JSON CLI and Streamlit interfaces over the same application runtime.
- Optional Mojo acceleration for fuzzy scoring and large semantic rankings.
- Repository allowlists, bounded snippets, cooperative MCP cancellation, and
  cross-process index locking.
- Contribution, security, conduct, issue, pull request, and CI guidance for the
  open-source community.

[Unreleased]: https://github.com/aslanbayli/firelens/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/aslanbayli/firelens/releases/tag/v1.0.0

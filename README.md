# FireLens

FireLens is a local-first code retrieval engine for Python repositories.

It indexes repository symbols, semantic chunks, and embeddings into SQLite so
other layers can build exact, fuzzy, and semantic search on top of a stable
local index.

## Current scope

- Python-only parsing via the standard library `ast` module
- SQLite-backed repository index storage
- Incremental reindexing based on file content changes
- Embedding reuse when chunk content has not changed
- Root `.gitignore` support during repository walking
- Optional progress callbacks for indexing status updates

FireLens is not a chatbot. Retrieval and indexing are the core product.

## Requirements

- Python `>=3.14,<3.15`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

For real semantic embeddings, install project dependencies and provide a
Hugging Face token in `.env` or the shell as `HF_TOKEN` if the model requires
authentication.

## Install

```bash
git clone https://github.com/aslanbayli/firelens.git
cd firelens
uv sync
```

## Index a repository

Use the persisted indexer entrypoint:

```python
from app.indexing.embedder import CodeRankEmbedder
from app.indexing.indexer import index_to_sqlite

report = index_to_sqlite(
    "~/projects/firelens",
    CodeRankEmbedder(),
)

print(report.database_path)
```

This creates a SQLite database under:

```text
data/indexes/<repository-key>/firelens.db
```

The index contains:

- `repositories`: repository metadata and embedding compatibility info
- `files`: indexed file metadata and content hashes
- `symbols`: parsed functions, classes, and methods
- `chunks`: semantic-search source chunks
- `embeddings`: serialized embedding vectors

## Incremental indexing

Reindexing the same repository does not rebuild everything.

FireLens now:

- reuses the same persisted repository identity
- hashes current files and compares them to stored file metadata
- parses and embeds only added or changed files
- removes records for deleted files
- reuses stored embeddings when chunk content hashes still match
- preserves previous valid records if a changed file fails parsing

## Progress reporting

`index_to_sqlite()` accepts an optional `progress_callback` so callers can
render indexing progress in a CLI, Streamlit UI, or logs.

```python
from app.indexing.embedder import CodeRankEmbedder
from app.indexing.indexer import index_to_sqlite

def show_progress(event):
    print(f"[{event.stage}] {event.current}/{event.total} {event.message}")

report = index_to_sqlite(
    "~/projects/firelens",
    CodeRankEmbedder(),
    progress_callback=show_progress,
)
```

Progress stages currently include:

- `load`
- `walk`
- `compare`
- `index`
- `write`
- `complete`

## `.gitignore` behavior

If the indexed repository contains a root `.gitignore`, FireLens excludes
matching paths while walking the tree. The current implementation supports the
common cases needed for repository indexing:

- comments and blank lines
- directory rules such as `build/`
- anchored rules such as `/generated.py`
- glob rules such as `*.generated.py`
- negation rules such as `!keep.py`

FireLens also ignores built-in paths such as `.git`, virtualenv directories,
`node_modules`, caches, build outputs, and the local `data` directory.

## Embeddings

The real semantic embedder is `CodeRankEmbedder`, which loads:

```text
nomic-ai/CodeRankEmbed
```

through `sentence-transformers`.

The model runs locally through PyTorch. On Apple Silicon, that typically means
`mps` when available, otherwise CPU. Model files are cached by Hugging Face in
the user cache directory unless overridden by environment variables such as
`HF_HOME` or `TRANSFORMERS_CACHE`.

For tests and pipeline validation, `FakeEmbedder` provides deterministic
normalized vectors without requiring any model downloads.

## Inspect the SQLite index

```bash
sqlite3 data/indexes/<repository-key>/firelens.db
```

Useful queries:

```sql
.tables

SELECT COUNT(*) FROM files;
SELECT COUNT(*) FROM symbols;
SELECT COUNT(*) FROM chunks;
SELECT COUNT(*) FROM embeddings;

SELECT name, qualified_name, kind, relative_path, start_line, end_line
FROM symbols
LIMIT 20;
```

## Run tests

```bash
uv run python -m unittest \
  tests.test_indexing_basics \
  tests.test_storage_database \
  tests.test_indexing_persistence
```

## Near-term gaps

- Semantic search execution is not wired yet; vectors are persisted for the
  upcoming search layer.
- Only Python repositories are parsed today.
- `.gitignore` support is intentionally lightweight and limited to the root
  `.gitignore` file.

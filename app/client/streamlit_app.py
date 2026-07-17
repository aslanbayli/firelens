"""Streamlit interface for exact, fuzzy, and semantic code search."""

import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.models import SearchRequest
from app.indexing.embedder import CodeRankEmbedder
from app.indexing.indexer import IndexingProgress, index_to_sqlite
from app.search.exact import exact_search
from app.search.fuzzy import fuzzy_search
from app.search.semantic import semantic_search
from app.storage.database import SQLiteIndexStore, default_database_path

INDEX_FORMAT_VERSION = "1"


def main() -> None:
    st.set_page_config(page_title="FireLens", layout="wide")

    st.title("FireLens")

    with st.sidebar:
        source = st.radio(
            "Repository source",
            options=["Existing index", "New repository"],
        )

        if source == "Existing index":
            selected = choose_existing_index()
            if selected is None:
                st.info("No compatible indexes found.")
                return

            repository_path = selected["repository_path"]
            database_path = selected["database_path"]
            repository_id = selected["repository_id"]
            st.caption(database_path)
        else:
            repository_path = st.text_input(
                "Repository path",
                value=str(PROJECT_ROOT),
            )
            database_path = st.text_input(
                "Index database",
                value=str(default_database_path(repository_path)),
            )
            repository_id = None

        if st.button("Index / Re-index", use_container_width=True):
            run_index(repository_path, database_path)

    store = SQLiteIndexStore(database_path)
    repository = load_repository(store, repository_path)

    if repository is None:
        st.info("No compatible CodeRank index found. Use the Index button first.")
        return

    repository_id = repository_id or repository.id

    mode = st.segmented_control(
        "Mode",
        options=["exact", "fuzzy", "semantic"],
        default="exact",
    )
    query = st.text_input("Search")

    left_column, right_column = st.columns([1, 1])
    with left_column:
        top_k = st.number_input("Results", min_value=1, max_value=50, value=5)
    with right_column:
        path_filter = st.text_input("Path filter")

    if query:
        request = SearchRequest(
            query=query,
            request_mode=mode,
            top_k=int(top_k),
            path=path_filter.strip() or None,
            backend="python",
        )
        response = run_search(store, repository_id, request)
        render_response(response)


def choose_existing_index() -> dict[str, Any] | None:
    options = find_existing_indexes()
    if not options:
        return None

    labels = [option["label"] for option in options]
    selected_label = st.selectbox("Index", labels)

    return options[labels.index(selected_label)]


def find_existing_indexes() -> list[dict[str, Any]]:
    embedder = get_embedder()
    options: list[dict[str, Any]] = []

    for database_path in sorted((PROJECT_ROOT / "data/indexes").glob("*/firelens.db")):
        store = SQLiteIndexStore(database_path)
        repositories = store.list_compatible_repositories(
            index_format_version=INDEX_FORMAT_VERSION,
            embedding_model=embedder.model,
            embedding_dim=embedder.dimension,
        )

        for repository in repositories:
            path = Path(repository.absolute_path)
            options.append(
                {
                    "label": f"{path.name} - {repository.absolute_path}",
                    "repository_path": repository.absolute_path,
                    "database_path": str(database_path),
                    "repository_id": repository.id,
                }
            )

    return options


def run_index(repository_path: str, database_path: str) -> None:
    progress_area = st.empty()
    embedder = get_embedder()

    def show_progress(event: IndexingProgress) -> None:
        progress_area.caption(
            f"{event.stage}: {event.current}/{event.total} {event.message}"
        )

    with st.spinner("Indexing"):
        report = index_to_sqlite(
            repository_path,
            embedder,
            database_path,
            progress_callback=show_progress,
        )

    st.success(
        (
            f"Indexed {report.symbol_count} symbols, {report.chunk_count} chunks, "
            f"{report.embedding_count} embeddings."
        )
    )

    if report.errors:
        with st.expander("Indexing errors"):
            for error in report.errors:
                st.write(f"{error.relative_path}: {error.stage}: {error.message}")


def load_repository(store: SQLiteIndexStore, repository_path: str):
    root = Path(repository_path).expanduser().resolve()
    embedder = get_embedder()

    return store.load_repository_by_identity(
        absolute_path=str(root),
        index_format_version=INDEX_FORMAT_VERSION,
        embedding_model=embedder.model,
        embedding_dim=embedder.dimension,
    )


@st.cache_resource
def get_embedder() -> CodeRankEmbedder:
    return CodeRankEmbedder()


def run_search(
    store: SQLiteIndexStore,
    repository_id: UUID,
    request: SearchRequest,
):
    if request.request_mode == "exact":
        return exact_search(store, repository_id, request)

    if request.request_mode == "fuzzy":
        return fuzzy_search(store, repository_id, request)

    if request.request_mode == "semantic":
        return semantic_search(
            store,
            repository_id,
            request,
            get_embedder(),
        )

    raise ValueError(f"Unsupported search mode: {request.request_mode}")


def render_response(response) -> None:
    st.caption(
        (
            f"{len(response.ranked_results)} results in "
            f"{response.elapsed_time * 1000:.1f} ms"
        )
    )

    for index, result in enumerate(response.ranked_results, start=1):
        label = (
            f"{index}. {result.symbol_name or result.file_path} "
            f"({result.file_path}:{result.start_line}-{result.end_line}) "
            f"score {result.score:.2f}"
        )
        with st.expander(label):
            st.code(result.snippet, language="python")


if __name__ == "__main__":
    main()

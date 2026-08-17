"""Streamlit interface for indexing and searching local code repositories."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.models import RETRIEVAL_MODE_OPTIONS
from app.core.runtime import FireLensRuntime, build_runtime
from app.indexing.indexer import IndexingProgress
from app.indexing.service import AvailableIndex


def main() -> None:
    st.set_page_config(page_title="FireLens", layout="wide")
    st.title("FireLens")

    runtime = get_runtime()

    with st.sidebar:
        source = st.radio(
            "Repository source",
            options=["Existing index", "New repository"],
        )

        if source == "Existing index":
            selected = choose_existing_index(runtime)
            if selected is None:
                st.info("No compatible indexes found.")
                return

            repository_path = selected.repository_path
            st.caption(f"Database: {selected.database_path}")
            st.caption(
                f"Model: {selected.embedding_provider}/"
                f"{selected.embedding_model} "
                f"({selected.embedding_dim} dimensions)"
            )
        else:
            repository_path = st.text_input(
                "Repository path",
                value=str(PROJECT_ROOT),
            )
            st.caption(f"Indexes are stored under {settings.data_dir}")

        left_action, right_action = st.columns(2)
        with left_action:
            if st.button("Check status", use_container_width=True):
                show_index_status(runtime, repository_path)
        with right_action:
            if st.button("Index / Re-index", use_container_width=True):
                run_index(runtime, repository_path)

    mode = st.segmented_control(
        "Mode",
        options=RETRIEVAL_MODE_OPTIONS,
        default="auto",
    )
    query = st.text_input("Search")

    results_column, backend_column, snippet_column = st.columns(3)
    with results_column:
        top_k = st.number_input(
            "Results",
            min_value=1,
            max_value=settings.max_top_k,
            value=min(5, settings.max_top_k),
        )
    with backend_column:
        backend = st.selectbox(
            "Backend",
            options=["auto", "python", "mojo"],
            index=0,
        )
    with snippet_column:
        max_snippet_chars = st.number_input(
            "Maximum snippet characters",
            min_value=1,
            max_value=settings.max_snippet_chars,
            value=min(
                settings.default_max_snippet_chars,
                settings.max_snippet_chars,
            ),
        )

    path_filter = st.text_input("Path filter")

    if query:
        try:
            response = runtime.search_code(
                repository_path=repository_path,
                query=query,
                mode=mode or "auto",
                top_k=int(top_k),
                path=path_filter.strip() or None,
                backend=backend,
                max_snippet_chars=int(max_snippet_chars),
            )
        except (OSError, RuntimeError, ValueError) as error:
            st.error(str(error))
            return

        render_response(response)


def choose_existing_index(runtime: FireLensRuntime) -> AvailableIndex | None:
    options = find_existing_indexes(runtime)
    if not options:
        return None

    return st.selectbox(
        "Index",
        options=options,
        format_func=lambda option: (
            f"{Path(option.repository_path).name} - {option.repository_path}"
            + (" (rebuild required)" if option.status == "stale" else "")
        ),
    )


def find_existing_indexes(runtime: FireLensRuntime) -> list[AvailableIndex]:
    """Return persisted indexes available through the shared runtime."""

    return runtime.list_available_indexes()


def show_index_status(runtime: FireLensRuntime, repository_path: str) -> None:
    try:
        with st.spinner("Checking index freshness"):
            status = runtime.get_index_status(repository_path)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return

    message = (
        f"{status.status}: {status.file_count} files, "
        f"{status.symbol_count} symbols, {status.chunk_count} chunks, "
        f"{status.graph_edge_count} graph edges"
    )
    if status.status == "ready":
        st.success(message)
    elif status.status == "missing":
        st.info(message)
    else:
        st.warning(message)

    if status.changed_paths:
        st.caption(
            f"{status.added_file_count} added, "
            f"{status.changed_file_count} changed, "
            f"{status.deleted_file_count} deleted"
        )
        with st.expander("Changed paths"):
            for path in status.changed_paths:
                st.write(path)

    for warning in status.warnings:
        st.warning(warning)


def run_index(runtime: FireLensRuntime, repository_path: str) -> None:
    progress_bar = st.progress(0.0, text="Starting index")

    def show_progress(event: IndexingProgress) -> None:
        fraction = 0.0 if event.total <= 0 else event.current / event.total
        progress_bar.progress(
            min(max(fraction, 0.0), 1.0),
            text=f"{event.stage}: {event.message}",
        )

    try:
        with st.spinner("Indexing"):
            report = runtime.index_repository(
                repository_path,
                progress_callback=show_progress,
            )
    except (OSError, RuntimeError, ValueError) as error:
        progress_bar.empty()
        st.error(str(error))
        return

    progress_bar.progress(1.0, text="Indexing complete")
    message = (
        f"Indexed {report.file_count} files, {report.symbol_count} symbols, "
        f"{report.chunk_count} chunks, {report.embedding_count} embeddings, and "
        f"{report.graph_edge_count} graph edges "
        f"in {report.elapsed_time:.1f} seconds."
    )
    if report.status == "ready":
        st.success(message)
    else:
        st.warning(f"{message} The index remains stale because some files failed.")

    st.caption(
        f"{report.added_file_count} added, "
        f"{report.changed_file_count} changed, "
        f"{report.deleted_file_count} deleted; "
        f"{report.reused_embedding_count} embeddings reused"
    )

    if report.errors:
        with st.expander(f"Indexing errors ({report.error_count})"):
            for error in report.errors:
                st.write(f"{error.relative_path}: {error.stage}: {error.message}")

    for warning in report.warnings:
        st.warning(warning)


@st.cache_resource
def get_runtime() -> FireLensRuntime:
    return build_runtime()


def render_response(response) -> None:
    st.caption(
        f"{len(response.ranked_results)} results in "
        f"{response.elapsed_time * 1000:.1f} ms · "
        f"mode {response.requested_mode} → {response.mode} · "
        f"backend {response.requested_backend} → {response.backend}"
    )

    for warning in response.warnings:
        st.warning(warning)

    for index, result in enumerate(response.ranked_results, start=1):
        label = (
            f"{index}. {result.symbol_name or result.file_path} "
            f"({result.file_path}:{result.start_line}-{result.end_line}) "
            f"score {result.score:.2f}"
        )
        with st.expander(label):
            if result.fusion_method is not None:
                component_evidence = [
                    evidence
                    for evidence in result.retrieval_evidence
                    if evidence.channel in {"lexical", "semantic"}
                ]
                details = " · ".join(
                    f"{evidence.channel} rank {evidence.rank}, "
                    f"normalized {evidence.score:.2f}, "
                    f"backend {evidence.backend}"
                    for evidence in component_evidence
                )
                st.caption(f"Fusion: {result.fusion_method} · {details}")
            if result.graph_evidence:
                graph = result.graph_evidence[0]
                st.caption(
                    "Graph: "
                    f"{graph.direction} {graph.edge_kind} from "
                    f"{graph.originating_seed_path} · hop {graph.hop_count} · "
                    f"confidence {graph.edge_confidence:.2f} · "
                    f"contribution {graph.graph_contribution:.2f}"
                )
            if result.snippet_truncated:
                st.caption("Snippet truncated to the configured output limit.")
            st.code(result.snippet, language=_display_language(result.language))


def _display_language(language: str) -> str:
    if language == "restructuredtext":
        return "rst"
    return language if language in {"python", "markdown", "rst"} else "text"


if __name__ == "__main__":
    main()

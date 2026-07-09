from app.core.models import SearchRequest
from app.indexing.embedder import CodeRankEmbedder
from app.indexing.indexer import index_to_sqlite
from app.search.exact import search_exact
from app.storage.database import SQLiteIndexStore


def show_progress(event):
    print(f"[{event.stage}] {event.current}/{event.total} {event.message}")


# # test indexing
# report = index_to_sqlite(
#     "~/projects/firelens",
#     CodeRankEmbedder(),
#     progress_callback=show_progress,
# )
# print(report.database_path)
# database_path = report.database_path
# repo_id = report.repository.id


# use existing index without reindexing
store = SQLiteIndexStore("data/indexes/firelens-de72b1b6d5a7/firelens.db")
repository = store.load_repository_by_identity(
    absolute_path="/Users/aslanbayli/Documents/projects/firelens",
    index_format_version="1",
    embedding_model=CodeRankEmbedder().model,
    embedding_dim=CodeRankEmbedder().dimension,
)
if repository is None:
    raise RuntimeError("No compatible repository found in existing index")

database_path = store.db_path
repo_id = repository.id


# test exact search
query = input("Search query: ")
store = SQLiteIndexStore(database_path)
request = SearchRequest(
    query=query,
    request_mode="exact",
    top_k=10,
    path=None,
    backend="python",
)
response = search_exact(store, repo_id, request)

print(f"\nexact search results for {request.query!r}:")
for result in response.ranked_results:
    print(
        f"- {result.symbol_name} "
        f"({result.file_path}:{result.start_line}-{result.end_line})"
    )

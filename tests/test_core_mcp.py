import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.core.config import DEFAULT_DATA_DIR, Settings, default_data_directory
from app.core.models import (
    IndexRepositoryResponse,
    IndexStatusResponse,
    IndexingErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.core.repositories import RepositoryResolver
from app.indexing.embedder import CodeRankEmbedder, embedding_model_identity
from app.search.fuzzy import fuzzy_score, levenshtein_distance


class FuzzySafetyTests(unittest.TestCase):
    def test_levenshtein_uses_the_expected_edit_distance(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("same", "same"), 0)

    def test_fuzzy_scoring_rejects_pathological_candidate_lengths(self) -> None:
        self.assertEqual(fuzzy_score("query", "x" * 4_097), 0.0)


class SearchContractTests(unittest.TestCase):
    def test_search_request_uses_mcp_safe_defaults(self) -> None:
        request = SearchRequest(query="find authentication")

        self.assertEqual(request.request_mode, "auto")
        self.assertEqual(request.top_k, 5)
        self.assertEqual(request.backend, "auto")
        self.assertEqual(request.max_snippet_chars, 2_000)

    def test_search_request_rejects_empty_query_and_out_of_range_limits(self) -> None:
        invalid_requests = [
            {"query": "   "},
            {"query": "valid", "top_k": 0},
            {"query": "valid", "top_k": 21},
            {"query": "x" * 2_001},
            {"query": "valid", "max_snippet_chars": 0},
            {"query": "valid", "max_snippet_chars": 4_001},
        ]

        for arguments in invalid_requests:
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                SearchRequest(**arguments)

    def test_search_response_records_requested_and_actual_execution(self) -> None:
        result = SearchResult(
            id=uuid.uuid4(),
            result_type="symbol",
            file_path="app/service.py",
            start_line=10,
            end_line=12,
            symbol_name="Service.run",
            snippet="def run(self): ...",
            snippet_truncated=True,
            score=1.0,
            mode="exact",
            backend="python",
        )

        response = SearchResponse(
            original_query="Service.run",
            requested_mode="auto",
            mode="exact",
            requested_backend="auto",
            backend="python",
            elapsed_time=0.01,
            ranked_results=[result],
        )

        self.assertEqual(response.requested_mode, "auto")
        self.assertEqual(response.mode, "exact")
        self.assertEqual(response.requested_backend, "auto")
        self.assertEqual(response.backend, "python")
        self.assertTrue(response.ranked_results[0].snippet_truncated)

    def test_index_response_samples_are_bounded(self) -> None:
        errors = [
            IndexingErrorResponse(
                relative_path=f"file_{index}.py",
                stage="parse",
                message="invalid syntax",
            )
            for index in range(21)
        ]

        with self.assertRaises(ValidationError):
            IndexRepositoryResponse(
                repository_path="/repo",
                database_path="/data/index.db",
                status="stale",
                index_format_version="1",
                timestamp_of_index=1,
                embedding_provider="test",
                embedding_model="fake",
                embedding_dim=8,
                error_count=len(errors),
                errors=errors,
            )

        with self.assertRaises(ValidationError):
            IndexStatusResponse(
                repository_path="/repo",
                database_path="/data/index.db",
                status="stale",
                changed_paths=[f"file_{index}.py" for index in range(21)],
            )


class SettingsTests(unittest.TestCase):
    def test_settings_have_local_project_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configured = Settings(_env_file=None)

        self.assertEqual(configured.data_dir, DEFAULT_DATA_DIR.resolve())
        self.assertEqual(configured.allowed_roots, [Path.cwd().resolve()])
        self.assertEqual(configured.embedding_provider, "sentence-transformers")
        self.assertEqual(configured.embedding_model, CodeRankEmbedder.DEFAULT_MODEL)
        self.assertEqual(
            configured.embedding_revision,
            CodeRankEmbedder.DEFAULT_REVISION,
        )
        self.assertEqual(configured.embedding_dimension, 768)
        self.assertEqual(configured.embedding_batch_size, 32)
        self.assertIsNone(configured.embedding_device)
        self.assertEqual(configured.fuzzy_threshold, 0.55)
        self.assertEqual(configured.max_fuzzy_candidates, 512)
        self.assertEqual(configured.max_semantic_candidates, 50_000)
        self.assertEqual(configured.max_semantic_index_bytes, 192 * 1024 * 1024)
        self.assertEqual(configured.max_chunks_per_file, 2_048)
        self.assertEqual(configured.max_top_k, 20)
        self.assertEqual(configured.default_max_snippet_chars, 2_000)
        self.assertEqual(configured.max_snippet_chars, 4_000)
        self.assertEqual(configured.max_total_snippet_chars, 12_000)

    def test_allowed_roots_use_the_platform_path_separator(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            environment = {
                "FIRELENS_ALLOWED_ROOTS": os.pathsep.join([first, second]),
                "FIRELENS_DATA_DIR": str(Path(first) / "indexes"),
                "FIRELENS_EMBEDDING_BATCH_SIZE": "16",
                "FIRELENS_EMBEDDING_DEVICE": "cpu",
                "FIRELENS_EMBEDDING_DIMENSION": "384",
            }
            with patch.dict(os.environ, environment, clear=True):
                configured = Settings(_env_file=None)

        self.assertEqual(
            configured.allowed_roots,
            [Path(first).resolve(), Path(second).resolve()],
        )
        self.assertEqual(configured.data_dir, (Path(first) / "indexes").resolve())
        self.assertEqual(configured.embedding_batch_size, 16)
        self.assertEqual(configured.embedding_device, "cpu")
        self.assertEqual(configured.embedding_dimension, 384)

    def test_installed_package_data_default_is_outside_the_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            with patch.dict(os.environ, {}, clear=True):
                installed_default = default_data_directory(package_root)

        self.assertFalse(installed_default.is_relative_to(package_root))
        self.assertEqual(installed_default.name, "indexes")

    def test_target_repository_dotenv_cannot_override_runtime_or_hf_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_repository = Path(temp_dir)
            (target_repository / ".env").write_text(
                "FIRELENS_EMBEDDING_MODEL=attacker/model\n"
                "HF_TOKEN=attacker-token\n",
                encoding="utf-8",
            )
            original_directory = Path.cwd()
            try:
                os.chdir(target_repository)
                with patch.dict(os.environ, {}, clear=True):
                    configured = Settings()
                    embedder = CodeRankEmbedder()
            finally:
                os.chdir(original_directory)

        self.assertNotEqual(configured.embedding_model, "attacker/model")
        self.assertNotEqual(embedder.hf_token, "attacker-token")

    def test_embedding_identity_includes_the_pinned_revision(self) -> None:
        identity = embedding_model_identity(
            CodeRankEmbedder.DEFAULT_MODEL,
            CodeRankEmbedder.DEFAULT_REVISION,
        )

        self.assertEqual(
            identity,
            f"{CodeRankEmbedder.DEFAULT_MODEL}@{CodeRankEmbedder.DEFAULT_REVISION}",
        )
        self.assertEqual(CodeRankEmbedder().model, identity)

    def test_output_settings_reject_inconsistent_or_unsafe_caps(self) -> None:
        invalid_settings = [
            {"embedding_dimension": 0},
            {"max_top_k": 4},
            {"default_max_snippet_chars": 3_000, "max_snippet_chars": 2_000},
            {"max_total_snippet_chars": 12_001},
            {"max_fuzzy_candidates": 513},
            {"max_semantic_candidates": 50_001},
            {"max_semantic_index_bytes": 256 * 1024 * 1024 + 1},
            {"max_chunks_per_file": 4_097},
        ]

        for overrides in invalid_settings:
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                Settings(_env_file=None, **overrides)


class RepositoryResolverTests(unittest.TestCase):
    def test_resolver_canonicalizes_and_derives_stable_database_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed_root = root / "allowed"
            repository = allowed_root / "project"
            data_dir = root / "indexes"
            repository.mkdir(parents=True)
            resolver = _resolver(data_dir, [allowed_root])

            first = resolver.resolve(repository / ".")
            second = resolver.resolve(str(repository))

        self.assertEqual(first.root, repository.resolve())
        self.assertEqual(first.database_path, second.database_path)
        self.assertEqual(first.database_path.parent.parent, data_dir.resolve())
        self.assertEqual(first.database_path.name, "firelens.db")

    def test_resolver_rejects_missing_files_and_outside_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed_root = root / "allowed"
            outside_root = root / "outside"
            allowed_root.mkdir()
            outside_root.mkdir()
            file_path = allowed_root / "file.py"
            file_path.write_text("pass\n", encoding="utf-8")
            resolver = _resolver(root / "indexes", [allowed_root])

            invalid_paths = [root / "missing", file_path, outside_root]
            for invalid_path in invalid_paths:
                with self.subTest(path=invalid_path), self.assertRaises(ValueError):
                    resolver.resolve(invalid_path)

    def test_resolver_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed_root = root / "allowed"
            outside_repository = root / "outside"
            allowed_root.mkdir()
            outside_repository.mkdir()
            symlink = allowed_root / "linked-repository"
            symlink.symlink_to(outside_repository, target_is_directory=True)
            resolver = _resolver(root / "indexes", [allowed_root])

            with self.assertRaises(ValueError):
                resolver.resolve(symlink)

    def test_path_filters_are_relative_portable_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resolver = _resolver(root / "indexes", [root])

            self.assertIsNone(resolver.validate_path_filter(None))
            self.assertIsNone(resolver.validate_path_filter("  "))
            self.assertIsNone(resolver.validate_path_filter("."))
            self.assertEqual(
                resolver.validate_path_filter("./app\\core/"),
                "app/core",
            )

            invalid_filters = [
                "/absolute/path.py",
                "C:\\absolute\\path.py",
                "C:drive-relative.py",
                "\\rooted\\path.py",
                "../outside.py",
                "app/../outside.py",
                "bad\x00path.py",
            ]
            for path_filter in invalid_filters:
                with (
                    self.subTest(path_filter=path_filter),
                    self.assertRaises(ValueError),
                ):
                    resolver.validate_path_filter(path_filter)


def _resolver(data_dir: Path, allowed_roots: list[Path]) -> RepositoryResolver:
    configured = Settings(
        _env_file=None,
        data_dir=data_dir,
        allowed_roots=allowed_roots,
    )
    return RepositoryResolver(configured)


if __name__ == "__main__":
    unittest.main()

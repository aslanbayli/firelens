"""Environment-backed FireLens configuration.

All defaults support local use without credentials or a ``.env`` file.
"""

import math
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_data_directory(project_root: Path = PROJECT_ROOT) -> Path:
    """Choose a writable default for source checkouts and installed wheels."""

    if (project_root / "pyproject.toml").is_file():
        return project_root / "data" / "indexes"

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base_directory = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return base_directory / "FireLens" / "indexes"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FireLens" / "indexes"

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base_directory = (
        Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    )
    return base_directory / "firelens" / "indexes"


DEFAULT_DATA_DIR = default_data_directory()
DEFAULT_ENV_FILE = (
    PROJECT_ROOT / ".env" if (PROJECT_ROOT / "pyproject.toml").is_file() else None
)
HARD_MAX_FUZZY_CANDIDATES = 512
HARD_MAX_SEMANTIC_CANDIDATES = 50_000
HARD_MAX_SEMANTIC_INDEX_BYTES = 256 * 1024 * 1024
HARD_MAX_CHUNKS_PER_FILE = 4_096
HARD_MAX_LEXICAL_CANDIDATES = 5_000
HARD_MAX_HYBRID_POOL_SIZE = 20
HARD_MAX_GRAPH_NEIGHBORS = 100
HARD_MAX_GRAPH_EXPANDED_NODES = 200


class Settings(BaseSettings):
    """Runtime settings shared by indexing, search, CLI, and MCP."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_prefix="FIRELENS_",
        extra="ignore",
    )

    data_dir: Path = DEFAULT_DATA_DIR
    allowed_roots: Annotated[list[Path], NoDecode] = Field(
        default_factory=lambda: [Path.cwd().resolve()]
    )
    embedding_provider: str = Field(
        default="sentence-transformers",
        min_length=1,
    )
    embedding_model: str = Field(
        default="nomic-ai/CodeRankEmbed",
        min_length=1,
    )
    embedding_revision: str | None = Field(
        default="3c4b60807d71f79b43f3c4363786d9493691f8b1",
        min_length=1,
    )
    embedding_dimension: int = Field(default=768, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: str | None = None
    mojo_library_path: Path | None = None
    mojo_fuzzy_min_candidates: int = Field(
        default=4,
        ge=1,
        le=HARD_MAX_FUZZY_CANDIDATES,
    )
    mojo_semantic_min_candidates: int = Field(
        default=30_000,
        ge=1,
        le=HARD_MAX_SEMANTIC_CANDIDATES,
    )
    fuzzy_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    retrieval_config_name: str = Field(default="default", min_length=1, max_length=64)
    lexical_exact_qualified_bonus: float = Field(default=1.0, ge=0.0, le=1.0)
    lexical_exact_short_bonus: float = Field(default=0.96, ge=0.0, le=1.0)
    lexical_path_bonus: float = Field(default=0.86, ge=0.0, le=1.0)
    lexical_identifier_bonus: float = Field(default=0.76, ge=0.0, le=1.0)
    lexical_bm25_bonus: float = Field(default=0.66, ge=0.0, le=1.0)
    lexical_fuzzy_bonus: float = Field(default=0.36, ge=0.0, le=1.0)
    lexical_exact_candidate_limit: int = Field(
        default=50, ge=1, le=HARD_MAX_LEXICAL_CANDIDATES
    )
    lexical_path_candidate_limit: int = Field(
        default=50, ge=1, le=HARD_MAX_LEXICAL_CANDIDATES
    )
    lexical_identifier_candidate_limit: int = Field(
        default=100, ge=1, le=HARD_MAX_LEXICAL_CANDIDATES
    )
    lexical_bm25_candidate_limit: int = Field(
        default=100, ge=1, le=HARD_MAX_LEXICAL_CANDIDATES
    )
    lexical_fuzzy_candidate_limit: int = Field(
        default=512, ge=1, le=HARD_MAX_LEXICAL_CANDIDATES
    )
    max_lexical_documents_ranked: int = Field(
        default=500, ge=1, le=HARD_MAX_LEXICAL_CANDIDATES
    )
    bm25_name_weight: float = Field(default=8.0, ge=0.0)
    bm25_qualified_name_weight: float = Field(default=6.0, ge=0.0)
    bm25_identifier_weight: float = Field(default=4.0, ge=0.0)
    bm25_path_weight: float = Field(default=2.0, ge=0.0)
    bm25_content_weight: float = Field(default=1.0, ge=0.0)
    semantic_score_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    hybrid_lexical_pool_size: int = Field(
        default=20,
        ge=1,
        le=HARD_MAX_HYBRID_POOL_SIZE,
    )
    hybrid_semantic_pool_size: int = Field(
        default=20,
        ge=1,
        le=HARD_MAX_HYBRID_POOL_SIZE,
    )
    hybrid_rrf_k: float = Field(default=60.0, gt=0.0)
    hybrid_rrf_lexical_weight: float = Field(default=1.0, ge=0.0)
    hybrid_rrf_semantic_weight: float = Field(default=1.0, ge=0.0)
    hybrid_weighted_lexical_weight: float = Field(default=0.5, ge=0.0)
    hybrid_weighted_semantic_weight: float = Field(default=0.5, ge=0.0)
    hybrid_weighted_missing_source_value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    hybrid_tie_breaking_version: str = Field(
        default="source-location-v1",
        min_length=1,
        max_length=64,
    )
    graph_seed_mode: Literal["hybrid_rrf", "lexical", "semantic"] = "hybrid_rrf"
    graph_seed_count: int = Field(default=5, ge=1, le=HARD_MAX_HYBRID_POOL_SIZE)
    graph_max_hops: int = Field(default=1, ge=1, le=2)
    graph_max_neighbors_per_node: int = Field(
        default=20,
        ge=1,
        le=HARD_MAX_GRAPH_NEIGHBORS,
    )
    graph_max_expanded_nodes: int = Field(
        default=50,
        ge=1,
        le=HARD_MAX_GRAPH_EXPANDED_NODES,
    )
    graph_allowed_edge_kinds: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "calls",
            "inherits",
            "tests",
            "references",
            "imports",
            "depends_on",
        ]
    )
    graph_directions: Annotated[list[Literal["incoming", "outgoing"]], NoDecode] = (
        Field(default_factory=lambda: ["outgoing", "incoming"])
    )
    graph_min_edge_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    graph_hop_decay: float = Field(default=0.7, gt=0.0, le=1.0)
    graph_calls_weight: float = Field(default=0.9, ge=0.0, le=1.0)
    graph_imports_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    graph_inherits_weight: float = Field(default=0.9, ge=0.0, le=1.0)
    graph_references_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    graph_depends_on_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    graph_tests_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    max_fuzzy_candidates: int = Field(
        default=512,
        ge=1,
        le=HARD_MAX_FUZZY_CANDIDATES,
    )
    max_semantic_candidates: int = Field(
        default=50_000,
        ge=1,
        le=HARD_MAX_SEMANTIC_CANDIDATES,
    )
    max_semantic_index_bytes: int = Field(
        default=192 * 1024 * 1024,
        ge=1,
        le=HARD_MAX_SEMANTIC_INDEX_BYTES,
    )
    max_file_size_bytes: int = Field(default=1_000_000, ge=1)
    max_chunks_per_file: int = Field(
        default=2_048,
        ge=1,
        le=HARD_MAX_CHUNKS_PER_FILE,
    )
    max_repository_files: int = Field(default=10_000, ge=1)
    max_walk_entries: int = Field(default=100_000, ge=1)
    max_top_k: int = Field(default=20, ge=5, le=20)
    default_max_snippet_chars: int = Field(default=2_000, ge=1, le=4_000)
    max_snippet_chars: int = Field(default=4_000, ge=2_000, le=4_000)
    max_total_snippet_chars: int = Field(default=12_000, ge=1, le=12_000)

    @field_validator("data_dir", mode="before")
    @classmethod
    def normalize_data_dir(cls, value: Any) -> Path:
        """Expand and canonicalize the index data directory."""

        return Path(value).expanduser().resolve()

    @field_validator("mojo_library_path", mode="before")
    @classmethod
    def normalize_mojo_library_path(cls, value: Any) -> Path | None:
        """Canonicalize an optional explicitly configured Mojo library."""

        if value is None or value == "":
            return None
        return Path(value).expanduser().resolve()

    @field_validator("embedding_provider", "embedding_model")
    @classmethod
    def normalize_embedding_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("embedding names must not be empty")
        return normalized

    @field_validator("embedding_revision")
    @classmethod
    def normalize_embedding_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("embedding_revision must not be empty")
        return normalized

    @field_validator("allowed_roots", mode="before")
    @classmethod
    def parse_allowed_roots(cls, value: Any) -> list[Path]:
        """Parse ``FIRELENS_ALLOWED_ROOTS`` using the OS path separator."""

        if isinstance(value, str):
            values = [part for part in value.split(os.pathsep) if part.strip()]
        else:
            values = list(value)

        if not values:
            raise ValueError("allowed_roots must contain at least one path")

        return [Path(item).expanduser().resolve() for item in values]

    @field_validator("graph_allowed_edge_kinds", mode="before")
    @classmethod
    def parse_graph_edge_kinds(cls, value: Any) -> list[str]:
        values = value.split(",") if isinstance(value, str) else list(value)
        normalized = [str(item).strip() for item in values if str(item).strip()]
        if not normalized:
            raise ValueError("graph_allowed_edge_kinds must not be empty")
        if any(not item.replace("-", "_").isidentifier() for item in normalized):
            raise ValueError("graph edge kinds must be identifiers")
        return list(dict.fromkeys(normalized))

    @field_validator("graph_directions", mode="before")
    @classmethod
    def parse_graph_directions(cls, value: Any) -> list[str]:
        values = value.split(",") if isinstance(value, str) else list(value)
        normalized = [str(item).strip() for item in values if str(item).strip()]
        if not normalized or any(
            item not in {"incoming", "outgoing"} for item in normalized
        ):
            raise ValueError("graph_directions must contain incoming or outgoing")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_output_limits(self) -> "Settings":
        if self.default_max_snippet_chars > self.max_snippet_chars:
            raise ValueError(
                "default_max_snippet_chars must not exceed max_snippet_chars"
            )
        fusion_values = (
            self.hybrid_rrf_k,
            self.hybrid_rrf_lexical_weight,
            self.hybrid_rrf_semantic_weight,
            self.hybrid_weighted_lexical_weight,
            self.hybrid_weighted_semantic_weight,
            self.hybrid_weighted_missing_source_value,
        )
        if not all(math.isfinite(value) for value in fusion_values):
            raise ValueError("hybrid fusion values must be finite")
        if (
            self.hybrid_rrf_lexical_weight
            + self.hybrid_rrf_semantic_weight
            <= 0.0
        ):
            raise ValueError(
                "hybrid RRF source weights must sum to a positive value"
            )
        if (
            self.hybrid_weighted_lexical_weight
            + self.hybrid_weighted_semantic_weight
            <= 0.0
        ):
            raise ValueError(
                "hybrid weighted source weights must sum to a positive value"
            )
        return self


settings = Settings()

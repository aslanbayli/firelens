"""Environment-backed FireLens configuration.

All defaults support local use without credentials or a ``.env`` file.
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Any

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
    fuzzy_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
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

    @model_validator(mode="after")
    def validate_output_limits(self) -> "Settings":
        if self.default_max_snippet_chars > self.max_snippet_chars:
            raise ValueError(
                "default_max_snippet_chars must not exceed max_snippet_chars"
            )
        return self


settings = Settings()

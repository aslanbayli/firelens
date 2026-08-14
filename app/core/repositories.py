"""Resolve and validate repository paths used by public interfaces."""

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from app.core.config import Settings
from app.storage.database import default_database_path


REPOSITORY_UNAVAILABLE_MESSAGE = "Repository path is unavailable or not allowed"


@dataclass(frozen=True)
class ResolvedRepository:
    """Canonical repository root and its deterministic index location."""

    root: Path
    database_path: Path


class RepositoryResolver:
    """Apply the MCP repository allowlist and path-safety policy."""

    def __init__(self, settings: Settings) -> None:
        self.data_dir = settings.data_dir.expanduser().resolve()
        self.allowed_roots = tuple(
            root.expanduser().resolve() for root in settings.allowed_roots
        )

    def resolve(self, repository_path: str | Path) -> ResolvedRepository:
        """Return a canonical allowed directory and its SQLite index path."""

        path_text = str(repository_path)
        if not path_text or len(path_text) > 4_096:
            raise ValueError(REPOSITORY_UNAVAILABLE_MESSAGE)

        try:
            expanded_path = Path(repository_path).expanduser()
            lexical_root = Path(os.path.abspath(os.fspath(expanded_path)))
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(REPOSITORY_UNAVAILABLE_MESSAGE) from error

        # Reject paths that are lexically outside the allowlist before probing
        # the filesystem. This prevents the error path from becoming an
        # existence or file-type oracle for arbitrary local paths.
        lexically_allowed = any(
            _is_within(lexical_root, allowed_root)
            for allowed_root in self.allowed_roots
        )
        if not lexically_allowed:
            # Platform aliases such as macOS ``/var`` -> ``/private/var`` can
            # make a safe path look lexically different from an already
            # canonicalized allowlist entry. Resolve that case while keeping
            # the same public error for missing, disallowed, and non-directory
            # paths.
            try:
                canonical_candidate = lexical_root.resolve(strict=False)
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(REPOSITORY_UNAVAILABLE_MESSAGE) from error
            if not any(
                _is_within(canonical_candidate, allowed_root)
                for allowed_root in self.allowed_roots
            ):
                raise ValueError(REPOSITORY_UNAVAILABLE_MESSAGE)

        try:
            root = lexical_root.resolve(strict=True)
            is_directory = root.is_dir()
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(REPOSITORY_UNAVAILABLE_MESSAGE) from error

        if not is_directory or not any(
            _is_within(root, allowed_root) for allowed_root in self.allowed_roots
        ):
            raise ValueError(REPOSITORY_UNAVAILABLE_MESSAGE)

        database_path = default_database_path(root, data_directory=self.data_dir)
        return ResolvedRepository(root=root, database_path=database_path)

    def validate_path_filter(self, path_filter: str | None) -> str | None:
        """Normalize a repository-relative file or directory-prefix filter."""

        if path_filter is None:
            return None

        value = path_filter.strip()
        if value == "":
            return None
        if len(value) > 4_096:
            raise ValueError("Path filter is too long")
        if "\x00" in value:
            raise ValueError("Path filter must not contain null bytes")
        windows_path = PureWindowsPath(value)
        portable_path = PurePosixPath(value.replace("\\", "/"))
        if (
            PurePosixPath(value).is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or portable_path.is_absolute()
        ):
            raise ValueError("Path filter must be relative to the repository")

        if ".." in portable_path.parts:
            raise ValueError("Path filter must not contain '..'")

        normalized = portable_path.as_posix()
        if normalized == ".":
            return None
        return normalized.rstrip("/")


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is equal to or nested beneath ``parent``."""

    return path == parent or parent in path.parents

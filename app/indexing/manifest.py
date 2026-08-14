"""Build and compare deterministic repository file manifests."""

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.indexing.file_io import read_regular_file
from app.indexing.walker import walk
from app.storage.database import IndexedFile


@dataclass(frozen=True)
class ManifestDiff:
    """Repository-relative file changes between disk and an index."""

    added_paths: list[str]
    changed_paths: list[str]
    deleted_paths: list[str]

    @property
    def is_current(self) -> bool:
        return not (self.added_paths or self.changed_paths or self.deleted_paths)

    @property
    def all_changed_paths(self) -> list[str]:
        return sorted(
            set(self.added_paths + self.changed_paths + self.deleted_paths)
        )


def build_file_manifest(
    root: str | Path,
    repository_id: uuid.UUID,
    relative_paths: list[Path] | None = None,
    max_file_size: int = 1_000_000,
    max_files: int = 10_000,
    max_entries: int = 100_000,
    cancellation_callback: CancellationCallback | None = None,
) -> dict[str, IndexedFile]:
    """Hash every supported source file in deterministic path order."""

    raise_if_cancelled(cancellation_callback)
    canonical_root = Path(root).expanduser().resolve()
    records: dict[str, IndexedFile] = {}

    paths = (
        relative_paths
        if relative_paths is not None
        else walk(
            canonical_root,
            max_file_size=max_file_size,
            max_files=max_files,
            max_entries=max_entries,
            cancellation_callback=cancellation_callback,
        )
    )
    if len(paths) > max_files:
        raise ValueError(f"Repository exceeds the {max_files} file limit")

    for relative_path in paths:
        raise_if_cancelled(cancellation_callback)
        absolute_path = canonical_root / relative_path
        if absolute_path.is_symlink():
            continue
        resolved_path = absolute_path.resolve(strict=True)
        if (
            resolved_path != canonical_root
            and canonical_root not in resolved_path.parents
        ):
            continue
        file_data = _read_manifest_file(absolute_path, max_file_size)
        raise_if_cancelled(cancellation_callback)
        if file_data is None:
            continue
        stat, contents = file_data
        relative_path_text = relative_path.as_posix()
        records[relative_path_text] = IndexedFile(
            repository_id=repository_id,
            relative_path=relative_path_text,
            modified_time_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
            content_hash=hashlib.sha256(contents).hexdigest(),
        )

    return records


def compare_file_manifests(
    current: dict[str, IndexedFile],
    stored: dict[str, IndexedFile],
    cancellation_callback: CancellationCallback | None = None,
) -> ManifestDiff:
    """Return added, content-changed, and deleted repository paths."""

    raise_if_cancelled(cancellation_callback)
    current_paths = set(current)
    stored_paths = set(stored)

    changed_paths: list[str] = []
    for path in current_paths.intersection(stored_paths):
        raise_if_cancelled(cancellation_callback)
        if current[path].content_hash != stored[path].content_hash:
            changed_paths.append(path)

    raise_if_cancelled(cancellation_callback)

    return ManifestDiff(
        added_paths=sorted(current_paths - stored_paths),
        changed_paths=sorted(changed_paths),
        deleted_paths=sorted(stored_paths - current_paths),
    )


def _read_manifest_file(
    absolute_path: Path,
    max_file_size: int,
) -> tuple[os.stat_result, bytes] | None:
    """Read bounded bytes and metadata from one non-symlink file handle."""

    file_status, contents = read_regular_file(
        absolute_path,
        byte_limit=max_file_size + 1,
    )

    if len(contents) > max_file_size:
        return None
    return file_status, contents

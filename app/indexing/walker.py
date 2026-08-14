"""Discover safe, supported source files inside a local repository.

The walker is deliberately responsible only for file discovery and basic file
eligibility. It does not read source as text, parse Python, create chunks, or
generate embeddings. Keeping those stages separate makes each behavior easier
to test and lets the indexer report exactly which stage failed.
"""

import os
import stat
from fnmatch import fnmatch
from pathlib import Path

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.indexing.file_io import read_regular_file

DEFAULT_IGNORED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
}
ROOT_IGNORED_NAMES = {"data"}

# FireLens currently parses only Python using the standard-library AST. Files
# from other languages must be excluded until an appropriate parser exists.
SUPPORTED_SUFFIXES = {".py"}
MAX_GITIGNORE_RULES = 512


def is_binary(path: Path, sample_size: int = 8192) -> bool:
    """Return True when the beginning of a file contains a null byte.

    This is a lightweight heuristic rather than complete file-type detection.
    Source files should not contain null bytes, while many binary formats do.
    Reading only a sample prevents scanning an entire large file twice.
    """

    _, sample = read_regular_file(path, byte_limit=sample_size)

    return b"\x00" in sample


class GitIgnoreRule:
    """One parsed .gitignore rule."""

    def __init__(
        self,
        pattern: str,
        negated: bool,
        directory_only: bool,
        anchored: bool,
    ) -> None:
        self.pattern = pattern
        self.negated = negated
        self.directory_only = directory_only
        self.anchored = anchored

    def matches(self, relative_path: Path, is_directory: bool) -> bool:
        """Return True when this rule applies to the relative path."""

        path_text = relative_path.as_posix()

        if self.directory_only and not is_directory:
            parent_parts = relative_path.parts[:-1]
            if self.anchored:
                parent_paths = [
                    "/".join(relative_path.parts[:index])
                    for index in range(1, len(relative_path.parts))
                ]
                return any(fnmatch(parent, self.pattern) for parent in parent_paths)
            elif "/" in self.pattern:
                parent_path = "/".join(parent_parts)
                return fnmatch(parent_path, self.pattern) or fnmatch(
                    parent_path, f"*/{self.pattern}"
                )
            return any(fnmatch(part, self.pattern) for part in parent_parts)

        if self.anchored:
            return fnmatch(path_text, self.pattern)

        if "/" in self.pattern:
            return fnmatch(path_text, self.pattern) or fnmatch(
                path_text,
                f"*/{self.pattern}",
            )

        return any(fnmatch(part, self.pattern) for part in relative_path.parts)


def load_gitignore_rules(
    root: Path,
    max_size: int = 1_000_000,
    cancellation_callback: CancellationCallback | None = None,
    max_rules: int = MAX_GITIGNORE_RULES,
) -> list[GitIgnoreRule]:
    """Parse root .gitignore into simple matching rules."""

    if max_rules < 1:
        raise ValueError("max_rules must be greater than 0")
    raise_if_cancelled(cancellation_callback)
    gitignore_path = root / ".gitignore"
    try:
        gitignore_status = os.lstat(gitignore_path)
    except FileNotFoundError:
        return []
    if _is_link_or_reparse_point(gitignore_status):
        return []
    _, gitignore_bytes = read_regular_file(
        gitignore_path,
        byte_limit=max_size + 1,
    )
    if len(gitignore_bytes) > max_size:
        raise ValueError(f".gitignore exceeds the {max_size} byte limit")
    raise_if_cancelled(cancellation_callback)

    rules: list[GitIgnoreRule] = []

    for line in gitignore_bytes.decode("utf-8").splitlines():
        raise_if_cancelled(cancellation_callback)
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        negated = stripped_line.startswith("!")
        if negated:
            stripped_line = stripped_line[1:]

        if not stripped_line:
            continue

        anchored = stripped_line.startswith("/")
        if anchored:
            stripped_line = stripped_line[1:]

        directory_only = stripped_line.endswith("/")
        if directory_only:
            stripped_line = stripped_line.rstrip("/")

        if not stripped_line:
            continue

        if len(rules) >= max_rules:
            raise ValueError(f".gitignore exceeds the {max_rules} rule limit")
        rules.append(
            GitIgnoreRule(
                pattern=stripped_line,
                negated=negated,
                directory_only=directory_only,
                anchored=anchored,
            )
        )

    return rules


def is_gitignored(
    relative_path: Path,
    is_directory: bool,
    rules: list[GitIgnoreRule],
    cancellation_callback: CancellationCallback | None = None,
) -> bool:
    """Return True when .gitignore rules exclude a path."""

    ignored = False

    for rule in rules:
        raise_if_cancelled(cancellation_callback)
        if rule.matches(relative_path, is_directory):
            ignored = not rule.negated

    return ignored


def walk(
    # Accept both strings from CLI/UI input and Path objects from Python code.
    path: str | Path,
    ignore_rules: set[str] | None = None,
    # Skip individual files larger than one megabyte by default.
    max_file_size: int = 1_000_000,
    # Abort unusually large traversals instead of consuming unbounded resources.
    max_files: int = 10_000,
    # Bound all visited directory entries, including ignored and unsupported files.
    max_entries: int = 100_000,
    cancellation_callback: CancellationCallback | None = None,
) -> list[Path]:
    """Return deterministic source-file paths relative to a repository root."""

    # Convert strings to Path objects, expand "~", resolve "..", and create a
    # canonical absolute root. Canonicalization is important for safe relative
    # paths and consistent repository identity.
    raise_if_cancelled(cancellation_callback)
    root = Path(path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    # Indexing expects a directory tree. A valid file path is still invalid as
    # a repository root, so distinguish it from a missing path.
    if not root.is_dir():
        raise NotADirectoryError(root)

    ignored_names = DEFAULT_IGNORED_NAMES
    if ignore_rules is not None:
        ignored_names = ignored_names.union(ignore_rules)

    gitignore_rules = load_gitignore_rules(
        root,
        max_size=max_file_size,
        cancellation_callback=cancellation_callback,
    )
    paths: list[Path] = []
    visited_entries = 0
    pending_directories = [root]

    while pending_directories:
        raise_if_cancelled(cancellation_callback)
        current_path = pending_directories.pop()
        child_directories: list[Path] = []

        try:
            current_status = os.lstat(current_path)
            resolved_directory = current_path.resolve(strict=True)
        except OSError:
            continue
        if _is_link_or_reparse_point(current_status):
            continue
        if not _is_within(resolved_directory, root):
            continue

        with os.scandir(current_path) as entries:
            for entry in entries:
                # Check and count immediately after scandir yields one entry.
                # This keeps a huge single directory from being materialized
                # before the configured traversal bound is enforced.
                raise_if_cancelled(cancellation_callback)
                visited_entries += 1
                _validate_entry_count(visited_entries, max_entries)

                candidate = current_path / entry.name
                relative_path = candidate.relative_to(root)
                try:
                    entry_status = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if _is_link_or_reparse_point(entry_status):
                    continue
                if (
                    relative_path.parts[0] in ROOT_IGNORED_NAMES
                    or any(part in ignored_names for part in relative_path.parts)
                ):
                    continue

                if entry.is_dir(follow_symlinks=False):
                    if is_gitignored(
                        relative_path,
                        True,
                        gitignore_rules,
                        cancellation_callback,
                    ):
                        continue
                    child_directories.append(candidate)
                    continue

                if not entry.is_file(follow_symlinks=False):
                    continue
                if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if is_gitignored(
                    relative_path,
                    False,
                    gitignore_rules,
                    cancellation_callback,
                ):
                    continue
                if entry_status.st_size > max_file_size:
                    continue

                try:
                    resolved_candidate = candidate.resolve(strict=True)
                except OSError:
                    continue
                if not _is_within(resolved_candidate, root):
                    continue
                if is_binary(candidate):
                    continue
                raise_if_cancelled(cancellation_callback)

                paths.append(relative_path)
                if len(paths) > max_files:
                    raise ValueError(
                        f"Repository exceeds the {max_files} file limit"
                    )

        # Reverse the local sort because the stack is last-in, first-out.
        # Retained directory names are bounded by the remaining entry budget.
        pending_directories.extend(
            sorted(
                child_directories,
                key=lambda directory: directory.name,
                reverse=True,
            )
        )

    # TODO: Make supported suffixes and generated-file detection configurable
    # when support for languages beyond Python is added.

    # Filesystem traversal order is not guaranteed. Sorting by POSIX-style path
    # produces repeatable indexes and repeatable tests on every run.
    raise_if_cancelled(cancellation_callback)
    return sorted(paths, key=lambda item: item.as_posix())


def _validate_entry_count(entry_count: int, maximum: int) -> None:
    if entry_count > maximum:
        raise ValueError(f"Repository exceeds the {maximum} entry scan limit")


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_link_or_reparse_point(path_status: os.stat_result) -> bool:
    if stat.S_ISLNK(path_status.st_mode):
        return True
    reparse_mask = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(path_status, "st_file_attributes", 0)
    return bool(reparse_mask and file_attributes & reparse_mask)

"""Discover safe, supported source files inside a local repository.

The walker is deliberately responsible only for file discovery and basic file
eligibility. It does not read source as text, parse Python, create chunks, or
generate embeddings. Keeping those stages separate makes each behavior easier
to test and lets the indexer report exactly which stage failed.
"""

from fnmatch import fnmatch
from pathlib import Path

DEFAULT_IGNORED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
}

# FireLens currently parses only Python using the standard-library AST. Files
# from other languages must be excluded until an appropriate parser exists.
SUPPORTED_SUFFIXES = {".py"}


def is_binary(path: Path, sample_size: int = 8192) -> bool:
    """Return True when the beginning of a file contains a null byte.

    This is a lightweight heuristic rather than complete file-type detection.
    Source files should not contain null bytes, while many binary formats do.
    Reading only a sample prevents scanning an entire large file twice.
    """

    with path.open("rb") as file:
        # Read at most `sample_size` bytes so the check has bounded cost.
        sample = file.read(sample_size)

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


def load_gitignore_rules(root: Path) -> list[GitIgnoreRule]:
    """Parse root .gitignore into simple matching rules."""

    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return []

    rules: list[GitIgnoreRule] = []

    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
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
) -> bool:
    """Return True when .gitignore rules exclude a path."""

    ignored = False

    for rule in rules:
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
) -> list[Path]:
    """Return deterministic source-file paths relative to a repository root."""

    # Convert strings to Path objects, expand "~", resolve "..", and create a
    # canonical absolute root. Canonicalization is important for safe relative
    # paths and consistent repository identity.
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

    gitignore_rules = load_gitignore_rules(root)
    paths: list[Path] = []

    # `rglob("*")` recursively yields every descendant beneath the root. The
    # following guards progressively reject entries that are not indexable.
    for candidate in root.rglob("*"):
        relative_path = candidate.relative_to(root)

        if any(part in ignored_names for part in relative_path.parts):
            continue

        if is_gitignored(relative_path, candidate.is_dir(), gitignore_rules):
            continue

        # Directories cannot be parsed as source files.
        if not candidate.is_file():
            continue

        # currently accepts only `.py` files because only Python is parsed.
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if candidate.stat().st_size > max_file_size:
            continue

        if is_binary(candidate):
            continue

        paths.append(relative_path)

        if len(paths) > max_files:
            raise ValueError(f"Repository exceeds the {max_files} file limit")

    # TODO: Make supported suffixes and generated-file detection configurable
    # when support for languages beyond Python is added.

    # Filesystem traversal order is not guaranteed. Sorting by POSIX-style path
    # produces repeatable indexes and repeatable tests on every run.
    return sorted(paths, key=lambda item: item.as_posix())

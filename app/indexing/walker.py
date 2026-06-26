"""Discover safe, supported source files inside a local repository.

The walker is deliberately responsible only for file discovery and basic file
eligibility. It does not read source as text, parse Python, create chunks, or
generate embeddings. Keeping those stages separate makes each behavior easier
to test and lets the indexer report exactly which stage failed.
"""

from pathlib import Path

DEFAULT_IGNORED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    "data",
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

    paths: list[Path] = []

    # `rglob("*")` recursively yields every descendant beneath the root. The
    # following guards progressively reject entries that are not indexable.
    for candidate in root.rglob("*"):
        # Directories cannot be parsed as source files.
        if not candidate.is_file():
            continue

        # Convert the absolute candidate into a path anchored at the repository
        # root, such as `app/indexing/walker.py`.
        relative_path = candidate.relative_to(root)

        # Compare complete path components rather than substrings. For example,
        # a rule named "build" should reject `build/output.py` but must not
        # reject a legitimate file named `rebuild_index.py`.
        if any(part in ignored_names for part in relative_path.parts):
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

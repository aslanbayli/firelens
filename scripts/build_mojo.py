"""Build the FireLens Mojo CPU kernels as a shared library."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "mojo_engine" / "exports.mojo"
EXPECTED_ABI_VERSION = 1


def shared_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def default_output_path() -> Path:
    return (
        PROJECT_ROOT
        / "build"
        / "mojo"
        / f"libfirelens_mojo{shared_library_suffix()}"
    )


def find_mojo_executable(explicit_path: str | None) -> str:
    candidates = [
        explicit_path,
        os.environ.get("MOJO_EXECUTABLE"),
        str(PROJECT_ROOT / ".venv" / "bin" / "mojo"),
        shutil.which("mojo"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_file():
            return str(candidate_path.resolve())
    raise FileNotFoundError(
        "Mojo compiler not found; pass --mojo or set MOJO_EXECUTABLE"
    )


def build_shared_library(mojo_executable: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    command = [
        mojo_executable,
        "build",
        "--emit",
        "shared-lib",
        "--optimization-level",
        "3",
        str(SOURCE_FILE),
        "-o",
        str(temporary_path),
    ]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        validate_shared_library(temporary_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path.resolve()


def validate_shared_library(library_path: Path) -> None:
    """Check the exported ABI before replacing a working library."""

    try:
        library = ctypes.CDLL(str(library_path))
        version_function = library.firelens_mojo_abi_version
        version_function.argtypes = []
        version_function.restype = ctypes.c_int32
        version = int(version_function())
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Built Mojo library does not expose the FireLens ABI"
        ) from error
    if version != EXPECTED_ABI_VERSION:
        raise RuntimeError(
            "Built Mojo library ABI mismatch: "
            f"expected {EXPECTED_ABI_VERSION}, found {version}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mojo", help="Path to the Mojo compiler")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Shared-library output path",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    mojo_executable = find_mojo_executable(arguments.mojo)
    library_path = build_shared_library(mojo_executable, arguments.output)
    print(library_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

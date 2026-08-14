"""Bounded source-file reads that reject final-component links and devices."""

import os
import stat
from pathlib import Path


def read_regular_file(
    path: str | Path,
    *,
    byte_limit: int,
) -> tuple[os.stat_result, bytes]:
    """Read at most ``byte_limit`` bytes from one verified regular file."""

    if byte_limit < 0:
        raise ValueError("byte_limit must not be negative")

    file_path = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK

    verify_path_after_open = not hasattr(os, "O_NOFOLLOW")
    if not verify_path_after_open:
        flags |= os.O_NOFOLLOW
    else:
        path_status = os.lstat(file_path)
        if _is_link_or_reparse_point(path_status):
            raise OSError(f"Refusing symbolic-link source file: {file_path}")
        if not stat.S_ISREG(path_status.st_mode):
            raise OSError(f"Source path is not a regular file: {file_path}")

    file_descriptor = os.open(file_path, flags)
    try:
        opened_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_status.st_mode):
            raise OSError(f"Source path is not a regular file: {file_path}")

        if verify_path_after_open:
            current_path_status = os.lstat(file_path)
            if _is_link_or_reparse_point(current_path_status):
                raise OSError(f"Refusing symbolic-link source file: {file_path}")
            if not os.path.samestat(current_path_status, opened_status):
                raise OSError(f"Source file changed while opening: {file_path}")

        with os.fdopen(file_descriptor, "rb") as source_file:
            file_descriptor = -1
            contents = source_file.read(byte_limit)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)

    return opened_status, contents


def _is_link_or_reparse_point(path_status: os.stat_result) -> bool:
    if stat.S_ISLNK(path_status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(path_status, "st_file_attributes", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)

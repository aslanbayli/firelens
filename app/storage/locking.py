"""Cross-process locks for one FireLens SQLite index.

The lock files live beside the database and must not be deleted while callers
may still be using them. POSIX systems support concurrent shared readers.
Windows uses an exclusive byte-range lock for both modes, which preserves
correctness at the cost of serializing readers.
"""

import errno
import os
import stat
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Literal


LockMode = Literal["shared", "exclusive"]
CancellationCheck = Callable[[], None]
LOCK_WAIT_SECONDS = 0.05


class DatabaseLockBusyError(RuntimeError):
    """Raised when a nonblocking database lock cannot be acquired."""


def database_lock_path(database_path: str | Path) -> Path:
    """Return the persistent lock-file path adjacent to a database."""

    path = Path(database_path)
    return path.with_name(f"{path.name}.lock")


def database_writer_intent_path(database_path: str | Path) -> Path:
    """Return the lock file used by writers to close reader admission."""

    lock_path = database_lock_path(database_path)
    return lock_path.with_name(f"{lock_path.name}.intent")


@contextmanager
def shared_database_lock(
    database_path: str | Path,
    *,
    blocking: bool = True,
) -> Iterator[None]:
    """Hold a cross-process read lock for one database."""

    with _database_lock(database_path, mode="shared", blocking=blocking):
        yield


@contextmanager
def exclusive_database_lock(
    database_path: str | Path,
    *,
    blocking: bool = True,
    cancellation_check: CancellationCheck | None = None,
) -> Iterator[None]:
    """Hold a cross-process write lock for one database."""

    with _database_lock(
        database_path,
        mode="exclusive",
        blocking=blocking,
        cancellation_check=cancellation_check,
    ):
        yield


@contextmanager
def _database_lock(
    database_path: str | Path,
    *,
    mode: LockMode,
    blocking: bool,
    cancellation_check: CancellationCheck | None = None,
) -> Iterator[None]:
    lock_path = database_lock_path(database_path)
    intent_path = database_writer_intent_path(database_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "exclusive":
        # POSIX writers share the intent lock, so every queued writer announces
        # itself immediately. Readers require the incompatible exclusive intent
        # lock and cannot barge until all queued and active writers are gone.
        with _file_lock(
            intent_path,
            mode="shared",
            blocking=blocking,
            busy_path=lock_path,
            cancellation_check=cancellation_check,
        ):
            with _file_lock(
                lock_path,
                mode="exclusive",
                blocking=blocking,
                busy_path=lock_path,
                cancellation_check=cancellation_check,
            ):
                yield
        return

    # Readers serialize only the short admission step. Once the shared database
    # lock is acquired, the intent lock is released and POSIX readers coexist.
    with ExitStack() as held_database_lock:
        with _file_lock(
            intent_path,
            mode="exclusive",
            blocking=blocking,
            busy_path=lock_path,
        ):
            held_database_lock.enter_context(
                _file_lock(
                    lock_path,
                    mode="shared",
                    blocking=blocking,
                    busy_path=lock_path,
                    cancellation_check=cancellation_check,
                )
            )
        yield


@contextmanager
def _file_lock(
    lock_path: Path,
    *,
    mode: LockMode,
    blocking: bool,
    busy_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> Iterator[None]:
    with _open_lock_file(lock_path) as lock_file:
        _acquire(
            lock_file,
            mode=mode,
            blocking=blocking,
            lock_path=busy_path,
            cancellation_check=cancellation_check,
        )
        try:
            yield
        finally:
            _release(lock_file)


def _open_lock_file(lock_path: Path) -> BinaryIO:
    """Open a lock file without following a hostile final-component symlink."""

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
        file_descriptor = os.open(lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise OSError(f"Database lock is not a regular file: {lock_path}")
        except Exception:
            os.close(file_descriptor)
            raise
    elif os.name == "nt":
        file_descriptor = _open_windows_lock_file(lock_path)
    else:
        file_descriptor = _open_verified_lock_file(lock_path)
    return os.fdopen(file_descriptor, "r+b")


def _open_verified_lock_file(lock_path: Path) -> int:
    """Open a regular lock file safely on platforms without ``O_NOFOLLOW``."""

    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    open_flags = os.O_RDWR

    while True:
        try:
            return os.open(lock_path, create_flags, 0o600)
        except FileExistsError:
            pass

        try:
            path_status = os.lstat(lock_path)
        except FileNotFoundError:
            continue
        if _is_reparse_or_symbolic_link(path_status):
            raise OSError(f"Refusing reparse-point database lock: {lock_path}")

        try:
            file_descriptor = os.open(lock_path, open_flags)
        except FileNotFoundError:
            continue

        try:
            current_path_status = os.lstat(lock_path)
            opened_status = os.fstat(file_descriptor)
            if _is_reparse_or_symbolic_link(current_path_status):
                raise OSError(f"Refusing reparse-point database lock: {lock_path}")
            if not os.path.samestat(current_path_status, opened_status):
                raise OSError(f"Database lock changed while opening: {lock_path}")
            if not stat.S_ISREG(opened_status.st_mode):
                raise OSError(f"Database lock is not a regular file: {lock_path}")
            return file_descriptor
        except FileNotFoundError:
            os.close(file_descriptor)
            continue
        except Exception:
            os.close(file_descriptor)
            raise


def _is_reparse_or_symbolic_link(path_status: os.stat_result) -> bool:
    if stat.S_ISLNK(path_status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(path_status, "st_file_attributes", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)


if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_ALWAYS = 4
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _create_file = _kernel32.CreateFileW
    _create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _create_file.restype = wintypes.HANDLE
    _close_handle = _kernel32.CloseHandle
    _close_handle.argtypes = [wintypes.HANDLE]
    _close_handle.restype = wintypes.BOOL

    def _open_windows_lock_file(lock_path: Path) -> int:
        """Open or create a lock file without resolving a final reparse point."""

        handle = _create_file(
            os.fspath(lock_path),
            _GENERIC_READ | _GENERIC_WRITE,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE,
            None,
            _OPEN_ALWAYS,
            _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())

        try:
            file_descriptor = msvcrt.open_osfhandle(
                handle,
                os.O_RDWR | os.O_BINARY,
            )
        except Exception:
            _close_handle(handle)
            raise

        try:
            opened_status = os.fstat(file_descriptor)
            if _is_reparse_or_symbolic_link(opened_status):
                raise OSError(f"Refusing reparse-point database lock: {lock_path}")
            if not stat.S_ISREG(opened_status.st_mode):
                raise OSError(f"Database lock is not a regular file: {lock_path}")
            return file_descriptor
        except Exception:
            os.close(file_descriptor)
            raise

    def _acquire(
        lock_file: BinaryIO,
        *,
        mode: LockMode,
        blocking: bool,
        lock_path: Path,
        cancellation_check: CancellationCheck | None,
    ) -> None:
        del mode  # Windows has no shared equivalent in the standard library.
        while True:
            if cancellation_check is not None:
                cancellation_check()
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if not blocking:
                    raise DatabaseLockBusyError(
                        f"Database is busy: {lock_path}"
                    ) from error
                time.sleep(LOCK_WAIT_SECONDS)

    def _release(lock_file: BinaryIO) -> None:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire(
        lock_file: BinaryIO,
        *,
        mode: LockMode,
        blocking: bool,
        lock_path: Path,
        cancellation_check: CancellationCheck | None,
    ) -> None:
        operation = fcntl.LOCK_SH if mode == "shared" else fcntl.LOCK_EX
        if not blocking or cancellation_check is not None:
            operation |= fcntl.LOCK_NB

        while True:
            if cancellation_check is not None:
                cancellation_check()
            try:
                fcntl.flock(lock_file.fileno(), operation)
                return
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if not blocking:
                    raise DatabaseLockBusyError(
                        f"Database is busy: {lock_path}"
                    ) from error
                time.sleep(LOCK_WAIT_SECONDS)

    def _release(lock_file: BinaryIO) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

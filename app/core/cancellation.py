"""Small cooperative-cancellation primitives shared by FireLens services."""

from collections.abc import Callable


CancellationCallback = Callable[[], bool]


class OperationCancelledError(RuntimeError):
    """Raised when a caller cooperatively cancels a synchronous operation."""


def raise_if_cancelled(
    cancellation_callback: CancellationCallback | None,
) -> None:
    """Raise at a safe operation boundary when cancellation was requested."""

    if cancellation_callback is not None and cancellation_callback():
        raise OperationCancelledError("FireLens operation was cancelled")

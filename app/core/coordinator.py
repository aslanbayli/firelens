"""Coordinate indexing and search activity for local repositories."""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


COORDINATOR_WAIT_SECONDS = 0.05
CancellationCheck = Callable[[], None]


class RepositoryBusyError(RuntimeError):
    """Raised when search is requested while a repository is being indexed."""


@dataclass
class _RepositoryState:
    active_searches: int = 0
    indexing: bool = False
    waiting_indexes: int = 0


class RepositoryCoordinator:
    """Provide writer-priority read/write leases for each repository."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._states: dict[str, _RepositoryState] = {}

    @contextmanager
    def indexing(
        self,
        repository_root: str | Path,
        cancellation_check: CancellationCheck | None = None,
    ) -> Iterator[None]:
        """Wait for searches and prior writers, then own an indexing lease."""

        key = _repository_key(repository_root)
        with self._condition:
            state = self._states.setdefault(key, _RepositoryState())
            state.waiting_indexes += 1
            acquired_lease = False
            try:
                while state.indexing or state.active_searches:
                    if cancellation_check is not None:
                        cancellation_check()
                    self._condition.wait(
                        timeout=(
                            COORDINATOR_WAIT_SECONDS
                            if cancellation_check is not None
                            else None
                        )
                    )
                if cancellation_check is not None:
                    cancellation_check()
                state.indexing = True
                acquired_lease = True
            finally:
                state.waiting_indexes -= 1
                if not acquired_lease:
                    self._condition.notify_all()
                    self._remove_idle_state(key, state)

        try:
            yield
        finally:
            with self._condition:
                state.indexing = False
                self._condition.notify_all()
                self._remove_idle_state(key, state)

    @contextmanager
    def searching(self, repository_root: str | Path) -> Iterator[None]:
        """Hold a search lease for the full database read operation."""

        key = _repository_key(repository_root)
        with self._condition:
            state = self._states.setdefault(key, _RepositoryState())
            if state.indexing or state.waiting_indexes:
                raise RepositoryBusyError(
                    f"Repository is currently being indexed: {repository_root}"
                )
            state.active_searches += 1

        try:
            yield
        finally:
            with self._condition:
                state.active_searches -= 1
                self._condition.notify_all()
                self._remove_idle_state(key, state)

    def is_indexing(self, repository_root: str | Path) -> bool:
        """Return whether an indexing run is active or waiting for readers."""

        key = _repository_key(repository_root)
        with self._condition:
            state = self._states.get(key)
            return bool(state and (state.indexing or state.waiting_indexes))

    def require_search_available(self, repository_root: str | Path) -> None:
        """Reject a new search once a writer is active or waiting."""

        key = _repository_key(repository_root)
        with self._condition:
            state = self._states.get(key)
            if state and (state.indexing or state.waiting_indexes):
                raise RepositoryBusyError(
                    f"Repository is currently being indexed: {repository_root}"
                )

    def _remove_idle_state(self, key: str, state: _RepositoryState) -> None:
        if (
            not state.indexing
            and not state.active_searches
            and not state.waiting_indexes
        ):
            self._states.pop(key, None)


def _repository_key(repository_root: str | Path) -> str:
    return str(Path(repository_root).expanduser().resolve())

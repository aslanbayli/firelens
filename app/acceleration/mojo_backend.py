"""ctypes adapter for the optional FireLens Mojo CPU shared library."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from app.acceleration.protocol import (
    AccelerationError,
    CapabilityName,
    Float32Array,
    Int64Array,
    RankedScores,
    SymbolCandidate,
)
from app.acceleration.python_backend import (
    _validate_candidates,
    _validate_float32_array,
    _validate_minimum_score,
    _validate_query,
    _validate_top_k,
    normalize_identifier,
)
from app.search.limits import MAX_FUZZY_CANDIDATE_CHARS, MAX_FUZZY_QUERY_CHARS


MOJO_ABI_VERSION = 1
MOJO_LIBRARY_ENVIRONMENT_VARIABLE = "FIRELENS_MOJO_LIBRARY_PATH"

_FLOAT32_POINTER = ctypes.POINTER(ctypes.c_float)
_FLOAT64_POINTER = ctypes.POINTER(ctypes.c_double)
_INT64_POINTER = ctypes.POINTER(ctypes.c_int64)
_UINT32_POINTER = ctypes.POINTER(ctypes.c_uint32)


class MojoBackendUnavailableError(RuntimeError):
    """Raised when the optional Mojo library cannot be loaded safely."""


class MojoBackend:
    """Run FireLens pure-compute kernels through a versioned C ABI."""

    name = "mojo"

    def __init__(self, library_path: str | Path | None = None) -> None:
        resolved_path = self.discover_library(library_path)
        if resolved_path is None:
            if library_path is not None:
                detail = f" at {Path(library_path).expanduser()}"
            else:
                detail = ""
            raise MojoBackendUnavailableError(
                f"FireLens Mojo shared library was not found{detail}"
            )

        try:
            library = ctypes.CDLL(str(resolved_path))
        except OSError as error:
            raise MojoBackendUnavailableError(
                f"FireLens Mojo shared library could not be loaded: {error}"
            ) from error

        try:
            self._configure_library(library)
            abi_version = int(library.firelens_mojo_abi_version())
        except (AttributeError, TypeError, ValueError) as error:
            raise MojoBackendUnavailableError(
                "FireLens Mojo shared library does not expose the required API"
            ) from error

        if abi_version != MOJO_ABI_VERSION:
            raise MojoBackendUnavailableError(
                "FireLens Mojo ABI version mismatch: "
                f"expected {MOJO_ABI_VERSION}, found {abi_version}"
            )

        self._library = library
        self.abi_version = abi_version
        self.library_path = resolved_path

    @classmethod
    def try_create(
        cls,
        library_path: str | Path | None = None,
    ) -> tuple[Self | None, str | None]:
        """Return an available backend or a user-facing unavailability reason."""

        try:
            return cls(library_path=library_path), None
        except MojoBackendUnavailableError as error:
            return None, str(error)

    @staticmethod
    def discover_library(library_path: str | Path | None = None) -> Path | None:
        """Resolve an explicit path, environment override, or default build."""

        if library_path is not None:
            explicit_path = Path(library_path).expanduser().resolve()
            return explicit_path if explicit_path.is_file() else None

        environment_path = os.environ.get(MOJO_LIBRARY_ENVIRONMENT_VARIABLE)
        if environment_path:
            configured_path = Path(environment_path).expanduser().resolve()
            return configured_path if configured_path.is_file() else None

        project_root = Path(__file__).resolve().parents[2]
        default_path = (
            project_root
            / "build"
            / "mojo"
            / f"libfirelens_mojo{_shared_library_suffix()}"
        )
        return default_path if default_path.is_file() else None

    def supports(self, capability: CapabilityName) -> bool:
        """The compiled CPU library implements every acceleration operation."""

        return capability in {"semantic", "top_k", "fuzzy", "exact"}

    def semantic_top_k(
        self,
        matrix: Float32Array,
        query: Float32Array,
        top_k: int,
    ) -> RankedScores:
        """Return stable top-k clipped matrix-vector scores."""

        _validate_float32_array(matrix, name="matrix", dimensions=2)
        _validate_float32_array(query, name="query", dimensions=1)
        _validate_top_k(top_k)
        if query.shape[0] == 0:
            raise ValueError("Query vector must not be empty")
        if matrix.shape[1] != query.shape[0]:
            raise ValueError("Matrix and query dimensions must match")

        selected_count = min(top_k, matrix.shape[0])
        if selected_count == 0:
            return _empty_ranked_scores()

        output_indices = np.empty(selected_count, dtype=np.int64)
        output_scores = np.empty(selected_count, dtype=np.float32)
        status = self._library.firelens_mojo_semantic_top_k(
            matrix.ctypes.data_as(_FLOAT32_POINTER),
            query.ctypes.data_as(_FLOAT32_POINTER),
            matrix.shape[0],
            matrix.shape[1],
            selected_count,
            output_indices.ctypes.data_as(_INT64_POINTER),
            output_scores.ctypes.data_as(_FLOAT32_POINTER),
        )
        _raise_for_status("semantic_top_k", status)
        return _validated_ranked_scores(
            "semantic_top_k",
            output_indices,
            output_scores,
            upper_index_bound=matrix.shape[0],
        )

    def top_k(self, scores: Float32Array, top_k: int) -> RankedScores:
        """Return stable descending indexes and their unchanged scores."""

        _validate_float32_array(scores, name="scores", dimensions=1)
        _validate_top_k(top_k)
        selected_count = min(top_k, scores.shape[0])
        if selected_count == 0:
            return _empty_ranked_scores()

        output_indices = np.empty(selected_count, dtype=np.int64)
        output_scores = np.empty(selected_count, dtype=np.float32)
        status = self._library.firelens_mojo_top_k(
            scores.ctypes.data_as(_FLOAT32_POINTER),
            scores.shape[0],
            selected_count,
            output_indices.ctypes.data_as(_INT64_POINTER),
            output_scores.ctypes.data_as(_FLOAT32_POINTER),
        )
        _raise_for_status("top_k", status)
        return _validated_ranked_scores(
            "top_k",
            output_indices,
            output_scores,
            upper_index_bound=scores.shape[0],
        )

    def fuzzy_scores(
        self,
        query: str,
        candidates: Sequence[SymbolCandidate],
        minimum_score: float,
    ) -> npt.NDArray[np.float64]:
        """Score normalized short and qualified names in one native call."""

        _validate_query(query)
        validated_candidates = _validate_candidates(candidates)
        threshold = _validate_minimum_score(minimum_score)
        candidate_count = len(validated_candidates)
        if candidate_count == 0:
            return np.empty(0, dtype=np.float64)
        if len(query) > MAX_FUZZY_QUERY_CHARS:
            return np.zeros(candidate_count, dtype=np.float64)

        normalized_query = normalize_identifier(query)
        if normalized_query == "":
            return np.zeros(candidate_count, dtype=np.float64)

        short_names: list[str] = []
        qualified_names: list[str] = []
        for short_name, qualified_name in validated_candidates:
            short_names.append(
                normalize_identifier(short_name)
                if len(short_name) <= MAX_FUZZY_CANDIDATE_CHARS
                else ""
            )
            qualified_names.append(
                normalize_identifier(qualified_name)
                if len(qualified_name) <= MAX_FUZZY_CANDIDATE_CHARS
                else ""
            )

        query_buffer, query_length = _pack_query(normalized_query)
        short_buffer, short_offsets = _pack_strings(short_names)
        qualified_buffer, qualified_offsets = _pack_strings(qualified_names)
        output_scores = np.empty(candidate_count, dtype=np.float64)

        status = self._library.firelens_mojo_fuzzy_scores(
            query_buffer.ctypes.data_as(_UINT32_POINTER),
            query_length,
            short_buffer.ctypes.data_as(_UINT32_POINTER),
            short_offsets.ctypes.data_as(_INT64_POINTER),
            qualified_buffer.ctypes.data_as(_UINT32_POINTER),
            qualified_offsets.ctypes.data_as(_INT64_POINTER),
            candidate_count,
            threshold,
            output_scores.ctypes.data_as(_FLOAT64_POINTER),
        )
        _raise_for_status("fuzzy_scores", status)
        return output_scores

    def exact_match_indices(
        self,
        query: str,
        candidates: Sequence[SymbolCandidate],
    ) -> Int64Array:
        """Return qualified matches followed by remaining short-name matches."""

        _validate_query(query)
        validated_candidates = _validate_candidates(candidates)
        normalized_query = query.strip()
        candidate_count = len(validated_candidates)
        if normalized_query == "" or candidate_count == 0:
            return np.empty(0, dtype=np.int64)

        short_names = [candidate[0] for candidate in validated_candidates]
        qualified_names = [candidate[1] for candidate in validated_candidates]
        query_buffer, query_length = _pack_query(normalized_query)
        short_buffer, short_offsets = _pack_strings(short_names)
        qualified_buffer, qualified_offsets = _pack_strings(qualified_names)
        output_indices = np.empty(candidate_count, dtype=np.int64)
        output_count = np.zeros(1, dtype=np.int64)

        status = self._library.firelens_mojo_exact_match_indices(
            query_buffer.ctypes.data_as(_UINT32_POINTER),
            query_length,
            short_buffer.ctypes.data_as(_UINT32_POINTER),
            short_offsets.ctypes.data_as(_INT64_POINTER),
            qualified_buffer.ctypes.data_as(_UINT32_POINTER),
            qualified_offsets.ctypes.data_as(_INT64_POINTER),
            candidate_count,
            output_indices.ctypes.data_as(_INT64_POINTER),
            output_count.ctypes.data_as(_INT64_POINTER),
        )
        _raise_for_status("exact_match_indices", status)
        match_count = int(output_count[0])
        if not 0 <= match_count <= candidate_count:
            raise AccelerationError(
                "Mojo exact_match_indices returned an invalid result count"
            )
        matches = np.asarray(output_indices[:match_count], dtype=np.int64)
        if np.any(matches < 0) or np.any(matches >= candidate_count):
            raise AccelerationError(
                "Mojo exact_match_indices returned an invalid candidate index"
            )
        return matches

    @staticmethod
    def _configure_library(library: ctypes.CDLL) -> None:
        library.firelens_mojo_abi_version.argtypes = []
        library.firelens_mojo_abi_version.restype = ctypes.c_int32

        library.firelens_mojo_top_k.argtypes = [
            _FLOAT32_POINTER,
            ctypes.c_int64,
            ctypes.c_int64,
            _INT64_POINTER,
            _FLOAT32_POINTER,
        ]
        library.firelens_mojo_top_k.restype = ctypes.c_int32

        library.firelens_mojo_semantic_top_k.argtypes = [
            _FLOAT32_POINTER,
            _FLOAT32_POINTER,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            _INT64_POINTER,
            _FLOAT32_POINTER,
        ]
        library.firelens_mojo_semantic_top_k.restype = ctypes.c_int32

        library.firelens_mojo_fuzzy_scores.argtypes = [
            _UINT32_POINTER,
            ctypes.c_int64,
            _UINT32_POINTER,
            _INT64_POINTER,
            _UINT32_POINTER,
            _INT64_POINTER,
            ctypes.c_int64,
            ctypes.c_double,
            _FLOAT64_POINTER,
        ]
        library.firelens_mojo_fuzzy_scores.restype = ctypes.c_int32

        library.firelens_mojo_exact_match_indices.argtypes = [
            _UINT32_POINTER,
            ctypes.c_int64,
            _UINT32_POINTER,
            _INT64_POINTER,
            _UINT32_POINTER,
            _INT64_POINTER,
            ctypes.c_int64,
            _INT64_POINTER,
            _INT64_POINTER,
        ]
        library.firelens_mojo_exact_match_indices.restype = ctypes.c_int32


def _shared_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _empty_ranked_scores() -> RankedScores:
    return RankedScores(
        indices=np.empty(0, dtype=np.int64),
        scores=np.empty(0, dtype=np.float32),
    )


def _validated_ranked_scores(
    operation: str,
    indices: Int64Array,
    scores: npt.NDArray[np.float32],
    *,
    upper_index_bound: int,
) -> RankedScores:
    if np.any(indices < 0) or np.any(indices >= upper_index_bound):
        raise AccelerationError(
            f"Mojo {operation} returned an invalid candidate index"
        )
    if not np.all(np.isfinite(scores)):
        raise AccelerationError(f"Mojo {operation} returned a non-finite score")
    return RankedScores(indices=indices, scores=scores)


def _pack_query(value: str) -> tuple[npt.NDArray[np.uint32], int]:
    logical_length = len(value)
    buffer = np.empty(max(logical_length, 1), dtype=np.uint32)
    for index, character in enumerate(value):
        buffer[index] = ord(character)
    if logical_length == 0:
        buffer[0] = 0
    return buffer, logical_length


def _pack_strings(
    values: Sequence[str],
) -> tuple[npt.NDArray[np.uint32], Int64Array]:
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    total_length = 0
    for index, value in enumerate(values, start=1):
        total_length += len(value)
        offsets[index] = total_length

    buffer = np.empty(max(total_length, 1), dtype=np.uint32)
    write_index = 0
    for value in values:
        for character in value:
            buffer[write_index] = ord(character)
            write_index += 1
    if total_length == 0:
        buffer[0] = 0
    return buffer, offsets


def _raise_for_status(operation: str, status: int) -> None:
    status_code = int(status)
    if status_code == 0:
        return
    if status_code == 1:
        detail = "invalid arguments"
    else:
        detail = f"unknown status {status_code}"
    raise AccelerationError(f"Mojo {operation} failed: {detail}")

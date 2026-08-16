"""Versioned C ABI for FireLens Mojo CPU kernels."""

from exact import exact_match_indices
from fuzzy import batch_fuzzy_scores
from topk import select_top_k
from vector import semantic_top_k


comptime ABI_VERSION = Int32(1)
comptime STATUS_OK = Int32(0)
comptime STATUS_INVALID_ARGUMENT = Int32(1)


@export
def firelens_mojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def firelens_mojo_top_k(
    scores: Pointer[Float32, MutAnyOrigin],
    score_count: Int64,
    requested_count: Int64,
    output_indices: Pointer[Int64, MutAnyOrigin],
    output_scores: Pointer[Float32, MutAnyOrigin],
) abi("C") -> Int32:
    if score_count < 0 or requested_count < 1:
        return STATUS_INVALID_ARGUMENT
    _ = select_top_k(
        scores,
        Int(score_count),
        Int(requested_count),
        output_indices,
        output_scores,
    )
    return STATUS_OK


@export
def firelens_mojo_semantic_top_k(
    matrix: Pointer[Float32, MutAnyOrigin],
    query: Pointer[Float32, MutAnyOrigin],
    row_count: Int64,
    column_count: Int64,
    requested_count: Int64,
    output_indices: Pointer[Int64, MutAnyOrigin],
    output_scores: Pointer[Float32, MutAnyOrigin],
) abi("C") -> Int32:
    if row_count < 0 or column_count < 1 or requested_count < 1:
        return STATUS_INVALID_ARGUMENT
    semantic_top_k(
        matrix,
        query,
        Int(row_count),
        Int(column_count),
        Int(requested_count),
        output_indices,
        output_scores,
    )
    return STATUS_OK


@export
def firelens_mojo_fuzzy_scores(
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int64,
    short_names: Pointer[UInt32, MutAnyOrigin],
    short_offsets: Pointer[Int64, MutAnyOrigin],
    qualified_names: Pointer[UInt32, MutAnyOrigin],
    qualified_offsets: Pointer[Int64, MutAnyOrigin],
    candidate_count: Int64,
    minimum_score: Float64,
    output_scores: Pointer[Float64, MutAnyOrigin],
) abi("C") -> Int32:
    if query_length < 0 or candidate_count < 0:
        return STATUS_INVALID_ARGUMENT
    if minimum_score < 0.0 or minimum_score > 1.0:
        return STATUS_INVALID_ARGUMENT
    batch_fuzzy_scores(
        query,
        Int(query_length),
        short_names,
        short_offsets,
        qualified_names,
        qualified_offsets,
        Int(candidate_count),
        minimum_score,
        output_scores,
    )
    return STATUS_OK


@export
def firelens_mojo_exact_match_indices(
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int64,
    short_names: Pointer[UInt32, MutAnyOrigin],
    short_offsets: Pointer[Int64, MutAnyOrigin],
    qualified_names: Pointer[UInt32, MutAnyOrigin],
    qualified_offsets: Pointer[Int64, MutAnyOrigin],
    candidate_count: Int64,
    output_indices: Pointer[Int64, MutAnyOrigin],
    output_count: Pointer[Int64, MutAnyOrigin],
) abi("C") -> Int32:
    if query_length < 0 or candidate_count < 0:
        return STATUS_INVALID_ARGUMENT
    output_count[unsafe_offset=0] = Int64(
        exact_match_indices(
            query,
            Int(query_length),
            short_names,
            short_offsets,
            qualified_names,
            qualified_offsets,
            Int(candidate_count),
            output_indices,
        )
    )
    return STATUS_OK

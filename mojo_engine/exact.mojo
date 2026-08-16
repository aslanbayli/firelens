"""Batch exact matching over Unicode-code-point buffers."""


def _matches(
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int,
    names: Pointer[UInt32, MutAnyOrigin],
    name_start: Int,
    name_end: Int,
) -> Bool:
    var name_length = name_end - name_start
    if query_length != name_length:
        return False
    for index in range(query_length):
        if (
            query[unsafe_offset=index]
            != names[unsafe_offset=name_start + index]
        ):
            return False
    return True


def exact_match_indices(
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int,
    short_names: Pointer[UInt32, MutAnyOrigin],
    short_offsets: Pointer[Int64, MutAnyOrigin],
    qualified_names: Pointer[UInt32, MutAnyOrigin],
    qualified_offsets: Pointer[Int64, MutAnyOrigin],
    candidate_count: Int,
    output_indices: Pointer[Int64, MutAnyOrigin],
) -> Int:
    """Write qualified matches, then non-duplicate short-name matches."""

    var output_count = 0
    for candidate_index in range(candidate_count):
        var qualified_start = Int(
            qualified_offsets[unsafe_offset=candidate_index]
        )
        var qualified_end = Int(
            qualified_offsets[unsafe_offset=candidate_index + 1]
        )
        if _matches(
            query,
            query_length,
            qualified_names,
            qualified_start,
            qualified_end,
        ):
            output_indices[unsafe_offset=output_count] = Int64(candidate_index)
            output_count += 1

    for candidate_index in range(candidate_count):
        var qualified_start = Int(
            qualified_offsets[unsafe_offset=candidate_index]
        )
        var qualified_end = Int(
            qualified_offsets[unsafe_offset=candidate_index + 1]
        )
        if _matches(
            query,
            query_length,
            qualified_names,
            qualified_start,
            qualified_end,
        ):
            continue

        var short_start = Int(short_offsets[unsafe_offset=candidate_index])
        var short_end = Int(short_offsets[unsafe_offset=candidate_index + 1])
        if _matches(
            query,
            query_length,
            short_names,
            short_start,
            short_end,
        ):
            output_indices[unsafe_offset=output_count] = Int64(candidate_index)
            output_count += 1

    return output_count

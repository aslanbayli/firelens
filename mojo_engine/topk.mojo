"""Stable top-k selection over contiguous float32 scores."""


def insert_top_k_candidate(
    candidate_score: Float32,
    candidate_index: Int,
    selected_count: Int,
    selected_limit: Int,
    output_indices: Pointer[Int64, MutAnyOrigin],
    output_scores: Pointer[Float32, MutAnyOrigin],
) -> Int:
    """Insert one candidate and return the updated selected count."""

    var insertion_index = selected_count
    # Candidates arrive in ascending original-index order. Inserting only on a
    # strictly greater score therefore preserves stable tie ordering.
    for result_index in range(selected_count):
        if candidate_score > output_scores[unsafe_offset=result_index]:
            insertion_index = result_index
            break

    if insertion_index >= selected_limit:
        return selected_count

    var last_result_index = selected_count
    if last_result_index >= selected_limit:
        last_result_index = selected_limit - 1

    var result_index = last_result_index
    while result_index > insertion_index:
        output_scores[unsafe_offset=result_index] = output_scores[
            unsafe_offset=result_index - 1
        ]
        output_indices[unsafe_offset=result_index] = output_indices[
            unsafe_offset=result_index - 1
        ]
        result_index -= 1

    output_scores[unsafe_offset=insertion_index] = candidate_score
    output_indices[unsafe_offset=insertion_index] = Int64(candidate_index)

    if selected_count < selected_limit:
        return selected_count + 1
    return selected_count


def select_top_k[
    input_origin: MutOrigin
](
    scores: Pointer[Float32, input_origin],
    score_count: Int,
    requested_count: Int,
    output_indices: Pointer[Int64, MutAnyOrigin],
    output_scores: Pointer[Float32, MutAnyOrigin],
) -> Int:
    """Write stable descending results and return the number selected."""

    var selected_limit = requested_count
    if selected_limit > score_count:
        selected_limit = score_count

    var selected_count = 0
    for candidate_index in range(score_count):
        var candidate_score = scores[unsafe_offset=candidate_index]
        selected_count = insert_top_k_candidate(
            candidate_score,
            candidate_index,
            selected_count,
            selected_limit,
            output_indices,
            output_scores,
        )

    return selected_count

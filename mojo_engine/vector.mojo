"""CPU vector kernels for semantic ranking."""

from topk import insert_top_k_candidate


def semantic_top_k(
    matrix: Pointer[Float32, MutAnyOrigin],
    query: Pointer[Float32, MutAnyOrigin],
    row_count: Int,
    column_count: Int,
    requested_count: Int,
    output_indices: Pointer[Int64, MutAnyOrigin],
    output_scores: Pointer[Float32, MutAnyOrigin],
):
    """Compute clipped row dot products and select stable top-k results."""

    comptime SIMD_WIDTH = 8
    var selected_limit = requested_count
    if selected_limit > row_count:
        selected_limit = row_count
    var selected_count = 0

    for row_index in range(row_count):
        var score: Float32 = 0.0
        var row_offset = row_index * column_count
        var vector_score = SIMD[DType.float32, SIMD_WIDTH](0.0)
        var column_index = 0
        while column_index + SIMD_WIDTH <= column_count:
            var matrix_values = matrix.unsafe_load[width=SIMD_WIDTH](
                row_offset + column_index
            )
            var query_values = query.unsafe_load[width=SIMD_WIDTH](column_index)
            vector_score += matrix_values * query_values
            column_index += SIMD_WIDTH
        score += vector_score.reduce_add()

        while column_index < column_count:
            score += (
                matrix[unsafe_offset=row_offset + column_index]
                * query[unsafe_offset=column_index]
            )
            column_index += 1

        if score < -1.0:
            score = -1.0
        elif score > 1.0:
            score = 1.0
        selected_count = insert_top_k_candidate(
            score,
            row_index,
            selected_count,
            selected_limit,
            output_indices,
            output_scores,
        )

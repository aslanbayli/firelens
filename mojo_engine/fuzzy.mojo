"""Unicode-code-point fuzzy scoring kernels."""


def _equal_ranges(
    first: Pointer[UInt32, MutAnyOrigin],
    first_start: Int,
    first_length: Int,
    second: Pointer[UInt32, MutAnyOrigin],
    second_start: Int,
    second_length: Int,
) -> Bool:
    if first_length != second_length:
        return False
    for index in range(first_length):
        if (
            first[unsafe_offset=first_start + index]
            != second[unsafe_offset=second_start + index]
        ):
            return False
    return True


def _starts_with(
    candidate: Pointer[UInt32, MutAnyOrigin],
    candidate_start: Int,
    candidate_length: Int,
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int,
) -> Bool:
    if query_length > candidate_length:
        return False
    for index in range(query_length):
        if (
            candidate[unsafe_offset=candidate_start + index]
            != query[unsafe_offset=index]
        ):
            return False
    return True


def _contains(
    candidate: Pointer[UInt32, MutAnyOrigin],
    candidate_start: Int,
    candidate_length: Int,
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int,
) -> Bool:
    if query_length > candidate_length:
        return False
    for candidate_offset in range(candidate_length - query_length + 1):
        if _equal_ranges(
            candidate,
            candidate_start + candidate_offset,
            query_length,
            query,
            0,
            query_length,
        ):
            return True
    return False


def _bounded_levenshtein_distance(
    first: Pointer[UInt32, MutAnyOrigin],
    first_start: Int,
    first_length: Int,
    second: Pointer[UInt32, MutAnyOrigin],
    second_start: Int,
    second_length: Int,
    maximum_distance: Int,
) -> Int:
    if first_length - second_length > maximum_distance:
        return maximum_distance + 1
    if second_length - first_length > maximum_distance:
        return maximum_distance + 1
    if _equal_ranges(
        first,
        first_start,
        first_length,
        second,
        second_start,
        second_length,
    ):
        return 0

    var longer = first
    var longer_start = first_start
    var longer_length = first_length
    var shorter = second
    var shorter_start = second_start
    var shorter_length = second_length
    if first_length < second_length:
        longer = second
        longer_start = second_start
        longer_length = second_length
        shorter = first
        shorter_start = first_start
        shorter_length = first_length

    var outside_bound = maximum_distance + 1
    var previous_row = List[Int](capacity=shorter_length + 1)
    var current_row = List[Int](capacity=shorter_length + 1)
    for column in range(shorter_length + 1):
        if column <= maximum_distance:
            previous_row.append(column)
        else:
            previous_row.append(outside_bound)
        current_row.append(outside_bound)

    for row_number in range(1, longer_length + 1):
        for column in range(shorter_length + 1):
            current_row[column] = outside_bound
        if row_number <= maximum_distance:
            current_row[0] = row_number

        var first_column = row_number - maximum_distance
        if first_column < 1:
            first_column = 1
        var last_column = row_number + maximum_distance
        if last_column > shorter_length:
            last_column = shorter_length

        var row_has_value = False
        for column_number in range(first_column, last_column + 1):
            var insertion_cost = current_row[column_number - 1] + 1
            var deletion_cost = previous_row[column_number] + 1
            var substitution_cost = previous_row[column_number - 1]
            if (
                longer[unsafe_offset=longer_start + row_number - 1]
                != shorter[unsafe_offset=shorter_start + column_number - 1]
            ):
                substitution_cost += 1

            var distance = insertion_cost
            if deletion_cost < distance:
                distance = deletion_cost
            if substitution_cost < distance:
                distance = substitution_cost
            if distance <= maximum_distance:
                current_row[column_number] = distance
                row_has_value = True

        if not row_has_value:
            return outside_bound

        var row_to_reuse = previous_row^
        previous_row = current_row^
        current_row = row_to_reuse^

    var result = previous_row[shorter_length]
    if result > maximum_distance:
        return outside_bound
    return result


def fuzzy_score(
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int,
    candidate: Pointer[UInt32, MutAnyOrigin],
    candidate_start: Int,
    candidate_length: Int,
    minimum_score: Float64,
) -> Float64:
    if query_length == 0 or candidate_length == 0:
        return 0.0
    if _equal_ranges(
        query,
        0,
        query_length,
        candidate,
        candidate_start,
        candidate_length,
    ):
        return 1.0
    if _starts_with(
        candidate,
        candidate_start,
        candidate_length,
        query,
        query_length,
    ):
        return 0.95
    if _contains(
        candidate,
        candidate_start,
        candidate_length,
        query,
        query_length,
    ):
        return 0.85

    var maximum_length = query_length
    if candidate_length > maximum_length:
        maximum_length = candidate_length
    var maximum_distance = Int(
        ((1.0 - minimum_score) * Float64(maximum_length)) + 1e-12
    )
    var distance = _bounded_levenshtein_distance(
        query,
        0,
        query_length,
        candidate,
        candidate_start,
        candidate_length,
        maximum_distance,
    )
    if distance > maximum_distance:
        return 0.0

    var score = 1.0 - (Float64(distance) / Float64(maximum_length))
    if score < 0.0:
        return 0.0
    return score


def batch_fuzzy_scores(
    query: Pointer[UInt32, MutAnyOrigin],
    query_length: Int,
    short_names: Pointer[UInt32, MutAnyOrigin],
    short_offsets: Pointer[Int64, MutAnyOrigin],
    qualified_names: Pointer[UInt32, MutAnyOrigin],
    qualified_offsets: Pointer[Int64, MutAnyOrigin],
    candidate_count: Int,
    minimum_score: Float64,
    output_scores: Pointer[Float64, MutAnyOrigin],
):
    for candidate_index in range(candidate_count):
        var short_start = Int(short_offsets[unsafe_offset=candidate_index])
        var short_end = Int(short_offsets[unsafe_offset=candidate_index + 1])
        var qualified_start = Int(
            qualified_offsets[unsafe_offset=candidate_index]
        )
        var qualified_end = Int(
            qualified_offsets[unsafe_offset=candidate_index + 1]
        )

        var short_score = fuzzy_score(
            query,
            query_length,
            short_names,
            short_start,
            short_end - short_start,
            minimum_score,
        )
        var qualified_score = fuzzy_score(
            query,
            query_length,
            qualified_names,
            qualified_start,
            qualified_end - qualified_start,
            minimum_score,
        )
        if qualified_score > short_score:
            short_score = qualified_score
        output_scores[unsafe_offset=candidate_index] = short_score

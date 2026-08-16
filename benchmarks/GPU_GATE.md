# Apple GPU acceleration gate

## Current pinned-toolchain result

Status: **unavailable for production**.

The local Mojo 1.0.0 compiler recognizes the `apple-m4` accelerator target,
but its `std.gpu.host` package does not provide `DeviceContext`. A dry-run with
MAX 26.4.0 requires replacing `mojo-compiler==1.0.0` with the older
`mojo-compiler==1.0.0b2`. FireLens will not add MAX or downgrade its pinned
compiler to make this experiment pass, so the current implementation remains
CPU-only.

## Gate for a future compatible toolchain

Record the toolchain version, hardware, initial matrix upload time, ten query
latencies with the matrix resident, Mojo CPU latencies for the same inputs,
median, p95, and parity at 10,000 and 50,000 rows with 768 dimensions.

Enable GPU routing only when the upload cost amortized across the ten-query
workload is at least 20% faster in median than Mojo CPU at both sizes, p95 does
not regress, and semantic indices and float32 scores pass the normal parity
gate. Otherwise retain CPU routing and record the negative result here.

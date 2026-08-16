# CPU acceleration baseline

This baseline records the release profile used to choose FireLens's automatic
backend crossover points. It measures complete in-memory backend calls,
including Python-to-native buffer preparation, after deterministic dataset
construction has finished.

## Environment

- Date: 2026-08-16
- Hardware: Apple M4 Max, 16 logical CPUs, 64 GiB memory
- Operating system: macOS 15.7.4, arm64
- Python: CPython 3.14.3
- NumPy: 2.5.0
- Mojo: 1.0.0
- Warm-ups: 5 per backend and case
- Measured iterations: 30 per backend and case

## Release profile

| Operation | Candidates | Python median | Python p95 | Mojo median | Mojo p95 | Speedup | Parity |
|---|---:|---:|---:|---:|---:|---:|---|
| semantic, 768 dimensions | 1,000 | 0.075 ms | 0.083 ms | 0.103 ms | 0.120 ms | 0.73x | passed |
| semantic, 768 dimensions | 10,000 | 1.472 ms | 1.795 ms | 1.159 ms | 1.271 ms | 1.27x | passed |
| semantic, 768 dimensions | 50,000 | 7.696 ms | 8.039 ms | 6.490 ms | 7.456 ms | 1.19x | passed |
| fuzzy | 128 | 2.872 ms | 2.912 ms | 0.712 ms | 0.730 ms | 4.03x | passed |
| fuzzy | 512 | 11.257 ms | 11.388 ms | 2.788 ms | 2.850 ms | 4.04x | passed |
| exact | 10,000 | 0.628 ms | 0.730 ms | 20.115 ms | 20.691 ms | 0.03x | passed |
| exact | 50,000 | 3.175 ms | 3.239 ms | 100.072 ms | 103.276 ms | 0.03x | passed |
| exact | 100,000 | 6.602 ms | 6.776 ms | 202.548 ms | 207.503 ms | 0.03x | passed |

Semantic crossover measurements at 30,000, 40,000, and 50,000 rows showed
1.10x, 1.14x, and 1.16x median speedups in a separate 50-run sweep. Fuzzy
crossed over at four candidates and remained approximately 4.0x faster at the
production limit of 512.

## Routing decision

- Automatically select Mojo fuzzy scoring from four candidates onward.
- Automatically select Mojo semantic ranking from 30,000 candidates onward.
- Keep indexed SQLite exact search in production; retain the native exact
  kernel only for parity and benchmark experiments.
- Allow explicit `mojo` requests to exercise fuzzy and semantic kernels below
  the automatic thresholds.

The thresholds are configuration defaults rather than universal hardware
claims. Re-run `python -m benchmarks` and override the two crossover settings
when deploying to a materially different CPU or toolchain.

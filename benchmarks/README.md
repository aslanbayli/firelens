# FireLens acceleration benchmarks

This directory contains deterministic, in-memory benchmarks for the Python and
Mojo compute backends. Dataset construction happens before timing. Each case
records warm-up and timed iteration counts, every sample, median, p95, minimum,
parity, speedup, and environment metadata in a JSON report.

The default smoke profile is intentionally small enough for local iteration:

```bash
uv run python -m benchmarks --comparison-backend auto \
  --output build/benchmarks/smoke.json
```

Run the release-gate profile after building the Mojo shared library:

```bash
uv run python -m benchmarks --profile full --comparison-backend mojo \
  --output build/benchmarks/full.json \
  --table-output build/benchmarks/full.md
```

The full profile uses five warm-ups and 30 measured iterations for:

- Semantic ranking at 1,000, 10,000, and 50,000 rows with 768 dimensions.
- Fuzzy scoring at 128 and 512 candidate pairs.
- Exact matching at 10,000, 50,000, and 100,000 candidate pairs.

Use `--semantic-sizes`, `--semantic-dimension`, `--fuzzy-sizes`,
`--exact-sizes`, `--warmups`, and `--runs` to define another profile. Each
sizes option accepts comma-separated integers. Use `--operations` to select a
comma-separated subset of `semantic`, `fuzzy`, and `exact`.

`--comparison-backend auto` records Mojo as skipped when its library is not
available. `--comparison-backend mojo` treats an unavailable library as an
error. `--mojo-library` selects a specific compiled library.

Every backend timing includes calls per second and candidate throughput. The
JSON `comparison_table` contains one flattened, side-by-side row per case;
`--table-output` writes the same median, p95, speedup, and parity columns as a
Markdown table.

The checked-in [CPU baseline](CPU_BASELINE.md) records the release-profile
result used for the default automatic crossover thresholds. Treat those
thresholds as machine-specific starting points and rerun the profile on
materially different hardware.

GPU measurements are a separate, conditional experiment because they require
a resident device matrix and GPU synchronization. The JSON report includes the
gate criteria as an unexecuted template. See [GPU_GATE.md](GPU_GATE.md) for the
current pinned-toolchain result.

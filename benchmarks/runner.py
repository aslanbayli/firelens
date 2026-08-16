"""Run deterministic Python-versus-Mojo acceleration benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from benchmarks.datasets import (
    CandidateNames,
    make_exact_dataset,
    make_fuzzy_dataset,
    make_semantic_dataset,
)


class AccelerationBackend(Protocol):
    """The backend surface exercised by this benchmark suite."""

    name: str

    def semantic_top_k(
        self,
        matrix: np.ndarray,
        query: np.ndarray,
        top_k: int,
    ) -> Any: ...

    def fuzzy_scores(
        self,
        query: str,
        candidates: CandidateNames,
        minimum_score: float,
    ) -> np.ndarray: ...

    def exact_match_indices(
        self,
        query: str,
        candidates: CandidateNames,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class BenchmarkConfig:
    """Dataset sizes and measurement settings for one benchmark run."""

    semantic_sizes: tuple[int, ...] = (100, 1_000)
    semantic_dimension: int = 128
    fuzzy_sizes: tuple[int, ...] = (32, 128)
    exact_sizes: tuple[int, ...] = (1_000, 5_000)
    top_k: int = 10
    minimum_fuzzy_score: float = 0.55
    warmups: int = 2
    runs: int = 5
    seed: int = 20260816
    operations: tuple[str, ...] = ("semantic", "fuzzy", "exact")

    @classmethod
    def full(cls) -> BenchmarkConfig:
        """Return the release-gate sizes from the Mojo implementation plan."""

        return cls(
            semantic_sizes=(1_000, 10_000, 50_000),
            semantic_dimension=768,
            fuzzy_sizes=(128, 512),
            exact_sizes=(10_000, 50_000, 100_000),
            warmups=5,
            runs=30,
        )

    def validate(self) -> None:
        """Reject configurations that cannot produce useful measurements."""

        if any(size < 1 for size in self.semantic_sizes):
            raise ValueError("semantic sizes must be greater than 0")
        if any(size < 1 for size in self.fuzzy_sizes):
            raise ValueError("fuzzy sizes must be greater than 0")
        if any(size < 1 for size in self.exact_sizes):
            raise ValueError("exact sizes must be greater than 0")
        if self.semantic_dimension < 1:
            raise ValueError("semantic_dimension must be greater than 0")
        if self.top_k < 1:
            raise ValueError("top_k must be greater than 0")
        if not 0.0 <= self.minimum_fuzzy_score <= 1.0:
            raise ValueError("minimum_fuzzy_score must be between 0 and 1")
        if self.warmups < 0:
            raise ValueError("warmups must not be negative")
        if self.runs < 1:
            raise ValueError("runs must be greater than 0")
        allowed_operations = {"semantic", "fuzzy", "exact"}
        if not self.operations or not set(self.operations) <= allowed_operations:
            raise ValueError("operations must contain semantic, fuzzy, or exact")


@dataclass(frozen=True)
class TimingSummary:
    """Wall-clock measurements for repeated calls to one backend operation."""

    warmups: int
    runs: int
    median_ms: float
    p95_ms: float
    min_ms: float
    samples_ms: tuple[float, ...]


def run_benchmarks(
    *,
    config: BenchmarkConfig | None = None,
    python_backend: AccelerationBackend | None = None,
    comparison_backend: AccelerationBackend | None = None,
    comparison_skip_reason: str | None = None,
) -> dict[str, Any]:
    """Benchmark a Python backend and, when supplied, a comparison backend."""

    selected_config = config or BenchmarkConfig()
    selected_config.validate()
    reference_backend = python_backend or _load_python_backend()

    cases: list[dict[str, Any]] = []
    if "semantic" in selected_config.operations:
        for candidate_count in selected_config.semantic_sizes:
            cases.append(
                _run_semantic_case(
                    selected_config,
                    candidate_count,
                    reference_backend,
                    comparison_backend,
                    comparison_skip_reason,
                )
            )
    if "fuzzy" in selected_config.operations:
        for candidate_count in selected_config.fuzzy_sizes:
            cases.append(
                _run_fuzzy_case(
                    selected_config,
                    candidate_count,
                    reference_backend,
                    comparison_backend,
                    comparison_skip_reason,
                )
            )
    if "exact" in selected_config.operations:
        for candidate_count in selected_config.exact_sizes:
            cases.append(
                _run_exact_case(
                    selected_config,
                    candidate_count,
                    reference_backend,
                    comparison_backend,
                    comparison_skip_reason,
                )
            )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment_metadata(
            reference_backend,
            comparison_backend,
        ),
        "config": _config_dict(selected_config),
        "cases": cases,
        "summary": _summarize_cases(cases),
        "comparison_table": _comparison_table(cases),
        "gpu_gate": {
            "status": "unavailable",
            "workload": "resident matrix, 10 semantic queries",
            "sizes": [10_000, 50_000],
            "pass_criteria": {
                "minimum_median_speedup_percent": 20,
                "allow_p95_regression": False,
                "include_initial_upload_amortized_across_queries": True,
            },
            "notes": (
                "The pinned Mojo 1.0.0 stack lacks the required host API; "
                "see benchmarks/GPU_GATE.md"
            ),
        },
    }


def _run_semantic_case(
    config: BenchmarkConfig,
    candidate_count: int,
    reference_backend: AccelerationBackend,
    comparison_backend: AccelerationBackend | None,
    comparison_skip_reason: str | None,
) -> dict[str, Any]:
    dataset = make_semantic_dataset(
        candidate_count,
        config.semantic_dimension,
        top_k=config.top_k,
        seed=config.seed,
    )

    def operation(backend: AccelerationBackend) -> Any:
        return backend.semantic_top_k(dataset.matrix, dataset.query, dataset.top_k)

    return _run_case(
        operation="semantic",
        candidate_count=candidate_count,
        parameters={
            "dimension": config.semantic_dimension,
            "top_k": dataset.top_k,
            "dataset_bytes": int(dataset.matrix.nbytes + dataset.query.nbytes),
        },
        reference_backend=reference_backend,
        comparison_backend=comparison_backend,
        comparison_skip_reason=comparison_skip_reason,
        benchmark_operation=operation,
        parity_check=_semantic_parity,
        warmups=config.warmups,
        runs=config.runs,
    )


def _run_fuzzy_case(
    config: BenchmarkConfig,
    candidate_count: int,
    reference_backend: AccelerationBackend,
    comparison_backend: AccelerationBackend | None,
    comparison_skip_reason: str | None,
) -> dict[str, Any]:
    dataset = make_fuzzy_dataset(
        candidate_count,
        minimum_score=config.minimum_fuzzy_score,
        seed=config.seed,
    )

    def operation(backend: AccelerationBackend) -> Any:
        return backend.fuzzy_scores(
            dataset.query,
            dataset.candidates,
            dataset.minimum_score,
        )

    return _run_case(
        operation="fuzzy",
        candidate_count=candidate_count,
        parameters={
            "minimum_score": dataset.minimum_score,
            "dataset_bytes": _candidate_text_bytes(dataset.query, dataset.candidates),
        },
        reference_backend=reference_backend,
        comparison_backend=comparison_backend,
        comparison_skip_reason=comparison_skip_reason,
        benchmark_operation=operation,
        parity_check=_numeric_array_parity,
        warmups=config.warmups,
        runs=config.runs,
    )


def _run_exact_case(
    config: BenchmarkConfig,
    candidate_count: int,
    reference_backend: AccelerationBackend,
    comparison_backend: AccelerationBackend | None,
    comparison_skip_reason: str | None,
) -> dict[str, Any]:
    dataset = make_exact_dataset(candidate_count, seed=config.seed)

    def operation(backend: AccelerationBackend) -> Any:
        return backend.exact_match_indices(dataset.query, dataset.candidates)

    return _run_case(
        operation="exact",
        candidate_count=candidate_count,
        parameters={
            "dataset_bytes": _candidate_text_bytes(dataset.query, dataset.candidates)
        },
        reference_backend=reference_backend,
        comparison_backend=comparison_backend,
        comparison_skip_reason=comparison_skip_reason,
        benchmark_operation=operation,
        parity_check=_integer_array_parity,
        warmups=config.warmups,
        runs=config.runs,
    )


def _run_case(
    *,
    operation: str,
    candidate_count: int,
    parameters: dict[str, Any],
    reference_backend: AccelerationBackend,
    comparison_backend: AccelerationBackend | None,
    comparison_skip_reason: str | None,
    benchmark_operation: Callable[[AccelerationBackend], Any],
    parity_check: Callable[[Any, Any], tuple[bool, str]],
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    reference_timing, reference_output = measure_operation(
        lambda: benchmark_operation(reference_backend),
        warmups=warmups,
        runs=runs,
    )
    case = {
        "operation": operation,
        "candidate_count": candidate_count,
        "parameters": parameters,
        "reference": {
            "backend": _backend_name(reference_backend, "python"),
            "timing": asdict(reference_timing),
            "throughput": _throughput(reference_timing, candidate_count),
        },
        "comparison": None,
        "parity": {
            "status": "not_run",
            "passed": None,
            "details": comparison_skip_reason or "No comparison backend supplied",
        },
        "speedup": None,
    }
    if comparison_backend is None:
        return case

    try:
        comparison_timing, comparison_output = measure_operation(
            lambda: benchmark_operation(comparison_backend),
            warmups=warmups,
            runs=runs,
        )
    except Exception as error:  # The JSON report should retain failed cases.
        case["comparison"] = {
            "backend": _backend_name(comparison_backend, "comparison"),
            "error": f"{type(error).__name__}: {error}",
        }
        case["parity"] = {
            "status": "error",
            "passed": False,
            "details": "The comparison backend did not complete the operation",
        }
        return case

    parity_passed, parity_details = parity_check(
        reference_output,
        comparison_output,
    )
    case["comparison"] = {
        "backend": _backend_name(comparison_backend, "comparison"),
        "timing": asdict(comparison_timing),
        "throughput": _throughput(comparison_timing, candidate_count),
    }
    case["parity"] = {
        "status": "passed" if parity_passed else "failed",
        "passed": parity_passed,
        "details": parity_details,
    }
    if comparison_timing.median_ms > 0.0:
        case["speedup"] = reference_timing.median_ms / comparison_timing.median_ms
    return case


def measure_operation(
    operation: Callable[[], Any],
    *,
    warmups: int,
    runs: int,
) -> tuple[TimingSummary, Any]:
    """Warm an operation, then return wall-clock samples and its last output."""

    if warmups < 0:
        raise ValueError("warmups must not be negative")
    if runs < 1:
        raise ValueError("runs must be greater than 0")

    output: Any = None
    for _ in range(warmups):
        output = operation()

    samples_ms: list[float] = []
    for _ in range(runs):
        start_time = time.perf_counter_ns()
        output = operation()
        elapsed_nanoseconds = time.perf_counter_ns() - start_time
        samples_ms.append(elapsed_nanoseconds / 1_000_000)

    sorted_samples = sorted(samples_ms)
    middle = len(sorted_samples) // 2
    if len(sorted_samples) % 2:
        median_ms = sorted_samples[middle]
    else:
        median_ms = (sorted_samples[middle - 1] + sorted_samples[middle]) / 2
    p95_index = max(0, math.ceil(0.95 * len(sorted_samples)) - 1)
    summary = TimingSummary(
        warmups=warmups,
        runs=runs,
        median_ms=median_ms,
        p95_ms=sorted_samples[p95_index],
        min_ms=sorted_samples[0],
        samples_ms=tuple(samples_ms),
    )
    return summary, output


def _semantic_parity(reference: Any, comparison: Any) -> tuple[bool, str]:
    reference_indices, reference_scores = _ranked_arrays(reference)
    comparison_indices, comparison_scores = _ranked_arrays(comparison)
    if not np.array_equal(reference_indices, comparison_indices):
        return False, "Ranked indices differ"
    if not np.allclose(
        reference_scores,
        comparison_scores,
        rtol=1e-5,
        atol=1e-5,
        equal_nan=False,
    ):
        maximum_error = float(
            np.max(np.abs(reference_scores - comparison_scores), initial=0.0)
        )
        return False, f"Semantic scores differ; maximum error is {maximum_error:.8g}"
    return True, "Indices match and float32 scores are within 1e-5"


def _numeric_array_parity(reference: Any, comparison: Any) -> tuple[bool, str]:
    reference_array = np.asarray(reference)
    comparison_array = np.asarray(comparison)
    if reference_array.shape != comparison_array.shape:
        return False, "Output shapes differ"
    if not np.allclose(
        reference_array,
        comparison_array,
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ):
        maximum_error = float(
            np.max(np.abs(reference_array - comparison_array), initial=0.0)
        )
        return False, f"Fuzzy scores differ; maximum error is {maximum_error:.8g}"
    return True, "Scores match within 1e-12"


def _integer_array_parity(reference: Any, comparison: Any) -> tuple[bool, str]:
    reference_array = np.asarray(reference, dtype=np.int64)
    comparison_array = np.asarray(comparison, dtype=np.int64)
    if not np.array_equal(reference_array, comparison_array):
        return False, "Exact-match indices differ"
    return True, "Exact-match indices match"


def _ranked_arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(result, "indices") and hasattr(result, "scores"):
        indices = result.indices
        scores = result.scores
    elif isinstance(result, tuple) and len(result) == 2:
        indices, scores = result
    else:
        raise TypeError("semantic_top_k must return indices and scores")
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
    )


def _summarize_cases(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    compared_cases = [case for case in cases if case["comparison"] is not None]
    passed_cases = [case for case in compared_cases if case["parity"]["passed"]]
    speedups = [
        float(case["speedup"])
        for case in passed_cases
        if case["speedup"] is not None
    ]
    return {
        "case_count": len(cases),
        "compared_case_count": len(compared_cases),
        "parity_passed_count": len(passed_cases),
        "all_compared_cases_passed": (
            len(compared_cases) > 0 and len(passed_cases) == len(compared_cases)
        ),
        "median_speedup": float(np.median(speedups)) if speedups else None,
    }


def _throughput(
    timing: TimingSummary,
    candidate_count: int,
) -> dict[str, float | None]:
    if timing.median_ms <= 0.0:
        return {"calls_per_second": None, "candidates_per_second": None}
    calls_per_second = 1_000.0 / timing.median_ms
    return {
        "calls_per_second": calls_per_second,
        "candidates_per_second": calls_per_second * candidate_count,
    }


def _comparison_table(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten case results into side-by-side machine-readable rows."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        comparison = case["comparison"]
        comparison_timing = comparison.get("timing") if comparison else None
        comparison_throughput = comparison.get("throughput") if comparison else None
        rows.append(
            {
                "operation": case["operation"],
                "candidate_count": case["candidate_count"],
                "reference_backend": case["reference"]["backend"],
                "reference_median_ms": case["reference"]["timing"]["median_ms"],
                "reference_p95_ms": case["reference"]["timing"]["p95_ms"],
                "reference_candidates_per_second": case["reference"][
                    "throughput"
                ]["candidates_per_second"],
                "comparison_backend": comparison.get("backend") if comparison else None,
                "comparison_median_ms": (
                    comparison_timing["median_ms"] if comparison_timing else None
                ),
                "comparison_p95_ms": (
                    comparison_timing["p95_ms"] if comparison_timing else None
                ),
                "comparison_candidates_per_second": (
                    comparison_throughput["candidates_per_second"]
                    if comparison_throughput
                    else None
                ),
                "speedup": case["speedup"],
                "parity": case["parity"]["status"],
            }
        )
    return rows


def format_markdown_table(report: dict[str, Any]) -> str:
    """Render the report's comparison rows as a compact Markdown table."""

    lines = [
        "| Operation | Candidates | Python median | Python p95 | "
        "Comparison median | Comparison p95 | Speedup | Parity |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["comparison_table"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["operation"]),
                    f"{row['candidate_count']:,}",
                    _format_milliseconds(row["reference_median_ms"]),
                    _format_milliseconds(row["reference_p95_ms"]),
                    _format_milliseconds(row["comparison_median_ms"]),
                    _format_milliseconds(row["comparison_p95_ms"]),
                    _format_speedup(row["speedup"]),
                    str(row["parity"]),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _format_milliseconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f} ms"


def _format_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _candidate_text_bytes(query: str, candidates: CandidateNames) -> int:
    return len(query.encode("utf-8")) + sum(
        len(short_name.encode("utf-8")) + len(qualified_name.encode("utf-8"))
        for short_name, qualified_name in candidates
    )


def _environment_metadata(
    reference_backend: AccelerationBackend,
    comparison_backend: AccelerationBackend | None,
) -> dict[str, Any]:
    metadata = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "reference_backend": _backend_metadata(reference_backend),
        "comparison_backend": (
            _backend_metadata(comparison_backend)
            if comparison_backend is not None
            else None
        ),
        "mojo_version": _mojo_version(),
    }
    metadata.update(_git_metadata())
    return metadata


def _backend_metadata(backend: AccelerationBackend) -> dict[str, Any]:
    backend_type = type(backend)
    metadata: dict[str, Any] = {
        "name": _backend_name(backend, backend_type.__name__),
        "type": f"{backend_type.__module__}.{backend_type.__qualname__}",
    }
    for attribute_name in ("abi_version", "library_path"):
        attribute_value = getattr(backend, attribute_name, None)
        if attribute_value is not None:
            metadata[attribute_name] = str(attribute_value)
    return metadata


def _git_metadata() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_branch": None, "git_dirty": None}
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        completed_process = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    output = completed_process.stdout.strip() or completed_process.stderr.strip()
    return output or None


def _mojo_version() -> str | None:
    environment_mojo = Path(sys.executable).with_name("mojo")
    if environment_mojo.is_file():
        version = _command_version(str(environment_mojo))
        if version is not None:
            return version
    return _command_version("mojo")


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    else:
        return int(page_size * page_count)

    if platform.system() != "Darwin":
        return None
    try:
        completed_process = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(completed_process.stdout.strip())
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def _backend_name(backend: AccelerationBackend, fallback: str) -> str:
    name = getattr(backend, "name", fallback)
    return name if isinstance(name, str) and name else fallback


def _config_dict(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "semantic_sizes": list(config.semantic_sizes),
        "semantic_dimension": config.semantic_dimension,
        "fuzzy_sizes": list(config.fuzzy_sizes),
        "exact_sizes": list(config.exact_sizes),
        "top_k": config.top_k,
        "minimum_fuzzy_score": config.minimum_fuzzy_score,
        "warmups": config.warmups,
        "runs": config.runs,
        "seed": config.seed,
        "operations": list(config.operations),
    }


def _load_python_backend() -> AccelerationBackend:
    from app.acceleration.python_backend import PythonBackend

    return PythonBackend()


def _load_mojo_backend(
    library_path: Path | None,
) -> tuple[AccelerationBackend | None, str | None]:
    try:
        from app.acceleration.mojo_backend import MojoBackend

        backend = MojoBackend(library_path=library_path)
    except Exception as error:
        reason = f"Mojo unavailable: {type(error).__name__}: {error}"
        return None, reason
    return backend, None


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must contain positive integers")
    return sizes


def _parse_operations(value: str) -> tuple[str, ...]:
    operations = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed_operations = {"semantic", "fuzzy", "exact"}
    if not operations or not set(operations) <= allowed_operations:
        raise argparse.ArgumentTypeError(
            "operations must contain semantic, fuzzy, or exact"
        )
    return operations


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark FireLens Python and Mojo compute backends",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--semantic-sizes", type=_parse_sizes)
    parser.add_argument("--semantic-dimension", type=int)
    parser.add_argument("--fuzzy-sizes", type=_parse_sizes)
    parser.add_argument("--exact-sizes", type=_parse_sizes)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--operations", type=_parse_operations)
    parser.add_argument(
        "--comparison-backend",
        choices=("auto", "mojo", "none"),
        default="auto",
    )
    parser.add_argument("--mojo-library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table-output", type=Path)
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> BenchmarkConfig:
    config = (
        BenchmarkConfig.full()
        if arguments.profile == "full"
        else BenchmarkConfig()
    )
    overrides = {
        "semantic_sizes": arguments.semantic_sizes,
        "semantic_dimension": arguments.semantic_dimension,
        "fuzzy_sizes": arguments.fuzzy_sizes,
        "exact_sizes": arguments.exact_sizes,
        "top_k": arguments.top_k,
        "warmups": arguments.warmups,
        "runs": arguments.runs,
        "seed": arguments.seed,
        "operations": arguments.operations,
    }
    return replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run selected cases, write JSON, and fail on comparison errors."""

    parsed_arguments = _argument_parser().parse_args(arguments)
    config = _config_from_arguments(parsed_arguments)
    comparison_backend = None
    comparison_skip_reason = "Comparison disabled by --comparison-backend=none"
    if parsed_arguments.comparison_backend != "none":
        comparison_backend, comparison_skip_reason = _load_mojo_backend(
            parsed_arguments.mojo_library
        )
        if (
            parsed_arguments.comparison_backend == "mojo"
            and comparison_backend is None
        ):
            print(comparison_skip_reason, file=sys.stderr)
            return 2

    report = run_benchmarks(
        config=config,
        comparison_backend=comparison_backend,
        comparison_skip_reason=comparison_skip_reason,
    )
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if parsed_arguments.output is None:
        sys.stdout.write(json_text)
    else:
        parsed_arguments.output.parent.mkdir(parents=True, exist_ok=True)
        parsed_arguments.output.write_text(json_text, encoding="utf-8")
        print(f"Wrote benchmark report to {parsed_arguments.output}")
    if parsed_arguments.table_output is not None:
        parsed_arguments.table_output.parent.mkdir(parents=True, exist_ok=True)
        parsed_arguments.table_output.write_text(
            format_markdown_table(report),
            encoding="utf-8",
        )
        print(f"Wrote comparison table to {parsed_arguments.table_output}")

    compared_case_count = report["summary"]["compared_case_count"]
    if compared_case_count and not report["summary"]["all_compared_cases_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

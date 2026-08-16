"""Deterministic performance benchmarks for FireLens acceleration backends."""

from benchmarks.datasets import (
    ExactDataset,
    FuzzyDataset,
    SemanticDataset,
    make_exact_dataset,
    make_fuzzy_dataset,
    make_semantic_dataset,
)
from benchmarks.runner import BenchmarkConfig, format_markdown_table, run_benchmarks

__all__ = [
    "BenchmarkConfig",
    "ExactDataset",
    "FuzzyDataset",
    "SemanticDataset",
    "format_markdown_table",
    "make_exact_dataset",
    "make_fuzzy_dataset",
    "make_semantic_dataset",
    "run_benchmarks",
]

#!/usr/bin/env python3
"""
Analyze connected component sizes for late-time N=256 matrix files.

This script searches recursively under results/ for matrix CSV files and
restricts analysis to the late-time N=256 family:
- T in {480, 560}
- alpha in {1.74, 1.80}

For each selected matrix:
- load safely from CSV
- symmetrize numerically
- zero the diagonal
- build an unweighted support graph at each cutoff in {0.0, 1e-12}
- compute connected-component statistics

Outputs:
- results/n256_late_time_component_sizes_report.csv
- results/n256_late_time_component_sizes_summary.csv
- results/n256_late_time_component_sizes_skipped.csv   (only if needed)
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


RESULTS_DIR = Path("results")

TARGET_N = 256
TARGET_T_VALUES = {480, 560}
TARGET_ALPHA_VALUES = {1.74, 1.80}
CUTOFFS = [0.0, 1e-12]

FULL_REPORT_PATH = RESULTS_DIR / "n256_late_time_component_sizes_report.csv"
SUMMARY_REPORT_PATH = RESULTS_DIR / "n256_late_time_component_sizes_summary.csv"
SKIPPED_REPORT_PATH = RESULTS_DIR / "n256_late_time_component_sizes_skipped.csv"

MATRIX_FILE_PATTERN = re.compile(
    r"^matrix_N(?P<n>\d+)_a(?P<alpha>\d+(?:\.\d+)?)_T(?P<T>\d+)_seed(?P<seed>\d+)\.csv$"
)


def parse_matrix_metadata(path: Path) -> Optional[Dict[str, object]]:
    match = MATRIX_FILE_PATTERN.match(path.name)
    if match is None:
        return None

    n = int(match.group("n"))
    alpha = float(match.group("alpha"))
    T = int(match.group("T"))
    seed = int(match.group("seed"))

    return {
        "file_path": path.as_posix(),
        "file_name": path.name,
        "n_from_name": n,
        "alpha": alpha,
        "T": T,
        "seed": seed,
    }


def is_target_late_time_case(meta: Dict[str, object]) -> bool:
    if int(meta["n_from_name"]) != TARGET_N:
        return False

    if int(meta["T"]) not in TARGET_T_VALUES:
        return False

    alpha_value = round(float(meta["alpha"]), 2)
    return alpha_value in TARGET_ALPHA_VALUES


def discover_target_files(results_dir: Path) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []

    for path in results_dir.rglob("matrix_*.csv"):
        meta = parse_matrix_metadata(path)
        if meta is None:
            continue
        if is_target_late_time_case(meta):
            matches.append(meta)

    matches.sort(
        key=lambda x: (
            int(x["T"]),
            float(x["alpha"]),
            int(x["seed"]),
            str(x["file_name"]),
        )
    )
    return matches


def load_matrix_safely(path: Path) -> np.ndarray:
    last_error: Optional[Exception] = None

    loaders = [
        lambda p: np.loadtxt(p, delimiter=",", dtype=float),
        lambda p: np.genfromtxt(p, delimiter=",", dtype=float),
    ]

    for loader in loaders:
        try:
            matrix = loader(path)
            matrix = np.asarray(matrix, dtype=float)
            return matrix
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load matrix CSV: {last_error}")


def validate_and_prepare_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got ndim={matrix.ndim}")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape={matrix.shape}")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Matrix contains non-finite values")

    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)

    return matrix


def connected_component_sizes(adjacency: np.ndarray) -> List[int]:
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    sizes: List[int] = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        size = 0

        while stack:
            node = stack.pop()
            size += 1

            unvisited_neighbors = np.flatnonzero(adjacency[node] & (~visited))
            if unvisited_neighbors.size > 0:
                visited[unvisited_neighbors] = True
                stack.extend(unvisited_neighbors.tolist())

        sizes.append(size)

    sizes.sort(reverse=True)
    return sizes


def compute_component_stats(matrix: np.ndarray, cutoff: float) -> Dict[str, object]:
    adjacency = matrix > cutoff
    np.fill_diagonal(adjacency, False)

    sizes = connected_component_sizes(adjacency)
    size_counter = Counter(sizes)

    return {
        "cutoff": cutoff,
        "n": int(matrix.shape[0]),
        "n_components": len(sizes),
        "largest_component_size": sizes[0] if len(sizes) >= 1 else 0,
        "second_largest_component_size": sizes[1] if len(sizes) >= 2 else 0,
        "n_singletons": int(size_counter.get(1, 0)),
        "n_dimers": int(size_counter.get(2, 0)),
        "n_trimers": int(size_counter.get(3, 0)),
        "component_sizes_sorted_desc": json.dumps(sizes),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(full_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []

    for cutoff in CUTOFFS:
        rows = [row for row in full_rows if float(row["cutoff"]) == cutoff]
        if not rows:
            continue

        n_components_values = np.array([int(row["n_components"]) for row in rows], dtype=float)
        largest_values = np.array([int(row["largest_component_size"]) for row in rows], dtype=float)
        second_values = np.array([int(row["second_largest_component_size"]) for row in rows], dtype=float)

        summary_rows.append(
            {
                "cutoff": cutoff,
                "n_rows": len(rows),
                "n_unique_files": len({str(row["file_name"]) for row in rows}),
                "median_n_components": float(np.median(n_components_values)),
                "min_n_components": int(np.min(n_components_values)),
                "max_n_components": int(np.max(n_components_values)),
                "median_largest_component_size": float(np.median(largest_values)),
                "min_largest_component_size": int(np.min(largest_values)),
                "max_largest_component_size": int(np.max(largest_values)),
                "median_second_largest_component_size": float(np.median(second_values)),
                "total_singletons": int(sum(int(row["n_singletons"]) for row in rows)),
                "total_dimers": int(sum(int(row["n_dimers"]) for row in rows)),
                "total_trimers": int(sum(int(row["n_trimers"]) for row in rows)),
                "files_with_singletons": int(sum(int(row["n_singletons"]) > 0 for row in rows)),
                "files_with_dimers": int(sum(int(row["n_dimers"]) > 0 for row in rows)),
                "files_with_trimers": int(sum(int(row["n_trimers"]) > 0 for row in rows)),
            }
        )

    return summary_rows


def print_case_header(total_files: int) -> None:
    print("=" * 80)
    print("Late-time N=256 component-size analysis")
    print(f"Results directory : {RESULTS_DIR.as_posix()}")
    print(f"Target T values   : {sorted(TARGET_T_VALUES)}")
    print(f"Target alpha      : {sorted(TARGET_ALPHA_VALUES)}")
    print(f"Cutoffs           : {CUTOFFS}")
    print(f"Matched files     : {total_files}")
    print("=" * 80)


def print_per_file_stats(file_name: str, cutoff_stats: List[Dict[str, object]]) -> None:
    print(f"[OK] {file_name}")
    for stats in cutoff_stats:
        sizes = json.loads(str(stats["component_sizes_sorted_desc"]))
        preview = sizes[:10]
        suffix = " ..." if len(sizes) > 10 else ""
        print(
            "  "
            f"cutoff={stats['cutoff']:<8g} "
            f"n_components={stats['n_components']:<4d} "
            f"largest={stats['largest_component_size']:<4d} "
            f"second={stats['second_largest_component_size']:<4d} "
            f"singletons={stats['n_singletons']:<4d} "
            f"dimers={stats['n_dimers']:<4d} "
            f"trimers={stats['n_trimers']:<4d} "
            f"top_sizes={preview}{suffix}"
        )


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR.as_posix()}")

    target_files = discover_target_files(RESULTS_DIR)
    print_case_header(len(target_files))

    if not target_files:
        print("No matching late-time N=256 matrix files were found.")
        return

    full_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    for meta in target_files:
        path = Path(str(meta["file_path"]))
        file_cutoff_rows: List[Dict[str, object]] = []

        try:
            matrix = load_matrix_safely(path)
            matrix = validate_and_prepare_matrix(matrix)

            if matrix.shape[0] != TARGET_N:
                raise ValueError(
                    f"Matrix size does not match expected N={TARGET_N}; got shape={matrix.shape}"
                )

            for cutoff in CUTOFFS:
                stats = compute_component_stats(matrix, cutoff)
                row = {
                    "file_path": str(meta["file_path"]),
                    "file_name": str(meta["file_name"]),
                    "alpha": float(meta["alpha"]),
                    "T": int(meta["T"]),
                    "seed": int(meta["seed"]),
                    "cutoff": float(stats["cutoff"]),
                    "n": int(stats["n"]),
                    "n_components": int(stats["n_components"]),
                    "largest_component_size": int(stats["largest_component_size"]),
                    "second_largest_component_size": int(stats["second_largest_component_size"]),
                    "n_singletons": int(stats["n_singletons"]),
                    "n_dimers": int(stats["n_dimers"]),
                    "n_trimers": int(stats["n_trimers"]),
                    "component_sizes_sorted_desc": str(stats["component_sizes_sorted_desc"]),
                }
                full_rows.append(row)
                file_cutoff_rows.append(row)

            print_per_file_stats(str(meta["file_name"]), file_cutoff_rows)

        except Exception as exc:
            skipped_rows.append(
                {
                    "file_path": str(meta["file_path"]),
                    "file_name": str(meta["file_name"]),
                    "alpha": float(meta["alpha"]),
                    "T": int(meta["T"]),
                    "seed": int(meta["seed"]),
                    "reason": str(exc),
                }
            )
            print(f"[SKIP] {meta['file_name']} :: {exc}")

    full_fieldnames = [
        "file_path",
        "file_name",
        "alpha",
        "T",
        "seed",
        "cutoff",
        "n",
        "n_components",
        "largest_component_size",
        "second_largest_component_size",
        "n_singletons",
        "n_dimers",
        "n_trimers",
        "component_sizes_sorted_desc",
    ]
    write_csv(FULL_REPORT_PATH, full_rows, full_fieldnames)

    summary_rows = build_summary_rows(full_rows)
    summary_fieldnames = [
        "cutoff",
        "n_rows",
        "n_unique_files",
        "median_n_components",
        "min_n_components",
        "max_n_components",
        "median_largest_component_size",
        "min_largest_component_size",
        "max_largest_component_size",
        "median_second_largest_component_size",
        "total_singletons",
        "total_dimers",
        "total_trimers",
        "files_with_singletons",
        "files_with_dimers",
        "files_with_trimers",
    ]
    write_csv(SUMMARY_REPORT_PATH, summary_rows, summary_fieldnames)

    if skipped_rows:
        skipped_fieldnames = [
            "file_path",
            "file_name",
            "alpha",
            "T",
            "seed",
            "reason",
        ]
        write_csv(SKIPPED_REPORT_PATH, skipped_rows, skipped_fieldnames)

    print("-" * 80)
    print(f"Full report written    : {FULL_REPORT_PATH.as_posix()}")
    print(f"Summary report written : {SUMMARY_REPORT_PATH.as_posix()}")
    if skipped_rows:
        print(f"Skipped report written : {SKIPPED_REPORT_PATH.as_posix()}")
    else:
        print("Skipped report written : none")
    print(f"Analyzed rows          : {len(full_rows)}")
    print(f"Skipped files          : {len(skipped_rows)}")
    print("-" * 80)


if __name__ == "__main__":
    main()

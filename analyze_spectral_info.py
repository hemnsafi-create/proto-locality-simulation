from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


DEFAULT_ZERO_EPS = 1e-10
SKIP_NAME_TOKENS = (
    "manifest",
    "table",
    "summary",
    "spectral_info_report",
    "spectral_info_skipped",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Laplacian spectral information for square matrix CSV files "
            "found recursively under a results directory."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results",
        help="Root directory to search recursively for CSV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for the spectral report. Default: <root>/spectral_info_report.csv",
    )
    parser.add_argument(
        "--skipped-output",
        type=str,
        default=None,
        help="Output CSV path for skipped files. Default: <root>/spectral_info_skipped.csv",
    )
    parser.add_argument(
        "--zero-eps",
        type=float,
        default=DEFAULT_ZERO_EPS,
        help="Threshold for treating eigenvalues as numerically zero.",
    )
    return parser.parse_args()


def should_skip_by_name(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(token in lower_name for token in SKIP_NAME_TOKENS)


def find_candidate_csv_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*.csv")
        if path.is_file() and not should_skip_by_name(path)
    )


def read_raw_csv_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned = [cell.strip() for cell in row]
            if any(cell != "" for cell in cleaned):
                rows.append(cleaned)
    return rows


def rows_to_numeric_array(rows: List[List[str]]) -> np.ndarray:
    if not rows:
        raise ValueError("CSV is empty")

    row_lengths = {len(row) for row in rows}
    if len(row_lengths) != 1:
        raise ValueError("CSV has irregular row lengths")

    try:
        array = np.array(rows, dtype=float)
    except ValueError as exc:
        raise ValueError(f"CSV is not fully numeric: {exc}") from exc

    if array.ndim != 2:
        raise ValueError("CSV did not produce a 2D array")

    if not np.all(np.isfinite(array)):
        raise ValueError("CSV contains non-finite values")

    return array


def try_extract_square_numeric_matrix(path: Path) -> Tuple[np.ndarray, str]:
    raw_rows = read_raw_csv_rows(path)

    candidates: List[Tuple[int, int, np.ndarray, str]] = []

    for drop_first_row in (0, 1):
        for drop_first_col in (0, 1):
            candidate_rows = raw_rows[drop_first_row:]
            if not candidate_rows:
                continue

            candidate_rows = [row[drop_first_col:] for row in candidate_rows]
            if not candidate_rows:
                continue
            if any(len(row) == 0 for row in candidate_rows):
                continue

            try:
                candidate = rows_to_numeric_array(candidate_rows)
            except ValueError:
                continue

            n_rows, n_cols = candidate.shape
            if n_rows != n_cols:
                continue
            if n_rows < 2:
                continue

            trim_score = drop_first_row + drop_first_col
            trim_label = f"drop_first_row={drop_first_row},drop_first_col={drop_first_col}"
            candidates.append((n_rows, -trim_score, candidate, trim_label))

    if not candidates:
        raise ValueError("Could not extract a finite square numeric matrix")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_matrix = candidates[0][2]
    best_trim_label = candidates[0][3]
    return best_matrix.astype(float, copy=False), best_trim_label


def symmetrize_and_zero_diagonal(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def compute_weighted_combinatorial_laplacian(W: np.ndarray) -> np.ndarray:
    degrees = W.sum(axis=1)
    return np.diag(degrees) - W


def compute_spectral_info(W: np.ndarray, zero_eps: float) -> dict:
    n = W.shape[0]
    L = compute_weighted_combinatorial_laplacian(W)
    eigvals = np.linalg.eigvalsh(L)

    lambda2 = float(eigvals[1]) if n >= 2 else np.nan
    lambda3 = float(eigvals[2]) if n >= 3 else np.nan

    zero_mask = np.abs(eigvals) < zero_eps
    n_zero_eps = int(np.count_nonzero(zero_mask))

    positive_indices = np.where(eigvals > zero_eps)[0]
    if positive_indices.size > 0:
        first_pos_idx_1based = int(positive_indices[0] + 1)
        lambda_first_pos = float(eigvals[positive_indices[0]])
    else:
        first_pos_idx_1based = np.nan
        lambda_first_pos = np.nan

    return {
        "n": int(n),
        "lambda2": lambda2,
        "lambda3": lambda3,
        "n_zero_eps": n_zero_eps,
        "lambda_first_pos_idx": first_pos_idx_1based,
        "lambda_first_pos": lambda_first_pos,
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    output_path = Path(args.output) if args.output else root / "spectral_info_report.csv"
    skipped_output_path = (
        Path(args.skipped_output) if args.skipped_output else root / "spectral_info_skipped.csv"
    )

    csv_files = find_candidate_csv_files(root)

    if not csv_files:
        print(f"No candidate CSV files found under: {root}")
        return

    rows: List[dict] = []
    skipped_rows: List[dict] = []

    print(f"Searching under: {root}")
    print(f"Found {len(csv_files)} candidate CSV files")
    print()

    for path in csv_files:
        try:
            W_raw, trim_label = try_extract_square_numeric_matrix(path)
            W = symmetrize_and_zero_diagonal(W_raw)
            spec = compute_spectral_info(W, zero_eps=args.zero_eps)

            row = {
                "file_path": str(path),
                "file_name": path.name,
                "trim_strategy": trim_label,
                **spec,
            }
            rows.append(row)

            print(
                f"[OK] {path} | "
                f"n={row['n']} | "
                f"lambda2={row['lambda2']:.12e} | "
                f"lambda3={row['lambda3']:.12e} | "
                f"n_zero_eps={row['n_zero_eps']} | "
                f"lambda_first_pos_idx={row['lambda_first_pos_idx']} | "
                f"lambda_first_pos={row['lambda_first_pos']:.12e}"
            )

        except Exception as exc:
            skipped_rows.append(
                {
                    "file_path": str(path),
                    "file_name": path.name,
                    "reason": str(exc),
                }
            )
            print(f"[SKIP] {path} | reason={exc}")

    print()
    print(f"Processed matrix files: {len(rows)}")
    print(f"Skipped files: {len(skipped_rows)}")

    if rows:
        rows.sort(key=lambda row: row["file_path"])
        write_csv(
            output_path,
            rows,
            [
                "file_path",
                "file_name",
                "trim_strategy",
                "n",
                "lambda2",
                "lambda3",
                "n_zero_eps",
                "lambda_first_pos_idx",
                "lambda_first_pos",
            ],
        )
        print(f"Saved spectral report: {output_path}")

    if skipped_rows:
        skipped_rows.sort(key=lambda row: row["file_path"])
        write_csv(
            skipped_output_path,
            skipped_rows,
            ["file_path", "file_name", "reason"],
        )
        print(f"Saved skipped-file report: {skipped_output_path}")


if __name__ == "__main__":
    main()

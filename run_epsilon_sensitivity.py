
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

from proto_locality_core import SimConfig, run_simulation, save_summary, save_final_matrix


OUTPUT_ROOT = Path("results") / "epsilon_sensitivity"

BASE_CFG = {
    "alpha": 1.80,
    "beta": 1.00,
    "gamma": 1.00,
    "dt": 0.02,
    "sample_every": 25,
    "target_initial_load": 1.0,
    "noise_fraction": 0.02,
    "device": "cuda",
    "dtype": "float64",
    "model_variant": "full",
}

EPS_LIST = [1e-6, 1e-12, 1e-18]

CASES = [
    {"case_tag": "n64_a180_seed101_t200", "N": 64, "seed": 101, "T": 200.0},
    {"case_tag": "n64_a180_seed123_t200", "N": 64, "seed": 123, "T": 200.0},
    {"case_tag": "n256_a180_seed101_t400", "N": 256, "seed": 101, "T": 400.0},
]

BASELINE_EPS = 1e-12


def make_cfg(case: Dict, eps: float) -> SimConfig:
    cfg_dict = dict(BASE_CFG)
    cfg_dict.update(
        {
            "N": case["N"],
            "seed": case["seed"],
            "T": case["T"],
            "eps": eps,
        }
    )
    return SimConfig(**cfg_dict)


def eps_tag(eps: float) -> str:
    return f"{eps:.0e}".replace("+", "")


def case_output_dir(case_tag: str, eps: float) -> Path:
    return OUTPUT_ROOT / case_tag / f"eps_{eps_tag(eps)}"
def trajectory_to_map(summary: Dict) -> Dict[float, Dict]:
    data = {}
    for row in summary["trajectory"]:
        t_key = round(float(row["t"]), 12)
        data[t_key] = row
    return data


def max_abs_diff_against_baseline(
    baseline_summary: Dict,
    test_summary: Dict,
    key: str,
) -> float:
    baseline_map = trajectory_to_map(baseline_summary)
    test_map = trajectory_to_map(test_summary)

    baseline_times = set(baseline_map.keys())
    test_times = set(test_map.keys())
    if baseline_times != test_times:
        missing_in_test = sorted(baseline_times - test_times)
        missing_in_baseline = sorted(test_times - baseline_times)
        raise ValueError(
            f"Trajectory time grids differ for key={key}. "
            f"Missing in test: {missing_in_test[:5]}; "
            f"Missing in baseline: {missing_in_baseline[:5]}"
        )

    max_diff = 0.0
    for t in sorted(baseline_times):
        diff = abs(float(test_map[t][key]) - float(baseline_map[t][key]))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def final_value(summary: Dict, key: str) -> float:
    return float(summary["trajectory"][-1][key])


def write_trajectory_rows(
    rows: List[Dict],
    case: Dict,
    eps: float,
    summary: Dict,
) -> None:
    for row in summary["trajectory"]:
        rows.append(
            {
                "case_tag": case["case_tag"],
                "N": case["N"],
                "alpha": BASE_CFG["alpha"],
                "seed": case["seed"],
                "T": case["T"],
                "eps": eps,
                "t": float(row["t"]),
                "chi_hhi_star": float(row["chi_hhi_star"]),
                "lambda2": float(row["lambda2"]),
                "total_weight": float(row["total_weight"]),
                "load_cv": float(row["load_cv"]),
            }
        )
def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_paths_are_new() -> Tuple[Path, Path, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_ROOT / "epsilon_sensitivity_manifest.csv"
    summary_path = OUTPUT_ROOT / "epsilon_sensitivity_summary.csv"
    trajectories_path = OUTPUT_ROOT / "epsilon_sensitivity_trajectories.csv"

    for path in [manifest_path, summary_path, trajectories_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    return manifest_path, summary_path, trajectories_path


def main() -> None:
    manifest_path, summary_path, trajectories_path = ensure_output_paths_are_new()

    manifest_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    trajectory_rows: List[Dict] = []

    for case in CASES:
        case_summaries: Dict[float, Dict] = {}

        for eps in EPS_LIST:
            outdir = case_output_dir(case["case_tag"], eps)
            if outdir.exists():
                raise FileExistsError(f"Refusing to overwrite existing directory: {outdir}")

            cfg = make_cfg(case, eps)

            print("=" * 100)
            print(f"Running case={case['case_tag']} eps={eps_tag(eps)}")
            print(cfg)

            summary = run_simulation(cfg)
            summary_json_path = save_summary(summary, outdir=outdir)
            final_matrix_path = save_final_matrix(summary, outdir=outdir)

            case_summaries[eps] = summary
            write_trajectory_rows(trajectory_rows, case, eps, summary)

            manifest_rows.append(
                {
                    "case_tag": case["case_tag"],
                    "N": case["N"],
                    "alpha": BASE_CFG["alpha"],
                    "seed": case["seed"],
                    "T": case["T"],
                    "eps": eps,
                    "summary_json_path": str(summary_json_path),
                    "final_matrix_path": str(final_matrix_path),
                    "runtime_seconds": float(summary["runtime_seconds"]),
                }
            )

        baseline_summary = case_summaries[BASELINE_EPS]

        for eps in EPS_LIST:
            summary = case_summaries[eps]

            max_abs_diff_chi = max_abs_diff_against_baseline(
                baseline_summary, summary, "chi_hhi_star"
            )
            max_abs_diff_lambda2 = max_abs_diff_against_baseline(
                baseline_summary, summary, "lambda2"
            )
            max_abs_diff_total_weight = max_abs_diff_against_baseline(
                baseline_summary, summary, "total_weight"
            )

            final_chi = final_value(summary, "chi_hhi_star")
            final_lambda2 = final_value(summary, "lambda2")
            final_total_weight = final_value(summary, "total_weight")

            baseline_final_chi = final_value(baseline_summary, "chi_hhi_star")
            baseline_final_lambda2 = final_value(baseline_summary, "lambda2")
            baseline_final_total_weight = final_value(baseline_summary, "total_weight")

            summary_rows.append(
                {
                    "case_tag": case["case_tag"],
                    "N": case["N"],
                    "alpha": BASE_CFG["alpha"],
                    "seed": case["seed"],
                    "T": case["T"],
                    "eps": eps,
                    "baseline_eps": BASELINE_EPS,
                    "final_chi_hhi_star": final_chi,
                    "final_lambda2": final_lambda2,
                    "final_total_weight": final_total_weight,
                    "max_abs_diff_chi_hhi_star_vs_baseline": max_abs_diff_chi,
                    "max_abs_diff_lambda2_vs_baseline": max_abs_diff_lambda2,
                    "max_abs_diff_total_weight_vs_baseline": max_abs_diff_total_weight,
                    "abs_diff_final_chi_hhi_star_vs_baseline": abs(final_chi - baseline_final_chi),
                    "abs_diff_final_lambda2_vs_baseline": abs(final_lambda2 - baseline_final_lambda2),
                    "abs_diff_final_total_weight_vs_baseline": abs(
                        final_total_weight - baseline_final_total_weight
                    ),
                    "runtime_seconds": float(summary["runtime_seconds"]),
                }
            )

    write_csv(
        manifest_path,
        [
            "case_tag",
            "N",
            "alpha",
            "seed",
            "T",
            "eps",
            "summary_json_path",
            "final_matrix_path",
            "runtime_seconds",
        ],
        manifest_rows,
    )

    write_csv(
        summary_path,
        [
            "case_tag",
            "N",
            "alpha",
            "seed",
            "T",
            "eps",
            "baseline_eps",
            "final_chi_hhi_star",
            "final_lambda2",
            "final_total_weight",
            "max_abs_diff_chi_hhi_star_vs_baseline",
            "max_abs_diff_lambda2_vs_baseline",
            "max_abs_diff_total_weight_vs_baseline",
            "abs_diff_final_chi_hhi_star_vs_baseline",
            "abs_diff_final_lambda2_vs_baseline",
            "abs_diff_final_total_weight_vs_baseline",
            "runtime_seconds",
        ],
        summary_rows,
    )

    write_csv(
        trajectories_path,
        [
            "case_tag",
            "N",
            "alpha",
            "seed",
            "T",
            "eps",
            "t",
            "chi_hhi_star",
            "lambda2",
            "total_weight",
            "load_cv",
        ],
        trajectory_rows,
    )

    print("=" * 100)
    print("Done.")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Trajectories saved to: {trajectories_path}")


if __name__ == "__main__":
    main()        
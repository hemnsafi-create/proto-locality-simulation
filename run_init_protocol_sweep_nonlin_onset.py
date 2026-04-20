from __future__ import annotations

from pathlib import Path
import csv
import argparse
from proto_locality_core import SimConfig, run_simulation


PYTHON_EXE = r"C:\Users\hemns\miniforge3\envs\torchgpu\python.exe"
CAMPAIGN_ROOT = Path("results/init_protocol_sweep_nonlin_onset")
RAW_DIR = CAMPAIGN_ROOT / "raw"
PLOTS_DIR = CAMPAIGN_ROOT / "plots"
LOGS_DIR = CAMPAIGN_ROOT / "logs"
MANIFEST_PATH = CAMPAIGN_ROOT / "manifest.csv"
MASTER_SUMMARY_PATH = CAMPAIGN_ROOT / "master_summary.csv"

ALPHAS = [1.80, 1.74]
NS = [64, 128, 256]
SEEDS = [101, 102, 103, 104, 105]

ETA_ABS_VALUES = [1e-4, 3e-4, 1e-3, 3e-3]
ETA_REL_C_VALUES = [0.05, 0.10, 0.20, 0.50, 1.00]

DT = 0.02
T_MAX = 600.0
EPSILON = 1e-12
BETA = 1.0
GAMMA = 1.0

PROTOCOLS = ["absolute", "relative"]
def format_alpha_label(alpha: float) -> str:
    return f"a{alpha:.2f}".replace(".", "p")


def format_eta_rel_label(c_value: float) -> str:
    return f"etarel_c{c_value:.2f}".replace(".", "p")


def format_eta_abs_label(eta_abs: float) -> str:
    return f"etaabs_{eta_abs:.0e}".replace("-", "m")


def build_run_stem(protocol: str, alpha: float, n: int, seed: int, amplitude_label: str) -> str:
    alpha_label = format_alpha_label(alpha)
    return f"{protocol}_{alpha_label}_N{n}_seed{seed}_{amplitude_label}"


def build_raw_csv_path(protocol: str, alpha: float, n: int, seed: int, amplitude_label: str) -> Path:
    stem = build_run_stem(protocol, alpha, n, seed, amplitude_label)
    return RAW_DIR / f"{stem}.csv"


def build_log_path(protocol: str, alpha: float, n: int, seed: int, amplitude_label: str) -> Path:
    stem = build_run_stem(protocol, alpha, n, seed, amplitude_label)
    return LOGS_DIR / f"{stem}.log"


def compute_w_star(alpha: float, n: int) -> float:
    return (alpha - BETA) / ((n - 1) * GAMMA)
def get_amplitude_label(protocol: str, amplitude_value: float) -> str:
    if protocol == "absolute":
        return format_eta_abs_label(amplitude_value)
    if protocol == "relative":
        return format_eta_rel_label(amplitude_value)
    raise ValueError(f"Unknown protocol: {protocol}")


def get_protocol_amplitude_values(protocol: str) -> list[float]:
    if protocol == "absolute":
        return ETA_ABS_VALUES
    if protocol == "relative":
        return ETA_REL_C_VALUES
    raise ValueError(f"Unknown protocol: {protocol}")
def get_effective_eta(protocol: str, alpha: float, n: int, amplitude_value: float) -> float:
    if protocol == "absolute":
        return amplitude_value
    if protocol == "relative":
        return amplitude_value * compute_w_star(alpha, n)
    raise ValueError(f"Unknown protocol: {protocol}")
def build_sim_config_for_case(case) -> SimConfig:
    protocol = str(case["protocol"]).strip()
    alpha = float(case["alpha"])
    n = int(case["N"])
    seed = int(case["seed"])
    amplitude_value = float(case["amplitude_value"])

    w_star = compute_w_star(alpha, n)
    if w_star <= 0.0:
        raise ValueError(
            f"Non-positive w_star for alpha={alpha}, N={n}: {w_star}"
        )

    if protocol == "absolute":
        noise_fraction = amplitude_value / w_star
    elif protocol == "relative":
        noise_fraction = amplitude_value
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    target_initial_load = (n - 1) * w_star

    return SimConfig(
        N=n,
        alpha=alpha,
        beta=BETA,
        gamma=GAMMA,
        dt=DT,
        T=T_MAX,
        sample_every=25,
        seed=seed,
        target_initial_load=target_initial_load,
        noise_fraction=noise_fraction,
        eps=EPSILON,
        device="cuda",
        dtype="float64",
        model_variant="full",
    )
def describe_run_case(protocol: str, alpha: float, n: int, seed: int, amplitude_value: float) -> dict[str, object]:
    amplitude_label = get_amplitude_label(protocol, amplitude_value)
    effective_eta = get_effective_eta(protocol, alpha, n, amplitude_value)
    return {
        "protocol": protocol,
        "alpha": alpha,
        "N": n,
        "seed": seed,
        "amplitude_value": amplitude_value,
        "amplitude_label": amplitude_label,
        "effective_eta": effective_eta,
        "run_stem": build_run_stem(protocol, alpha, n, seed, amplitude_label),
    }
def build_case_list() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for alpha in ALPHAS:
            for n in NS:
                for seed in SEEDS:
                    for amplitude_value in get_protocol_amplitude_values(protocol):
                        cases.append(
                            describe_run_case(protocol, alpha, n, seed, amplitude_value)
                        )
    return cases
def get_manifest_fieldnames() -> list[str]:
    return [
        "protocol",
        "alpha",
        "N",
        "seed",
        "amplitude_value",
        "amplitude_label",
        "effective_eta",
        "run_stem",
    ]
def get_master_summary_fieldnames() -> list[str]:
    return [
        "protocol",
        "alpha",
        "N",
        "seed",
        "amplitude_value",
        "amplitude_label",
        "effective_eta",
        "run_stem",
        "raw_csv_path",
        "log_path",
        "reached_0p1",
        "reached_0p4",
        "t_onset_0p1",
        "t_onset_0p4",
        "chi_hhi_final",
        "lambda2_final",
        "w_total_final",
    ]
def initialize_raw_csv(case: dict[str, object]) -> None:
    raw_csv_path = build_raw_csv_path(
        case["protocol"],
        case["alpha"],
        case["N"],
        case["seed"],
        case["amplitude_label"],
    )
    
    if raw_csv_path.exists():
        raise FileExistsError(f"Raw CSV already exists: {raw_csv_path}")

    fieldnames = get_raw_timeseries_fieldnames()
    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def initialize_all_raw_csvs(cases: list[dict[str, object]]) -> None:
    for case in cases:
        initialize_raw_csv(case)
def get_raw_timeseries_fieldnames() -> list[str]:
    return [
        "t",
        "chi_hhi",
        "lambda2",
        "w_total",
    ]

def initialize_log_file(case: dict[str, object]) -> None:
    log_path = build_log_path(
        case["protocol"],
        case["alpha"],
        case["N"],
        case["seed"],
        case["amplitude_label"],
    )
    if log_path.exists():
        raise FileExistsError(f"Log file already exists: {log_path}")

    log_path.write_text("", encoding="utf-8")


def initialize_all_log_files(cases: list[dict[str, object]]) -> None:
    for case in cases:
        initialize_log_file(case)
def write_manifest(cases: list[dict[str, object]]) -> None:
    fieldnames = get_manifest_fieldnames()
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({name: case[name] for name in fieldnames})
            


def build_master_summary_row(case: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": case["protocol"],
        "alpha": case["alpha"],
        "N": case["N"],
        "seed": case["seed"],
        "amplitude_value": case["amplitude_value"],
        "amplitude_label": case["amplitude_label"],
        "effective_eta": case["effective_eta"],
        "run_stem": case["run_stem"],
        "raw_csv_path": str(build_raw_csv_path(case["protocol"], case["alpha"], case["N"], case["seed"], case["amplitude_label"])),
        "log_path": str(build_log_path(case["protocol"], case["alpha"], case["N"], case["seed"], case["amplitude_label"])),
        "reached_0p1": "",
        "reached_0p4": "",
        "t_onset_0p1": "",
        "t_onset_0p4": "",
        "chi_hhi_final": "",
        "lambda2_final": "",
        "w_total_final": "",
    }
def write_master_summary(cases: list[dict[str, object]]) -> None:
    fieldnames = get_master_summary_fieldnames()
    with MASTER_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(build_master_summary_row(case))


def main() -> None:
    cases = build_case_list()
    assert len(cases) == len({case["run_stem"] for case in cases}), "Duplicate run_stem detected"
    print(f"Initialization / protocol sweep scaffold ready. Cases={len(cases)} UniqueStems={len({case['run_stem'] for case in cases})}")
    write_manifest(cases)
    write_master_summary(cases)
    initialize_all_raw_csvs(cases)
    initialize_all_log_files(cases)





def select_single_case_from_manifest(manifest_path, case_index=0):
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    if case_index < 0 or case_index >= len(rows):
        raise IndexError(
            f"case_index={case_index} is out of range for manifest with {len(rows)} rows"
        )

    row = rows[case_index]

    required_keys = [
        "protocol",
        "alpha",
        "N",
        "seed",
        "amplitude_value",
        "amplitude_label",
        "effective_eta",
        "run_stem",
    ]

    missing = [k for k in required_keys if k not in row or not str(row[k]).strip()]
    if missing:
        raise KeyError(
            f"Manifest row {case_index} is missing required fields: {missing}"
        )

    run_stem = row["run_stem"].strip()
    row["raw_csv_path"] = str(RAW_DIR / f"{run_stem}.csv")
    row["log_path"] = str(LOGS_DIR / f"{run_stem}.log")

    return row
def write_raw_csv_from_summary(summary, raw_csv_path) -> None:
    raw_csv_path = Path(raw_csv_path)
    trajectory = summary["trajectory"]

    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "chi_hhi", "lambda2", "w_total"])

        for row in trajectory:
            writer.writerow([
                row["t"],
                row["chi_hhi_star"],
                row["lambda2"],
                row["total_weight"],
            ])

def execute_single_case(case):
    protocol = str(case["protocol"]).strip()
    alpha = float(case["alpha"])
    N = int(case["N"])
    seed = int(case["seed"])
    amplitude_value = float(case["amplitude_value"])
    amplitude_label = str(case["amplitude_label"]).strip()
    effective_eta = float(case["effective_eta"])
    run_stem = str(case["run_stem"]).strip()

    raw_csv_path = Path(case["raw_csv_path"])
    log_path = Path(case["log_path"])

    if protocol not in {"absolute", "relative"}:
        raise ValueError(f"Unsupported protocol: {protocol}")

    if N <= 0:
        raise ValueError(f"Invalid N: {N}")

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    expected_header = "t,chi_hhi,lambda2,w_total"
    with raw_csv_path.open("r", encoding="utf-8", newline="") as f:
        header_line = f.readline().strip()

    if header_line != expected_header:
        raise RuntimeError(
            f"Unexpected raw CSV header in {raw_csv_path}: {header_line!r}"
        )

    lines = [
        "SINGLE_CASE_EXECUTION_STUB_START",
        f"protocol={protocol}",
        f"alpha={alpha}",
        f"N={N}",
        f"seed={seed}",
        f"amplitude_value={amplitude_value}",
        f"amplitude_label={amplitude_label}",
        f"effective_eta={effective_eta}",
        f"run_stem={run_stem}",
        f"raw_csv_path={raw_csv_path}",
        f"log_path={log_path}",
        f"raw_csv_header={header_line}",
        "SINGLE_CASE_EXECUTION_STUB_END",
        "",
    ]

    with log_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print("Single-case execution stub completed.")
    print(f"protocol: {protocol}")
    print(f"alpha: {alpha}")
    print(f"N: {N}")
    print(f"seed: {seed}")
    print(f"amplitude_value: {amplitude_value}")
    print(f"amplitude_label: {amplitude_label}")
    print(f"effective_eta: {effective_eta}")
    print(f"run_stem: {run_stem}")
    print(f"raw_csv_path: {raw_csv_path}")
    print(f"log_path: {log_path}")
    print(f"raw_csv_header: {header_line}")

def execute_single_case(case):
    protocol = str(case["protocol"]).strip()
    alpha = float(case["alpha"])
    N = int(case["N"])
    seed = int(case["seed"])
    amplitude_value = float(case["amplitude_value"])
    amplitude_label = str(case["amplitude_label"]).strip()
    effective_eta = float(case["effective_eta"])
    run_stem = str(case["run_stem"]).strip()

    raw_csv_path = Path(case["raw_csv_path"])
    log_path = Path(case["log_path"])

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    cfg = build_sim_config_for_case(case)
    summary = run_simulation(cfg)

    write_raw_csv_from_summary(summary, raw_csv_path)

    final_chi_hhi_star = summary["final_chi_hhi_star"]
    final_lambda2 = summary["final_lambda2"]
    final_total_weight = summary["final_total_weight"]
    final_load_cv = summary["final_load_cv"]
    runtime_seconds = summary["runtime_seconds"]
    n_samples = len(summary["trajectory"])

    lines = [
        "SINGLE_CASE_RUN_START",
        f"protocol={protocol}",
        f"alpha={alpha}",
        f"N={N}",
        f"seed={seed}",
        f"amplitude_value={amplitude_value}",
        f"amplitude_label={amplitude_label}",
        f"effective_eta={effective_eta}",
        f"run_stem={run_stem}",
        f"raw_csv_path={raw_csv_path}",
        f"log_path={log_path}",
        f"target_initial_load={cfg.target_initial_load}",
        f"noise_fraction={cfg.noise_fraction}",
        f"runtime_seconds={runtime_seconds}",
        f"n_samples={n_samples}",
        f"final_chi_hhi_star={final_chi_hhi_star}",
        f"final_lambda2={final_lambda2}",
        f"final_total_weight={final_total_weight}",
        f"final_load_cv={final_load_cv}",
        "SINGLE_CASE_RUN_END",
        "",
    ]

    with log_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print("Single-case execution completed.")
    print(f"protocol: {protocol}")
    print(f"alpha: {alpha}")
    print(f"N: {N}")
    print(f"seed: {seed}")
    print(f"amplitude_value: {amplitude_value}")
    print(f"amplitude_label: {amplitude_label}")
    print(f"effective_eta: {effective_eta}")
    print(f"run_stem: {run_stem}")
    print(f"target_initial_load: {cfg.target_initial_load}")
    print(f"noise_fraction: {cfg.noise_fraction}")
    print(f"runtime_seconds: {runtime_seconds}")
    print(f"n_samples: {n_samples}")
    print(f"final_chi_hhi_star: {final_chi_hhi_star}")
    print(f"final_lambda2: {final_lambda2}")
    print(f"final_total_weight: {final_total_weight}")
    print(f"final_load_cv: {final_load_cv}")
    print(f"raw_csv_path: {raw_csv_path}")
    print(f"log_path: {log_path}")
def get_pilot_batch_case_indices():
     return [0, 20, 40, 60, 80, 100, 120, 145, 170, 195, 220, 245]
def execute_pilot_batch(manifest_path):
    case_indices = get_pilot_batch_case_indices()
    print(f"Running pilot batch with {len(case_indices)} cases: {case_indices}")

    for batch_pos, case_index in enumerate(case_indices, start=1):
        print("=" * 80)
        print(f"Pilot batch item {batch_pos}/{len(case_indices)}")
        print(f"Selecting manifest row: {case_index}")

        case = select_single_case_from_manifest(manifest_path, case_index=case_index)

        print("Selected pilot case:")
        for key, value in case.items():
            print(f"{key}: {value}")

        execute_single_case(case)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single init-protocol sweep case by manifest index."
    )

    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Zero-based row index in manifest.csv",
    )
    parser.add_argument(
        "--pilot-batch",
        action="store_true",
        help="Run a small representative pilot batch across protocol/alpha/N groups",
    )

    return parser.parse_args()

        
if __name__ == "__main__":
    manifest_path = Path("results/init_protocol_sweep_nonlin_onset/manifest.csv")
    args = parse_args()

    if args.pilot_batch:
        execute_pilot_batch(manifest_path)
    else:
        case_index = args.case_index
        case = select_single_case_from_manifest(manifest_path, case_index=case_index)

        print("Selected single case:")
        for key, value in case.items():
            print(f"{key}: {value}")

        execute_single_case(case)

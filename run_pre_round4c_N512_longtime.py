
from pathlib import Path
import csv

from proto_locality_core import SimConfig, run_simulation, save_summary, save_final_matrix

BASE_CFG = {
    "N": 64,
    "alpha": 1.8,
    "beta": 1.0,
    "gamma": 1.0,
    "dt": 0.02,
    "T": 2000.0,
    "sample_every": 100,
    "seed": 123,
    "target_initial_load": 1.0,
    "noise_fraction": 0.02,
    "eps": 1e-12,
    "device": "cuda",
    "dtype": "float64",
    "model_variant": "full",
}

CASES = [
    

       # N=512: all five seeds
    {"N": 512, "seed": 101},
    {"N": 512, "seed": 102},
    {"N": 512, "seed": 103},
    {"N": 512, "seed": 104},
    {"N": 512, "seed": 105},
]     


def make_cfg(**updates):
    cfg = BASE_CFG.copy()
    cfg.update(updates)
    return cfg


def save_trajectory_csv(summary: dict, outdir: Path, N: int, seed: int) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"trajectory_N{N}_seed{seed}.csv"
    traj = summary["trajectory"]
    if not traj:
        raise ValueError("Trajectory is empty.")
    fieldnames = list(traj[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(traj)
    return path


rows = []

for i, case in enumerate(CASES, start=1):
    cfg_dict = make_cfg(N=case["N"], seed=case["seed"])
    label = f"N{cfg_dict['N']}_a1.8_T2000_seed{cfg_dict['seed']}"
    outdir = Path("results") / "pre_round4c_N512_longtime"

    print("=" * 90)
    print(f"[{i}/{len(CASES)}] Running: {label}")
    print(cfg_dict)

    cfg = SimConfig(**cfg_dict)
    summary = run_simulation(cfg)

    summary_path = save_summary(summary, outdir=outdir)
    matrix_path = save_final_matrix(summary, outdir=outdir)
    traj_path = save_trajectory_csv(summary, outdir, cfg_dict["N"], cfg_dict["seed"])

    row = {
        "label": label,
        "N": cfg_dict["N"],
        "T": cfg_dict["T"],
        "seed": cfg_dict["seed"],
        "chi_hhi_star": summary["final_chi_hhi_star"],
        "lambda2": summary["final_lambda2"],
        "total_weight": summary["final_total_weight"],
        "runtime_seconds": summary["runtime_seconds"],
        "summary_path": str(summary_path),
        "matrix_path": str(matrix_path),
        "trajectory_path": str(traj_path),
    }
    rows.append(row)

    print(f"Final chi*_HHI : {row['chi_hhi_star']}")
    print(f"Final lambda2  : {row['lambda2']}")
    print(f"Final total w  : {row['total_weight']}")
    print(f"Saved summary  : {summary_path}")
    print(f"Saved matrix   : {matrix_path}")
    print(f"Saved traj csv : {traj_path}")

manifest_path = Path("results") / "pre_round4c_N512_longtime_manifest.csv"
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "N",
            "T",
            "seed",
            "chi_hhi_star",
            "lambda2",
            "total_weight",
            "runtime_seconds",
            "summary_path",
            "matrix_path",
            "trajectory_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 90)
print("Done.")
print(f"Manifest saved to: {manifest_path}")
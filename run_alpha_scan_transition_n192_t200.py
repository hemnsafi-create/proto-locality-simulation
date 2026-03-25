
from pathlib import Path
import csv

from proto_locality_core import SimConfig, run_simulation, save_summary, save_final_matrix

BASE_CFG = {
    "N": 192,
    "alpha": 1.8,
    "beta": 1.0,
    "gamma": 1.0,
    "dt": 0.01,
    "T": 200.0,
    "sample_every": 10,
    "seed": 123,
    "target_initial_load": 1.0,
    "noise_fraction": 0.02,
    "eps": 1e-12,
    "device": "cuda",
    "dtype": "float64",
    "model_variant": "full",
}

ALPHA_LIST = [1.66, 1.68, 1.70, 1.74, 1.76, 1.80]
SEEDS = [101, 102, 103]
N_FIXED = 192
T_FIXED = 200.0


def make_cfg(**updates):
    cfg = BASE_CFG.copy()
    cfg.update(updates)
    return cfg


cases = []
for alpha in ALPHA_LIST:
    for seed in SEEDS:
        cases.append(
            {
                "label": f"N{N_FIXED}_a{alpha:.2f}_T{int(T_FIXED)}_seed{seed}",
                "outdir": Path("results") / "alpha_scan_transition_N192_T200",
                "cfg": make_cfg(
                    N=N_FIXED,
                    alpha=alpha,
                    T=T_FIXED,
                    seed=seed,
                    sample_every=10,
                ),
            }
        )

rows = []

for i, case in enumerate(cases, start=1):
    label = case["label"]
    outdir = case["outdir"]
    cfg_dict = case["cfg"]

    print("=" * 90)
    print(f"[{i}/{len(cases)}] Running: {label}")
    print(cfg_dict)

    cfg = SimConfig(**cfg_dict)
    summary = run_simulation(cfg)

    summary_path = save_summary(summary, outdir=outdir)
    matrix_path = save_final_matrix(summary, outdir=outdir)

    row = {
        "label": label,
        "N": cfg_dict["N"],
        "alpha": cfg_dict["alpha"],
        "beta": cfg_dict["beta"],
        "gamma": cfg_dict["gamma"],
        "T": cfg_dict["T"],
        "seed": cfg_dict["seed"],
        "chi_hhi_star": summary["final_chi_hhi_star"],
        "lambda2": summary["final_lambda2"],
        "total_weight": summary["final_total_weight"],
        "runtime_seconds": summary["runtime_seconds"],
        "summary_path": str(summary_path),
        "matrix_path": str(matrix_path),
    }
    rows.append(row)

    print(f"Final chi*_HHI : {row['chi_hhi_star']}")
    print(f"Final lambda2  : {row['lambda2']}")
    print(f"Final total w  : {row['total_weight']}")
    print(f"Saved summary  : {summary_path}")
    print(f"Saved matrix   : {matrix_path}")

manifest_path = Path("results") / "alpha_scan_transition_N192_T200_manifest.csv"
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "N",
            "alpha",
            "beta",
            "gamma",
            "T",
            "seed",
            "chi_hhi_star",
            "lambda2",
            "total_weight",
            "runtime_seconds",
            "summary_path",
            "matrix_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 90)
print("Done.")
print(f"Manifest saved to: {manifest_path}")
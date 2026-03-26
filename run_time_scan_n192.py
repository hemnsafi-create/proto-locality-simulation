
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

ALPHA_LIST = [1.74, 1.80]
T_LIST = [250.0, 300.0, 350.0, 400.0]
SEEDS = [101, 102, 103]
N_FIXED = 192


def make_cfg(**updates):
    cfg = BASE_CFG.copy()
    cfg.update(updates)
    return cfg


cases = []
for alpha in ALPHA_LIST:
    for T in T_LIST:
        for seed in SEEDS:
            cases.append(
                {
                    "label": f"N{N_FIXED}_a{alpha:.2f}_T{int(T)}_seed{seed}",
                    "outdir": Path("results") / "time_scan_N192",
                    "cfg": make_cfg(
                        N=N_FIXED,
                        alpha=alpha,
                        T=T,
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

    tmp_summary_path = Path(save_summary(summary, outdir=outdir))
    tmp_matrix_path = Path(save_final_matrix(summary, outdir=outdir))

    summary_path = outdir / f"summary_{label}{tmp_summary_path.suffix}"
    matrix_path = outdir / f"matrix_{label}{tmp_matrix_path.suffix}"

    tmp_summary_path.replace(summary_path)
    tmp_matrix_path.replace(matrix_path)

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

manifest_path = Path("results") / "time_scan_N192_manifest.csv"
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
table_path = Path("results") / "time_scan_N192_table.csv"
with open(table_path, "w", newline="", encoding="utf-8") as f:
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
def mean_std(vals):
    n = len(vals)
    mean_val = sum(vals) / n
    if n > 1:
        var = sum((x - mean_val) ** 2 for x in vals) / (n - 1)
        std_val = var ** 0.5
    else:
        std_val = 0.0
    return mean_val, std_val


summary_groups = {}
for row in rows:
    key = (row["alpha"], row["T"])
    if key not in summary_groups:
        summary_groups[key] = {
            "chi": [],
            "lam": [],
            "rt": [],
        }
    summary_groups[key]["chi"].append(row["chi_hhi_star"])
    summary_groups[key]["lam"].append(row["lambda2"])
    summary_groups[key]["rt"].append(row["runtime_seconds"])


summary_path = Path("results") / "time_scan_N192_alpha_time_summary.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "alpha",
            "T",
            "n_runs",
            "chi_hhi_star_mean",
            "chi_hhi_star_std",
            "lambda2_mean",
            "lambda2_std",
            "runtime_seconds_mean",
            "runtime_seconds_std",
        ],
    )
    writer.writeheader()

    for alpha, T in sorted(summary_groups):
        chi_mean, chi_std = mean_std(summary_groups[(alpha, T)]["chi"])
        lam_mean, lam_std = mean_std(summary_groups[(alpha, T)]["lam"])
        rt_mean, rt_std = mean_std(summary_groups[(alpha, T)]["rt"])

        writer.writerow(
            {
                "alpha": alpha,
                "T": T,
                "n_runs": len(summary_groups[(alpha, T)]["chi"]),
                "chi_hhi_star_mean": chi_mean,
                "chi_hhi_star_std": chi_std,
                "lambda2_mean": lam_mean,
                "lambda2_std": lam_std,
                "runtime_seconds_mean": rt_mean,
                "runtime_seconds_std": rt_std,
            }
        )    
print("=" * 90)
print("Done.")
print(f"Manifest saved to: {manifest_path}")
print(f"Table saved to: {table_path}")
print(f"Summary saved to: {summary_path}")
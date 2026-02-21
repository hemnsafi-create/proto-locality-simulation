import numpy as np
import json
from pathlib import Path
from simulation import run_simulation
from order_parameters import compute_all

def run_scaling_sweep(N_list=None, n_seeds=5, alpha=2.0,
                      beta=0.5, gamma=0.1, t_end=25.0,
                      n_steps=300, output_dir="results"):
    if N_list is None:
        N_list = [40, 100, 200, 400, 600, 1000]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}
    for N in N_list:
        print(f"Running N={N}...")
        seed_results = []
        for seed in range(n_seeds):
            sim = run_simulation(N=N, alpha=alpha, beta=beta,
                                 gamma=gamma, t_end=t_end,
                                 n_steps=n_steps, seed=seed)
            W_final = sim["W_series"][-1]
            ops = compute_all(W_final, seed=seed)
            seed_results.append({
                "N": N, "seed": seed,
                "chi_hhi":  float(ops["chi_hhi"]),
                "Q_star":   float(ops["Q_star"]),
                "lambda2":  float(ops["lambda2"]),
            })
            print(f"  seed={seed}: chi_hhi={ops['chi_hhi']:.3f}")
        all_results[str(N)] = seed_results
    with open(f"{output_dir}/scaling_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results

def summarise(results):
    print(f"\n{'N':>6} {'chi_hhi':>10} {'lambda2':>10} {'Q*':>8}")
    print("-" * 40)
    for N_str, seeds in sorted(results.items(), key=lambda x: int(x[0])):
        chi = np.mean([r["chi_hhi"] for r in seeds])
        lam = np.mean([r["lambda2"] for r in seeds])
        q   = np.mean([r["Q_star"]  for r in seeds])
        print(f"{int(N_str):>6} {chi:>10.4f} {lam:>10.5f} {q:>8.4f}")

if __name__ == "__main__":
    results = run_scaling_sweep(N_list=[40, 100, 200],
                                n_seeds=3, t_end=20.0, n_steps=200)
    summarise(results)

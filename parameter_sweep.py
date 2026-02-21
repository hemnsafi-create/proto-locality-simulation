import numpy as np
import json
from pathlib import Path
import itertools
from simulation import run_simulation
from order_parameters import compute_all

def run_parameter_sweep(N=100, alpha_vals=None, beta_vals=None,
                        gamma_vals=None, seeds=None,
                        t_end=20.0, n_steps=200, output_dir="results"):
    if alpha_vals is None: alpha_vals = [1.5, 2.0, 3.0]
    if beta_vals  is None: beta_vals  = [0.3, 0.5, 0.8]
    if gamma_vals is None: gamma_vals = [0.05, 0.1, 0.2]
    if seeds      is None: seeds      = [0, 1, 2]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for alpha, beta, gamma, seed in itertools.product(
            alpha_vals, beta_vals, gamma_vals, seeds):
        S0 = (N-2)/(N-1)
        if alpha * S0 <= beta:
            continue
        print(f"alpha={alpha}, beta={beta}, gamma={gamma}, seed={seed}")
        sim = run_simulation(N=N, alpha=alpha, beta=beta, gamma=gamma,
                             t_end=t_end, n_steps=n_steps, seed=seed)
        W = sim["W_series"][-1]
        ops = compute_all(W, seed=seed)
        results.append({
            "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
            "chi_hhi": float(ops["chi_hhi"]),
            "Q_star":  float(ops["Q_star"]),
            "lambda2": float(ops["lambda2"]),
        })
    with open(f"{output_dir}/parameter_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    results = run_parameter_sweep(
        N=80, alpha_vals=[1.5, 2.0], beta_vals=[0.3, 0.5],
        gamma_vals=[0.1, 0.2], seeds=[0, 1], t_end=20.0, n_steps=150)
    print(f"\nTotal runs: {len(results)}")
    chi_vals = [r["chi_hhi"] for r in results]
    print(f"chi_hhi range: {min(chi_vals):.3f} - {max(chi_vals):.3f}")
    print(f"All > 1: {all(c > 1 for c in chi_vals)}")

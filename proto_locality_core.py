
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import torch


@dataclass
class SimConfig:
    N: int = 64
    alpha: float = 1.10
    beta: float = 1.00
    gamma: float = 1.00
    dt: float = 0.02
    T: float = 40.0
    sample_every: int = 25
    seed: int = 123
    target_initial_load: float = 1.0
    noise_fraction: float = 0.02
    eps: float = 1e-12
    device: str = "cuda"
    dtype: str = "float64"
    model_variant: str = "full"  # full | no_similarity | no_load


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_constraints(W: torch.Tensor) -> torch.Tensor:
    W = 0.5 * (W + W.T)
    W = torch.clamp(W, min=0.0)
    W.fill_diagonal_(0.0)
    return W


def make_initial_W(cfg: SimConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    N = cfg.N
    c = cfg.target_initial_load / (N - 1)
    sigma = cfg.noise_fraction * c

    Xi = torch.randn((N, N), device=device, dtype=dtype)
    Xi = 0.5 * (Xi + Xi.T)
    Xi.fill_diagonal_(0.0)

    W = c * (torch.ones((N, N), device=device, dtype=dtype) - torch.eye(N, device=device, dtype=dtype))
    W = W + sigma * Xi
    W = project_constraints(W)
    return W


def compute_loads(W: torch.Tensor) -> torch.Tensor:
    return W.sum(dim=1)


def compute_similarity(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = (W * W).sum(dim=1)
    numer = W @ W
    wij2 = W * W

    denom_left = q[:, None] - wij2
    denom_right = q[None, :] - wij2
    denom_left = torch.clamp(denom_left, min=0.0)
    denom_right = torch.clamp(denom_right, min=0.0)
    denom = torch.sqrt(denom_left * denom_right) + eps

    S = numer / denom
    S = torch.clamp(S, min=0.0, max=1.0)
    S.fill_diagonal_(0.0)
    return S


def rhs(W: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    B = compute_loads(W)

    if cfg.model_variant == "no_similarity":
        S = torch.ones_like(W)
        S.fill_diagonal_(0.0)
    else:
        S = compute_similarity(W, eps=cfg.eps)

    gamma = 0.0 if cfg.model_variant == "no_load" else cfg.gamma
    load_term = 0.5 * gamma * (B[:, None] + B[None, :])

    dW = (cfg.alpha * S - cfg.beta - load_term) * W
    dW.fill_diagonal_(0.0)
    return dW


def rk4_step(W: torch.Tensor, dt: float, cfg: SimConfig) -> torch.Tensor:
    k1 = rhs(W, cfg)
    k2 = rhs(project_constraints(W + 0.5 * dt * k1), cfg)
    k3 = rhs(project_constraints(W + 0.5 * dt * k2), cfg)
    k4 = rhs(project_constraints(W + dt * k3), cfg)

    W_next = W + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    W_next = project_constraints(W_next)
    return W_next


def chi_hhi_star(W: torch.Tensor, eps: float = 1e-12) -> float:
    N = W.shape[0]
    B = compute_loads(W)
    P = W / (B[:, None] + eps)
    H = (P * P).sum(dim=1)
    baseline = 1.0 / (N - 1)
    chi = ((H - baseline) / (1.0 - baseline)).mean()
    return float(chi.item())


def load_cv(W: torch.Tensor, eps: float = 1e-12) -> float:
    B = compute_loads(W)
    mean_B = B.mean()
    std_B = B.std(unbiased=False)
    return float((std_B / (mean_B + eps)).item())


def total_weight(W: torch.Tensor) -> float:
    return float((0.5 * W.sum()).item())


def lambda2_combinatorial_laplacian(W: torch.Tensor) -> float:
    B = compute_loads(W)
    L = torch.diag(B) - W
    evals = torch.linalg.eigvalsh(L)
    evals = torch.sort(evals).values
    if len(evals) < 2:
        return 0.0
    return float(evals[1].item())


def constraint_report(W: torch.Tensor) -> Dict[str, float | bool]:
    sym_err = float((W - W.T).abs().max().item())
    diag_err = float(torch.diag(W).abs().max().item())
    min_entry = float(W.min().item())
    return {
        "symmetric": sym_err < 1e-10,
        "zero_diagonal": diag_err < 1e-12,
        "nonnegative": min_entry >= -1e-12,
        "symmetry_max_abs_error": sym_err,
        "diagonal_max_abs_error": diag_err,
        "minimum_entry": min_entry,
    }


def run_simulation(cfg: SimConfig) -> Dict:
    if cfg.N < 3:
        raise ValueError("N must be at least 3 for the similarity definition to make sense.")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if cfg.T <= 0.0:
        raise ValueError("T must be positive.")
    if cfg.model_variant not in {"full", "no_similarity", "no_load"}:
        raise ValueError("model_variant must be one of: full, no_similarity, no_load")

    device = get_device(cfg.device)
    dtype = DTYPE_MAP[cfg.dtype]
    set_seed(cfg.seed)

    W = make_initial_W(cfg, device=device, dtype=dtype)

    n_steps = int(round(cfg.T / cfg.dt))
    if n_steps <= 0:
        raise ValueError("Computed n_steps <= 0; check dt and T.")

    t0 = time.perf_counter()

    trajectory = []
    chi_max = -math.inf

    for step in range(n_steps + 1):
        t = step * cfg.dt

        if step % cfg.sample_every == 0 or step == n_steps:
            chi = chi_hhi_star(W, eps=cfg.eps)
            lam2 = lambda2_combinatorial_laplacian(W)
            tw = total_weight(W)
            cv = load_cv(W, eps=cfg.eps)
            chi_max = max(chi_max, chi)
            trajectory.append(
                {
                    "t": t,
                    "chi_hhi_star": chi,
                    "lambda2": lam2,
                    "total_weight": tw,
                    "load_cv": cv,
                }
            )

        if step < n_steps:
            W = rk4_step(W, cfg.dt, cfg)

    elapsed = time.perf_counter() - t0

    summary = {
        "config": asdict(cfg),
        "device_used": str(device),
        "dtype_used": str(dtype),
        "runtime_seconds": elapsed,
        "final_constraint_report": constraint_report(W),
        "final_chi_hhi_star": trajectory[-1]["chi_hhi_star"],
        "max_chi_hhi_star": chi_max,
        "final_lambda2": trajectory[-1]["lambda2"],
        "final_total_weight": trajectory[-1]["total_weight"],
        "final_load_cv": trajectory[-1]["load_cv"],
        "W_final": W.detach().cpu().numpy().tolist(),
        "trajectory": trajectory,
    }
    return summary


def save_summary(summary: Dict, outdir: str | Path = "results") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = summary["config"]
    filename = (
        f"run_{cfg['model_variant']}_N{cfg['N']}_"
        f"a{cfg['alpha']}_b{cfg['beta']}_g{cfg['gamma']}_seed{cfg['seed']}.json"
    )
    path = outdir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path

def save_final_matrix(summary: Dict, outdir: str | Path = "results") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = summary["config"]
    variant = cfg["model_variant"]
    N = cfg["N"]
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    gamma = cfg["gamma"]
    seed = cfg["seed"]

    filename = (
        f"W_{variant}_N{N}_a{alpha}_b{beta}_g{gamma}_seed{seed}.csv"
    )
    path = outdir / filename

    W_final = summary["W_final"]
    import numpy as np
    np.savetxt(path, np.array(W_final), delimiter=",")

    return path
if __name__ == "__main__":
    with open("default_v0.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg = SimConfig(**cfg_dict)
    summary = run_simulation(cfg)
    path = save_summary(summary, outdir="results")
    matrix_path = save_final_matrix(summary, outdir="results")

    print("Final constraints:", summary["final_constraint_report"])
    print("Final chi*_HHI   :", summary["final_chi_hhi_star"])
    print("Final lambda2    :", summary["final_lambda2"])
    print("Final total w    :", summary["final_total_weight"])
    print("Saved summary to :", path)
    print("Saved final W to :", matrix_path)
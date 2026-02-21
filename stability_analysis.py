import numpy as np
from dataclasses import dataclass

@dataclass
class StabilityResult:
    N: int
    alpha: float
    beta: float
    gamma: float
    S0: float
    w_star: float
    lambda_uniform: float
    lambda_node: float
    lambda_contrast: float
    is_positive_fixed_point: bool
    lambda_mf_initial: float

def analytic_stability(N, alpha, beta, gamma):
    S0 = (N - 2) / (N - 1)
    w_star = (alpha * S0 - beta) / (gamma * (N - 1))
    is_positive = (alpha * S0 > beta) and (w_star > 0)
    lam_uniform = -gamma * (N - 1) * w_star
    lam_node = (-alpha * S0 * N / ((N-1)*(N-2))
                - gamma * (N-2) * w_star / 2.0)
    lam_contrast = (-alpha * S0 * (2*N-1) / ((N-1)*(N-2))
                    + (alpha*S0 - beta) / (2.0*(N-1)))
    lambda_mf_initial = alpha * S0 - beta
    return StabilityResult(
        N=N, alpha=alpha, beta=beta, gamma=gamma,
        S0=S0, w_star=w_star,
        lambda_uniform=lam_uniform,
        lambda_node=lam_node,
        lambda_contrast=lam_contrast,
        is_positive_fixed_point=is_positive,
        lambda_mf_initial=lambda_mf_initial,
    )

if __name__ == "__main__":
    r = analytic_stability(200, 2.0, 0.5, 0.1)
    print(f"N={r.N}, w*={r.w_star:.4f}")
    print(f"lambda_uniform  = {r.lambda_uniform:.4f}")
    print(f"lambda_node     = {r.lambda_node:.4f}")
    print(f"lambda_contrast = {r.lambda_contrast:.4f}  ~ O(1/N)={1/r.N:.4f}")

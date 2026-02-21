import numpy as np
from scipy.integrate import solve_ivp

def cosine_similarity_matrix(W, eps=1e-10):
    norms = np.linalg.norm(W, axis=1)
    denom = np.outer(norms, norms) + eps
    S = (W @ W.T) / denom
    np.fill_diagonal(S, 0.0)
    return S

def interaction_loads(W):
    return W.sum(axis=1)

def dW_dt(t, w_flat, N, alpha, beta, gamma, eps=1e-10):
    W = np.zeros((N, N))
    idx = np.triu_indices(N, k=1)
    W[idx] = w_flat
    W = W + W.T
    W = np.maximum(W, 0.0)
    S = cosine_similarity_matrix(W, eps)
    B = interaction_loads(W)
    B_sum = B[:, None] + B[None, :]
    dW = alpha * W * S - beta * W - (gamma / 2.0) * W * B_sum
    dW = (dW + dW.T) / 2.0
    np.fill_diagonal(dW, 0.0)
    return dW[idx]

def run_simulation(N=100, alpha=2.0, beta=0.5, gamma=0.1,
                   t_end=20.0, n_steps=500, seed=42, w_init_scale=0.1):
    rng = np.random.default_rng(seed)
    idx = np.triu_indices(N, k=1)
    W0 = np.abs(rng.uniform(0, w_init_scale, size=len(idx[0]))) + 1e-4
    t_eval = np.linspace(0, t_end, n_steps)
    sol = solve_ivp(dW_dt, (0, t_end), W0,
                    args=(N, alpha, beta, gamma),
                    method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-8)
    W_series = np.zeros((n_steps, N, N))
    for k in range(n_steps):
        W = np.zeros((N, N))
        W[idx] = np.maximum(sol.y[:, k], 0.0)
        W_series[k] = W + W.T
    return {"t": sol.t, "W_series": W_series,
            "params": dict(N=N, alpha=alpha, beta=beta, gamma=gamma)}

if __name__ == "__main__":
    r = run_simulation(N=50, seed=0)
    print("Done. Final mean weight:",
          r["W_series"][-1][r["W_series"][-1]>0].mean().round(4))

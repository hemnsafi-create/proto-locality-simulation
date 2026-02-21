import numpy as np

def node_entropy(W):
    N = W.shape[0]
    B = W.sum(axis=1)
    H = np.zeros(N)
    for i in range(N):
        if B[i] < 1e-15:
            continue
        p = W[i] / B[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.where(p > 0, np.log(p), 0.0)
        H[i] = -np.sum(p * log_p)
    return H

def network_mean_entropy(W):
    return float(np.mean(node_entropy(W)))

def entropy_time_series(W_series):
    T, N, _ = W_series.shape
    H_mean = np.array([network_mean_entropy(W_series[t]) for t in range(T)])
    H_max = np.log(N - 1) if N > 1 else 1.0
    return {
        "H_mean":       H_mean,
        "H_normalised": H_mean / H_max,
        "H_max":        H_max,
        "dH_dt":        np.diff(H_mean),
    }

if __name__ == "__main__":
    from simulation import run_simulation
    sim = run_simulation(N=60, alpha=2.0, beta=0.5, gamma=0.1,
                         t_end=20.0, n_steps=200, seed=0)
    ets = entropy_time_series(sim["W_series"])
    print(f"H(t=0)   = {ets['H_mean'][0]:.4f}")
    print(f"H(t_end) = {ets['H_mean'][-1]:.4f}")
    print(f"Delta H  = {ets['H_mean'][-1]-ets['H_mean'][0]:.4f}  (should be < 0)")

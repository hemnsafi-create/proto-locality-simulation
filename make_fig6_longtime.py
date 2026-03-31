
# make_fig6_longtime.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(str(p) for p in paths))


def load_matrix_csv(path: Path) -> np.ndarray:
    # Try plain numeric CSV (no header)
    try:
        W = np.loadtxt(path, delimiter=",")
        if W.ndim == 2:
            return W
    except Exception:
        pass
    # Fallback: pandas (handles odd formatting)
    df = pd.read_csv(path, header=None)
    return df.to_numpy(dtype=float)


def main():
    repo = Path(".").resolve()

    # --- Inputs (based on your terminal output) ---
    traj_candidates = [
        repo / "results" / "pre_round3_targeted_dynamics" / "trajectory_seed123.csv",
    ]
    W_candidates = [
        repo / "results" / "pre_round2_seed202_T1000" / "w_full_N64_a1.8_b1.0_g1.0_seed202.csv",
        # (Optional fallbacks if you later want to swap sources)
        repo / "results" / "pre_round1_seed202_T600" / "w_full_N64_a1.8_b1.0_g1.0_seed202.csv",
        repo / "results" / "pre_round1_seed202_T400" / "w_full_N64_a1.8_b1.0_g1.0_seed202.csv",
    ]

    traj_path = first_existing(traj_candidates)
    W_path = first_existing(W_candidates)

    # --- Load trajectory ---
    traj = pd.read_csv(traj_path)
    # Try to infer column names robustly
    cols = {c.lower(): c for c in traj.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    t_col = pick("t", "time", "step")
    chi_col = pick("chi_hhi_star", "chi_hhi", "chi", "chi_hhi*")
    lam_col = pick("lambda2", "lambda_2", "lambda2_dag", "lambda2dag", "lambda2†")
    wt_col = pick("total_weight", "w_total", "wtotal", "w_tot", "w_total_sum")

    if t_col is None:
        # assume first column is time if not labeled
        t = np.arange(len(traj))
    else:
        t = traj[t_col].to_numpy()

    # If columns are missing, raise a friendly message
    missing = []
    if chi_col is None: missing.append("chi_hhi_star (or similar)")
    if lam_col is None: missing.append("lambda2 (or similar)")
    if wt_col is None: missing.append("total_weight (or similar)")
    if missing:
        raise ValueError(
            "Could not find required columns in trajectory file.\n"
            f"Missing: {missing}\n"
            f"Available columns: {list(traj.columns)}\n"
            "Fix: rename columns or adjust picks in make_fig6_longtime.py."
        )

    chi = traj[chi_col].to_numpy()
    lam = traj[lam_col].to_numpy()
    wtot = traj[wt_col].to_numpy()

    # --- Load final W and load-sort ---
    W = load_matrix_csv(W_path)

    # "load" definition: q_i = sum_j w_ij^2 (consistent with q_i in your similarity formula)
    q = np.sum(W * W, axis=1)
    order = np.argsort(-q)  # descending
    W_sorted = W[np.ix_(order, order)]

    # --- Plot layout: left (3 stacked), right (heatmap) ---
    fig = plt.figure(figsize=(12.5, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1.05, 1.0], wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    axH = fig.add_subplot(gs[:, 1])

    ax1.plot(t, chi)
    ax1.set_title("Seed 123: trajectory diagnostics", fontsize=11)
    ax1.set_ylabel(r"$\chi^*_{\mathrm{HHI}}$")

    ax2.plot(t, lam)
    ax2.set_ylabel(r"$\lambda_2$")

    ax3.plot(t, wtot)
    ax3.set_ylabel(r"$W_{\mathrm{total}}$")
    ax3.set_xlabel("t")

    im = axH.imshow(W_sorted, aspect="auto")
    axH.set_title("Seed 202: final W (load-sorted)", fontsize=11)
    axH.set_xlabel("node index (sorted by load)")
    axH.set_ylabel("node index (sorted by load)")
    cbar = fig.colorbar(im, ax=axH, fraction=0.046, pad=0.04)
    cbar.set_label(r"$w_{ij}$")

    # --- Save outputs ---
    out_dir = repo / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "Fig6_longer_time_diagnostics.png"
    pdf_path = out_dir / "Fig6_longer_time_diagnostics.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print("Saved:", png_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()
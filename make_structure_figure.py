
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

results_dir = Path("results")

files = {
    "Nearly homogeneous\nα = 1.20": results_dir / "W_full_N64_a1.2_b1.0_g1.0_seed123.csv",
    "Transition regime\nα = 1.69": results_dir / "W_full_N64_a1.69_b1.0_g1.0_seed123.csv",
    "Clustered regime\nα = 1.80": results_dir / "W_full_N64_a1.8_b1.0_g1.0_seed123.csv",
}

# تأكد أن الملفات موجودة
for label, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file for {label}: {path}")

# حمّل المصفوفات
mats = {label: np.loadtxt(path, delimiter=",") for label, path in files.items()}

# استخدم ترتيبًا طيفيًا ثابتًا مشتقًا من الحالة المتكتلة
W_ref = mats["Clustered regime\nα = 1.80"]
D = np.diag(W_ref.sum(axis=1))
L = D - W_ref

evals, evecs = np.linalg.eigh(L)
# المتجه الثاني (Fiedler-like) لترتيب العقد
fiedler = evecs[:, 1]
order = np.argsort(fiedler)

# أعد ترتيب جميع المصفوفات بنفس الترتيب
ordered = {}
for label, W in mats.items():
    W_ord = W[np.ix_(order, order)].copy()

    # تطبيع داخل كل panel لإظهار البنية بوضوح
    vmax = W_ord.max()
    if vmax > 0:
        W_disp = W_ord / vmax
    else:
        W_disp = W_ord

    # إخفاء القطر الرئيسي بصريًا حتى لا يشتت الانتباه
    np.fill_diagonal(W_disp, np.nan)

    ordered[label] = W_disp

# احفظ الترتيب نفسه للرجوع إليه لاحقًا
order_path = results_dir / "node_order_N64_seed123_from_alpha1.8.csv"
np.savetxt(order_path, order, fmt="%d", delimiter=",")

# ارسم الشكل
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)

im = None
for ax, (label, W_disp) in zip(axes, ordered.items()):
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        W_disp,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, shrink=0.85)
cbar.set_label("Normalized weight", rotation=90)

fig.suptitle(
    "Coordinate-free weighted network: structural progression (N = 64, seed = 123)\n"
    "Common node ordering derived from the clustered case (α = 1.80)",
    fontsize=11
)

out_png = results_dir / "structure_heatmaps_N64_seed123.png"
out_pdf = results_dir / "structure_heatmaps_N64_seed123.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print("Saved figure to:", out_png)
print("Saved figure to:", out_pdf)
print("Saved node order to:", order_path)
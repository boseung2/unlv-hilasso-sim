"""3방법(LASSO·ElasticNet·Random) F1·AUCPR 박스플롯 — Stochastic LASSO Fig 1 형식.

per-rep 값 수집:
  - LASSO·ElasticNet: R glmnet(solver) 로컬 실행, 검증셋 λ 선택 + F1·AUCPR.
  - Random: I·II 로컬 실행 / III·IV 는 클러스터 NPZ 의 beta_hats 재사용
            (AUCPR=|β̂| 순위 직접, F1=1-SE 임계 재적용).
결과: results/all_methods_perrep.csv + results/compare_fig.png
"""
import os, sys, warnings
os.environ.setdefault("R_HOME", os.path.join(sys.prefix, "lib", "R"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import generate_data as gd
import scoring
import random_lasso as rl

DSN = ["Dataset I", "Dataset II", "Dataset III", "Dataset IV"]
LABELS = {"Dataset I": "Dataset I\n(n:50, p:100)", "Dataset II": "Dataset II\n(n:100, p:1,000)",
          "Dataset III": "Dataset III\n(n:200, p:10,000)", "Dataset IV": "Dataset IV\n(n:400, p:10,000)"}
rows = []


def add(method, ds, rep, bh_for_f1, beta, bh_for_auc):
    rows.append(dict(method=method, dataset=ds, rep=rep,
                     f1=scoring.f1_selection(bh_for_f1, beta)["f1"],
                     aucpr=scoring.aucpr(bh_for_auc, beta)))


# ---- LASSO · ElasticNet (R glmnet) ----
import solver
for method, ratios in [("LASSO", [1.0]),
                       ("ElasticNet", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]:
    for ds in DSN:
        cfg = gd.DATASETS[ds]; beta = gd.build_beta(cfg)
        for rep in range(10):
            (Xtr, ytr), (Xval, yval), _ = gd.generate_split(cfg, beta, rep)
            best, _ = solver.select_model((Xtr, ytr), (Xval, yval), ratios)
            add(method, ds, rep, best["bhat"], beta, best["bhat"])
        print(f"{method} {ds} done", flush=True)

# ---- Random LASSO ----
npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results_random_lasso.npz"))
for ds in DSN:
    cfg = gd.DATASETS[ds]; beta = gd.build_beta(cfg)
    if ds in ("Dataset I", "Dataset II"):
        for rep in range(10):
            (Xtr, ytr), (Xval, yval), _ = gd.generate_split(cfg, beta, rep)
            r = rl.fit_random_lasso(Xtr, ytr, Xval, yval, L=30, random_state=rep, n_jobs=-1)
            add("Random", ds, rep, r["coef_"], beta, r["beta_hat_"])
    else:
        BH = npz[f"{ds}__beta_hats"]                       # (10, p) 클러스터
        for rep in range(len(BH)):
            bh = BH[rep]
            (Xtr, ytr), (Xval, yval), _ = gd.generate_split(cfg, beta, rep)
            t, _, _ = rl._select_threshold(bh, Xval, yval, n_se=1.0)
            add("Random", ds, rep, np.where(np.abs(bh) >= t, bh, 0.0), beta, bh)
    print(f"Random {ds} done", flush=True)

df = pd.DataFrame(rows)
outdir = os.path.dirname(os.path.abspath(__file__))
df.to_csv(os.path.join(outdir, "all_methods_perrep.csv"), index=False)

# ---- 박스플롯 (Fig 1 형식: A=F1, B=AUCPR) ----
METHODS = ["LASSO", "ElasticNet", "Random"]
COLORS = {"LASSO": "#a3c34c", "ElasticNet": "#5aa9e0", "Random": "#e8873a"}
fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
group_w = 0.8
for ax, metric, title in [(axes[0], "f1", "F1-score"), (axes[1], "aucpr", "AUCPR")]:
    for gi, ds in enumerate(DSN):
        for mi, m in enumerate(METHODS):
            vals = df[(df.method == m) & (df.dataset == ds)][metric].values
            pos = gi + (mi - 1) * (group_w / 3)
            bp = ax.boxplot([vals], positions=[pos], widths=group_w / 3 * 0.82,
                            patch_artist=True, medianprops=dict(color="black", lw=1),
                            flierprops=dict(marker="", ), showcaps=True)
            bp["boxes"][0].set(facecolor=COLORS[m], edgecolor="black", lw=0.8)
            ax.scatter(np.full_like(vals, pos) + np.random.uniform(-0.04, 0.04, len(vals)),
                       vals, s=9, color="black", alpha=0.55, zorder=3)
        if gi < len(DSN) - 1:
            ax.axvline(gi + 0.5, color="#bbbbbb", ls=":", lw=1)
    ax.set_ylabel(title, fontsize=13)
    ax.set_ylim(0, max(0.05, df[metric].max() * 1.12))
    ax.set_xlim(-0.6, len(DSN) - 0.4)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
axes[1].set_xticks(range(len(DSN)))
axes[1].set_xticklabels([LABELS[d] for d in DSN], fontsize=11)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[m], edgecolor="black", lw=0.8) for m in METHODS]
axes[1].legend(handles, ["LASSO", "Elastic-Net", "Random LASSO"], ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=11)
fig.suptitle("Feature selection performance (10 reps) — LASSO · Elastic-Net · Random LASSO",
             fontsize=12.5, y=0.98)
plt.tight_layout(rect=(0, 0.02, 1, 0.97))
plt.savefig(os.path.join(outdir, "compare_fig.png"), dpi=150, bbox_inches="tight")
print("saved: results/all_methods_perrep.csv + results/compare_fig.png", flush=True)

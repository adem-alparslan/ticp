from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; GEN = ROOT / "paper" / "gen"
KEY = ["dataset", "fold", "target"]
MODELS = [("tabicl-n8", "TabICL", "#1e5631", "white"), ("tabpfn35-n8", "TabPFN", "#3f8f52", "white"), ("catboost", "CatBoost", "#9fd0a6", "#143a22")]
M = pd.read_parquet(ROOT / "data" / "measurements.parquet", columns=["model", "arm", "b", "clean_correct", "flipped"] + KEY)
for c in ("model", "dataset", "arm"): M[c] = M[c].astype(str)
ok = M[M.arm == "clean"].set_index(["model"] + KEY).clean_correct == 1
A = M[M.arm == "attack"].join(ok.rename("ok"), on=["model"] + KEY); A = A[A.ok]
D = 100 * A.groupby(["model", "b", "dataset"]).flipped.mean()
S = D.groupby(["model", "b"]).agg(["mean", "std", "size"]).reset_index()
def zeichne(models, name, S=S, w=0.25, fs=8, fehler=True):
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for i, (m, lab, col, tcol) in enumerate(models):
        s = S[S.model == m].set_index("b"); x = s.index.to_numpy() + (i - (len(models) - 1) / 2) * (w + 0.02)
        lo, hi = np.minimum(s["std"], s["mean"]), np.minimum(s["std"], 100 - s["mean"])
        ax.bar(x, s["mean"], width=w, color=col, label=lab, zorder=2)
        if fehler: ax.errorbar(x, s["mean"], yerr=[lo, hi], fmt="none", ecolor="#222222", elinewidth=0.9, capsize=2.5, zorder=3)
        for xi, v in zip(x, s["mean"]): ax.text(xi, 3, f"{v:.0f}", ha="center", va="bottom", fontsize=fs, color=tcol, zorder=4)
    ax.set_xticks(sorted(S.b.unique())); ax.set_xlabel("poison rows $b$", fontsize=10); ax.set_ylabel("ASR (%)", fontsize=10)
    ax.set_ylim(0, 104); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=9); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.6, handlelength=1.2)
    fig.tight_layout(); fig.savefig(GEN / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}); plt.close(fig)
zeichne(MODELS, "fig_asr"); zeichne(MODELS[:2], "fig_asr_tfm")
R = pd.read_parquet(ROOT / "data" / "reference_models.parquet", columns=["model", "arm", "b", "clean_correct", "flipped", "dataset"])
for c in ("model", "dataset", "arm"): R[c] = R[c].astype(str)
R = R[(R.arm == "attack") & (R.clean_correct == 1)]
SA = pd.concat([S, (100 * R.groupby(["model", "b", "dataset"]).flipped.mean()).groupby(["model", "b"]).agg(["mean", "std", "size"]).reset_index()])
ALLE = MODELS[:2] + [("catboost", "CatBoost", "#6f6f6f", "white"), ("random-forest", "Random forest", "#a3a3a3", "#111111"),
                     ("logistic-regression", "Logistic regression", "#d2d2d2", "#111111")]
zeichne(ALLE, "fig_asr_alle", SA[SA.b <= 3], w=0.15, fs=7.5, fehler=False)
# Section 3.3 of ticp_alparslan_v5.0_de: every model at every number of poison rows, the TFMs first, then the trained models
REIHE = MODELS[:2] + [("random-forest", "Random forest", "#a3a3a3", "#111111"), ("catboost", "CatBoost", "#6f6f6f", "white"),
                      ("logistic-regression", "Logistic regression", "#d2d2d2", "#111111")]
zeichne(REIHE, "fig_asr_trainiert", SA, w=0.15, fs=6.5, fehler=False)
S.rename(columns={"mean": "ASR_mean", "std": "ASR_sd", "size": "datasets"}).to_csv(GEN / "csv" / "fig_asr.csv", index=False)
print(S.pivot(index="model", columns="b", values="mean").round(0).astype(int).astype(str) + " ± " + S.pivot(index="model", columns="b", values="std").round(0).astype(int).astype(str))

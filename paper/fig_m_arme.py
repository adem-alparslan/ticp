"""Figure 2 of ticp_alparslan_v5.0_de (Section 3.2.1), tests (2a) and (2b): TabICL, box plots of the mean m per dataset
under attack, per number of poison rows b, over the held (blue) and the flipped (orange) target rows, against the line
m = 1 of the read-out rule. Box: quartiles over the datasets, line: median, diamond: mean, whiskers: 1.5 times the
interquartile range, dots beyond. Below, the number of datasets with the group. Reads gen/csv/app_m_ds.csv from
zahlen_lage.py (held and flipped agree with gen/csv/tab_test_ii_v5_ds.csv of tests_v5.py, asserted) and writes
gen/fig_m_arme.pdf.

    python paper/fig_m_arme.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "gen"
WAHR, GIFT = "#2a78d6", "#eb6834"
BUDGETS = (1, 2, 3, 4, 5)
OFF, W = 0.2, 0.3                             # offset of a group from b, width of its box
YLIM = (0.7, 1.5)


def box(ax, x, v, col):
    v = v[~np.isnan(v)]
    if not len(v):
        return
    ax.boxplot([v], positions=[x], widths=W, manage_ticks=False, patch_artist=True, whis=1.5,
               showmeans=True, meanprops=dict(marker="D", ms=3.5, mfc="white", mec=col, mew=1.1),
               boxprops=dict(facecolor=col, alpha=0.35, edgecolor=col, lw=1), medianprops=dict(color=col, lw=1.8),
               whiskerprops=dict(color=col, lw=1), capprops=dict(color=col, lw=1),
               flierprops=dict(marker="o", ms=3, mfc=col, mec="none", alpha=0.9))
    ax.text(x, YLIM[0] + 0.01, str(len(v)), color=col, fontsize=9, ha="center", va="bottom")


def main():
    D = pd.read_csv(GEN / "csv" / "app_m_ds.csv")
    T = pd.read_csv(GEN / "csv" / "tab_test_ii_v5_ds.csv").pivot_table(index=["dataset", "b"], columns="group", values="mean_m")
    X = D.set_index(["dataset", "b"])
    for g in ("held", "flipped"):
        assert np.allclose(X[g].dropna(), T[g].reindex(X[g].dropna().index), atol=1e-5), f"{g}: m differs from tests_v5.py"
    assert D[["held", "flipped"]].stack().between(*YLIM).all(), "a value lies outside the axis"
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhline(1, color="0.35", ls="--", lw=0.8, zorder=0)
    for b in BUDGETS:
        t = D[D.b == b]
        box(ax, b - OFF, t.held.to_numpy(), WAHR)
        box(ax, b + OFF, t.flipped.to_numpy(), GIFT)
    ax.text(5.56, YLIM[0] + 0.01, "datasets", color="0.35", fontsize=9, ha="left", va="bottom", clip_on=False)
    ax.text(5.56, 1.0, "$m = 1$", color="0.35", fontsize=9.5, ha="left", va="center", clip_on=False)
    ax.legend([Patch(facecolor=c, alpha=0.5, edgecolor=c) for c in (WAHR, GIFT)], ["held", "flipped"], loc="upper left",
              ncol=2, frameon=False, fontsize=9.5, handlelength=1.0, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
    ax.set_xticks(BUDGETS)
    ax.set_xlim(0.5, 5.5); ax.set_ylim(*YLIM)
    ax.set_xlabel("number of poison rows $b$", fontsize=11)
    ax.set_ylabel("$m$ under attack", fontsize=11)
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(GEN / "fig_m_arme.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("->", GEN / "fig_m_arme.pdf")


if __name__ == "__main__":
    main()

"""Appendix D and E of ticp_alparslan_v5.0_de, per dataset: the tables that replace the means over the datasets of D5 to D8
(TabICL: all attacked target rows with push and control arm, movement) and E1 to E3 (TabPFN: vote, weight of the poison rows, strongest row).

Reads only per-dataset CSVs in gen/csv, so it runs after anhang.ipynb, zahlen_lage.py and zahlen_staerkste.py:
  app_m_ds.csv, app_m_liste_b.csv, app_m_bewegung_ds.csv (zahlen_lage.py)
  app_pfn_b_ds.csv, app_pfn_aF_ds.csv, app_ausserhalb_ds.csv (anhang.ipynb)
  app_rang_staerkste_ds.csv (zahlen_staerkste.py)
Writes gen/app_b_bewegt_icl.tex, app_b_bewegt_pfn.tex, app_b_dauc.tex, app_d_alle.tex, app_d_zz.tex, app_d_zentroide.tex, app_e_eins.tex, app_e_gift.tex,
app_e_staerkste.tex. Every row is a dataset, the last row the mean +- SD over the datasets with a value (with fewer than
five datasets the mean alone). Shares are decimal fractions, never percent. Nothing new is computed except the mean of
the centroid cosines over all attacked target rows, the means of the flipped and the held weighted by their counts.

    ~/miniconda3/envs/tabicl_env/bin/python paper/anhang_ds.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

GEN = Path(__file__).resolve().parent / "gen"; CSV = GEN / "csv"
BUDGETS = [1, 2, 3, 4, 5]
MIN_DS = 5
KURZ = {"banknote-authentication": "banknote", "climate-model-simulation-crashes": "climate-model",
        "Pulsar-Dataset-HTRU2": "Pulsar-Dataset", "electrical_grid_stability": "electrical_grid",
        "hazelnut-spread-contaminant": "hazelnut-spread", "polish_companies_bankruptcy": "polish_companies"}


def tn(ds):
    return "\\texttt{" + KURZ.get(ds, ds).replace("_", r"\_") + "}"


def zelle(v, nd):
    return "--" if pd.isna(v) else f"${v:.{nd}f}$"


def msd(s, nd):
    s = pd.Series(s, dtype=float).dropna()
    if s.empty:
        return "--"
    return f"${s.mean():.{nd}f}$" if len(s) < MIN_DS else f"${s.mean():.{nd}f} \\pm {s.std():.{nd}f}$"


def tabelle(name, T, gruppen, nd, eng=False):
    """T: DataFrame indexed by dataset; gruppen: list of (title, [(column, subheader), ...])."""
    spalten = [c for _, cs in gruppen for c, _ in cs]
    kopf = "Dataset" + "".join(f" & \\multicolumn{{{len(cs)}}}{{c}}{{{t}}}" for t, cs in gruppen) + r" \\ "
    pos = 2
    for _, cs in gruppen:
        kopf += f"\\cmidrule(lr){{{pos}-{pos + len(cs) - 1}}}"
        pos += len(cs)
    kopf += "\n" + "".join(f" & {s}" for _, cs in gruppen for _, s in cs)
    zeilen = [f"{tn(ds)} & " + " & ".join(zelle(T.loc[ds, c], nd) for c in spalten) + r" \\" for ds in T.index]
    zeilen.append(r"\midrule mean $\pm$ SD & " + " & ".join(msd(T[c], nd) for c in spalten) + r" \\")
    tex = ("\\begin{tabular}{@{}l" + "c" * len(spalten) + "@{}}\\toprule\n" + kopf + r" \\ \midrule" + "\n"
           + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}")
    # eng: slightly tighter rows, where a text has to fit below the table on the same landscape page
    (GEN / f"{name}.tex").write_text(("\\renewcommand{\\arraystretch}{0.93}" if eng else "") + "\\centering\\scriptsize\\maxbreite{" + tex + "}\n")


def b_spalten(prefix):
    return [(f"{prefix}{b}", f"$b = {b}$") for b in BUDGETS]


def main():
    ordnung = lambda idx: sorted(idx, key=lambda d: KURZ.get(d, d).lower())
    # D5: all attacked target rows of a dataset (the same rows at every b): m_0 and its share below 1 without attack, the push
    # m - m_0 under attack and m in the control arm per b. The push of the flipped and the held is D1 minus D2 per dataset.
    m = pd.read_csv(CSV / "app_m_ds.csv")
    assert (m.groupby("dataset").start.nunique() == 1).all(), "m_0 differs between budgets"
    T = pd.DataFrame(index=ordnung(m.dataset.unique()))
    s1 = m[m.b == 1].set_index("dataset")
    T["m0"], T["unter"] = s1.start, s1.start_below / 100
    for col, p in (("push_all", "s"), ("control", "c")):
        w = m.pivot(index="dataset", columns="b", values=col)
        for b in BUDGETS:
            T[f"{p}{b}"] = w[b]
    tabelle("app_d_alle", T, [("without attack", [("m0", "$m_0$"), ("unter", "$m_0 < 1$, share")]),
                              ("$m - m_0$, under attack", b_spalten("s")), ("$m$, control arm", b_spalten("c"))], 3)
    # D7 and D8: the movement, cos(z, z_0) per group, and the centroids over all attacked target rows
    bw = pd.read_csv(CSV / "app_m_bewegung_ds.csv")
    n = pd.read_csv(CSV / "app_m_liste_b.csv")[["dataset", "b", "n_flipped", "n_held"]]
    T = pd.DataFrame(index=ordnung(bw.dataset.unique()))
    z = bw[bw.quantity == "cos_z_z0"]
    for col, p in (("flipped", "f"), ("held", "h"), ("control", "c")):
        w = z.pivot(index="dataset", columns="b", values=col)
        for b in BUDGETS:
            T[f"{p}{b}"] = w[b]
    tabelle("app_d_zz", T, [(r"$\cos(z, z_0)$, flipped", b_spalten("f")), (r"$\cos(z, z_0)$, held", b_spalten("h")),
                            (r"$\cos(z, z_0)$, control arm", b_spalten("c"))], 3)
    j = bw[bw.quantity != "cos_z_z0"].merge(n, on=["dataset", "b"], validate="many_to_one")
    j["all"] = (j.flipped.fillna(0) * j.n_flipped + j.held.fillna(0) * j.n_held) / (j.n_flipped + j.n_held)
    T = pd.DataFrame(index=ordnung(bw.dataset.unique()))
    for q, p in (("cos_muT_muT0", "t"), ("cos_muF_muF0", "f")):
        w = j[j.quantity == q].pivot(index="dataset", columns="b", values="all")
        for b in BUDGETS:
            T[f"{p}{b}"] = w[b]
    tabelle("app_d_zentroide", T, [(r"$\cos(\mu_T, \mu_{T,0})$", b_spalten("t")), (r"$\cos(\mu_F, \mu_{F,0})$", b_spalten("f"))], 4)
    # B3 to B5: the remaining test rows of the fold, per dataset: rows whose prediction changes (mean over the attacked target
    # rows, a mean count, not a share) under attack and in the control arm, and the change of their AUC under attack
    au = pd.read_csv(CSV / "app_ausserhalb_ds.csv")
    for lab, name in (("TabICL", "app_b_bewegt_icl"), ("TabPFN", "app_b_bewegt_pfn")):
        x = au[au.model == lab]
        T = pd.DataFrame(index=ordnung(x.dataset.unique()))
        for col, p in (("moved_attack", "a"), ("moved_control", "c")):
            w = x.pivot(index="dataset", columns="b", values=col)
            for b in BUDGETS:
                T[f"{p}{b}"] = w[b]
        tabelle(name, T, [(f"moved rows, attack, {lab}", b_spalten("a")), (f"moved rows, control arm, {lab}", b_spalten("c"))], 1,
                eng=(lab == "TabICL"))
    T = pd.DataFrame(index=ordnung(au.dataset.unique()))
    for lab, p in (("TabICL", "i"), ("TabPFN", "p")):
        w = au[au.model == lab].pivot(index="dataset", columns="b", values="dAUC_attack")
        for b in BUDGETS:
            T[f"{p}{b}"] = w[b]
    tabelle("app_b_dauc", T, [(r"$\Delta$AUC, attack, TabICL", b_spalten("i")), (r"$\Delta$AUC, attack, TabPFN", b_spalten("p"))], 4)
    # D1 to D5: the vote per number of poison rows b, built like Figure fig_vote: per condition the weights of the vote, which
    # sum to 1 per row and condition; for the flipped and the held target rows also a_F,0 of the same rows without attack
    v = pd.read_csv(CSV / "app_pfn_b_ds.csv")
    a = pd.read_csv(CSV / "app_pfn_aF_ds.csv")
    ohne = v[v.arm == "clean"].set_index("dataset")
    for b in BUDGETS:
        T = pd.DataFrame(index=ordnung(v.dataset.unique()))
        T["o_T"], T["o_F"] = ohne.a_T, ohne.a_F
        c = v[(v.arm == "control") & (v.b == b)].set_index("dataset")
        T["c_T"], T["c_C"], T["c_F"] = c.a_T, c.a_P, c.a_F
        for g in ("flipped", "held"):
            x = v[(v.arm == "attack") & (v.group == g) & (v.b == b)].set_index("dataset")
            T[f"{g}_F0"] = a[(a.comparison == g) & (a.b == b)].set_index("dataset").a_F0
            T[f"{g}_T"], T[f"{g}_F"], T[f"{g}_P"] = x.a_T, x.a_F, x.a_P
        assert np.allclose(a[(a.comparison == "control") & (a.b == b)].set_index("dataset").a_F0.reindex(T.index), T.o_F), "a_F,0 differs"
        gruppe = lambda g: [(f"{g}_F0", "$a_{F,0}$"), (f"{g}_T", "$a_T$"), (f"{g}_F", "$a_{F,G}$"), (f"{g}_P", "$a_{F,P}$")]
        tabelle(f"app_e_vote_{b}", T, [("without attack", [("o_T", "$a_{T,0}$"), ("o_F", "$a_{F,0}$")]),
                                       ("control arm", [("c_T", "$a_{T,G}$"), ("c_C", "$a_{T,C}$"), ("c_F", "$a_{F,G}$")]),
                                       ("under attack, flipped", gruppe("flipped")), ("under attack, held", gruppe("held"))], 3)
    # E3: the share of target rows for which a poison row is the strongest row, one and two poison rows
    r = pd.read_csv(CSV / "app_rang_staerkste_ds.csv")
    T = pd.DataFrame(index=ordnung(r.dataset.unique()))
    for g, p in (("all", "a"), ("flipped", "f"), ("held", "h")):
        w = r[r.group == g].pivot(index="dataset", columns="b", values="strongest_pct") / 100
        for b in BUDGETS:
            T[f"{p}{b}"] = w[b] if b in w else np.nan
    tabelle("app_e_staerkste", T, [("share strongest, all attacked", b_spalten("a")), ("share strongest, flipped", b_spalten("f")),
                                   ("share strongest, held", b_spalten("h"))], 2)
    print("->", ", ".join(f"gen/{x}.tex" for x in ("app_b_bewegt_icl", "app_b_bewegt_pfn", "app_b_dauc", "app_d_alle", "app_d_zz", "app_d_zentroide", "app_e_eins",
                                                    "app_e_gift", "app_e_afg", "app_e_kontrolle", "app_e_staerkste")))


if __name__ == "__main__":
    main()

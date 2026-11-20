"""Neighbourhood of the TabICL target row, nu_F (Sections 2.4, 3.2.1, 4.2, 4.4; Tables B4 and B5 of the appendix).

nu_F = column k_opp_gen_25 of data/measurements.parquet: of the 25 nearest genuine context rows of the target row by
cosine at the input of the MLP of TabICL (the copies left out), how many carry the wrong label, per member, then
averaged. The wrong class holds the majority at nu_F > 12.5. Attacked target rows: correct without attack, as everywhere.
The result file holds the same count for TabPFN, at the output of its last norm; the manuscript does not use it, since
the decoder of TabPFN weights the context rows by query and key projections, not by the cosine of the embeddings
(Section 2.4).

Writes gen/app_kopp.tex (TabICL per number of poison rows b: flipped target rows with and without a majority of the
wrong class, held ones with it, and the AUC of nu_F, flipped against held, mean +- SD over the datasets with both
groups), gen/app_kopp_ds_icl.tex (per dataset at b = 1: without attack, held and flipped under attack, control arm;
mean of nu_F and share with a majority of the wrong class), gen/csv/app_kopp.csv, gen/csv/app_kopp_ds.csv and
gen/zahlen_kopp.tex.

    python paper/zahlen_kopp.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from zahlen_lage import KURZ, msd, tn

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "gen"; CSV = GEN / "csv"
ICL, PFN = "tabicl-n8", "tabpfn35-n8"
BUDGETS = (1, 2, 3, 4, 5)
MEHRHEIT = 12.5


def angriff(P, model, b):
    return P[(P.model == model) & (P.arm == "attack") & (P.b == b) & (P.clean_correct == 1)]


def main():
    P = pd.read_parquet(ROOT / "data" / "measurements.parquet",
                        columns=["model", "dataset", "fold", "target", "arm", "b", "clean_correct", "flipped", "pi_T",
                                 "m", "a_F", "a_copy", "k_opp_gen_25"])
    for c in ("model", "dataset", "arm"):
        P[c] = P[c].astype(str)
    P = P.rename(columns={"k_opp_gen_25": "k"})
    assert P[P.model.isin([ICL, PFN])].k.notna().all(), "k_opp is missing"

    T = []
    for b in BUDGETS:
        for model in (ICL,):
            a = angriff(P, model, b); f, h = a[a.flipped == 1], a[a.flipped == 0]
            auc = pd.Series([roc_auc_score(s.flipped, s.k) for _, s in a.groupby("dataset") if 0 < s.flipped.mean() < 1], dtype=float)
            T.append(dict(model=model, b=b, flipped=len(f), flipped_majority=int((f.k > MEHRHEIT).sum()),
                          flipped_no_majority=int((f.k <= MEHRHEIT).sum()), held=len(h), held_majority=int((h.k > MEHRHEIT).sum()),
                          auc=auc.mean(), auc_sd=auc.std(), auc_ds=len(auc)))
    T = pd.DataFrame(T); T.to_csv(CSV / "app_kopp.csv", index=False)

    def zelle(r):
        a = "--" if r.auc_ds < 2 else f"${r.auc:.2f} \\pm {r.auc_sd:.2f}$"
        return f"{r.flipped_majority} & {r.flipped_no_majority} & {r.held_majority} & {a}"
    rows = []
    for b in BUDGETS:
        ri = T[(T.model == ICL) & (T.b == b)].iloc[0]
        rows.append(f"{b} & {zelle(ri)} \\\\")
    head = (r"$b$ & \multicolumn{2}{c}{flipped} & held & AUC \\"
            "\n" r"& majority & no majority & majority & of $\nu_F$")
    t = ("\\begin{tabular}{@{}rcccc@{}}\\toprule\n" + head + r" \\ \midrule" + "\n" + "\n".join(rows)
         + "\n\\bottomrule\\end{tabular}")
    (GEN / "app_kopp.tex").write_text("\\centering\\scriptsize\\maxbreite{" + t + "}\n")

    # per dataset at b = 1 (Table B5)
    DS = []
    for model, tag in ((ICL, "icl"),):
        c = P[(P.model == model) & (P.arm == "clean") & (P.clean_correct == 1)]
        a, k = angriff(P, model, 1), P[(P.model == model) & (P.arm == "control") & (P.b == 1) & (P.clean_correct == 1)]
        L = []
        for ds in sorted(c.dataset.unique(), key=lambda d: KURZ.get(d, d).lower()):
            cd, ad, kd = c[c.dataset == ds], a[a.dataset == ds], k[k.dataset == ds]
            r = dict(model=model, dataset=ds, n=len(cd), clean=cd.k.mean(), clean_maj=(cd.k > MEHRHEIT).mean(),
                     control=kd.k.mean(), control_maj=(kd.k > MEHRHEIT).mean())
            for grp, g in (("held", ad[ad.flipped == 0]), ("flipped", ad[ad.flipped == 1])):
                r.update({f"n_{grp}": len(g), grp: g.k.mean() if len(g) else np.nan,
                          f"{grp}_maj": (g.k > MEHRHEIT).mean() if len(g) else np.nan})
            L.append(r)
        L = pd.DataFrame(L); DS.append(L)
        zahl = lambda v, nd: "--" if pd.isna(v) else f"${v:.{nd}f}$"
        rows = [f"{tn(r.dataset)} & {r.n} & {zahl(r.clean, 1)} & {zahl(r.clean_maj, 2)} & {r.n_held} & {zahl(r.held, 1)} & "
                f"{zahl(r.held_maj, 2)} & {r.n_flipped} & {zahl(r.flipped, 1)} & {zahl(r.flipped_maj, 2)} & "
                f"{zahl(r.control, 1)} & {zahl(r.control_maj, 2)} \\\\" for r in L.itertuples()]
        rows.append(r"\midrule mean $\pm$ SD & " + f"{L.n.sum()} & {msd(L.clean, 1)} & {msd(L.clean_maj, 2)} & {L.n_held.sum()} & "
                    f"{msd(L.held.dropna(), 1)} & {msd(L.held_maj.dropna(), 2)} & {L.n_flipped.sum()} & {msd(L.flipped.dropna(), 1)} & "
                    f"{msd(L.flipped_maj.dropna(), 2)} & {msd(L.control, 1)} & {msd(L.control_maj, 2)} \\\\")
        head = (r"& & \multicolumn{2}{c}{without attack} & \multicolumn{3}{c}{held} & \multicolumn{3}{c}{flipped} & \multicolumn{2}{c}{control arm} \\"
                r" \cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}" "\n"
                r"Dataset & $n$ & $\nu_F$ & majority & $n$ & $\nu_F$ & majority & $n$ & $\nu_F$ & majority & $\nu_F$ & majority")
        t = ("\\begin{tabular}{@{}lccccccccccc@{}}\\toprule\n" + head + r" \\ \midrule" + "\n" + "\n".join(rows)
             + "\n\\bottomrule\\end{tabular}")
        (GEN / f"app_kopp_ds_{tag}.tex").write_text("\\centering\\scriptsize\\maxbreite{" + t + "}\n")
    pd.concat(DS, ignore_index=True).to_csv(CSV / "app_kopp_ds.csv", index=False)

    mac = {}
    # Section 4.2: TabICL, flipped target rows without a majority of the wrong class, b = 1
    r1 = T[(T.model == ICL) & (T.b == 1)].iloc[0]
    mac["zKoppICLohne"], mac["zKoppICLgekippt"] = str(r1.flipped_no_majority), str(r1.flipped)
    # Section 4.4: the picture holds at the thresholds 10 and 15 (TabICL flips almost only with a majority)
    a1 = angriff(P, ICL, 1); f1, h1 = a1[a1.flipped == 1], a1[a1.flipped == 0]
    for thr in (10, MEHRHEIT, 15):
        assert (f1.k > thr).mean() >= 0.85 and (h1.k > thr).mean() <= 0.25, \
            f"Section 4.4: at the threshold {thr} the majority no longer separates flipped from held TabICL target rows"
    # Section 3.2.1: TabICL, flipped target rows below and above the line m = 1, b = 1
    a = angriff(P, ICL, 1); f = a[a.flipped == 1]; un, ue = f[f.m <= 1], f[f.m > 1]
    # per dataset, then the mean over the datasets with such rows
    proDS = lambda g, wert: wert(g).groupby(g.dataset).mean().mean()
    mac["zKoppVerfehltMaj"] = f"{100 * proDS(un, lambda g: g.k > MEHRHEIT):.0f}"
    mac["zKoppGetroffenMaj"] = f"{100 * proDS(ue, lambda g: g.k > MEHRHEIT):.0f}"
    mac["zKoppPiFverfehlt"] = f"{proDS(un, lambda g: 1 - g.pi_T):.2f}"
    mac["zKoppPiFgetroffen"] = f"{proDS(ue, lambda g: 1 - g.pi_T):.2f}"
    # Section 3.2.1: TabICL per dataset at b = 1 (Table B5): majority among the flipped and the held, k_opp without attack
    # and in the control arm
    L = DS[0]; assert (L.model == ICL).all()
    mac["zKoppICLgekipptMaj"] = f"{100 * L.flipped_maj.dropna().mean():.0f}"
    mac["zKoppICLgehaltenMaj"] = f"{100 * L.held_maj.dropna().mean():.0f}"
    mac["zKoppICLohneMittel"] = f"{L.clean.mean():.1f}"
    mac["zKoppICLgekipptMittel"] = f"{L.flipped.dropna().mean():.1f}"
    mac["zKoppICLgehaltenMittel"] = f"{L.held.dropna().mean():.1f}"
    mac["zKoppICLkontrMittel"] = f"{L.control.mean():.1f}"
    assert L.control.mean() < L.clean.mean(), "Section 3.2.1: nu_F falls in the control arm"

    andere = set()
    for g in GEN.glob("zahlen*.tex"):
        if g.name != "zahlen_kopp.tex":
            andere |= set(re.findall(r"\\newcommand\{\\(z[A-Za-z]+)\}", g.read_text()))
    assert not andere & set(mac), f"also defined elsewhere: {sorted(andere & set(mac))}"
    (GEN / "zahlen_kopp.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    return mac


if __name__ == "__main__":
    for k, v in main().items():
        print(k, v)
    print("->", GEN / "app_kopp.tex", GEN / "zahlen_kopp.tex", CSV / "app_kopp.csv")

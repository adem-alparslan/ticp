import re

import pandas as pd

from zahlen_referenz import BUDGETS, CSV, GEN, MODELS, load

KURZ = {"banknote-authentication": "banknote", "climate-model-simulation-crashes": "climate-model",
        "Pulsar-Dataset-HTRU2": "Pulsar-Dataset", "electrical_grid_stability": "electrical_grid",
        "hazelnut-spread-contaminant": "hazelnut-spread", "polish_companies_bankruptcy": "polish_companies"}
TFM = MODELS[:2]
# the trained models in the order of Figure fig_asr and of the text: random forest, CatBoost, logistic regression
TRAINIERT = [next(m for m in MODELS if m[0] == k) for k in ("random-forest", "catboost", "logistic-regression")]
WEG = "CatBoost"
FEHLT = "Missing, no target row correct without attack in a class: "


def tex_name(ds):
    return "\\texttt{" + KURZ.get(ds, ds).replace("_", r"\_") + "}"


def pct(v):
    # a share as a decimal fraction (0.87), never in percent, so that it cannot be read as a count
    return "--" if pd.isna(v) else f"${v:.2f}$"


def msd(s, d, f=1.0):
    s = pd.Series(s, dtype=float).dropna() * f
    return f"${s.mean():.{d}f} \\pm {s.std():.{d}f}$"


def tabular(spec, kopf, zeilen, fuss=(), breit=True):
    body = kopf + r" \\ \midrule" + "\n" + "\n".join(z + r" \\" for z in zeilen)
    if fuss:
        body += "\n\\midrule " + "\n".join(z + r" \\" for z in fuss)
    innen = f"\\begin{{tabular}}{{{spec}}}\\toprule\n{body}\n\\bottomrule\\end{{tabular}}"
    return "\\centering\\scriptsize" + ("\\maxbreite{" + innen + "}" if breit else innen) + "\n"


def stapel(teile):
    return "\n".join(f"\\par\\medskip\\noindent\\textit{{{titel}}}\\par\\nopagebreak\n" + tex for titel, tex in teile)


def guete(x, ds, modelle, name):
    clean = x[x.arm == "clean"]
    auc = (clean.groupby(["model", "dataset", "fold"]).auc_clean.first().groupby(["model", "dataset"]).mean()
           .unstack("model").reindex(ds))
    n = clean[clean.clean_correct == 1].groupby(["dataset", "model"]).size().unstack("model").reindex(ds).fillna(0).astype(int)
    zeilen = [f"{tex_name(d)} & " + " & ".join(f"${auc.loc[d, m]:.3f}$ & {n.loc[d, m]}" for m, _, _ in modelle) for d in ds]
    fuss = ["mean $\\pm$ SD & " + " & ".join(f"{msd(auc[m], 3)} & " for m, _, _ in modelle),
            "sum & " + " & ".join(f" & {n[m].sum():,}" for m, _, _ in modelle)]
    kopf = ("Dataset & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab, _ in modelle) + " \\\\ "
            + " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(modelle)))
            + "\n & " + " & ".join("AUC & correct" for _ in modelle))
    (GEN / f"{name}.tex").write_text(tabular("@{}l" + "cc" * len(modelle) + "@{}", kopf, zeilen, fuss, breit=False))
    return pd.concat([auc[[m for m, _, _ in modelle]].rename(columns={m: f"AUC_{lab}" for m, lab, _ in modelle}),
                      n[[m for m, _, _ in modelle]].rename(columns={m: f"correct_{lab}" for m, lab, _ in modelle})],
                     axis=1).rename_axis("dataset").reset_index()


def asr(x, ds, modelle, name):
    a = x[(x.arm == "attack") & (x.clean_correct == 1)]
    T = {m: a[a.model == m].groupby(["dataset", "b"]).flipped.mean().unstack("b").reindex(ds) for m, _, _ in modelle}
    paare = [(m, b) for m, _, _ in modelle for b in BUDGETS]
    zeilen = [f"{tex_name(d)} & " + " & ".join(pct(T[m].loc[d, b]) for m, b in paare) for d in ds]
    fuss = ["mean $\\pm$ SD & " + " & ".join(msd(T[m][b], 2) for m, b in paare),
            r"datasets with ASR $\geq 0.5$ & " + " & ".join(str(int((T[m][b] >= 0.5).sum())) for m, b in paare),
            r"datasets with ASR $< 0.1$ & " + " & ".join(str(int((T[m][b] < 0.1).sum())) for m, b in paare)]
    nb = len(BUDGETS)
    kopf = ("Dataset & " + " & ".join(f"\\multicolumn{{{nb}}}{{c}}{{ASR, {lab}}}" for _, lab, _ in modelle) + " \\\\ "
            + " ".join(f"\\cmidrule(lr){{{2 + nb * i}-{1 + nb * (i + 1)}}}" for i in range(len(modelle)))
            + "\n & " + " & ".join(f"$b = {b}$" for _ in modelle for b in BUDGETS))
    (GEN / f"{name}.tex").write_text(tabular("@{}l" + "c" * (nb * len(modelle)) + "@{}", kopf, zeilen, fuss))


def asr_tfm(x, ds, modelle, name):
    """Table B1 of ticp_alparslan_v5.0_de: the ASR of the two TFMs per dataset, grouped by the number of poison rows b, the two
    TFMs side by side under each b. Below the means, per b: the mean over the datasets of the difference TabICL minus TabPFN,
    and Spearman's rho of the two ASR columns over the datasets (the order of the datasets), formerly Table C1."""
    assert len(modelle) == 2, modelle
    a = x[(x.arm == "attack") & (x.clean_correct == 1)]
    T = {m: a[a.model == m].groupby(["dataset", "b"]).flipped.mean().unstack("b").reindex(ds) for m, _, _ in modelle}
    (m1, l1, _), (m2, l2, _) = modelle
    paare = [(m, b) for b in BUDGETS for m, _, _ in modelle]
    zeilen = [f"{tex_name(d)} & " + " & ".join(pct(T[m].loc[d, b]) for m, b in paare) for d in ds]
    diff = {b: (T[m1][b] - T[m2][b]).mean() for b in BUDGETS}
    rho = {b: T[m1][b].corr(T[m2][b], method="spearman") for b in BUDGETS}
    # the numbers of Section 3.1 (formerly Table C1): TabPFN above TabICL at one and two poison rows, rho 0.80 at one
    k = pd.read_csv(CSV / "app_gepaart_tfm.csv").set_index("b")
    assert all(abs(100 * diff[b] - k.loc[b, "diff"]) < 1e-6 and abs(rho[b] - k.loc[b, "rho"]) < 1e-9 for b in BUDGETS), "differs from app_gepaart_tfm"
    zwei = lambda w: f"\\multicolumn{{2}}{{c}}{{{w}}}"
    fuss = ["mean $\\pm$ SD & " + " & ".join(msd(T[m][b], 2) for m, b in paare),
            r"datasets with ASR $\geq 0.5$ & " + " & ".join(str(int((T[m][b] >= 0.5).sum())) for m, b in paare),
            r"datasets with ASR $< 0.1$ & " + " & ".join(str(int((T[m][b] < 0.1).sum())) for m, b in paare),
            f"{l1} $-$ {l2}, mean & " + " & ".join(zwei(f"${diff[b]:+.3f}$") for b in BUDGETS),
            "Spearman's $\\rho$ & " + " & ".join(zwei(f"${rho[b]:.2f}$") for b in BUDGETS)]
    kopf = ("Dataset & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{ASR, $b = {b}$}}" for b in BUDGETS) + " \\\\ "
            + " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BUDGETS)))
            + "\n & " + " & ".join(lab for _ in BUDGETS for _, lab, _ in modelle))
    # slightly tighter rows, so that the table fits on one landscape page with the headings of Appendix B
    (GEN / f"{name}.tex").write_text("\\renewcommand{\\arraystretch}{0.93}"
                                     + tabular("@{}l" + "c" * (2 * len(BUDGETS)) + "@{}", kopf, zeilen, fuss))


def korrigiert(ds, modelle, name):
    """Control arm, ticp_alparslan_v5.0_de: per dataset and number of control rows the share of the target rows wrong without
    attack that the control rows correct (-- where the model predicts every target row of the dataset correctly). The values
    are those of app_kontrolle_reparatur_ds.csv of anhang.ipynb, the last row repeats the means of its Table app_kontrolle."""
    r = pd.read_csv(CSV / "app_kontrolle_reparatur_ds.csv")
    # the notebook aggregates a boolean column, so a block (model, b) in which every dataset is corrected completely is written
    # as True instead of 1.0
    assert set(r.corrected[pd.to_numeric(r.corrected, errors="coerce").isna()]) <= {"True"}, "unexpected text in corrected"
    r["corrected"] = pd.to_numeric(r.corrected.replace({"True": "1.0"}))
    rep = pd.read_csv(CSV / "app_kontrolle_reparatur.csv")
    T = {lab: r[r.model == lab].pivot(index="dataset", columns="b", values="corrected").reindex(ds) for _, lab, _ in modelle}
    for _, lab, _ in modelle:
        mean = rep[rep.model == lab].set_index("b").corrected_mean
        assert all(abs(T[lab][b].mean() - mean[b]) < 1e-9 for b in BUDGETS), f"{lab}: the means differ from app_kontrolle"
    paare = [(lab, b) for _, lab, _ in modelle for b in BUDGETS]
    zeilen = [f"{tex_name(d)} & " + " & ".join(pct(T[lab].loc[d, b]) for lab, b in paare) for d in ds]
    fuss = ["mean $\\pm$ SD & " + " & ".join(msd(T[lab][b], 2) for lab, b in paare)]
    nb = len(BUDGETS)
    kopf = ("Dataset & " + " & ".join(f"\\multicolumn{{{nb}}}{{c}}{{share corrected, {lab}}}" for _, lab, _ in modelle) + " \\\\ "
            + " ".join(f"\\cmidrule(lr){{{2 + nb * i}-{1 + nb * (i + 1)}}}" for i in range(len(modelle)))
            + "\n & " + " & ".join(f"$b = {b}$" for _ in modelle for b in BUDGETS))
    (GEN / f"{name}.tex").write_text(tabular("@{}l" + "c" * (nb * len(modelle)) + "@{}", kopf, zeilen, fuss))


def kontrolle_text(x, modelle, name):
    """Appendix B.2 of ticp_alparslan_v5.0_de: the attacked target rows that flip in the control arm, as a sentence."""
    k = x[(x.arm == "control") & (x.clean_correct == 1)]
    teile = []
    for m, lab, _ in modelle:
        km = k[k.model == m]
        n = int((km.b == BUDGETS[0]).sum())
        f = km[km.flipped == 1]
        if len(f) == 0:
            teile.append(f"none of the {n:,} of {lab}")
        else:
            bs = [b for b in BUDGETS if (f.b == b).any()]
            wo = (f"at $b = {bs[0]}$" if len(bs) == 1 else
                  "(" + ", ".join(f"{int((f.b == b).sum())} at $b = {b}$" for b in bs) + ")")
            teile.append(f"{len(f)} of the {n:,} attacked target rows of {lab} flip{'s' if len(f) == 1 else ''} {wo}")
    (GEN / f"{name}.tex").write_text("In the control arm, over all numbers of control rows, " + ", and ".join(teile) + ".\n")


def hill_text(name, ds_name="hill-valley"):
    """Appendix B.3 of ticp_alparslan_v5.0_de: on hill-valley the predictions of the remaining test rows of TabICL change as
    often in the control arm as under attack. Means over the attacked target rows of all numbers of poison rows; near the
    boundary: |pi - 0.5| < 0.1 without attack, over the target rows of the dataset (the only rows whose pi is stored)."""
    import pandas as _pd
    from zahlen_referenz import ROOT
    d = _pd.read_parquet(ROOT / "data" / "measurements.parquet",
                         columns=["model", "arm", "dataset", "clean_correct", "pi_T", "co_moved", "co_moved_wrong", "co_moved_right",
                                  "auc_rest", "auc_clean"])
    for c in ("model", "arm", "dataset"):
        d[c] = d[c].astype(str)
    icl, pfn = TFM[0][0], TFM[1][0]
    x = d[(d.model == icl) & (d.arm != "clean") & (d.clean_correct == 1)]
    meist = x.groupby("dataset").co_moved.mean().idxmax()
    assert meist == ds_name, f"the dataset with the most moved rows is {meist}"
    h = x[x.dataset == ds_name]
    ma, mc = (h[h.arm == a].co_moved.mean() for a in ("attack", "control"))
    hw, hr = h[h.arm == "attack"].co_moved_wrong.mean(), h[h.arm == "attack"].co_moved_right.mean()
    dauc = (h[h.arm == "attack"].auc_rest - h[h.arm == "attack"].auc_clean).mean()
    k = d[(d.arm == "clean") & (d.dataset == ds_name)].assign(knapp=lambda z: (z.pi_T - 0.5).abs() < 0.1)
    ki, n = int(k[k.model == icl].knapp.sum()), int((k.model == icl).sum())
    kp = int(k[k.model == pfn].knapp.sum())
    zahl = lambda z: "none" if z == 0 else str(z)
    t = (f"On \\texttt{{{ds_name}}}, the predictions of on average {ma:.1f} remaining test rows change per attacked target row for TabICL "
         f"under attack and {mc:.1f} in the control arm, means over all numbers of poison rows, "
         f"{hw:.1f} of them to the wrong and {hr:.1f} to the right label under attack, and their AUC stays "
         f"unchanged (${dauc:+.4f}$). This is no effect of the attack: as many predictions change in the control arm, and the predictions "
         f"of TabICL on this dataset react to any change of the context. Without attack, {ki} of the {n} target rows of this dataset lie close "
         f"to the decision boundary for TabICL ($|\\pi_F - 0.5| < 0.1$) and {zahl(kp)} for TabPFN, whose predictions of the remaining test "
         f"rows on this dataset barely change (Table~\\ref{{tab:app_b_bewegt_pfn}}).\n")
    (GEN / f"{name}.tex").write_text(t)


def credit_text(x, modelle, name, ds_name="creditcard"):
    """Appendix A of ticp_alparslan_v5.0_de: the AUC criterion says little on creditcard (one minority row per test fold), and
    leaving the dataset out changes the mean ASR of either TFM by at most this much at any number of poison rows (rounded up)."""
    import math
    a = x[(x.arm == "attack") & (x.clean_correct == 1)]
    d = 0.0
    for m, _, _ in modelle:
        for b in BUDGETS:
            s = a[(a.model == m) & (a.b == b)].groupby("dataset").flipped.mean()
            d = max(d, abs(s.drop(ds_name).mean() - s.mean()))
    grenze = math.ceil(d * 1000) / 1000
    (GEN / f"{name}.tex").write_text(
        f"In \\texttt{{{ds_name}}} each test fold holds a single row of the minority class, so its per-fold AUC rests on one positive "
        f"case and the AUC criterion of Section~\\ref{{sec:daten}} says little there. Without \\texttt{{{ds_name}}}, the mean ASR of "
        f"either TFM changes by at most {grenze:.3f} at any number of poison rows.\n")
    # the same bound for Section 4.3 of the manuscript
    (GEN / "zahlen_creditcard.tex").write_text(f"\\newcommand{{\\zOhneCreditMax}}{{{grenze:.3f}}}\n")


def daten_guete(x, ds, modelle, name):
    """Appendix A.1 of ticp_alparslan_v5.0_de: the datasets (app_daten.csv of anhang.ipynb) and the baseline performance of the
    TFMs, the AUC on the test rows (mean over the folds) and the number of target rows predicted correctly without attack."""
    d = pd.read_csv(CSV / "app_daten.csv").set_index("dataset").reindex(ds)
    clean = x[x.arm == "clean"]
    auc = (clean.groupby(["model", "dataset", "fold"]).auc_clean.first().groupby(["model", "dataset"]).mean()
           .unstack("model").reindex(ds))
    n = clean[clean.clean_correct == 1].groupby(["dataset", "model"]).size().unstack("model").reindex(ds).fillna(0).astype(int)
    dup = lambda r: f"{int(r.dup_targets)} ({int(r.dup_max)})" if r.dup_targets else "0"
    zeilen = [f"{tex_name(i)} & {int(r.did)} & {int(r.rows):,} & {int(r.n_ctx):,} & {int(r.features)} & {r.minority:.3f} & "
              f"{int(r.targets)} & {dup(r)} & " + " & ".join(f"${auc.loc[i, m]:.3f}$ & {n.loc[i, m]}" for m, _, _ in modelle)
              for i, r in d.iterrows()]
    leer = " & " * 7
    fuss = ["mean $\\pm$ SD" + leer + " & " + " & ".join(f"{msd(auc[m], 3)} & " for m, _, _ in modelle),
            "sum" + " & " * 5 + f" & {int(d.targets.sum()):,} & & " + " & ".join(f" & {n[m].sum():,}" for m, _, _ in modelle)]
    kopf = (" & & & & & & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab, _ in modelle) + " \\\\ "
            + " ".join(f"\\cmidrule(lr){{{9 + 2 * i}-{10 + 2 * i}}}" for i in range(len(modelle)))
            + "\nDataset & OpenML & rows & context & features & minority & target rows & with duplicate (max) & "
            + " & ".join("AUC & correct" for _ in modelle))
    (GEN / f"{name}.tex").write_text(tabular("@{}lrrrrrcc" + "cc" * len(modelle) + "@{}", kopf, zeilen, fuss))


def kontrolle(x, modelle, name):
    k = x[(x.arm == "control") & (x.clean_correct == 1)]
    f = x[(x.arm == "attack") & (x.clean_correct == 1) & (x.flipped == 1)]
    f = f.assign(sicher=(1 - f.pi_T) >= 0.9)
    oben, unten, recs = [], [], []
    for m, lab, _ in modelle:
        o, u = [], []
        for b in BUDGETS:
            kb, fb = k[(k.model == m) & (k.b == b)], f[(f.model == m) & (f.b == b)]
            s9 = fb.groupby("dataset").sicher.mean()
            o.append(f"{int(kb.flipped.sum())} of {len(kb):,}")
            u.append(pct(s9.mean()) if len(s9) else "--")
            recs.append(dict(model=lab, b=b, attacked=len(kb), flipped_control=int(kb.flipped.sum()), flipped=len(fb),
                             DS_flipped=len(s9), confident_wrong=int(fb.sicher.sum()), confident_wrong_share=s9.mean()))
        oben.append(f"{lab} & " + " & ".join(o))
        unten.append(f"{lab} & " + " & ".join(u))
    kopf = "Model & " + " & ".join(f"$b = {b}$" for b in BUDGETS)
    spec = "@{}l" + "c" * len(BUDGETS) + "@{}"
    (GEN / f"{name}.tex").write_text(stapel([
        ("attacked target rows that flip in the control arm", tabular(spec, kopf, oben, breit=False)),
        (r"flipped target rows that are wrong with $\pi_F \geq 0.9$, share", tabular(spec, kopf, unten, breit=False))]))
    pd.DataFrame(recs).to_csv(CSV / f"{name}.csv", index=False)


def ohne_trainierte(name):
    t = (GEN / f"{name}.tex").read_text().split("\n")
    assert any(z.startswith(WEG) for z in t) and all(any(z.startswith(lab) for z in t) for _, lab, _ in TFM), name
    (GEN / f"{name}_tfm.tex").write_text("\n".join(z for z in t if not z.startswith(WEG)))


def texte():
    t, k = re.subn(r"hold for TabICL, (\d+) for TabPFN and \d+ for CatBoost\. Of these, (\d+), (\d+) and \d+ have",
                   r"hold for TabICL and \1 for TabPFN. Of these, \2 and \3 have", (GEN / "app_daten_text.tex").read_text())
    assert k == 1 and WEG not in t, t
    (GEN / "app_daten_text_tfm.tex").write_text(t)
    # Appendix A of ticp_alparslan_v5.0_de: one sentence from the numbers of the notebook text, the TFMs only
    q = (GEN / "app_klassen_text.tex").read_text()
    m = re.search(r"^(\d+) datasets with a minority share below ([0-9.]+):", q)
    z = re.search(r"Without attack, TabICL, TabPFN and CatBoost predict (\d+), (\d+) and \d+ of their (\d+) target rows of the "
                  r"minority class correctly, and (\d+), (\d+) and \d+ of the (\d+) of the majority class\.", q)
    assert m and z, q
    fehlt = [p for p in q.partition(FEHLT)[2].strip().rstrip(".").split("; ") if p and any(f" for {lab} " in p for _, lab, _ in TFM)]
    assert not fehlt, fehlt                                    # every class has correct target rows for both TFMs
    mi1, mi2, n_mi, ma1, ma2, n_ma = z.groups()
    mehr = f"{ma1} of the {n_ma}" if ma1 == ma2 else f"{ma1} and {ma2} of the {n_ma}"
    t = (f"On the {m.group(1)} datasets with a clearly smaller class, a minority share below {m.group(2)} "
         f"(Table~\\ref{{tab:app_daten_guete}}), TabICL and TabPFN predict {mi1} and {mi2} of the {n_mi} target rows of the minority "
         f"class correctly, but {mehr} of the majority class.\n")
    assert WEG not in t, t
    (GEN / "app_klassen_text_tfm.tex").write_text(t)


def main():
    x = load()
    ds = sorted(x.dataset.unique(), key=str.lower)
    guete(x, ds, TFM, "app_guete_tfm")
    asr_tfm(x, ds, TFM, "app_asr_ds_b_tfm")
    korrigiert(ds, TFM, "app_kontrolle_ds_b_tfm")
    kontrolle_text(x, TFM, "app_kontrolle_text_tfm")
    daten_guete(x, ds, TFM, "app_daten_guete_tfm")
    hill_text("app_ausserhalb_hill_text")
    credit_text(x, TFM, "app_daten_credit_text")
    for name in ("app_kontrolle", "app_sicherheit", "app_sicherheit_auc", "app_klassen", "app_ausserhalb"):
        ohne_trainierte(name)
    texte()
    guete(x, ds, TRAINIERT, "app_trainiert_guete").to_csv(CSV / "app_trainiert_guete.csv", index=False)
    asr(x, ds, TRAINIERT, "app_trainiert_asr")
    kontrolle(x, TRAINIERT, "app_trainiert_kontrolle")
    print((GEN / "app_trainiert_kontrolle.tex").read_text())
    print("->", GEN / "app_*_tfm.tex", GEN / "app_trainiert_*.tex")


if __name__ == "__main__":
    main()

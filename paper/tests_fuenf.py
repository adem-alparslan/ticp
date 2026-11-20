from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "paper" / "gen" / "csv"; GEN = CSV.parent
BUDGETS, N_BOOT = range(1, 6), 10_000
WORT = {1: "Eins", 2: "Zwei", 3: "Drei", 4: "Vier", 5: "Fuenf"}
TESTS = [("(i)", "(i)", r"ASR, TabICL $-$ TabPFN (pp)", "ICLPFN"), ("(ii)+(iii)", "(ii)", r"ASR, TabICL $-$ CatBoost (pp)", "ICLCAT"),
         ("(ii)+(iii)", "(iii)", r"ASR, TabPFN $-$ CatBoost (pp)", "PFNCAT"), ("(iv)", "(iv)", r"TabICL, attacked: $m - m_0$", "MAlle"),
         ("(v)", "(v)", r"TabPFN, flipped: $a_{F,P} - 0.5$", "AP")]
ptex = lambda p: "$<0.001$" if p < 0.001 else f"${p:.3f}$"
DATA, KEY = ROOT / "data" / "measurements.parquet", ["dataset", "fold", "target"]
PAARE = [("(i)", "TabICL $-$ TabPFN", "tabicl-n8", "tabpfn35-n8"), ("(ii)", "TabICL $-$ CatBoost", "tabicl-n8", "catboost"),
         ("(ii)", "TabPFN $-$ CatBoost", "tabpfn35-n8", "catboost")]
LABEL = {"tabicl-n8": "TabICL", "tabpfn35-n8": "TabPFN", "catboost": "CatBoost"}


def test(x, y, skala, nd, b):
    d = (skala * (x - y)).to_numpy()
    bs = d[np.random.default_rng([20240711, b]).integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(b=b, diff=d.mean(), ci_lo=lo, ci_hi=hi, DS=len(d), hi=int((d > 0).sum()), eq=int((d == 0).sum()),
                lo=int((d < 0).sum()), p=float(wilcoxon(x, y).pvalue), nd=nd)


def sensitivitaet(T):
    x = pd.read_parquet(DATA, columns=["model", "arm", "b", "clean_correct", "flipped"] + KEY)
    for c in ("model", "dataset", "arm"):
        x[c] = x[c].astype(str)
    ok = x[x.arm == "clean"].set_index(["model"] + KEY).clean_correct
    a = x[x.arm == "attack"].join(ok.rename("ok"), on=["model"] + KEY)
    a = a[a.ok == 1]
    alle = ok.unstack("model")
    alle = alle.index[(alle == 1).all(axis=1)]
    mengen = {"own": a, "common": a[pd.MultiIndex.from_frame(a[KEY]).isin(alle)], "without hill-valley": a[a.dataset != "hill-valley"]}
    rows, ds = [], []
    for menge, s in mengen.items():
        asr = s.groupby(["model", "dataset", "b"]).flipped.mean().unstack("model")
        ds.append(asr.rename(columns=LABEL).add_prefix("ASR_").reset_index().assign(set=menge))
        for b in BUDGETS:
            t = asr.xs(b, level="b")
            for frage, vergleich, m1, m2 in PAARE:
                j = t[[m1, m2]].dropna()
                rows.append(dict(set=menge, question=frage, comparison=vergleich, **test(j[m1], j[m2], 100, 1, b)))
    P = pd.DataFrame(rows)
    eigen, haupt = P[P.set == "own"], T[T.test.isin(["(i)", "(ii)", "(iii)"])]
    assert np.allclose(eigen["diff"], haupt["diff"]) and np.allclose(eigen.p, haupt.p), "own sets differ from the main tests"
    P = P[P.set != "own"].copy()
    P["p_holm"] = pd.concat([pd.Series(multipletests(g.p, method="holm")[1], index=g.index) for _, g in P.groupby("set")]).sort_index()
    teile = []
    for menge, titel in (("common", f"target rows correct without attack for all three models, {len(alle):,} target rows"),
                         ("without hill-valley", r"attacked target rows of each model, without \texttt{hill-valley}")):
        Q = P[P.set == menge]
        assert Q.DS.nunique() == 1, (menge, Q.DS.unique())
        zeilen = [f"{b} & " + " & ".join(f"${r.diff:+.1f}$ & {r.hi}/{r.eq}/{r.lo} & {ptex(r.p_holm)}" for r in Q[Q.b == b].itertuples()) + r" \\"
                  for b in BUDGETS]
        kopf = (" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{q} {v}}}" for q, v, _, _ in PAARE) + r" \\ "
                + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(PAARE))) + "\n$b$" + " & pp & higher/equal/lower & $p$" * len(PAARE))
        teile.append(f"\\par\\medskip\\noindent\\textit{{{titel}, {int(Q.DS.iloc[0])} datasets}}\\par\\nopagebreak\n"
                     "\\centering\\scriptsize\\maxbreite{\\begin{tabular}{@{}r" + "ccc" * len(PAARE) + "@{}}\\toprule\n" + kopf
                     + r" \\ \midrule" + "\n" + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}}\n")
    (GEN / "app_sensitivity.tex").write_text("\n".join(teile))
    P.drop(columns="nd").rename(columns={"p": "p_raw"}).to_csv(CSV / "app_sensitivity.csv", index=False)
    pd.concat(ds[1:], ignore_index=True).to_csv(CSV / "app_sensitivity_ds.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 40):
        print(P.drop(columns=["nd"]))


def main():
    tfm0, cat0 = pd.read_csv(CSV / "app_gepaart_tfm_ds.csv"), pd.read_csv(CSV / "app_gepaart_cat_ds.csv")
    m0, pb0 = pd.read_csv(CSV / "app_m_ds.csv"), pd.read_csv(CSV / "app_pfn_b_ds.csv")
    rows = []
    for b in BUDGETS:
        tfm, cat = tfm0[tfm0.b == b], cat0[cat0.b == b]
        ic, pc = cat[cat.pair == "TabICL-CatBoost"], cat[cat.pair == "TabPFN-CatBoost"]
        ma = m0[m0.b == b]
        ap = pb0[(pb0.arm == "attack") & (pb0.b == b) & (pb0.group == "flipped")]
        werte = [(tfm.ASR_TabICL, tfm.ASR_TabPFN, 100, 1), (ic.ASR_TabICL, ic.ASR_CatBoost, 100, 1),
                 (pc.ASR_TabPFN, pc.ASR_CatBoost, 100, 1), (ma.m_all, ma.start, 1, 3),
                 (ap.a_P, pd.Series(0.5, index=ap.index), 1, 3)]
        rows += [dict(question=q, test=t, comparison=c, **test(*w, b)) for (q, t, c, _), w in zip(TESTS, werte)]
    T = pd.DataFrame(rows)
    T["p_holm"] = multipletests(T.p, method="holm")[1]
    T["p_holm_je_frage"] = pd.concat([pd.Series(multipletests(g.p, method="holm")[1], index=g.index)
                                      for _, g in T.groupby("question")]).sort_index()
    for t, datei, paar in (("(i)", "app_gepaart_tfm", "TabICL-TabPFN"), ("(ii)", "app_gepaart_cat", "TabICL-CatBoost"),
                           ("(iii)", "app_gepaart_cat", "TabPFN-CatBoost")):
        C = pd.read_csv(CSV / f"{datei}.csv").query("pair == @paar").set_index("b")
        for _, r in T[T.test == t].iterrows():
            c = C.loc[r.b]
            assert np.allclose(r[["diff", "ci_lo", "ci_hi"]].astype(float), c[["diff", "ci_lo", "ci_hi"]].astype(float)), (paar, r.b)
            assert (r[["DS", "hi", "eq", "lo"]].to_numpy() == c[["n", "hi", "eq", "lo"]].to_numpy()).all(), (paar, r.b)
    s = T.set_index(["test", "b"]).p_holm < 0.05
    assert s.loc["(i)"].tolist() == [True, True, False, False, False] and not s.loc["(ii)"].any() and not s.loc["(iii)"].any()
    assert s.loc["(iv)"].all() and s.loc["(v)"].tolist() == [True, False, True, True, True], T[["test", "b", "p_holm"]]
    for tests, datei, diffkopf in ((("(i)",), "tab_test_a", r"TabICL $-$ TabPFN (pp)"), (("(ii)", "(iii)"), "tab_test_bc", r"TFM $-$ CatBoost (pp)"),
                                   (("(iv)",), "tab_test_d", "$m - m_0$"), (("(v)",), "tab_test_e", "$a_{F,P} - 0.5$")):
        zeilen = []
        for k, t in enumerate(tests):
            if k:
                zeilen.append(r"\midrule")
            for i, r in enumerate(T[T.test == t].sort_values("b").itertuples()):
                vorn = "" if len(tests) == 1 else (r.comparison.split(", ")[1].split(" ")[0] if i == 0 else "") + " & "
                zeilen.append(f"{vorn}{r.b} & ${r.diff:+.{r.nd}f}$ & {r.DS} & "
                              f"{r.hi}/{r.eq}/{r.lo} & {ptex(r.p_holm)} \\\\")
        kopf = ("TFM & " if len(tests) > 1 else "") + f"$b$ & {diffkopf} & DS & higher/equal/lower & $p$"
        spalten = ("l" if len(tests) > 1 else "") + "ccccc"
        (GEN / f"{datei}.tex").write_text("\\centering\\footnotesize\\begin{tabular}{@{}" + spalten + "@{}}\\toprule\n" + kopf
                                          + r" \\ \midrule" + "\n" + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}\n")
    zeilen = []
    for b in BUDGETS:
        mf = m0[(m0.b == b) & (m0.n_flipped >= 1)]
        d = (mf.flipped - mf.start_flipped).to_numpy()
        zeilen.append(f"{b} & ${d.mean():+.3f}$ & {len(d)} & {int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum())} \\\\")
    (GEN / "app_m_gekippt.tex").write_text("\\centering\\footnotesize\\begin{tabular}{@{}cccc@{}}\\toprule\n"
                                           r"$b$ & $m - m_0$ & DS & higher/equal/lower \\ \midrule" + "\n"
                                           + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}\n")
    T.drop(columns="nd").rename(columns={"p": "p_raw"}).to_csv(CSV / "tab_tests.csv", index=False)
    name = {t: n for _, t, _, n in TESTS}
    mac = {}
    for r in T.itertuples():
        k = f"{name[r.test]}{WORT[r.b]}"
        mac[f"zTestHolm{k}"] = "< 0.001" if r.p_holm < 0.001 else f"= {r.p_holm:.3f}"
        mac[f"zTestHolmWert{k}"] = "<0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
    (GEN / "zahlen_tests.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 40):
        print(T.drop(columns=["nd", "comparison"]))
    sensitivitaet(T)
    print("->", GEN / "tab_test_*.tex")


if __name__ == "__main__":
    main()

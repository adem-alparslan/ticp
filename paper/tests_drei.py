import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from tests_fuenf import BUDGETS, CSV, DATA, GEN, KEY, LABEL, WORT, ptex, test

TESTS = [("(i)", r"TabICL $-$ TabPFN (pp)", "ICLPFN", "tab_test_i"), ("(ii)", "$m - m_0$", "M", "tab_test_ii"),
         ("(iii)", "$a_{F,P} - 0.5$", "AP", "tab_test_iii")]
TFMS = ["tabicl-n8", "tabpfn35-n8"]
pmac = lambda p: "< 0.001" if p < 0.001 else f"= {p:.3f}"


def tabelle(R, kopf, groesse="footnotesize"):
    zeilen = [f"{r.b} & ${r.diff:+.{r.nd}f}$ & {r.DS} & {r.hi}/{r.eq}/{r.lo} & {ptex(r.p_holm)} \\\\"
              for r in R.sort_values("b").itertuples()]
    return (f"\\centering\\{groesse}\\begin{{tabular}}{{@{{}}ccccc@{{}}}}\\toprule\n$b$ & {kopf} & DS & higher/equal/lower & $p$"
            + r" \\ \midrule" + "\n" + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}\n")


def sensitivitaet(T, mac):
    x = pd.read_parquet(DATA, columns=["model", "arm", "b", "clean_correct", "flipped"] + KEY)
    for c in ("model", "dataset", "arm"):
        x[c] = x[c].astype(str)
    x = x[x.model.isin(TFMS)]
    ok = x[x.arm == "clean"].set_index(["model"] + KEY).clean_correct
    a = x[x.arm == "attack"].join(ok.rename("ok"), on=["model"] + KEY)
    a = a[a.ok == 1]
    beide = ok.unstack("model")
    beide = beide.index[(beide == 1).all(axis=1)]
    rows, ds = [], []
    for menge, s in (("own", a), ("common", a[pd.MultiIndex.from_frame(a[KEY]).isin(beide)])):
        asr = s.groupby(["model", "dataset", "b"]).flipped.mean().unstack("model")
        ds.append(asr.rename(columns=LABEL).add_prefix("ASR_").reset_index().assign(set=menge))
        for b in BUDGETS:
            t = asr.xs(b, level="b")[TFMS].dropna()
            rows.append(dict(set=menge, **test(t[TFMS[0]], t[TFMS[1]], 100, 1, b)))
    P = pd.DataFrame(rows)
    eigen, haupt = P[P.set == "own"], T[T.test == "(i)"]
    assert np.allclose(eigen["diff"], haupt["diff"]) and np.allclose(eigen.p, haupt.p), "own sets differ from test (i)"
    P = P[P.set == "common"].copy()
    P["p_holm"] = multipletests(P.p, method="holm")[1]
    assert P.DS.nunique() == 1 and (P.p_holm < 0.05).tolist() == [True, True, False, False, False], P
    (GEN / "app_sensitivity_tfm.tex").write_text(tabelle(P, TESTS[0][1], "scriptsize"))
    P.drop(columns="nd").rename(columns={"p": "p_raw"}).to_csv(CSV / "app_sensitivity_tfm.csv", index=False)
    ds[1].to_csv(CSV / "app_sensitivity_tfm_ds.csv", index=False)
    mac["zSensMenge"] = f"{len(beide):,}"
    mac["zSensPFNmehrEins"] = f"{-P.set_index('b')['diff'][1]:.1f}"
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(P.drop(columns=["nd"]))


def main():
    tfm0, m0, pb0 = (pd.read_csv(CSV / f"{n}.csv") for n in ("app_gepaart_tfm_ds", "app_m_ds", "app_pfn_b_ds"))
    rows = []
    for b in BUDGETS:
        tfm, ma = tfm0[tfm0.b == b], m0[m0.b == b]
        ap = pb0[(pb0.arm == "attack") & (pb0.b == b) & (pb0.group == "flipped")]
        werte = [(tfm.ASR_TabICL, tfm.ASR_TabPFN, 100, 1), (ma.m_all, ma.start, 1, 3),
                 (ap.a_P, pd.Series(0.5, index=ap.index), 1, 3)]
        rows += [dict(test=t, **test(*w, b)) for (t, _, _, _), w in zip(TESTS, werte)]
    T = pd.DataFrame(rows)
    T["p_holm"] = multipletests(T.p, method="holm")[1]
    T["p_holm_je_frage"] = pd.concat([pd.Series(multipletests(g.p, method="holm")[1], index=g.index)
                                      for _, g in T.groupby("test")]).sort_index()
    s = T.set_index(["test", "b"]).p_holm < 0.05
    assert s.loc["(i)"].tolist() == [True, True, False, False, False] and s.loc["(ii)"].all(), T[["test", "b", "p_holm"]]
    assert s.loc["(iii)"].tolist() == [True, False, True, True, True], T[["test", "b", "p_holm"]]
    roh = T[(T.p < 0.05) & (T.p_holm >= 0.05)]
    assert [(r.test, r.b) for r in roh.itertuples()] == [("(iii)", 2)] and (roh.p_holm_je_frage < 0.05).all(), roh
    for t, kopf, _, datei in TESTS:
        (GEN / f"{datei}.tex").write_text(tabelle(T[T.test == t], kopf))
    T.drop(columns="nd").rename(columns={"p": "p_raw"}).to_csv(CSV / "tab_tests_drei.csv", index=False)
    name = {t: n for t, _, n, _ in TESTS}
    mac = {}
    for r in T.itertuples():
        mac[f"zHolm{name[r.test]}{WORT[r.b]}"] = pmac(r.p_holm)
        mac[f"zRoh{name[r.test]}{WORT[r.b]}"] = pmac(r.p)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 40):
        print(T.drop(columns=["nd"]))
    sensitivitaet(T, mac)
    (GEN / "zahlen_tests_drei.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    print("->", GEN / "tab_test_i.tex", GEN / "tab_test_ii.tex", GEN / "tab_test_iii.tex")


if __name__ == "__main__":
    main()

"""The 25 tests of ticp_alparslan_v5.0_de (Section 2.5), five tests per question, one per number of poison rows b.

  (1a)  TabICL: the ASR in the attack arm against the share of the same attacked target rows that flip in the control
        arm (the copies carry the true label), per dataset, paired over the datasets
  (1b)  the same for TabPFN
  (2a)  TabICL, m of the flipped target rows against 1, the threshold of the read-out over the nearest class centroid
        reported for TabICLv2: per dataset the mean m over its flipped target rows, over the datasets with flipped rows
  (2b)  TabICL, m of the held target rows (correct without attack, not flipped under attack) against 1, likewise
  (3)   TabPFN, the weight a_{F,P} of the poison rows at the flipped target rows against 0.5

Two-sided Wilcoxon signed-rank tests over the datasets, Holm within each test over its five numbers of poison rows.
Writes gen/tab_test_i_v5.tex (1a, 1b), gen/tab_test_ii_v5.tex (2a, 2b), gen/tab_test_iii_v5.tex (3),
gen/zahlen_tests_v5.tex and gen/csv/tab_tests_v5.csv, gen/csv/tab_test_i_v5_ds.csv, gen/csv/tab_test_ii_v5_ds.csv.
History: until 2026-11-13 test (ii) was m against m_0 over all attacked target rows (the push, now descriptive); until
2026-11-14 the tests were named (i) to (iii), the held target rows were not tested and Holm ran over all 20 tests.
The files of tests_drei.py stay untouched.
Table 2 (tests (1a), (1b)) has one row per b with the two TFMs side by side, because DS and p are the same in every
row they stand in its caption. The table heads are German like the manuscript.

    python paper/tests_v5.py
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from tests_drei import TESTS, pmac
from tests_fuenf import BUDGETS, CSV, DATA, GEN, KEY, LABEL, WORT, ptex, test

TFMS = ["tabicl-n8", "tabpfn35-n8"]
KURZ = {"tabicl-n8": "ICL", "tabpfn35-n8": "PFN"}
ARM = {"tabicl-n8": "(1a)", "tabpfn35-n8": "(1b)"}
GRUPPE = {"flipped": ("(2a)", "flipped"), "held": ("(2b)", "held")}
proz = lambda v: f"{v / 100:.3f}"                   # a share as a decimal fraction, never in percent; three decimals for the control arm


def arme():
    """Per model, dataset and b: the share of the attacked target rows that flip, in the attack and in the control arm."""
    x = pd.read_parquet(DATA, columns=["model", "arm", "b", "clean_correct", "flipped"] + KEY)
    for c in ("model", "dataset", "arm"):
        x[c] = x[c].astype(str)
    x = x[x.model.isin(TFMS) & x.arm.isin(["attack", "control"]) & (x.clean_correct == 1)]
    n = x.groupby(["model", "dataset", "b", "arm"]).size().unstack("arm")
    assert (n.attack == n.control).all(), "the two arms do not cover the same target rows"
    return x.groupby(["model", "dataset", "b", "arm"]).flipped.mean().unstack("arm").reset_index()


def regel():
    """Tests (2a) and (2b): per dataset, b and group (flipped, held: correct without attack and flipped or not flipped
    under attack), the mean m over the TabICL target rows of the group and the share of them with m > 1. A dataset
    without a target row in a group at that b is left out of that group."""
    x = pd.read_parquet(DATA, columns=["model", "arm", "b", "m", "clean_correct", "flipped"] + KEY)
    for c in ("model", "dataset", "arm"):
        x[c] = x[c].astype(str)
    x = x[(x.model == TFMS[0]) & (x.arm == "attack") & (x.clean_correct == 1)]
    assert x.m.notna().all()
    x = x.assign(group=x.flipped.map({1: "flipped", 0: "held"}), m_above_1=(x.m > 1).astype(float))
    return (x.groupby(["dataset", "b", "group"])
            .agg(n=("m", "size"), mean_m=("m", "mean"), share_m_above_1=("m_above_1", "mean")).reset_index())


def tabelle_i(R):
    """Tests (1a) and (1b): one row per b, the two TFMs side by side, each with the mean share of flipped attacked target rows in
    the attack and in the control arm, the counts of datasets and the Holm-corrected p. DS is the same in every row and stands
    in the caption."""
    assert (R.DS == R.DS.iloc[0]).all(), "the caption states one DS for all ten tests"
    zelle = lambda r: f"{proz(r['attack'])} / {proz(r['control'])} & {r['hi']}/{r['eq']}/{r['lo']} & {ptex(r['p_holm'])}"
    zeilen = [f"{b} & " + " & ".join(zelle(R[(R.model == m) & (R.b == b)].iloc[0]) for m in TFMS) + r" \\" for b in BUDGETS]
    gruppen = " & ".join(r"\multicolumn{3}{c}{" + LABEL[m] + "}" for m in TFMS)
    linien = "".join(rf"\cmidrule({'lr' if i < len(TFMS) - 1 else 'l'}){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(TFMS)))
    spalten = r" & attack / control & higher/equal/lower & $p$" * len(TFMS)
    return ("\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}c" + "ccc" * len(TFMS) + "@{}}\\toprule\n"
            + " & " + gruppen + r" \\ " + linien + "\n"
            + "$b$" + spalten + r" \\ \midrule" + "\n"
            + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}\n")


def tabelle_m(R):
    """Tests (2a) and (2b): one row per b, the flipped and the held target rows side by side."""
    zelle = lambda r: f"${r['diff']:+.{r['nd']}f}$ & {r['DS']} & {r['hi']}/{r['eq']}/{r['lo']} & {ptex(r['p_holm'])}"
    tests = [(t, wort) for t, wort in GRUPPE.values()]
    zeilen = [f"{b} & " + " & ".join(zelle(R[(R.test == t) & (R.b == b)].iloc[0]) for t, _ in tests) + r" \\" for b in BUDGETS]
    gruppen = " & ".join(rf"\multicolumn{{4}}{{c}}{{{wort}, Test {t}}}" for t, wort in tests)
    linien = "".join(rf"\cmidrule({'lr' if i < len(tests) - 1 else 'l'}){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(tests)))
    spalten = r" & $m - 1$ & DS & higher/equal/lower & $p$" * len(tests)
    return ("\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}\\begin{tabular}{@{}c" + "cccc" * len(tests) + "@{}}\\toprule\n"
            + " & " + gruppen + r" \\ " + linien + "\n"
            + "$b$" + spalten + r" \\ \midrule" + "\n"
            + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}\n")


def tabelle_de(R, kopf):
    """Test (3) laid out as tests_drei.tabelle, with the heads in German."""
    zeilen = [f"{r.b} & ${r.diff:+.{r.nd}f}$ & {r.DS} & {r.hi}/{r.eq}/{r.lo} & {ptex(r.p_holm)} \\\\" for r in R.sort_values("b").itertuples()]
    return ("\\centering\\footnotesize\\begin{tabular}{@{}ccccc@{}}\\toprule\n$b$ & " + kopf
            + r" & DS & higher/equal/lower & $p$ \\ \midrule" + "\n" + "\n".join(zeilen) + "\n\\bottomrule\\end{tabular}\n")


def main():
    A, M = arme(), regel()
    tfm0, pb0 = (pd.read_csv(CSV / f"{n}.csv") for n in ("app_gepaart_tfm_ds", "app_pfn_b_ds"))
    for mod in TFMS:      # the attack arm must be the ASR the appendix reports
        a = A[A.model == mod].set_index(["dataset", "b"]).attack
        ref = tfm0.set_index(["dataset", "b"])[f"ASR_{LABEL[mod]}"]
        assert np.allclose(a.sort_index(), ref.sort_index()), mod
    rows = []
    for mod in TFMS:
        for b in BUDGETS:
            g = A[(A.model == mod) & (A.b == b)]
            rows.append(dict(test=ARM[mod], model=mod, attack=100 * g.attack.mean(), control=100 * g.control.mean(),
                             **test(g.attack, g.control, 100, 1, b)))
    for b in BUDGETS:
        for grp, (t, _) in GRUPPE.items():
            ma = M[(M.b == b) & (M.group == grp)]
            rows.append(dict(test=t, model=TFMS[0], **test(ma.mean_m, pd.Series(1.0, index=ma.index), 1, 3, b)))
        ap = pb0[(pb0.arm == "attack") & (pb0.b == b) & (pb0.group == "flipped")]
        rows.append(dict(test="(3)", model=TFMS[1], **test(ap.a_P, pd.Series(0.5, index=ap.index), 1, 3, b)))
    T = pd.DataFrame(rows)
    assert len(T) == 25 and (T.groupby("test").size() == 5).all()
    T["p_holm"] = T.groupby("test").p.transform(lambda p: multipletests(p, method="holm")[1])   # Holm within each test
    # Section 3.2.1 states p < 0.001 for all of (2a), and (2b) significant at one to three poison rows only
    assert (T[T.test == "(2a)"].p_holm < 0.001).all()
    assert (T[(T.test == "(2b)") & (T.b <= 3)].p_holm < 0.05).all() and (T[(T.test == "(2b)") & (T.b >= 4)].p_holm >= 0.05).all()
    (GEN / "tab_test_i_v5.tex").write_text(tabelle_i(T[T.test.isin(ARM.values())]))
    (GEN / "tab_test_ii_v5.tex").write_text(tabelle_m(T[T.test.isin(["(2a)", "(2b)"])]))
    (GEN / "tab_test_iii_v5.tex").write_text(tabelle_de(T[T.test == "(3)"], TESTS[2][1]))
    M.assign(model=LABEL[TFMS[0]]).to_csv(CSV / "tab_test_ii_v5_ds.csv", index=False)
    T.drop(columns="nd").rename(columns={"p": "p_raw"}).to_csv(CSV / "tab_tests_v5.csv", index=False)
    A.assign(model=A.model.map(LABEL)).rename(columns={"attack": "flipped_attack", "control": "flipped_control"}) \
        .to_csv(CSV / "tab_test_i_v5_ds.csv", index=False)
    name = {"(2a)": "M", "(2b)": "MH", "(3)": "AP"}
    mac = {}
    for r in T.itertuples():
        kern = f"Arm{KURZ[r.model]}" if r.test in ARM.values() else name[r.test]
        mac[f"zHolm{kern}{WORT[r.b]}"] = pmac(r.p_holm)
        mac[f"zRoh{kern}{WORT[r.b]}"] = pmac(r.p)
    mac["zHolmArmMax"] = pmac(T[T.test.isin(ARM.values())].p_holm.max())
    (GEN / "zahlen_tests_v5.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 40):
        print(T.drop(columns=["nd", "ci_lo", "ci_hi"]))
    print("->", GEN / "tab_test_i_v5.tex", GEN / "zahlen_tests_v5.tex")


if __name__ == "__main__":
    main()

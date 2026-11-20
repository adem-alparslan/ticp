"""Numbers of ticp_alparslan_v5.0_de.tex that no other generator writes (Sections 2 to 4).

Reads gen/csv/*.csv (anhang.ipynb, anhang_ds.py, tests_v5.py) and data/*.parquet (via zahlen_referenz.load) and writes
gen/zahlen_text.tex. Every macro is named after the place it is used; values are rounded as printed in the text.
"""
import math
import re

import pandas as pd

from zahlen_referenz import BUDGETS, CSV, GEN, ROOT, W, load

ICL, PFN, CAT = "tabicl-n8", "tabpfn35-n8", "catboost"
CREDIT = "creditcard"


def main():
    mac = {}

    def put(name, value):
        assert name not in mac, name
        mac[name] = value

    # Section 2.2: data
    d = pd.read_csv(CSV / "app_daten.csv").set_index("dataset")
    put("zFeatMin", f"{int(d.features.min())}")
    put("zFeatMax", f"{int(d.features.max())}")
    put("zRowsMin", f"{int(d.rows.min()):,}")  # Section 2.2: rows before the reduction to at most 3,000
    put("zRowsMax", f"{int(d.rows.max()):,}")
    put("zMinorityMinPct", f"{100 * d.minority.min():.1f}")
    put("zCtxMin", f"{int(d.n_ctx.min())}")
    put("zGiftAnteilMax", f"{100 * max(BUDGETS) / d.n_ctx.min():.1f}")  # Section 1: largest share of the poison rows in a context
    put("zCreditZiele", f"{int(d.loc[CREDIT, 'targets'])}")
    put("zDupZiele", f"{int(d.dup_targets.sum())}")

    # Section 2.3 and 3.1: attacked target rows, predictions behind the tests
    put("zVorhersagen", f"{int(d.targets.sum()) * (1 + 2 * len(BUDGETS)):,}")

    x = load()
    k = x[(x.arm == "control") & (x.clean_correct == 1) & (x.flipped == 1) & x.model.isin([ICL, PFN])]
    assert len(k) == 1 and k.model.iloc[0] == ICL, k
    put("zKontrolleKipptB", {1: "einer", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf"}[int(k.b.iloc[0])])
    b5 = max(BUDGETS)
    for m, t in ((ICL, "ICL"), (PFN, "PFN")):
        h = x[(x.model == m) & (x.arm == "attack") & (x.b == b5) & (x.clean_correct == 1) & (x.flipped == 0)]
        put(f"z{t}haltenFuenf", f"{len(h)}")
        put(f"z{t}haltenDupFuenf", f"{int((h.n_dup0 > 0).sum())}")
    # Table B1: at b = 5 the held target rows hold the more clearly, the more duplicates with the true label they have
    h = x[x.model.isin([ICL, PFN]) & (x.arm == "attack") & (x.b == b5) & (x.clean_correct == 1) & (x.flipped == 0)]
    assert h.n_dup0.corr(h.pi_T, method="spearman") > 0.5, "Table B1: pi_T of the held rows rises with the duplicates"
    viel = h[h.n_dup0 >= 100]
    assert set(viel.dataset) == {"mammography"}, set(viel.dataset)
    put("zHaltenPiTOhneMax", f"{h[h.n_dup0 == 0].pi_T.max():.2f}")
    put("zHaltenPiTVielMin", f"{viel.pi_T.min():.2f}")
    # Table B1 and Section 4.2: one poison row flips no attacked target row with a duplicate, five only those with few
    du = x[x.model.isin([ICL, PFN]) & (x.arm == "attack") & (x.clean_correct == 1) & (x.n_dup0 > 0)]
    assert (du[du.b == 1].groupby("model").size() == int(d.dup_targets.sum())).all(), "Table B1: all target rows with a duplicate are attacked"
    assert du[du.b == 1].flipped.sum() == 0, "Section 4.2: one poison row flips no target row with a duplicate"
    d5 = du[du.b == b5]
    assert d5[d5.flipped == 1].n_dup0.max() <= d5[d5.flipped == 0].n_dup0.min(), "Table B1: at b = 5 only few duplicates give way"
    put("zDupGekipptFuenfMax", f"{int(d5[d5.flipped == 1].n_dup0.max())}")

    # Section 3.1: the arms, the spread over the datasets, the side effects
    t = pd.read_csv(CSV / "tab_tests_v5.csv")
    for m, tt, lab in ((ICL, "(1a)", "ICL"), (PFN, "(1b)", "PFN")):
        r = t[(t.test == tt) & (t.model == m) & (t.b == 1)].iloc[0]
        put(f"z{lab}armHoeherEins", f"{int(r.hi)}")
    a = pd.read_csv(CSV / "app_asr_ds_b.csv").set_index("dataset")
    # Section 4.1: the lead of TabPFN over TabICL shrinks with the budget, in percentage points of the mean ASR
    put("zDiffPFNICLEins", f"{100 * (a.TabPFN_b1.mean() - a.TabICL_b1.mean()):.0f}")
    put("zDiffPFNICLDrei", f"{100 * (a.TabPFN_b3.mean() - a.TabICL_b3.mean()):.0f}")
    # \zICLasrZwei comes from anhang.ipynb
    assert a.TabICL_b2.mean() > a.TabPFN_b1.mean(), "Section 4.1: two poison rows in TabICL flip more than one in TabPFN"
    for lab in ("ICL", "PFN"):
        s = a[f"Tab{lab}_b1"]
        put(f"z{lab}dsHalbEins", f"{int((s >= 0.5).sum())}")
        put(f"z{lab}dsZehntelEins", f"{int((s < 0.1).sum())}")
    diff = [100 * (a[f"TabPFN_b{b}"] - a[f"TabICL_b{b}"]).mean() for b in (1, 2)]
    put("zDiffPFNICL", f"{round(sum(diff) / len(diff))}")
    put("zRhoTFMEins", f"{a.TabICL_b1.corr(a.TabPFN_b1, method='spearman'):.2f}")
    au = pd.read_csv(CSV / "app_ausserhalb.csv")
    au = au[au.model.isin(["TabICL", "TabPFN"])]
    mx = au.d_auc.abs().max()
    put("zDaucMittelMax", f"{math.ceil(mx * 1000) / 1000:.3f}")
    rep = pd.read_csv(CSV / "app_kontrolle_reparatur.csv")
    for lab in ("ICL", "PFN"):
        r = rep[(rep.model == f"Tab{lab}") & (rep.b == 1)].iloc[0]
        put(f"z{lab}korrigiertEins", f"{100 * r.corrected_mean:.0f}")

    # Section 3.2.1: TabICL
    m1 = t[(t.test == "(2a)") & (t.b == 1)].iloc[0]
    put("zMgekipptMinusEins", f"{m1['diff']:+.3f}")
    bw = pd.read_csv(CSV / "app_m_bewegung_ds.csv")
    c = bw[(bw.dataset == CREDIT) & (bw.b == b5) & (bw.quantity == "cos_muF_muF0")].flipped.iloc[0]
    put("zMcosMuFCredit", f"{c:.2f}")
    # the two centroids over all attacked target rows, as in Table app_d_zentroide (anhang_ds.py): smallest mean, rounded down
    nb = pd.read_csv(CSV / "app_m_liste_b.csv")[["dataset", "b", "n_flipped", "n_held"]]
    j = bw[bw.quantity != "cos_z_z0"].merge(nb, on=["dataset", "b"], validate="many_to_one")
    j["all"] = (j.flipped.fillna(0) * j.n_flipped + j.held.fillna(0) * j.n_held) / (j.n_flipped + j.n_held)
    put("zMcosMuAlleMin", f"{math.floor(1000 * j.groupby(['quantity', 'b'])['all'].mean().min()) / 1000:.3f}")

    # Section 3.2.2: TabPFN
    v = pd.read_csv(CSV / "app_pfn_b.csv")
    put("zPFNaFnull", f"{v[(v.arm == 'clean')].a_F.iloc[0]:.3f}")
    f1 = v[(v.arm == "attack") & (v.b == 1) & (v.group == "flipped")].iloc[0]
    f2 = v[(v.arm == "attack") & (v.b == 2) & (v.group == "flipped")].iloc[0]
    put("zPFNaPEins", f"{f1.a_P:.3f}")
    put("zPFNaPZwei", f"{f2.a_P:.3f}")
    put("zPFNfalschEins", f"{f1.a_F + f1.a_P:.3f}")
    for b, w in ((1, "Eins"), (2, "Zwei")):
        r = t[(t.test == "(3)") & (t.b == b)].iloc[0]
        put(f"zPFNaPDS{w}", f"{int(r.DS)}")
        put(f"zPFNaPunter{w}", f"{int(r.lo)}")
        put(f"zPFNaPueber{w}", f"{int(r.hi)}")
    # Figure fig_vote: the parts of the vote in percent at one to three poison rows (as the labels of the figure)
    pc = lambda v: f"{100 * v:.0f}"
    put("zPFNohneFalsch", pc(v[v.arm == "clean"].a_F.iloc[0]))
    for b in (1, 2, 3):
        fb = v[(v.arm == "attack") & (v.b == b) & (v.group == "flipped")].iloc[0]
        hb = v[(v.arm == "attack") & (v.b == b) & (v.group == "held")].iloc[0]
        cb = v[(v.arm == "control") & (v.b == b) & (v.group == "all")].iloc[0]
        put(f"zPFNgift{W[b]}", pc(fb.a_P))
        put(f"zPFNecht{W[b]}", pc(fb.a_F))
        put(f"zPFNhalt{W[b]}", pc(hb.a_F + hb.a_P))
        put(f"zPFNkontr{W[b]}", pc(cb.a_P))
        put(f"zPFNkontrFalsch{W[b]}", pc(cb.a_F))
        # Section 3.2.2: what is left for the true label of the flipped target rows, as 100 minus the figure's wrong-label total
        assert abs(fb.a_T + fb.a_F + fb.a_P - 1) < 1e-6, "Figure fig_vote: the parts of the vote sum to one"
        put(f"zPFNwahr{W[b]}", f"{100 - round(100 * (fb.a_F + fb.a_P))}")
    # Section 3.2.2: at three poison rows the genuine rows of the wrong class still get about three times their weight
    # without attack, and in held target rows the poison rows get only about a fifth
    af3 = pd.read_csv(CSV / "app_pfn_aF.csv").query("b == 3 and comparison == 'flipped'").iloc[0]
    assert 2.5 < af3.a_F / af3.a_F0 < 3.5, f"Section 3.2.2: about three times, {af3.a_F / af3.a_F0:.2f}"
    h3 = v[(v.arm == "attack") & (v.b == 3) & (v.group == "held")].iloc[0]
    assert 0.15 < h3.a_P < 0.25, f"Section 3.2.2: about a fifth, {h3.a_P:.3f}"
    af = pd.read_csv(CSV / "app_pfn_aF.csv")
    fl = af[(af.b == 1) & (af.comparison == "flipped")].iloc[0]
    he = af[(af.b == 1) & (af.comparison == "held")].iloc[0]
    put("zPFNaFnullGekippt", f"{fl.a_F0:.3f}")
    put("zPFNaFGgekippt", f"{fl.a_F:.3f}")
    put("zPFNaFnullGehalten", f"{he.a_F0:.3f}")
    # the same in percent of the vote, as the text gives them
    put("zPFNaFnullGekipptPct", f"{100 * fl.a_F0:.0f}")
    put("zPFNaFnullGehaltenPct", f"{100 * he.a_F0:.0f}")
    put("zPFNaPZweiPct", f"{100 * f2.a_P:.0f}")
    assert f"{100 * fl.a_F:.0f}" == f"{100 * f1.a_F:.0f}", "Section 3.2.2: a_F,G of the flipped differs between the two tables"
    st = pd.read_csv(CSV / "app_rang_staerkste.csv")
    put("zPFNstaerksteEins", f"{st[(st.b == 1) & (st.group == 'flipped')].strongest_pct.iloc[0]:.0f}")

    # Section 3.3: trained models
    for ds, tag in (("Pulsar-Dataset-HTRU2", "Pulsar"), ("wilt", "Wilt"), ("phoneme", "Phoneme"), ("electricity", "Electricity")):
        for col, lab in (("CatBoost", "CAT"), ("TabICL", "ICL"), ("TabPFN", "PFN")):
            put(f"z{lab}asr{tag}", f"{100 * a.loc[ds, f'{col}_b1']:.0f}")
    for b in (1, 2, 3):
        for lab in ("ICL", "PFN"):
            put(f"zRhoCAT{lab}{W[b]}", f"{a[f'CatBoost_b{b}'].corr(a[f'Tab{lab}_b{b}'], method='spearman'):.2f}")

    # Section 3.3: why the trained models differ; the random forest flips every attacked target row whose context holds
    # no genuine duplicate with the true label, the held ones all hold one; the logistic regression barely moves pi_T
    k = ["model", "dataset", "fold", "target"]
    rf = x[(x.model == "random-forest") & (x.arm == "attack") & (x.b == 1) & (x.clean_correct == 1)]
    assert rf[rf.n_dup0 == 0].flipped.all() and (rf[rf.flipped == 0].n_dup0 > 0).all(), "random forest: duplicates"
    put("zRFhaltenEins", f"{int((rf.flipped == 0).sum())}")
    lr0 = x[(x.model == "logistic-regression") & (x.arm == "clean")][k + ["pi_T"]].rename(columns={"pi_T": "pi_T0"})
    lr = x[(x.model == "logistic-regression") & (x.arm == "attack") & (x.b == 1) & (x.clean_correct == 1)].merge(lr0, on=k)
    put("zLRdeltaPiEins", f"{(lr.pi_T0 - lr.pi_T).median():.3f}")

    # Section 3.2.2: of the flipped TabPFN target rows, the share whose poison rows alone get more than half of the weight
    pa = pd.read_parquet(ROOT / "data" / "measurements.parquet",
                         columns=["model", "dataset", "arm", "b", "clean_correct", "flipped", "a_copy"])
    for c in ("model", "dataset", "arm"):
        pa[c] = pa[c].astype(str)
    pa = pa[(pa.model == PFN) & (pa.arm == "attack") & (pa.clean_correct == 1) & (pa.flipped == 1)]
    for b_, w in ((1, "Eins"), (3, "Drei")):
        s_ = (pa[pa.b == b_].a_copy > 0.5).groupby(pa[pa.b == b_].dataset).mean()  # per dataset, then over the datasets
        put(f"zPFNaPueberZiel{w}", f"{100 * s_.mean():.0f}")

    # Section 2.5: Holm over all 25 tests of tests_v5.py instead of per test over its five budgets; only (3) at b = 2 changes
    t = pd.read_csv(CSV / "tab_tests_v5.csv")
    p = t.p_raw.to_list(); order = sorted(range(len(p)), key=p.__getitem__); h, run = [0.0] * len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); h[i] = min(1.0, run)
    t["p_holm_all"] = h
    changed = t[(t.p_holm < 0.05) != (t.p_holm_all < 0.05)]
    assert len(t) == 25 and len(changed) == 1 and changed.test.iloc[0] == "(3)" and int(changed.b.iloc[0]) == 2, changed
    put("zHolmGesamtAPZwei", f"{changed.p_holm_all.iloc[0]:.3f}")

    # no macro may also come from another generator (anhang.ipynb writes every \\z macro the manuscript uses that it knows)
    andere = set()
    for f in GEN.glob("zahlen*.tex"):
        if f.name != "zahlen_text.tex":
            andere |= set(re.findall(r"\\newcommand\{\\(z[A-Za-z]+)\}", f.read_text()))
    assert not andere & set(mac), f"also defined elsewhere: {sorted(andere & set(mac))}"
    (GEN / "zahlen_text.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    return mac


if __name__ == "__main__":
    for k, v in main().items():
        print(k, v)

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "gen"; CSV = GEN / "csv"
MIN_N, MIN_DS, BUDGETS = 1, 5, range(1, 6)
KEY = ["dataset", "fold", "target"]
W = {1: "Eins", 2: "Zwei", 3: "Drei", 4: "Vier", 5: "Fuenf"}
BEW = ["cos_z_z0", "cos_muT_muT0", "cos_muF_muF0"]
KURZ = {"climate-model-simulation-crashes": "climate-model", "banknote-authentication": "banknote", "Pulsar-Dataset-HTRU2": "Pulsar-Dataset",
        "polish_companies_bankruptcy": "polish_companies", "hazelnut-spread-contaminant": "hazelnut-spread", "electrical_grid_stability": "electrical_grid"}
tn = lambda ds: "\\texttt{" + KURZ.get(ds, ds).replace("_", "\\_") + "}"
f = lambda x, nd: f"{x:.{nd}f}"
msd = lambda s, nd=3: "--" if len(s) == 0 else (f"${s.mean():.{nd}f}$" if len(s) < MIN_DS else f"${s.mean():.{nd}f} \\pm {s.std():.{nd}f}$")


def tab(spec, head, rows, breit=True):
    t = "\\begin{tabular}{" + spec + "}\\toprule\n" + head + r" \\ \midrule" + "\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}"
    return "\\centering\\scriptsize" + ("\\maxbreite{" + t + "}" if breit else t) + "\n"


def stapel(teile):
    return "\n".join(f"\\par\\medskip\\noindent\\textit{{{titel}}}\\par\\nopagebreak\n" + tex for titel, tex in teile)


def innen(i):
    """Within each dataset of one number of poison rows: Spearman's rho of m with pi_F over its attacked target rows, and the AUC with which m
    separates the flipped from the held target rows (only where both groups occur)."""
    rho = pd.Series({ds: spearmanr(s.m, s.pi_F).correlation for ds, s in i.groupby("dataset") if len(s) >= 5 and s.m.std() > 0 and s.pi_F.std() > 0}, dtype=float)
    auc = pd.Series({ds: roc_auc_score(s.flipped, s.m) for ds, s in i.groupby("dataset") if 0 < s.flipped.mean() < 1}, dtype=float)
    return rho, auc


def load():
    df = pd.read_parquet(ROOT / "data" / "measurements.parquet")
    d = df[df.model == "tabicl-n8"]
    c = d[d.arm == "clean"][KEY + ["m", "clean_correct"]].rename(columns={"m": "m0"})
    a = d[d.arm == "attack"][KEY + ["b", "m", "pi_T", "flipped"] + BEW]
    k = d[d.arm == "control"][KEY + ["b", "m"] + BEW].rename(columns={"m": "m_c", **{c: c + "_c" for c in BEW}})
    x = a.merge(k, on=KEY + ["b"]).merge(c, on=KEY)
    x = x[x.clean_correct == 1].copy()
    x["dataset"] = x.dataset.astype(str); x["pi_F"] = 1 - x.pi_T; x["push"] = x.m - x.m0; x["push_c"] = x.m_c - x.m0
    assert x[["m", "m0", "m_c"] + BEW + [c + "_c" for c in BEW]].notna().all().all(), "m or the movement is missing"
    return x


def main():
    CSV.mkdir(exist_ok=True)
    x = load(); mac = {}
    per = {}
    for b in BUDGETS:
        rows = []
        for ds, s in x[x.b == b].groupby("dataset"):
            fl, he = s[s.flipped == 1], s[s.flipped == 0]
            rows.append(dict(dataset=ds, n=len(s), n_flipped=len(fl), n_held=len(he), asr=s.flipped.mean(), start=s.m0.mean(), m_all=s.m.mean(),
                             control=s.m_c.mean(), push_all=s.push.mean(), flipped=fl.m.mean(), held=he.m.mean(), start_flipped=fl.m0.mean(), start_held=he.m0.mean(),
                             push_flipped=fl.push.mean(), push_held=he.push.mean(),
                             rule_flipped=100 * (fl.m > 1).mean() if len(fl) else np.nan, rule_held=100 * (he.m <= 1).mean() if len(he) else np.nan,
                             start_below=100 * (s.m0 < 1).mean()))
        per[b] = pd.DataFrame(rows)
    QU = [("m", "flipped", "held", "$m$ under attack"), ("m0", "start_flipped", "start_held", "$m_0$ without attack, the same target rows"),
          ("push", "push_flipped", "push_held", "push $m - m_0$")]
    T = []
    for b in BUDGETS:
        t = per[b]; ok = t[(t.n_held >= MIN_N) & (t.n_flipped >= MIN_N)]
        for key, cf, ch, _ in QU:
            vf, vh = t[t.n_flipped >= MIN_N][cf], t[t.n_held >= MIN_N][ch]
            T.append(dict(b=b, quantity=key, flipped=vf.mean(), flipped_sd=vf.std(), DS_flipped=len(vf), held=vh.mean(), held_sd=vh.std(), DS_held=len(vh),
                          pairs=len(ok), hi=int((ok[cf] > ok[ch]).sum()), eq=int((ok[cf] == ok[ch]).sum()), lo=int((ok[cf] < ok[ch]).sum()),
                          # for the push also over all attacked target rows of a dataset, the same rows at every number of poison rows
                          **({"above0_flipped": int((vf > 0).sum()), "above0_held": int((vh > 0).sum()), "all": t.push_all.mean(), "all_sd": t.push_all.std(),
                              "DS_all": len(t), "above0_all": int((t.push_all > 0).sum())} if key == "push" else {})))
    T = pd.DataFrame(T)
    teile = []
    for key, cf, ch, titel in QU:
        rows = []
        for r in T[T.quantity == key].itertuples():
            t = per[r.b]; vf, vh = t[t.n_flipped >= MIN_N][cf], t[t.n_held >= MIN_N][ch]
            alle = f" & {msd(t.push_all)}" if key == "push" else ""
            ueber = f" & {int(r.above0_all)}/{int(r.above0_flipped)}/{int(r.above0_held)}" if key == "push" else ""
            rows.append(f"{r.b}{alle} & {msd(vf)} & {msd(vh)} & {r.DS_flipped}/{r.DS_held}{ueber} & {r.pairs} & {r.hi}/{r.eq}/{r.lo} \\\\")
        kopf = (r"$b$" + (" & all" if key == "push" else "") + r" & flipped & held & DS flipped/held"
                + (r" & above 0, all/flipped/held" if key == "push" else "") + r" & pairs & higher/equal/lower")
        # the push block is wider by its column all, \maxbreite fits it to the line width
        teile.append((titel, tab("@{}rccccccc@{}" if key == "push" else "@{}rccccc@{}", kopf, rows, breit=(key == "push"))))
    (GEN / "app_m.tex").write_text(stapel(teile)); T.to_csv(CSV / "app_m.csv", index=False)
    pd.concat([per[b].assign(b=b) for b in BUDGETS], ignore_index=True).to_csv(CSV / "app_m_ds.csv", index=False)
    rows, N = [], []
    for b in BUDGETS:
        t = per[b]; d_ = t.control - t.start
        N.append(dict(b=b, DS=len(t), m0=t.start.mean(), m0_sd=t.start.std(), m0_below_1=t.start_below.mean(), m0_below_1_sd=t.start_below.std(),
                      control=t.control.mean(), control_sd=t.control.std(),
                      lo=int((d_ < 0).sum()), eq=int((d_ == 0).sum()), hi=int((d_ > 0).sum())))
        rows.append(f"{b} & {msd(t.start)} & {msd(t.start_below / 100, 2)} & {msd(t.control)} & {len(t)} & {N[-1]['lo']}/{N[-1]['eq']}/{N[-1]['hi']} \\\\")
    (GEN / "app_m_null.tex").write_text(tab("@{}rccccc@{}", r"$b$ & $m_0$, without attack & $m_0 < 1$, share & $m$, control arm & DS & lower/equal/higher", rows[:1], breit=False))
    N = pd.DataFrame(N); N.to_csv(CSV / "app_m_null.csv", index=False)
    BT = [("cos_z_z0", "target row, $\\cos(z, z_0)$"), ("cos_muT_muT0", "centroid of the true class, $\\cos(\\mu_T, \\mu_{T,0})$"),
          ("cos_muF_muF0", "centroid of the wrong class, $\\cos(\\mu_F, \\mu_{F,0})$")]
    teile, BW, rows_by, BWds = [], [], {}, []
    for col, titel in BT:
        rows = []; rows_by[col] = rows
        for b in BUDGETS:
            s = x[x.b == b]; g = s.groupby(["dataset", "flipped"])[col].mean().unstack()
            vf = g[1].dropna() if 1 in g else pd.Series(dtype=float); vh = g[0].dropna() if 0 in g else pd.Series(dtype=float)
            vc = s.groupby("dataset")[col + "_c"].mean(); ok = g.dropna() if {0, 1} <= set(g.columns) else g.iloc[:0]
            BWds.append(pd.DataFrame({"flipped": g[1] if 1 in g else np.nan, "held": g[0] if 0 in g else np.nan, "control": vc})
                        .rename_axis("dataset").reset_index().assign(b=b, quantity=col))
            BW.append(dict(b=b, quantity=col, flipped=vf.mean(), flipped_sd=vf.std(), DS_flipped=len(vf), held=vh.mean(), held_sd=vh.std(), DS_held=len(vh),
                           control=vc.mean(), control_sd=vc.std(), DS_control=len(vc), pairs=len(ok), hi=int((ok[1] > ok[0]).sum()) if len(ok) else 0,
                           eq=int((ok[1] == ok[0]).sum()) if len(ok) else 0, lo=int((ok[1] < ok[0]).sum()) if len(ok) else 0))
            r = BW[-1]
            if col == "cos_z_z0":
                rows.append(f"{b} & {msd(vf)} & {msd(vh)} & {len(vf)}/{len(vh)} & {r['pairs']} & {r['hi']}/{r['eq']}/{r['lo']} & {msd(vc)} \\\\")
            else:
                rows.append(f"{b} & {msd(vf, 4)} & {msd(vh, 4)} & {len(vf)}/{len(vh)} & {msd(vc, 4)} \\\\")
        head = (r"$b$ & flipped & held & DS flipped/held & pairs & higher/equal/lower & control arm" if col == "cos_z_z0"
                else r"$b$ & flipped & held & DS flipped/held & control arm")
        teile.append((titel, tab("@{}rcccccc@{}" if col == "cos_z_z0" else "@{}rcccc@{}", head, rows, breit=False)))
    (GEN / "app_m_bewegung.tex").write_text(stapel(teile))
    BW = pd.DataFrame(BW); BW.to_csv(CSV / "app_m_bewegung.csv", index=False)
    pd.concat(BWds, ignore_index=True)[["dataset", "b", "quantity", "flipped", "held", "control"]].to_csv(CSV / "app_m_bewegung_ds.csv", index=False)
    RG = []
    for b in BUDGETS:
        t = per[b]; rf, rh = t.rule_flipped.dropna(), t.rule_held.dropna()
        RG.append(dict(b=b, DS_flipped=len(rf), flipped_above=rf.mean(), flipped_above_sd=rf.std(), DS_held=len(rh), held_below=rh.mean(), held_below_sd=rh.std()))
    RG = pd.DataFrame(RG); RG.to_csv(CSV / "app_m_regel.csv", index=False)
    i = x[x.b == 1]
    g = i.groupby(["dataset", "flipped"]).agg(m0=("m0", "mean"), m=("m", "mean"), piF=("pi_F", "median")).unstack()
    L = pd.DataFrame(index=sorted(i.dataset.unique(), key=str.lower))
    for grp, flag in (("flipped", 1), ("held", 0)):
        for c in ("m0", "m", "piF"):
            L[f"{c}_{grp}"] = g[(c, flag)].reindex(L.index) if (c, flag) in g else np.nan
    rho1, auc1 = innen(i)
    L["rho_within"], L["AUC"] = rho1.reindex(L.index), auc1.reindex(L.index)
    t1r = per[1].set_index("dataset")
    L["rule_flipped"], L["rule_held"] = t1r.rule_flipped.reindex(L.index), t1r.rule_held.reindex(L.index)
    v = lambda val, nd: f"${val:.{nd}f}$" if np.isfinite(val) else "--"
    cols = [("m0_flipped", 2), ("m_flipped", 2), ("piF_flipped", 2), ("m0_held", 2), ("m_held", 2), ("piF_held", 2), ("rule_flipped", 0), ("rule_held", 0), ("rho_within", 2), ("AUC", 2)]
    rows = [f"{tn(ds)} & " + " & ".join(v(r[c], nd) for c, nd in cols) + " \\\\" for ds, r in L.iterrows()]
    rows.append(r"\midrule mean $\pm$ SD & " + " & ".join(f"${L[c].mean():.{nd}f} \\pm {L[c].std():.{nd}f}$" for c, nd in cols) + " \\\\")
    head = (r"Dataset & \multicolumn{3}{c}{flipped} & \multicolumn{3}{c}{held} & \multicolumn{2}{c}{rule $m > 1$ (\%)} & \multicolumn{2}{c}{within the dataset} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}" + "\n"
            r" & $m_0$ & $m$ & $\pi_F$ & $m_0$ & $m$ & $\pi_F$ & flipped, $m > 1$ & held, $m \leq 1$ & $\rho(m, \pi_F)$ & separation")
    (GEN / "app_m_liste.tex").write_text(tab("@{}lcccccccccc@{}", head, rows)); L.rename_axis("dataset").reset_index().to_csv(CSV / "app_m_liste.csv", index=False)
    # Appendix D.1 of ticp_alparslan_v5.0_de: the per-dataset values at every number of poison rows, one table each for m, the rule m > 1 and
    # m within the dataset; app_m_liste above (one poison row, with m_0 and pi_F) stays for the versions in archiv/
    DSL = list(L.index)
    LB = []
    for b in BUDGETS:
        t = per[b].set_index("dataset"); rho_b, auc_b = innen(x[x.b == b])
        LB.append(pd.DataFrame({"n_flipped": t.n_flipped, "n_held": t.n_held, "m_flipped": t.flipped, "m_held": t.held,
                                "m0_flipped": t.start_flipped, "m0_held": t.start_held, "rule_flipped": t.rule_flipped,
                                "rule_held": t.rule_held, "rho_within": rho_b, "AUC": auc_b}).reindex(DSL).rename_axis("dataset").reset_index().assign(b=b))
    LB = pd.concat(LB, ignore_index=True)[["dataset", "b", "n_flipped", "n_held", "m_flipped", "m_held", "m0_flipped", "m0_held", "rule_flipped", "rule_held", "rho_within", "AUC"]]
    SP = ["m_flipped", "m_held", "rule_flipped", "rule_held", "rho_within", "AUC"]
    assert np.allclose(LB[LB.b == 1].set_index("dataset").loc[DSL, SP].to_numpy(float), L.loc[DSL, SP].to_numpy(float), equal_nan=True), \
        "D.1 at one poison row differs from app_m_liste"
    wb = LB.set_index(["dataset", "b"])
    # m with three decimals as in tab_test_ii_v5 and app_m, so that a value just below or above 1 shows its side of the line
    # the rule is shown as a share (0.87), not in percent, so that it cannot be read as a count; the CSV keeps percent
    # columns: the group on top (flipped and held, or rho and AUC), the number of poison rows b = 1 to 5 below it
    nb_ = len(BUDGETS)
    # m_0 without attack of the same target rows, grouped by their outcome under attack at b
    for name, cols, nd in (("app_m_liste_m", ("m_flipped", "m_held"), 3), ("app_m_liste_m0", ("m0_flipped", "m0_held"), 3), ("app_m_liste_regel", ("rule_flipped", "rule_held"), 2),
                           ("app_m_liste_innen", ("AUC",), 2)):
        # every column header names its metric (the group on top, b below)
        sub = {"app_m_liste_m": ("$m$, flipped", "$m$, held"), "app_m_liste_m0": ("$m_0$, flipped", "$m_0$, held"),
               "app_m_liste_regel": ("share with $m > 1$, flipped", r"share with $m \leq 1$, held"),
               "app_m_liste_innen": ("AUC of $m$, flipped against held",)}[name]
        head = ("Dataset" + "".join(f" & \\multicolumn{{{nb_}}}{{c}}{{{s_}}}" for s_ in sub) + r" \\ "
                + "".join(f"\\cmidrule(lr){{{nb_ * k + 2}-{nb_ * k + nb_ + 1}}}" for k in range(len(sub))) + "\n"
                + "".join(f" & $b = {b}$" for _ in sub for b in BUDGETS))
        sk = 100 if name == "app_m_liste_regel" else 1
        rows = [f"{tn(ds)} & " + " & ".join(v(wb.loc[(ds, b), c] / sk, nd) for c in cols for b in BUDGETS) + " \\\\" for ds in DSL]
        rows.append(r"\midrule mean $\pm$ SD & " + " & ".join(msd(LB[LB.b == b][c].dropna() / sk, nd) for c in cols for b in BUDGETS) + " \\\\")
        (GEN / f"{name}.tex").write_text(tab("@{}l" + "c" * (len(cols) * len(BUDGETS)) + "@{}", head, rows))
    LB.to_csv(CSV / "app_m_liste_b.csv", index=False)
    AS = []
    for b in BUDGETS:
        t = per[b]
        AS.append(dict(b=b, DS=len(t), **{f"rho_{n_}": spearmanr(t[c], t.asr).correlation for n_, c in zip(("start", "push", "m"), ("start", "push_all", "m_all"))}))
    AS = pd.DataFrame(AS); AS.to_csv(CSV / "app_m_asr.csv", index=False)
    t1 = per[1]
    mac["zMnull"], mac["zMnullSD"] = f(t1.start.mean(), 2), f(t1.start.std(), 2)
    mac["zMnullUnterEins"] = f(t1.start_below.mean(), 0)
    # Section 3.2.1: the attacked target rows (correct without attack) that lie above the line without attack, b = 1
    # holds the same rows as every b
    x1 = x[x.b == 1]; ueber = x1[x1.m0 > 1]
    mac["zMnullUeberZeilen"], mac["zMnullUeberDS"] = str(len(ueber)), str(ueber.dataset.nunique())
    for b in BUDGETS:
        w = W[b]; r = T[(T.b == b) & (T.quantity == "m")].iloc[0]
        mac[f"zMgekippt{w}"], mac[f"zMgehalten{w}"] = f(r.flipped, 2), f(r.held, 2)
        mac[f"zMgekipptDS{w}"], mac[f"zMgehaltenDS{w}"], mac[f"zMpaare{w}"], mac[f"zMhoeher{w}"] = str(r.DS_flipped), str(r.DS_held), str(r.pairs), str(r.hi)
        n_ = N[N.b == b].iloc[0]
        mac[f"zMkontrolle{w}"], mac[f"zMkontrolleNiedriger{w}"] = f(n_.control, 2), str(int(n_.lo))
        g_ = RG[RG.b == b].iloc[0]
        mac[f"zMregelGekippt{w}"], mac[f"zMregelGehalten{w}"] = f(g_.flipped_above, 0), f(g_.held_below, 0)
    for key, name in (("m0", "start"), ("push", "schub")):
        r = T[(T.b == 1) & (T.quantity == key)].iloc[0]
        mac[f"zM{name}Gekippt"], mac[f"zM{name}Gehalten"], mac[f"zM{name}Hoeher"] = f(r.flipped, 2), f(r.held, 2), str(r.hi)
    for b in BUDGETS:
        r = T[(T.b == b) & (T.quantity == "m")].iloc[0]
        assert r.hi == r.pairs, f"b = {b}: m of the flipped not higher in every pair"
    assert T[(T.b >= 4) & (T.quantity == "m")].pairs.nunique() == 1, "Section 3.3.1: as many pairs at four as at five poison rows"
    assert T[T.quantity == "m"].sort_values("b").flipped.is_monotonic_increasing, "Section 3.3.1: m of the flipped rises with every poison row"
    pa = T[T.quantity == "push"].sort_values("b")
    assert (pa["all"].diff().dropna() > 0).all() and (pa.above0_all == pa.DS_all).all(), "Section 3.2.1: the push over all target rows rises with every poison row and is positive in every dataset"
    assert int(N[N.b == 1].lo.iloc[0]) == len(t1), "Section 3.3.1: m in the control arm lower than m_0 in every dataset"
    mac["zMschubKontrolle"] = f((t1.control - t1.start).mean(), 2)
    mac["zMkontrolleNiedrigerMin"] = str(int(N.lo.min()))
    auc, rho = L.AUC.dropna(), L.rho_within.dropna()
    mac["zMauc"], mac["zMaucSD"], mac["zMaucDS"] = f(auc.mean(), 2), f(auc.std(), 2), str(len(auc))
    mac["zMrhoPiF"], mac["zMrhoPiFSD"], mac["zMrhoPiFDS"], mac["zMrhoPiFpos"] = f(rho.mean(), 2), f(rho.std(), 2), str(len(rho)), str(int((rho > 0).sum()))
    assert (rho > 0).all(), "Section 3.3.1: rho(m, pi_F) positive in every dataset"
    a1 = AS[AS.b == 1].iloc[0]
    mac["zMrhoStartASR"], mac["zMrhoSchubASR"], mac["zMrhoASR"] = f(a1.rho_start, 2), f(a1.rho_push, 2), f(a1.rho_m, 2)
    bz = BW[(BW.b == 1) & (BW.quantity == "cos_z_z0")].iloc[0]
    mac["zMcosZgekippt"], mac["zMcosZgehalten"], mac["zMcosZkontrolle"] = f(bz.flipped, 2), f(bz.held, 2), f(bz.control, 2)
    mac["zMcosZpaare"], mac["zMcosZniedriger"] = str(bz.pairs), str(bz.lo)
    for b_, w_ in ((2, "Zwei"), (3, "Drei")):
        bb = BW[(BW.b == b_) & (BW.quantity == "cos_z_z0")].iloc[0]
        mac[f"zMcosZgekippt{w_}"], mac[f"zMcosZgehalten{w_}"] = f(bb.flipped, 2), f(bb.held, 2)
        mac[f"zMcosZpaare{w_}"], mac[f"zMcosZniedriger{w_}"] = str(bb.pairs), str(bb.lo)
        assert b_ != 3 or bb.lo == bb.pairs, "Section 3.3.1: cos(z, z_0) lower for the flipped in all pairs at three poison rows"
    gz = BW[BW.quantity == "cos_z_z0"].set_index("b"); gz = (gz.held - gz.flipped)[[1, 2, 3]]
    assert gz.is_monotonic_increasing and (gz > 0).all(), f"Section 3.3.1: the cos gap held - flipped widens up to three poison rows {gz.to_dict()}"
    mu = BW[BW.quantity != "cos_z_z0"]
    mac["zMcosMuMin"] = f(np.floor(1000 * np.nanmin(mu[["flipped", "held", "control"]].to_numpy())) / 1000, 3)
    bf = BW[(BW.b == 5) & (BW.quantity == "cos_z_z0")].iloc[0]
    mac["zMcosZgekipptFuenf"] = f(bf.flipped, 2)
    pu = t1.set_index("dataset").loc["Pulsar-Dataset-HTRU2"]
    mac["zMPulsar"], mac["zMPulsarNull"] = f(pu.m_all, 2), f(pu.start, 2)
    allc = pd.read_parquet(ROOT / "data" / "measurements.parquet", columns=["model", "cos_z_muT", "cos_z_muF", "m"])
    allc = allc[allc.model == "tabicl-n8"]
    cs = pd.concat([allc.cos_z_muT, allc.cos_z_muF])
    assert (cs > 0).all(), "a cosine behind m is not positive"
    mac["zMcosBereichMin"], mac["zMcosBereichMax"] = f(np.floor(100 * cs.min()) / 100, 2), f(np.ceil(100 * cs.max()) / 100, 2)
    mac["zMbereichMin"], mac["zMbereichMax"] = f(np.floor(100 * allc.m.min()) / 100, 2), f(np.ceil(100 * allc.m.max()) / 100, 2)
    # the single ensemble members behind the mean m (columns of the TabICL run of 2026-11-14)
    mit = pd.read_parquet(ROOT / "data" / "measurements.parquet", columns=["model", "m_members_dropped", "cos_z_muT_min", "cos_z_muF_min"])
    mit = mit[mit.model.astype(str) == "tabicl-n8"]
    assert mit.notna().all().all() and (mit.m_members_dropped == 0).all(), "the appendix states that no member is left out of the mean m"
    mac["zMcosTmin"], mac["zMcosFmin"] = (f(np.floor(100 * mit[c].min()) / 100, 2) for c in ("cos_z_muT_min", "cos_z_muF_min"))
    import re
    assert all(re.fullmatch(r"z[A-Za-z]+", k) for k in mac), "macro names: letters only"
    (GEN / "zahlen_lage.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v_}}}\n" for k, v_ in mac.items()))
    print("\n".join(f"{k} = {v_}" for k, v_ in mac.items()))
    print("->", GEN / "zahlen_lage.tex")


if __name__ == "__main__":
    main()

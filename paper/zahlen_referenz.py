from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "gen"; CSV = GEN / "csv"
KEY = ["model", "dataset", "fold", "target"]
BUDGETS = range(1, 6)
W = {1: "Eins", 2: "Zwei", 3: "Drei", 4: "Vier", 5: "Fuenf"}
MODELS = [("tabicl-n8", "TabICL", None), ("tabpfn35-n8", "TabPFN", None), ("catboost", "CatBoost", None),
          ("random-forest", "Random forest", "RF"), ("logistic-regression", "Logistic regression", "LR")]


def load():
    cols = KEY + ["arm", "b", "pi_T", "clean_correct", "flipped", "auc_clean", "n_dup0"]
    x = pd.concat([pd.read_parquet(ROOT / "data" / f, columns=cols) for f in ("measurements.parquet", "reference_models.parquet")],
                  ignore_index=True)
    for c in ("model", "dataset", "arm"):
        x[c] = x[c].astype(str)
    for c in ("b", "clean_correct", "flipped", "n_dup0"):
        x[c] = x[c].astype(int)
    return x


def ist(k):
    return "none is" if k == 0 else ("one is" if k == 1 else f"{k:,} are")


def main():
    CSV.mkdir(parents=True, exist_ok=True)
    x = load()
    clean = x[x.arm == "clean"]
    a = x[(x.arm == "attack") & (x.clean_correct == 1)]
    k = x[(x.arm == "control") & (x.clean_correct == 1)]
    asr = a.groupby(["model", "b", "dataset"]).flipped.agg(ASR="mean", n_attacked="size").reset_index()
    auc = clean.groupby(["model", "dataset", "fold"]).auc_clean.first().groupby(["model", "dataset"]).mean().groupby("model").mean()
    n = clean[clean.clean_correct == 1].groupby("model").size()
    n_ds = x.dataset.nunique()
    rows, haupt, recs, mac = [], [], [], {}
    for m, lab, tag in MODELS:
        s, zellen = asr[asr.model == m], []
        for b in BUDGETS:
            v = 100 * s[s.b == b].ASR
            assert len(v) == n_ds, (m, b, len(v))
            zellen.append(f"${v.mean():.1f} \\pm {v.std():.1f}$")
            recs.append(dict(model=lab, b=b, DS=len(v), ASR_mean=v.mean(), ASR_sd=v.std(), auc=auc[m], attacked=int(n[m])))
            if tag:
                mac[f"z{tag}asr{W[b]}"] = f"{v.mean():.1f}"
        if tag:
            mac[f"z{tag}auc"], mac[f"z{tag}angegriffen"] = f"{auc[m]:.3f}", f"{int(n[m]):,}"
        zeile = f"{lab} & " + " & ".join(zellen) + f" & ${auc[m]:.3f}$ & {int(n[m]):,} \\\\"
        rows.append(("\\midrule\n" if tag == "RF" else "") + zeile)
        haupt.append(("\\midrule\n" if m == "catboost" else "") + zeile)
    head = "model & " + " & ".join(f"$b = {b}$" for b in BUDGETS) + " & AUC & attacked"
    (GEN / "app_referenz.tex").write_text("\\centering\\scriptsize\\maxbreite{\\begin{tabular}{@{}lccccccr@{}}\\toprule\n" + head
                                          + r" \\ \midrule" + "\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n")
    (GEN / "tab_trainiert.tex").write_text("\\centering\\footnotesize\\maxbreite{\\begin{tabular}{@{}lccccccr@{}}\\toprule\n" + head
                                           + r" \\ \midrule" + "\n" + "\n".join(haupt) + "\n\\bottomrule\\end{tabular}}\n")
    pd.DataFrame(recs).to_csv(CSV / "app_referenz.csv", index=False)
    asr.assign(model=asr.model.map({m: lab for m, lab, _ in MODELS}))[["model", "dataset", "b", "ASR", "n_attacked"]].to_csv(
        CSV / "app_referenz_ds.csv", index=False)
    ref = [m for m, _, tag in MODELS if tag]
    kflip = {m: int(k[k.model == m].flipped.sum()) for m in ref}
    a1 = a[a.b == 1]
    rf, lr = a1[a1.model == "random-forest"], a1[a1.model == "logistic-regression"]
    held = rf[rf.flipped == 0]
    dup = int((held.n_dup0 > 0).sum())
    sicher = lambda d: int(((1 - d[d.flipped == 1].pi_T) >= 0.9).sum())
    text = (("In the control arm no target row flips for either reference model. " if not any(kflip.values()) else
             f"In the control arm {kflip['random-forest']} target rows flip for the random forest and "
             f"{kflip['logistic-regression']} for the logistic regression. ")
            + f"At one poison row, {len(held)} attacked target rows hold for the random forest, "
            + ("all of them" if dup == len(held) else f"{dup} of them") + " with genuine duplicates of the true label in the context. "
            + f"Of the {int(rf.flipped.sum()):,} target rows that flip for the random forest at one poison row, {ist(sicher(rf))} "
            + "wrong with a confidence of at least 90\\,\\% ($\\pi_F \\geq 0.9$). "
            + f"Of the {int(lr.flipped.sum()):,} that flip for the logistic regression, {ist(sicher(lr))}.")
    gleich = clean[(clean.model == "random-forest") & (clean.pi_T == 0.5)]
    if len(gleich):
        text += (" The random forest averages 100 trees, so its probability can be exactly 0.5. Without attack this holds for "
                 f"{len(gleich)} target rows, which follow the tie rule of Section~\\ref{{sec:wirkung}}, and "
                 f"{int((gleich.clean_correct == 1).sum())} of them count as predicted correctly.")
    text += "\n"
    (GEN / "app_referenz_text.tex").write_text(text)
    (GEN / "zahlen_referenz.tex").write_text("".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in mac.items()))
    print(pd.DataFrame(recs).round(3).to_string(index=False))
    print(text)
    print("->", GEN / "app_referenz.tex")


if __name__ == "__main__":
    main()

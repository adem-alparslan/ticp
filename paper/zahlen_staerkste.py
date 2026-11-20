from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "gen"
MIN_N, MIN_DS = 1, 5


def tab(spec, head, rows, breit=True):
    t = "\\begin{tabular}{" + spec + "}\\toprule\n" + head + r" \\ \midrule" + "\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}"
    return "\\centering\\scriptsize" + ("\\maxbreite{" + t + "}" if breit else t) + "\n"


def main():
    csv = GEN / "csv"; csv.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "measurements.parquet")
    pf = df[(df.model == "tabpfn35-n8") & (df.clean_correct == 1)
            & (df.arm == "attack") & df.copy_rank_weight.notna()]
    rows, recs, per_ds = [], [], []
    for b in range(1, 6):
        xb = pf[pf.b == b]; cells = []
        for grp, sub in (("all", xb), ("flipped", xb[xb.flipped == 1]), ("held", xb[xb.flipped == 0])):
            sh = 100 * sub.groupby("dataset", observed=True).copy_rank_weight.apply(lambda s: (s == 1).mean() if len(s) >= MIN_N else np.nan).dropna()
            cells.append(f"${sh.mean() / 100:.2f} \\pm {sh.std() / 100:.2f}$ & {len(sh)}" if len(sh) >= MIN_DS else (f"${sh.mean() / 100:.2f}$ & {len(sh)}" if len(sh) else "-- & 0"))  # as a share
            recs.append(dict(b=b, group=grp, DS=len(sh), strongest_pct=sh.mean(), strongest_sd=sh.std()))
            per_ds.append(sh.rename("strongest_pct").rename_axis("dataset").reset_index().assign(b=b, group=grp))
        if b <= 2:
            rows.append(f"{b} & " + " & ".join(cells) + " \\\\")
    head = (r"$b$ & \multicolumn{2}{c}{all attacked target rows} & \multicolumn{2}{c}{flipped} & \multicolumn{2}{c}{held} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" + "\n"
            + " & " + " & ".join(["strongest & DS"] * 3))
    (GEN / "app_rang_staerkste.tex").write_text(tab("@{}rcccccc@{}", head, rows, breit=False))
    pd.DataFrame(recs).to_csv(csv / "app_rang_staerkste.csv", index=False)
    pd.concat(per_ds, ignore_index=True)[["dataset", "b", "group", "strongest_pct"]].to_csv(csv / "app_rang_staerkste_ds.csv", index=False)
    print(pd.DataFrame(recs).round(1).to_string(index=False))
    print("->", GEN / "app_rang_staerkste.tex")


if __name__ == "__main__":
    main()

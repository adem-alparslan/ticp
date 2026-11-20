"""The duplicate check of ticp_alparslan_v5.0_de (Sections 2.6 and 3.3, Appendix G): an alarm when the context holds
a row equal to the query row in every feature (a missing value equals only a missing value, as n_dup0).

No model is needed. The contexts are rebuilt with core/ (data.py, context.py) from cache/, exactly as in the main run:
  attack, control  the target rows under attack and in the control arm, b = 1 to 5; the check finds the copies
  without attack   every test row of the fold as a query in the clean context
Per dataset the share of queries with an alarm, then the mean over the datasets. Asserts that every attacked and every
control target row raises an alarm and that the rebuilt contexts match data/measurements.parquet (n_ctx, n_dup0).
Writes gen/app_dup.tex (Table G1), gen/zahlen_duplikat.tex (macros \\zDup...) and gen/csv/app_dup_ds.csv.

    python paper/zahlen_duplikat.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import config as C, context as X, data as D  # noqa: E402
from zahlen_lage import KURZ, tn  # noqa: E402

GEN, CSV = HERE / "gen", HERE / "gen" / "csv"
TFMS = ("tabicl-n8", "tabpfn35-n8")


def equal_rows(A, q):
    return ((A == q) | (np.isnan(A) & np.isnan(q))).all(1)


def main():
    rows, arms = [], []
    for name in C.DATASETS:
        Xd, y = D.load(name)
        for fold in range(C.FOLDS):
            prep = D.prepare(Xd, y, fold)
            A, yc = prep["X_ctx"].to_numpy(float), np.asarray(prep["y_ctx"]).astype(int)
            Q, yq = prep["X_te"].to_numpy(float), np.asarray(prep["y_te"]).astype(int)
            for j in range(len(Q)):
                dup = equal_rows(A, Q[j])
                rows.append(dict(dataset=name, fold=fold, alarm=bool(dup.any()), both=len(set(yc[dup])) == 2))
            for t in D.targets(prep["y_te"], fold):
                od = X.orders(prep, t)
                q = prep["X_te"].iloc[int(t["test_idx"])].to_numpy(float)
                for b in C.BUDGETS:
                    for arm in ("attack", "control"):
                        Xa, _, _ = X.build(arm, b, prep, t, od)
                        arms.append(dict(dataset=name, fold=fold, target=int(t["target"]), arm=arm, b=b,
                                         n_ctx=len(Xa), n_dup0=od["n_dup0"],
                                         alarm=bool(equal_rows(Xa.to_numpy(float), q).any())))
    S, R = pd.DataFrame(rows), pd.DataFrame(arms)
    assert R.alarm.all(), "every poison row and every control row is a duplicate of its target row"
    m = pd.read_parquet(C.DATA / "measurements.parquet", columns=["model", "dataset", "fold", "target", "arm", "b",
                                                                  "n_ctx", "n_dup0", "flipped"])
    for c in ("model", "dataset", "arm"):
        m[c] = m[c].astype(str)
    m = m[m.model.isin(TFMS) & m.arm.isin(["attack", "control"])]
    j = m.merge(R, on=["dataset", "fold", "target", "arm", "b"], suffixes=("", "_r"), validate="many_to_one")
    assert len(j) == len(m) == 2 * len(R)
    assert (j.n_ctx == j.n_ctx_r).all() and (j.n_dup0 == j.n_dup0_r).all()

    ds = S.groupby("dataset").agg(queries=("alarm", "size"), alarm=("alarm", "mean"), both=("both", "mean"))
    ds = (100 * ds[["alarm", "both"]]).join(ds.queries).reset_index()
    ds.to_csv(CSV / "app_dup_ds.csv", index=False)
    mean = ds.alarm.mean()
    none = int((ds.alarm == 0).sum())
    order = sorted(ds.dataset, key=lambda d: KURZ.get(d, d).lower())
    pc = lambda v: f"{v / 100:.3f}"                # a share as a decimal fraction, three decimals for the shares below 0.01
    body = [f"{tn(d)} & {int(r.queries):,} & ${pc(r.alarm)}$ & ${pc(r.both)}$ \\\\"
            for d in order for r in [ds.set_index("dataset").loc[d]]]
    (GEN / "app_dup.tex").write_text(
        "\\centering\\scriptsize\\begin{tabular}{@{}lccc@{}}\\toprule\n"
        "Dataset & Queries & Alarm & Duplicates of both labels \\\\ \\midrule\n"
        + "\n".join(body) + "\n\\midrule\n"
        + f"Mean & & ${pc(mean)}$ & ${pc(ds.both.mean())}$ \\\\\n\\bottomrule\\end{{tabular}}\n")
    mac = dict(zDupSauber=f"{mean:.0f}", zDupNull=str(none), zDupMammo=f"{ds.set_index('dataset').alarm['mammography']:.0f}")
    (GEN / "zahlen_duplikat.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    print(ds.sort_values("alarm", ascending=False).round(2).head(12).to_string(index=False))
    print("mean alarm without attack", round(mean, 3), "| datasets without alarm", none, "| arms: all", len(R), "alarm")
    print("->", GEN / "app_dup.tex", GEN / "zahlen_duplikat.tex", CSV / "app_dup_ds.csv")


if __name__ == "__main__":
    main()

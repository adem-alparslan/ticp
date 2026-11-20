import numpy as np

from . import config as C
from .data import key

MIN_REMAIN = 2


def orders(prep, target):
    y = np.asarray(prep["y_ctx"]).astype(int)
    tl, ti = int(target["true_label"]), int(target["test_idx"])
    same = np.where(y == tl)[0]
    A = prep["X_ctx"].to_numpy(float)[same]
    q = prep["X_te"].iloc[ti].to_numpy(float)
    n_dup0 = int(((A == q) | (np.isnan(A) & np.isnan(q))).all(1).sum())
    rng = np.random.default_rng(key(ti, C.ROW_KEY))
    rng.random(len(same))  # this draw precedes the rows; removing it changes the poison rows
    rows = rng.permutation(len(y))[:int(C.MAX_BUDGET)]
    return dict(rows=rows, n_same=int(len(same)), n_dup0=n_dup0)


def build(arm, b, prep, target, od):
    X_ctx, y_ctx = prep["X_ctx"], prep["y_ctx"]
    if arm == "clean":
        return X_ctx, y_ctx, None
    if arm not in ("attack", "control"):
        raise ValueError(f"unknown arm {arm!r}")
    tl = int(target["true_label"])
    idx = np.sort(od["rows"][:int(b)])
    y = np.asarray(y_ctx).astype(int)
    keep = np.setdiff1d(np.arange(len(y)), idx)
    if min((y[keep] == 0).sum(), (y[keep] == 1).sum()) < MIN_REMAIN:
        return None
    Xm = X_ctx.copy()
    Xm.iloc[idx] = prep["X_te"].iloc[int(target["test_idx"])].to_numpy(float)
    ym = y_ctx.copy()
    ym.iloc[idx] = (1 - tl) if arm == "attack" else tl
    return Xm, ym, idx

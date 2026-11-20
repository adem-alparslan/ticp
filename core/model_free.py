import numpy as np


def vote(a, y, true_label, poison_idx):
    tl = int(true_label)
    y = np.asarray(y).astype(int)
    copy = np.zeros(len(a), bool)
    if poison_idx is not None and len(poison_idx):
        copy[np.asarray(poison_idx)] = True
    out = dict(a_T=float(a[~copy & (y == tl)].sum()), a_F=float(a[~copy & (y != tl)].sum()))
    if copy.any():
        out["a_copy"] = float(a[copy].sum())
        out["a_copy_max"] = float(a[copy].max())
        out["copy_rank_weight"] = float(1 + (a[~copy] > a[copy].max()).sum())
    return out


def mean_over_members(per):
    out = {}
    for k in sorted({k for d in per for k in d}):
        v = np.array([d.get(k, np.nan) for d in per], float)
        v = v[~np.isnan(v)]
        out[k] = float(v.mean()) if len(v) else np.nan
    return out


def copy_cosine(H, n, poison_idx):
    ctx, z = H[:n], H[n]
    cos = (ctx @ z) / (np.linalg.norm(ctx, axis=1) * np.linalg.norm(z) + 1e-12)
    return dict(cos_z_copy=float(cos[np.asarray(poison_idx)].mean()))


def position(H, n, y, true_label):
    ctx, z = H[:n], H[n]
    y, tl = np.asarray(y).astype(int), int(true_label)
    if (y == 0).sum() < 2 or (y == 1).sum() < 2:
        return {}
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cT, cF = cos(z, ctx[y == tl].mean(0)), cos(z, ctx[y != tl].mean(0))
    return dict(cos_z_muT=cT, cos_z_muF=cF, m=(cF / cT if cT > 0 else np.nan))


def movement(H, n, y, true_label, ref=None):
    ctx, z = H[:n], H[n]
    y, tl = np.asarray(y).astype(int), int(true_label)
    ok = (y == 0).sum() >= 2 and (y == 1).sum() >= 2
    now = dict(z=z, muT=ctx[y == tl].mean(0) if ok else None, muF=ctx[y != tl].mean(0) if ok else None)
    out = {}
    if ref is not None:
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        out["cos_z_z0"] = cos(z, ref["z"])
        if ok and ref["muT"] is not None:
            out["cos_muT_muT0"], out["cos_muF_muF0"] = cos(now["muT"], ref["muT"]), cos(now["muF"], ref["muF"])
    return out, now



def neighbours(H, n, y, true_label, poison_idx, k=25):
    ctx, z = H[:n].astype(np.float64), H[n].astype(np.float64)
    cos = ctx @ z / (np.linalg.norm(ctx, axis=1) * np.linalg.norm(z) + 1e-12)
    order = np.argsort(-cos)
    copy = np.zeros(n, bool)
    if poison_idx is not None and len(poison_idx):
        copy[np.asarray(poison_idx)] = True
    wrong = np.asarray(y).astype(int) != int(true_label)
    return {f"k_opp_gen_{k}": float(wrong[order[~copy[order]][:k]].sum())}

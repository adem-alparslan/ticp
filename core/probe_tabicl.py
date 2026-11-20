import numpy as np

from .model_free import copy_cosine, mean_over_members, movement, neighbours, position
from .models import classifier, positive, torch_module


def probe(spec, X_ctx, y_ctx, x_query, poison_idx, true_label, ref=None, target=0):
    n_est = int(spec["n_est"])
    clf = classifier(spec, X_ctx, y_ctx)
    n, n_q, q = len(y_ctx), len(x_query), int(target)
    has_copy = poison_idx is not None and len(poison_idx) > 0
    mats = []

    def keep(m, a, kw):
        x = a[0].detach().float().cpu().numpy()
        for R in (x if x.ndim == 3 else x[None]):
            if R.shape[0] == n + n_q:
                R = R.reshape(n + n_q, -1)
                mats.append(np.concatenate([R[:n], R[n + q:n + q + 1]]))

    hook = torch_module(clf).icl_predictor.decoder.register_forward_pre_hook(keep, with_kwargs=True)
    try:
        p_all = positive(clf, x_query)
    finally:
        hook.remove()
    if len(mats) != n_est:
        raise RuntimeError(f"decoder input of {len(mats)} members for n_estimators = {n_est}")

    per, refs = [], []
    for i, H in enumerate(mats):
        d = dict(copy_cosine(H, n, poison_idx)) if has_copy else {}
        d.update(position(H, n, y_ctx, true_label))
        d.update(neighbours(H, n, y_ctx, true_label, poison_idx))
        mv, now = movement(H, n, y_ctx, true_label, ref[i] if ref else None)
        d.update(mv)
        refs.append(now)
        per.append(d)
    inside = mean_over_members(per)
    cT, cF = ([d[k] for d in per if k in d] for k in ("cos_z_muT", "cos_z_muF"))
    if cT:
        # position() leaves m of a member empty where cos_z_muT <= 0, and the mean then runs over the others
        inside.update(m_members_dropped=int(sum(not c > 0 for c in cT)), cos_z_muT_min=float(min(cT)),
                      cos_z_muF_min=float(min(cF)))
    return dict(pi_pos=float(p_all[q]), p_all=p_all, inside=inside, ref=refs)


def clean_rows(spec, X_ctx, y_ctx, X_te):
    clf = classifier(spec, X_ctx, y_ctx)
    return np.asarray(positive(clf, X_te), float), {}

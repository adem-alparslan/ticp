import math

import numpy as np

from .model_free import copy_cosine, mean_over_members, neighbours, vote as _vote
from .models import classifier, positive, torch_module


def probe(spec, X_ctx, y_ctx, x_query, poison_idx, true_label, ref=None, target=0):
    import torch
    import tabpfn.architectures.tabpfn_v3_5 as arch

    n_est = int(spec["n_est"])
    clf = classifier(spec, X_ctx, y_ctx)
    n, n_q, q = len(y_ctx), len(x_query), int(target)
    y = np.asarray(y_ctx).astype(int)
    mod = torch_module(clf)
    has_copy = poison_idx is not None and len(poison_idx) > 0
    state, votes, mats = dict(on=False), [], []

    def read_representation(m, a, o):
        X = o.detach().float()
        if X.dim() != 3 or X.shape[1] != n + n_q or X.shape[0] > n_est:
            return
        Xc = torch.cat([X[:, :n], X[:, n + q:n + q + 1]], 1).cpu().numpy()
        mats.extend(Xc[b] for b in range(Xc.shape[0]))

    def read_attention(q_, k, scaling):
        if not state["on"] or k is None or q_.shape[1] != n_q or k.shape[1] != n or q_.shape[0] > n_est:
            return
        with torch.no_grad():
            qs = scaling(q_, k.shape[1]) if scaling is not None else q_
            with torch.autocast(device_type=q_.device.type, enabled=False):
                qf = qs[:, q].float() / math.sqrt(qs.shape[-1])
                kf = k.float()
                if kf.shape[2] != qf.shape[1]:
                    kf = kf.repeat_interleave(qf.shape[1] // kf.shape[2], dim=2)
                s = torch.einsum("bhd,bnhd->bhn", qf, kf)
                if not torch.isfinite(s).all():
                    raise RuntimeError("decoder attention scores not finite")
                a = torch.softmax(s, dim=-1).mean(1).cpu().numpy()
        votes.extend(dict(_vote(a[b], y, true_label, poison_idx), N_eff=float(1.0 / (a[b] ** 2).sum()))
                     for b in range(a.shape[0]))

    inner = arch._batched_scaled_dot_product_attention

    def wrapped(q_BSHD, k_BSJD, v_BSJD, softmax_scaling_layer=None, **kw):
        read_attention(q_BSHD, k_BSJD, softmax_scaling_layer)
        return inner(q_BSHD, k_BSJD, v_BSJD, softmax_scaling_layer=softmax_scaling_layer, **kw)

    dec = mod.heads.many_class_decoder
    hooks = [dec.register_forward_pre_hook(lambda m, a, kw: state.update(on=True), with_kwargs=True),
             dec.register_forward_hook(lambda m, a, o: state.update(on=False)),
             mod.output_norm.register_forward_hook(read_representation)]
    arch._batched_scaled_dot_product_attention = wrapped
    try:
        with torch.no_grad():
            p_all = positive(clf, x_query)
    finally:
        arch._batched_scaled_dot_product_attention = inner
        for h in hooks:
            h.remove()
    if len(votes) != n_est or len(mats) != n_est:
        raise RuntimeError(f"decoder attention of {len(votes)} and representation of {len(mats)} members "
                           f"for n_estimators = {n_est}")

    per = [dict(v, **(copy_cosine(H, n, poison_idx) if has_copy else {}), **neighbours(H, n, y, true_label, poison_idx))
           for v, H in zip(votes, mats)]
    return dict(pi_pos=float(p_all[q]), p_all=p_all, inside=mean_over_members(per), ref=[])


def clean_rows(spec, X_ctx, y_ctx, X_te, chunk=128):
    import torch
    import tabpfn.architectures.tabpfn_v3_5 as arch

    clf = classifier(spec, X_ctx, y_ctx)
    mod = torch_module(clf)
    n, n_q = len(y_ctx), len(X_te)
    state, kept = dict(on=False), []
    inner = arch._batched_scaled_dot_product_attention

    def wrapped(q_, k, v, softmax_scaling_layer=None, **kw):
        if state["on"] and k is not None and q_.shape[1] == n_q and k.shape[1] == n:
            with torch.no_grad():
                qs = softmax_scaling_layer(q_, k.shape[1]) if softmax_scaling_layer is not None else q_
                with torch.autocast(device_type=q_.device.type, enabled=False):
                    kf = k.float()
                    if kf.shape[2] != qs.shape[2]:
                        kf = kf.repeat_interleave(qs.shape[2] // kf.shape[2], dim=2)
                    out = torch.empty(qs.shape[0], n_q, n, dtype=torch.float32)
                    for lo in range(0, n_q, chunk):
                        qf = qs[:, lo:lo + chunk].float() / math.sqrt(qs.shape[-1])
                        s = torch.einsum("bqhd,bnhd->bqhn", qf, kf)
                        if not torch.isfinite(s).all():
                            raise RuntimeError("decoder attention scores not finite")
                        out[:, lo:lo + chunk] = torch.softmax(s, -1).mean(2).cpu()
            kept.extend(out.numpy())
        return inner(q_, k, v, softmax_scaling_layer=softmax_scaling_layer, **kw)

    dec = mod.heads.many_class_decoder
    hooks = [dec.register_forward_pre_hook(lambda m, a, kw: state.update(on=True), with_kwargs=True),
             dec.register_forward_hook(lambda m, a, o: state.update(on=False))]
    arch._batched_scaled_dot_product_attention = wrapped
    try:
        with torch.no_grad():
            p_all = np.asarray(positive(clf, X_te), float)
    finally:
        arch._batched_scaled_dot_product_attention = inner
        for h in hooks:
            h.remove()
    if len(kept) != int(spec["n_est"]):
        raise RuntimeError(f"decoder attention of {len(kept)} members, expected {spec['n_est']}")
    K = np.asarray(kept, dtype=np.float64)
    A = K.mean(0)
    lab = np.asarray(y_ctx).astype(int)
    return p_all, dict(w0=A[:, lab == 0].sum(1), w1=A[:, lab == 1].sum(1))

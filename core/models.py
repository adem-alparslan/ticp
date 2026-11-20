import os

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C

os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")

TABPFN35_ARCHITECTURE = "TabPFNV3p5"
_TABICL = {}


def positive(clf, X):
    return np.asarray(clf.predict_proba(X))[:, list(clf.classes_).index(1)]


def classifier(spec, X_ctx, y_ctx):
    kind = spec["kind"]
    yc = np.asarray(y_ctx).astype(int)

    if kind == "tabicl":
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(n_estimators=int(spec["n_est"]), random_state=C.SEED)
        if _TABICL:
            clf._load_model = lambda: [setattr(clf, k, v) for k, v in _TABICL.items()]
        clf.fit(X_ctx, yc)
        if not _TABICL:
            _TABICL.update(model_=clf.model_, model_config_=clf.model_config_, model_path_=clf.model_path_)
    elif kind == "tabpfn35":
        import tabpfn
        clf = tabpfn.TabPFNClassifier(n_estimators=int(spec["n_est"]), random_state=C.SEED).fit(X_ctx, yc)
        arch = type(clf.model_).__name__
        if arch != TABPFN35_ARCHITECTURE:
            raise RuntimeError(f"TabPFN loaded {arch}, expected {TABPFN35_ARCHITECTURE} (tabpfn {tabpfn.__version__})")
    elif kind == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(random_seed=C.SEED, verbose=0, allow_writing_files=False,
                                 thread_count=int(spec.get("threads", 6))).fit(X_ctx, yc)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return clf


def flipped(p_pos, true_label, p_clean_pos):
    clean_correct = int(int(p_clean_pos >= 0.5) == int(true_label))
    return clean_correct, int(clean_correct == 1 and int(p_pos >= 0.5) != int(true_label))


def on_true(p_pos, true_label):
    return float(p_pos) if int(true_label) == 1 else 1.0 - float(p_pos)


def auc(p_pos, y_true):
    y = np.asarray(y_true).astype(int)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p_pos, float)))


def torch_module(clf):
    import torch.nn as nn
    m = getattr(clf, "model_", None)
    if not isinstance(m, nn.Module):
        raise RuntimeError(f"{type(clf).__name__}.model_ is not a torch module")
    return m

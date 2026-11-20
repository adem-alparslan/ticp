import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C

_OML = "{http://openml.org/openml}"


def key(*parts):
    return [C.SEED] + [int(p) for p in parts]


def load(name: str, cap="design"):
    did = C.did(name)
    folder = C.CACHE / "org" / "openml" / "www" / "datasets" / str(did)
    root = ET.parse(folder / "description.xml").getroot()
    text = lambda tag: [e.text for e in root.findall(_OML + tag) if e.text]
    target = text("default_target_attribute")[0]
    drop = set(text("row_id_attribute")) | set(text("ignore_attribute"))
    pq = folder / f"dataset_{did}.pq"
    if pq.exists():
        data = pd.read_parquet(pq)
    else:
        with open(folder / "dataset.pkl.py3", "rb") as fh:
            matrix, _, names = pickle.load(fh)
        data = pd.DataFrame(matrix.toarray(), columns=list(names))
    X = data.drop(columns=[target] + [c for c in data.columns if c in drop])
    y = data[target]

    bad = [c for c in X.columns if X[c].dtype.kind not in "fiu"]
    if bad:
        raise ValueError(f"{name}: non-continuous feature columns {bad}")
    classes = sorted(y.unique())
    if len(classes) != 2:
        raise ValueError(f"{name}: {len(classes)} classes, expected 2.")
    ye = y.map({classes[0]: 0, classes[1]: 1}).astype("int64")

    cap = C.CAP if cap == "design" else int(cap or 0)
    if cap and len(X) > cap:
        idx, _ = train_test_split(np.arange(len(X)), train_size=cap, stratify=ye.values,
                                  random_state=C.SPLIT_SEED)
        idx = np.sort(idx)
        X = X.iloc[idx].reset_index(drop=True)
        ye = ye.iloc[idx].reset_index(drop=True)
    return X, ye


def folds(X, y):
    return list(StratifiedKFold(n_splits=C.FOLDS, shuffle=True, random_state=C.SPLIT_SEED).split(X, y))


def targets(y_te, fold):
    yte = np.asarray(y_te).astype(int)
    rng = np.random.default_rng(key(int(fold), C.TARGET_SEED))
    order = {lab: rng.permutation(np.where(yte == lab)[0]) for lab in (0, 1)}
    per = max(1, C.TARGETS // 2)
    sel = []
    for i in range(per):
        for lab in (0, 1):
            if i < len(order[lab]):
                sel.append(int(order[lab][i]))
    return [dict(target=i, test_idx=int(t), true_label=int(yte[t])) for i, t in enumerate(sel)]


def prepare(X, y, fold):
    tr, te = folds(X, y)[fold]
    X_ctx, y_ctx = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_te, y_te = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
    return dict(X_ctx=X_ctx, y_ctx=y_ctx, X_te=X_te, y_te=y_te)

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import config as C, context as X, data as D, models as M

MODELS = ("random-forest", "logistic-regression")
FILE = "reference_models.parquet"
TYPES = dict(model="string", dataset="string", fold="int8", target="int8", arm="string", b="int16", test_idx="int32",
             true_label="int8", n_ctx="int16", n_dup0="int16", pi_T="float32", clean_correct="int8", flipped="int8",
             auc_clean="float32")


def fit(model, X_ctx, y_ctx):
    y = np.asarray(y_ctx).astype(int)
    if model == "random-forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=C.SEED).fit(X_ctx, y)
    if model == "logistic-regression":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return make_pipeline(SimpleImputer(), StandardScaler(),
                                 LogisticRegression(max_iter=5000, random_state=C.SEED)).fit(X_ctx, y)
    raise ValueError(f"unknown model {model!r}")


def cells():
    return [("clean", 0)] + [(a, int(b)) for a in C.ARMS if a != "clean" for b in C.BUDGETS]


def dataset_rows(job):
    model, name = job
    Xd, y = D.load(name)
    rows = []
    for fold in range(C.FOLDS):
        prep = D.prepare(Xd, y, fold)
        clean = fit(model, prep["X_ctx"], prep["y_ctx"])
        auc_clean = M.auc(M.positive(clean, prep["X_te"]), prep["y_te"])
        for t in D.targets(prep["y_te"], fold):
            ti, tl = int(t["test_idx"]), int(t["true_label"])
            od = X.orders(prep, t)
            x_target = prep["X_te"].iloc[[ti]]
            p0 = float(M.positive(clean, x_target)[0])
            for arm, b in cells():
                built = X.build(arm, b, prep, t, od)
                if built is None:
                    raise RuntimeError(f"{name}|{fold}|{t['target']}|{arm}|{b}: cell left out")
                Xm, ym, _ = built
                p = p0 if arm == "clean" else float(M.positive(fit(model, Xm, ym), x_target)[0])
                cc, flip = M.flipped(p, tl, p0)
                rows.append(dict(model=model, dataset=name, fold=int(fold), target=int(t["target"]), arm=arm, b=b,
                                 test_idx=ti, true_label=tl, n_ctx=len(Xm), n_dup0=od["n_dup0"], pi_T=M.on_true(p, tl),
                                 clean_correct=cc, flipped=flip, auc_clean=auc_clean))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--data", default=None, help="another folder for the file (tests); default data/")
    a = ap.parse_args()
    bad = sorted(set(a.datasets or []) - set(C.DATASETS))
    if bad:
        ap.error(f"unknown datasets {bad}; see config.yaml")
    names = [n for n in C.DATASETS if a.datasets is None or n in a.datasets]
    jobs = [(m, n) for m in MODELS if m in a.models for n in names]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        parts = list(ex.map(dataset_rows, jobs))
    table = pd.DataFrame([r for part in parts for r in part]).astype(TYPES)
    out = (Path(a.data) if a.data else C.DATA) / FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f"{len(table)} rows in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()

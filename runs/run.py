import argparse
import hashlib
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import config as C, context as X, data as D, models as M, table as T

TOL = 1e-3


def cells():
    return [("clean", 0)] + [(a, int(b)) for a in C.ARMS if a != "clean" for b in C.BUDGETS]


def probe_module(kind):
    if kind == "tabicl":
        from core import probe_tabicl as P
    elif kind == "tabpfn35":
        from core import probe_tabpfn35 as P
    else:
        return None
    return P


def run_record(args, run_id):
    import importlib.metadata as im
    files = sorted((ROOT / "core").glob("*.py")) + [Path(__file__).resolve(), ROOT / "config.yaml"]
    pkgs = {}
    for p in ("torch", "tabpfn", "tabicl", "catboost", "scikit-learn", "pandas", "numpy", "pyarrow"):
        try:
            pkgs[p] = im.version(p)
        except im.PackageNotFoundError:
            pass
    return dict(run=run_id, model=args.model, command=" ".join(["runs/run.py"] + sys.argv[1:]),
                started=datetime.now().isoformat(timespec="seconds"), environment=Path(sys.prefix).name, packages=pkgs,
                code={f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files})


def clean_pass(spec, P, prep):
    X_ctx, y_ctx, X_te = prep["X_ctx"], prep["y_ctx"], prep["X_te"]
    if P is None:
        return np.asarray(M.positive(M.classifier(spec, X_ctx, y_ctx), X_te), float), None
    p_all, f = P.clean_rows(spec, X_ctx, y_ctx, X_te)
    if spec["kind"] == "tabpfn35":
        y_te = np.asarray(prep["y_te"]).astype(int)
        pi_T = np.where(y_te == 1, p_all, 1.0 - p_all)
        w_false = np.where(y_te == 1, f["w0"], f["w1"])
        off = np.abs((1.0 - pi_T) - w_false)
        if off.max() > TOL:
            raise RuntimeError(f"clean pass: P(wrong label) and the attention of the wrong class differ by up to "
                               f"{off.max():.2e} (row {int(off.argmax())})")
    return p_all, f


def check_target(spec, inside, p, ti, p_clean_all, f, y_te):
    if abs(p - float(p_clean_all[ti])) > TOL or (p >= 0.5) != (p_clean_all[ti] >= 0.5):
        raise RuntimeError(f"target row {ti}: P {p:.5f} in the probe, {p_clean_all[ti]:.5f} in the clean pass")
    if spec["kind"] == "tabpfn35":
        w_true = float(f["w1"][ti] if y_te[ti] == 1 else f["w0"][ti])
        if abs(inside["a_T"] - w_true) > TOL:
            raise RuntimeError(f"target row {ti}: a_T {inside['a_T']:.5f} in the probe, {w_true:.5f} in the clean pass")


def fold_rows(spec, name, fold, prep):
    kind, model = spec["kind"], spec["name"]
    P = probe_module(kind)
    X_te, y_te = prep["X_te"], np.asarray(prep["y_te"]).astype(int)
    p_clean_all, f = clean_pass(spec, P, prep)
    pred_clean_all = (p_clean_all >= 0.5).astype(int)
    auc_clean = M.auc(p_clean_all, y_te)
    rows, left_out = [], []
    for t in D.targets(prep["y_te"], fold):
        ti, tl = int(t["test_idx"]), int(t["true_label"])
        od = X.orders(prep, t)
        rest = np.arange(len(y_te)) != ti
        was_right = pred_clean_all[rest] == y_te[rest]
        p0, ref = None, None
        for arm, b in cells():
            built = X.build(arm, b, prep, t, od)
            if built is None:
                left_out.append(f"{name}|{fold}|{t['target']}|{arm}|{b}")
                continue
            Xm, ym, idx = built
            base = dict(model=model, dataset=name, fold=int(fold), target=int(t["target"]),
                        arm=arm, b=int(b), test_idx=ti, true_label=tl, n_ctx=len(Xm),
                        n_same=od["n_same"], n_dup0=od["n_dup0"])
            if P is None:
                if arm == "clean":
                    p, p_all = float(p_clean_all[ti]), p_clean_all
                else:
                    p_all = np.asarray(M.positive(M.classifier(spec, Xm, ym), X_te), float)
                    p = float(p_all[ti])
                inside = {}
            else:
                res = P.probe(spec, Xm, ym, X_te, idx, tl, ref=ref, target=ti)
                p, inside = res["pi_pos"], res["inside"]
                if arm == "clean":
                    ref = res["ref"] or None
                    p_all = p_clean_all
                    check_target(spec, inside, p, ti, p_clean_all, f, y_te)
                else:
                    p_all = res["p_all"]
            if arm == "clean":
                p0 = p
            cc, flip = M.flipped(p, tl, p0)
            moved = (np.asarray(p_all)[rest] >= 0.5).astype(int) != pred_clean_all[rest]
            rows.append(dict(base, pi_T=M.on_true(p, tl), clean_correct=cc,
                             flipped=flip, **inside,
                             auc_rest=M.auc(np.asarray(p_all)[rest], y_te[rest]), n_rest=int(rest.sum()),
                             co_moved=int(moved.sum()), co_moved_wrong=int((moved & was_right).sum()),
                             co_moved_right=int((moved & ~was_right).sum()), auc_clean=auc_clean))
    return rows, left_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, choices=range(C.FOLDS), default=list(range(C.FOLDS)))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data", default=None, help="another folder for the files (tests); default data/")
    a = ap.parse_args()
    bad = sorted(set(a.datasets or []) - set(C.DATASETS))
    if bad:
        ap.error(f"unknown datasets {bad}; see config.yaml")
    if a.data:
        C.DATA = Path(a.data)
        T.FILE, T.LOCK = C.DATA / "measurements.parquet", C.DATA / "measurements.lock"

    spec = C.MODELS[a.model]
    P = probe_module(spec["kind"])
    run_id = a.run_id or f"{datetime.now():%Y%m%d_%H%M%S}_{a.model}"
    record = run_record(a, run_id)
    print(f"=== {run_id}: {a.model}", flush=True)
    t_all = time.time()
    skipped = record["skipped"] = []
    record["left_out"] = []
    names = [n for n in C.DATASETS if a.datasets is None or n in a.datasets]
    for name in names:
        in_file = {f: T.has_fold(spec["name"], name, f) for f in a.folds}
        todo = [f for f in a.folds if not in_file[f]]
        if not todo:
            print(f"{name}: done", flush=True)
            continue
        Xd, y = D.load(name)
        for fold in todo:
            t0 = time.time()
            prep = D.prepare(Xd, y, fold)
            try:
                rows, left_out = fold_rows(spec, name, fold, prep)
                record["left_out"] = left_out
            except Exception as e:
                skipped.append(f"{name}|{fold}: {e!r}")
                print(f"{name} f{fold}: SKIPPED, nothing written: {e!r}\n{traceback.format_exc()}", flush=True)
                continue
            n = T.append(rows, record)
            extra_txt = f", {len(left_out)} cells left out" if left_out else ""
            print(f"{name} f{fold}: {n} rows in {(time.time() - t0) / 60:.1f} min{extra_txt}", flush=True)
    print(f"=== {run_id} done in {(time.time() - t_all) / 60:.1f} min, {len(skipped)} folds skipped", flush=True)
    if skipped:
        T.note(record)
        print("\n".join(skipped), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

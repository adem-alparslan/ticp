import fcntl
import json
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

SCHEMA_VERSION = 1
FILE = C.DATA / "measurements.parquet"
LOCK = C.DATA / "measurements.lock"

KEY = ["model", "dataset", "fold", "target", "arm", "b"]
SORT = KEY

COLUMNS = {
    "model": ("string", "tabicl-n8, tabpfn35-n8, catboost"),
    "dataset": ("string", "short name of the dataset"),
    "fold": ("int8", "test fold, 0-4"),
    "target": ("int8", "target row of the fold, 0-9"),
    "arm": ("string", "clean; attack (b random context rows overwritten by an exact copy of the target row with the WRONG label); control (the same rows, the copy with the TRUE label)"),
    "b": ("int16", "number of poison rows (budget, the overwritten rows); 0 in clean"),
    "test_idx": ("int32", "row of the target in the test fold"),
    "true_label": ("int8", "true label of the target row"),
    "n_ctx": ("int16", "number of context rows"),
    "n_same": ("int16", "context rows of the target row's true class before the overwrite"),
    "n_dup0": ("int16", "of these, rows identical to the target row in every feature (a missing value equals only a missing value)"),
    "pi_T": ("float32", "probability of the true label in this arm; in clean = pi_T,0"),
    "clean_correct": ("int8", "1 if predicted correctly without attack (class 1 at p >= 0.5, core/models.py flipped); derived"),
    "flipped": ("int8", "1 if clean_correct and the predicted label in this arm is not the true label (class 1 at p >= 0.5, core/models.py flipped); 0 in clean; derived"),
    "auc_rest": ("float32", "AUC on the other test rows of the fold in this arm"),
    "n_rest": ("int16", "number of these other test rows"),
    "co_moved": ("int16", "other test rows whose prediction changes against clean; co_moved_wrong + co_moved_right"),
    "co_moved_wrong": ("int16", "of these, rows predicted correctly without attack and wrongly in this arm"),
    "co_moved_right": ("int16", "of these, rows predicted wrongly without attack and correctly in this arm"),
    "auc_clean": ("float32", "AUC on all test rows of the fold without attack (the clean batch); one value per "
                             "model and fold, repeated on every row"),
    "a_T": ("float32", "TabPFN 3.5: decoder attention weight of the genuine rows with the true label"),
    "a_F": ("float32", "TabPFN 3.5: decoder attention weight of the genuine rows with the wrong label"),
    "a_copy": ("float32", "TabPFN 3.5: decoder attention weight of the poison rows, the copies (a_P under attack, a_C in control)"),
    "a_copy_max": ("float32", "TabPFN 3.5: largest decoder attention weight of a single copy"),
    "copy_rank_weight": ("float32", "TabPFN 3.5: rank of the most attended copy in the decoder (1 = largest weight)"),
    "cos_z_copy": ("float32", "TFMs: cosine of the target row to the copies at the read-out point, mean over them; per member, then averaged; empty in clean"),
    "cos_z_muT": ("float32", "TabICL: cosine of the target row to the mean of the context rows with its true label at the read-out point; per member, then averaged"),
    "cos_z_muF": ("float32", "TabICL: cosine of the target row to the mean of the context rows with the wrong label (a poison row counts to the class of its label)"),
    "k_opp_gen_25": ("float32", "TFMs: of the 25 nearest genuine context rows by cosine at the read-out point (the copies left out), how many carry the wrong label (nu_F); per member, then averaged"),
    "m": ("float32", "TabICL: position of the target row between the two class means, cos_z_muF / cos_z_muT per member, then averaged; m > 1: nearer to the wrong class"),
    "m_members_dropped": ("int8", "TabICL: number of the ensemble members with cos_z_muT <= 0, whose m is left empty and is not in the mean m; 0: m is the mean over all members"),
    "cos_z_muT_min": ("float32", "TabICL: smallest cos_z_muT of a single ensemble member"),
    "cos_z_muF_min": ("float32", "TabICL: smallest cos_z_muF of a single ensemble member"),
    "cos_z_z0": ("float32", "TabICL: cosine of the target row at the read-out point to itself without attack (the clean arm of the same target row); per member, then averaged; empty in clean"),
    "cos_muT_muT0": ("float32", "TabICL: cosine of the mean of the context rows with the true label to the same mean without attack; per member, then averaged; empty in clean"),
    "cos_muF_muF0": ("float32", "TabICL: cosine of the mean of the context rows with the wrong label (a poison row counts to the class of its label) to the same mean without attack; empty in clean"),
    "N_eff": ("float32", "TabPFN 3.5: effective number of context rows in the decoder vote, 1 / sum a_j^2 per member, then averaged"),
    "written_at": ("datetime64[s]", "when the row was written (local time; the same for every row of a fold)"),
}
REQUIRED = KEY + ["test_idx", "true_label"]


@contextmanager
def _locked():
    C.DATA.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _conform(df):
    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        raise ValueError(f"columns not in the schema: {extra}")
    out = pd.DataFrame(index=df.index)
    for col, (typ, _) in COLUMNS.items():
        s = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        if typ == "string":
            out[col] = s.astype(str)
        elif typ.startswith("datetime"):
            out[col] = pd.to_datetime(s).astype(typ)
        elif typ.startswith("int"):
            out[col] = pd.to_numeric(s).astype(f"Int{typ[3:]}")
        else:
            out[col] = pd.to_numeric(s).astype(typ)
    return out


def runs():
    return json.loads((pq.read_schema(FILE).metadata or {}).get(b"runs", b"[]")) if FILE.exists() else []


def has_fold(model, dataset, fold):
    if not FILE.exists():
        return False
    t = pq.read_table(FILE, columns=["model"], filters=[
        ("model", "=", model), ("dataset", "=", dataset), ("fold", "=", int(fold))])
    return t.num_rows > 0


def _write(df, runs_md):
    df = df.sort_values(SORT, na_position="last", kind="stable").reset_index(drop=True)
    schema_md = {b"schema_version": str(SCHEMA_VERSION).encode(),
                 b"columns": json.dumps({k: v[1] for k, v in COLUMNS.items()}).encode(),
                 b"runs": json.dumps(runs_md).encode()}
    tmp = FILE.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **schema_md})
    with pq.ParquetWriter(tmp, table.schema, compression="zstd") as w:
        for _, idx in df.groupby(["model", "dataset"], sort=False, observed=True).indices.items():
            w.write_table(table.take(pa.array(np.sort(idx))))
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, FILE)


def _record(runs_md, run):
    rec = next((r for r in runs_md if r.get("run") == run["run"]), None)
    if rec is None:
        rec = dict(rows=0, keys=[], left_out=[])
        runs_md.append(rec)
    left_out = rec["left_out"]
    rec.update(run)
    rec["left_out"] = left_out
    return rec


def note(run):
    with _locked():
        if not FILE.exists():
            return False
        runs_md = runs()
        _record(runs_md, run)
        _write(_conform(pd.read_parquet(FILE)), runs_md)
    return True


def append(rows, run):
    stamp = pd.Timestamp.now().floor("s")
    new = pd.DataFrame(rows)
    bad = [c for c in REQUIRED if c not in new.columns or new[c].isna().any()]
    if bad:
        raise ValueError(f"required columns empty: {bad}")
    new["written_at"] = stamp
    new = _conform(new)
    if new.duplicated(KEY).any():
        raise ValueError(f"{int(new.duplicated(KEY).sum())} keys repeat within the new rows")
    with _locked():
        old = pd.read_parquet(FILE) if FILE.exists() else None
        if old is not None:
            old_keys = old[KEY].astype({c: str for c in ("model", "dataset", "arm")})
            clash = new.merge(old_keys, on=KEY, how="inner")
            if len(clash):
                raise ValueError(f"{len(clash)} keys already in {FILE.name}, nothing written "
                                 f"(first: {clash.iloc[0][KEY].to_dict()})")
        runs_md = runs()
        rec = _record(runs_md, run)
        rec.update(written=stamp.isoformat(), rows=rec["rows"] + int(len(new)),
                   left_out=rec["left_out"] + list(run.get("left_out", [])),
                   keys=rec["keys"] + sorted({f"{r.model}|{r.dataset}|{r.fold}" for r in new.itertuples()}))
        both = new if old is None else pd.concat([_conform(old), new], ignore_index=True)
        _write(both, runs_md)
    return len(new)

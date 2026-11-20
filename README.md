# Targeted in-context poisoning of tabular foundation models: an empirical study

Code, result data and manuscript of the study. A context row of a tabular foundation model (TabICL 2.1.1,
TabPFN 3.5) is overwritten by an exact copy of the query row carrying the wrong label, and the model's
prediction for that query flips. CatBoost, a random forest and a logistic regression, retrained on the same
poisoned context, are the trained models for comparison. 30 OpenML datasets, 5 folds, 10 target rows per fold,
budgets of 1 to 5 poison rows, an attack arm and a control arm (the same rows, the copy with the TRUE label)
per target row.

The manuscript is `manuscript/ticp_alparslan_v5.0.tex` with the appendix `manuscript/anhang_v5.0.tex`, set
with the Springer Nature LaTeX template (`sn-jnl.cls`). Every number, table and figure in it is generated from
`data/measurements.parquet` and `data/reference_models.parquet` by the generators in `paper/`. Nothing is
typed in by hand.

## Layout

| path | what it is |
|---|---|
| `config.yaml` | the design: folds, target rows, budgets, the three models with their environments, the 30 datasets with their OpenML ids |
| `core/` | the study: `data.py` (datasets, capping, folds, target rows), `context.py` (the poisoned context: attack and control arm), `models.py` (the three models), `probe_tabicl.py`, `probe_tabpfn35.py` (one forward pass per cell: outcome and the read-out point), `model_free.py` (the quantities computed from the read-out point: vote, position for the relative closeness m, movement), `table.py` (the result file and its schema), `config.py` (paths and seeds) |
| `runs/run.py` | the run: one call per model, resumable per fold, writes `data/measurements.parquet` |
| `runs/fetch_openml.py` | fills the OpenML cache the study reads (see Data) |
| `data/measurements.parquet` | the result file the manuscript is built from, see `SCHEMA.md` |
| `runs/reference_models.py` | the two reference models (random forest, logistic regression, scikit-learn defaults), retrained on every cell of the main run; one call, writes `data/reference_models.parquet` |
| `data/reference_models.parquet` | the outcome of the two reference models, see `SCHEMA.md`; read by `paper/zahlen_referenz.py` and `paper/anhang_modelle.py` (Appendix F) |
| `paper/` | the generators of every number, table and figure: the notebook `anhang.ipynb` and the scripts (see Reproducing the manuscript) |
| `paper/gen/` | everything the generators write: `app_*.tex` (appendix tables), `fig_*.pdf` (figures), `zahlen*.tex` and `tab_test_*_v5.tex` (the `\z...` macros and the test tables), `csv/` (the per-dataset values behind every table, the Online Resource) |
| `manuscript/` | the manuscript as submitted: the two tex files, the Springer class `sn-jnl.cls` and style `sn-mathphys-num.bst`, and flat copies of the generated files it reads |
| `make_manuscript.py` | copies the generated files the manuscript reads from `paper/gen/` to `manuscript/`, compiles `manuscript/` on its own and writes `manuscript.zip` and `manuscript.pdf` |
| `manuscript.pdf` | the compiled manuscript, as written by `make_manuscript.py` |
| `environment/` | the two Python environments, `pip list --format=freeze` of each, and which packages decide a result |

Names: `zahlen` = numbers, `lage` = position (the relative closeness m of TabICL), `staerkste` = strongest row,
`tests_fuenf` = the five tests, `tests_drei` = the three tests, `anhang_modelle` = appendix by model group, `anhang` = appendix, `gen` = generated. The code is in English.

## Data

The runs read the 30 datasets from an OpenML cache in `cache/` (layout of openml 0.15: `description.xml` and
`dataset_<did>.pq`, or `dataset.pkl.py3` for the one sparse-ARFF dataset, codrna) without the openml package,
so that every model environment loads the data the same way (`core/data.py`). The cache is not in the
repository (410 MB). `python runs/fetch_openml.py` fills it with openml 0.15 from the ids in `config.yaml`,
`--check` only reports what is there.

The result file is in the repository, so the manuscript can be rebuilt without the runs:
`data/measurements.parquet`, 48,840 rows, the 38 columns of `core/table.py`; one row per model × dataset × fold ×
target row × arm × budget, see `SCHEMA.md`. The manuscript reads only this file and, for the two reference
models (Appendix F), `data/reference_models.parquet`.

Seeds: `random_state` of every model 0, folds and capping 42, target rows 20240711, the overwritten rows drawn
per target row with the key `[0, row, 0xf11b]` (`core/config.py`).

## Reproducing the manuscript from the result data

Environment `tabicl_env` (see `environment/README.md`). The notebook also reads the datasets in `cache/` for
Table A1, so `cache/` must be filled first (see Data). From this directory:

```
(cd paper && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 anhang.ipynb)
python paper/zahlen_lage.py       # TabICL: relative closeness m (Appendix C), app_m_ds.csv
python paper/zahlen_staerkste.py  # TabPFN: the strongest row (Appendix D)
python paper/zahlen_referenz.py   # the two reference models (Appendix F)
python paper/zahlen_duplikat.py   # duplicate check (Appendix G), reads cache/
python paper/zahlen_kopp.py       # neighbourhood of the target row (Appendix C)
python paper/tests_v5.py          # the 25 tests, Tables 2 to 4 (Appendix E)
python paper/fig_asr.py           # Figure 1
python paper/fig_m_arme.py        # Figure 2, from app_m_ds.csv of zahlen_lage.py
python paper/anhang_modelle.py    # Appendices A, B and F
python paper/anhang_ds.py         # Appendices B, C and D, per dataset
python paper/zahlen_text.py       # the remaining macros of the text
python make_manuscript.py         # copies the generated files to manuscript/, compiles, writes manuscript.zip and manuscript.pdf
```

Run the generators in this order: `fig_m_arme.py`, `anhang_modelle.py`, `anhang_ds.py` and `zahlen_text.py`
read the CSVs written before them. `tests_fuenf.py` and `tests_drei.py` hold the test functions that `tests_v5.py` imports;
they are not run on their own. The notebook writes Figure 3, the table of Appendix H, the per-dataset CSVs
in `gen/csv/` and `gen/zahlen.tex`.

## Reproducing the result data

Three runs, one per model, each in its environment (`config.yaml: models[].env`):

```
python runs/fetch_openml.py                                  # once, fills cache/
~/miniconda3/envs/tabicl_env/bin/python   runs/run.py --model tabicl-n8
~/miniconda3/envs/tabicl_env/bin/python   runs/run.py --model catboost
~/miniconda3/envs/tabfm_cu130/bin/python  runs/run.py --model tabpfn35-n8
```

The metadata of `data/measurements.parquet` hold one record per run: id, command, environment, package
versions, the hashes of `core/*.py`, `runs/run.py` and `config.yaml`, and the folds written (Appendix H).
A run is resumable per fold (`SCHEMA.md`); `--datasets` and `--folds` restrict it, `--data` writes into
another folder.

The two reference models run on the CPU in `tabicl_env`, one process per model and dataset:

```
~/miniconda3/envs/tabicl_env/bin/python   runs/reference_models.py --procs 16
```

The call retrains both models on every cell of the main run (the same folds, target rows, overwritten rows and
copies) and writes `data/reference_models.parquet`.

# data/measurements.parquet (exact-duplicate injection, r = 0)

The file holds 48,840 rows with exactly the 38 columns of `core/table.py` (schema version 1), the schema below;
every table, figure and number of the manuscript is computed from it. `runs/run.py` writes the file, one run per
model. The metadata of the file keep the schema version, the column descriptions and one record per run.

- **One row** = model × dataset × fold × target × arm × b.
- **Key**: `model`, `dataset`, `fold`, `target`, `arm`, `b`.
- **Arms**: `clean` (b = 0, the unchanged context); `attack`: b random context rows overwritten by an exact
  copy of the target row carrying the WRONG label (distance 0); `control`: the SAME rows overwritten by the same
  copy carrying the TRUE label (the control tests the label, not the duplication). Nested in b: the rows of b = 2
  contain the one of b = 1. The rows are drawn at random from the whole context, so no nearest supporter is removed.
- **Drawn rows**: the rows to overwrite are drawn per target row with the RNG keyed on it
  (`[seed, test_idx, ROW_KEY]`), nested in b; attack and control overwrite the same rows. `n_dup0` counts the
  context rows of the target row's true class that equal it in every feature.
- **Left-out cells**: a cell is built only if at least 2 rows of EACH class survive the overwrite; with 2,400
  context rows and b <= 5 this is never binding, so no cell is left out.
- **Inside** quantities per ensemble member, then averaged over the 8 members. `a_copy` is the weight of the
  exact copies (a_P under attack, a_C in control), `a_copy_max` the weight of the single strongest copy;
  `a_F` the weight of the genuine rows that carry the wrong label. TabICL centroid quantities
  (`m`, `cos_z_mu*`, `cos_mu*`) are NaN when a class has fewer than two context rows (cannot happen with the
  left-out rule above).
- **Empty values**: the copy columns (`a_copy`, `a_copy_max`, `copy_rank_weight`, `cos_z_copy`) and the
  movement columns (`cos_z_z0`, `cos_mu*_mu*0`) are empty in `clean`; model-specific columns are empty for the
  other models (the vote `a_*`, `copy_rank_weight`, `N_eff` is TabPFN's, the relative closeness `m`, `cos_z_mu*` and the
  movement are TabICL's, CatBoost has the outcome columns only); `auc_rest` is NaN when the other test rows hold
  one class only.

| column | type | description |
|---|---|---|
| `model` | string | tabicl-n8, tabpfn35-n8, catboost |
| `dataset` | string | short name of the dataset |
| `fold` | int8 | test fold, 0-4 |
| `target` | int8 | target row of the fold, 0-9 |
| `arm` | string | clean; attack (b random context rows overwritten by an exact copy of the target row with the WRONG label); control (the same rows, the copy with the TRUE label) |
| `b` | int16 | number of poison rows (budget, the overwritten rows); 0 in clean |
| `test_idx` | int32 | row of the target in the test fold |
| `true_label` | int8 | true label of the target row |
| `n_ctx` | int16 | number of context rows |
| `n_same` | int16 | context rows of the target row's true class before the overwrite |
| `n_dup0` | int16 | of these, rows identical to the target row in every feature (a missing value equals only a missing value) |
| `pi_T` | float32 | probability of the true label in this arm; in clean = pi_T,0 |
| `clean_correct` | int8 | 1 if predicted correctly without attack (class 1 at p >= 0.5, core/models.py flipped); derived |
| `flipped` | int8 | 1 if clean_correct and the predicted label in this arm is not the true label (class 1 at p >= 0.5, core/models.py flipped); 0 in clean; derived |
| `auc_rest` | float32 | AUC on the other test rows of the fold in this arm |
| `n_rest` | int16 | number of these other test rows |
| `co_moved` | int16 | other test rows whose prediction changes against clean; co_moved_wrong + co_moved_right |
| `co_moved_wrong` | int16 | of these, rows predicted correctly without attack and wrongly in this arm |
| `co_moved_right` | int16 | of these, rows predicted wrongly without attack and correctly in this arm |
| `auc_clean` | float32 | AUC on all test rows of the fold without attack (the clean batch); one value per model and fold, repeated on every row |
| `a_T` | float32 | TabPFN 3.5: decoder attention weight of the genuine rows with the true label |
| `a_F` | float32 | TabPFN 3.5: decoder attention weight of the genuine rows with the wrong label |
| `a_copy` | float32 | TabPFN 3.5: decoder attention weight of the poison rows, the copies (a_P under attack, a_C in control) |
| `a_copy_max` | float32 | TabPFN 3.5: largest decoder attention weight of a single copy |
| `copy_rank_weight` | float32 | TabPFN 3.5: rank of the most attended copy in the decoder (1 = largest weight) |
| `cos_z_copy` | float32 | TFMs: cosine of the target row to the copies at the read-out point, mean over them; per member, then averaged; empty in clean |
| `k_opp_gen_25` | float32 | TFMs: of the 25 nearest genuine context rows by cosine at the read-out point (the copies left out), how many carry the wrong label (nu_F); per member, then averaged |
| `cos_z_muT` | float32 | TabICL: cosine of the target row to the mean of the context rows with its true label at the read-out point; per member, then averaged |
| `cos_z_muF` | float32 | TabICL: cosine of the target row to the mean of the context rows with the wrong label (a poison row counts to the class of its label) |
| `m` | float32 | TabICL: position of the target row between the two class means, cos_z_muF / cos_z_muT per member, then averaged; m > 1: nearer to the wrong class |
| `m_members_dropped` | int8 | TabICL: number of the ensemble members with cos_z_muT <= 0, whose m is left empty and is not in the mean m; 0: m is the mean over all members |
| `cos_z_muT_min` | float32 | TabICL: smallest cos_z_muT of a single ensemble member |
| `cos_z_muF_min` | float32 | TabICL: smallest cos_z_muF of a single ensemble member |
| `cos_z_z0` | float32 | TabICL: cosine of the target row at the read-out point to itself without attack (the clean arm of the same target row); per member, then averaged; empty in clean |
| `cos_muT_muT0` | float32 | TabICL: cosine of the mean of the context rows with the true label to the same mean without attack; per member, then averaged; empty in clean |
| `cos_muF_muF0` | float32 | TabICL: cosine of the mean of the context rows with the wrong label (a poison row counts to the class of its label) to the same mean without attack; empty in clean |
| `N_eff` | float32 | TabPFN 3.5: effective number of context rows in the decoder vote, 1 / sum a_j^2 per member, then averaged |
| `written_at` | datetime64[s] | when the row was written (local time; the same for every row of a fold) |

## Derived (not stored, computed by the reader)

| quantity | from |
|---|---|
| pi_T,0, m_0, N_eff,0, a_F,0 | the `clean` row of the same target (`arm = clean`, `b = 0`) |
| pi_F | 1 - `pi_T` |
| Delta m | `m` - `m` of the clean row |
| a_P, a_C | `a_copy` with `arm = attack` / `arm = control` |
| TabPFN true-label vote, wrong-label vote | w_true = `a_T`, w_wrong = `a_F` + `a_copy`, in BOTH arms |
| ASR | mean of `flipped` over the target rows with `clean_correct = 1` |

## Resume and writing

A fold is written all or nothing: its rows at once (tmp + os.replace, one writer at a time via a lock). A fold
counts as done when its rows are in the file. `runs/run.py` skips a fold whose computation raises (any
exception, traceback in the log), continues, and exits with status 1. The record of a run holds its id, command,
environment, package versions, code hashes, the folds written, left-out cells and skipped folds.
Before writing, the run checks every target row of a fold against the clean pass (one forward pass on the clean
context with all test rows: P and TabPFN a_T, tolerance 1e-3) and, TabPFN, that P(wrong label) equals the
attention of the wrong class for every test row.

# data/reference_models.parquet (two further trained models)

The outcome of two reference models, a random forest and a logistic regression with the defaults of
scikit-learn, retrained on every cell of the main run like CatBoost (`runs/reference_models.py`, one pass).
The cells are those of `data/measurements.parquet`: the same folds, target rows, overwritten rows and copies
(`core/data.py`, `core/context.py`). 32,560 rows, one per model × dataset × fold × target × arm × b, with the
same key. The file holds the outcome only and no run records. The manuscript reports the two models
descriptively (Appendix F of the manuscript, `paper/zahlen_referenz.py`).

- **random-forest**: `RandomForestClassifier(random_state=0)`, 100 trees, each grown on a bootstrap sample until
  its leaves are pure; missing values are handled by the trees.
- **logistic-regression**: mean imputation and z-scores on the context, then `LogisticRegression(max_iter=5000,
  random_state=0)` (L2, C = 1, lbfgs); a fit that does not converge stops the run.

| column | type | description |
|---|---|---|
| `model` | string | random-forest, logistic-regression |
| `dataset`, `fold`, `target`, `arm`, `b` | string, int8, int8, string, int16 | as in `data/measurements.parquet` |
| `test_idx`, `true_label`, `n_ctx`, `n_dup0` | int32, int8, int16, int16 | as in `data/measurements.parquet` |
| `pi_T` | float32 | probability of the true label in this arm; in clean = pi_T,0 |
| `clean_correct` | int8 | 1 if predicted correctly without attack (class 1 at p >= 0.5) |
| `flipped` | int8 | 1 if clean_correct and the predicted label in this arm is not the true label; 0 in clean |
| `auc_clean` | float32 | AUC on all test rows of the fold without attack; one value per model and fold, repeated on every row |

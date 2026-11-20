# Environments

Two Python environments, because TabPFN 3.5 (tabpfn 9.1.0) and TabICL 2.1.1 do not share a torch build on the
machine of the study (NVIDIA GB10, sm_121, driver 580.159.03). `tabicl_env.txt` and `tabfm_cu130.txt` are the
full `pip list --format=freeze` of each; the packages that decide a result are:

| environment | python | used for | packages |
|---|---|---|---|
| `tabicl_env` | 3.10.19 | TabICL, CatBoost, the evaluation (`paper/anhang.ipynb`, `paper/*.py`) | tabicl 2.1.1, catboost 1.2.8, torch 2.11.0.dev20260119+cu128, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.15.3, pyarrow 20.0.0, matplotlib 3.10.8, nbconvert 7.16.6, openml 0.15.1 |
| `tabfm_cu130` | 3.11.14 | TabPFN 3.5, the random forest and the logistic regression (`runs/reference_models.py`) | tabpfn 9.1.0 (checkpoint tabpfn-v3.5-20260909, TabPFNV3p5), torch 2.10.0+cu130, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.15.3, pyarrow 25.0.1 |

`config.yaml` names the environment of each model (`models[].env`). The run records in the result file keep the
package versions and the code hashes of every run (Appendix A of the manuscript, table `app_repro`).

To rebuild: create the two environments with the python versions above and `pip install -r <file>`; the torch
builds are platform specific (a nightly cu128 wheel for sm_121 in `tabicl_env`, cu130 in `tabfm_cu130`), on
another GPU the matching torch build of the same major version should be used instead. The model checkpoints are
downloaded by the packages on first use (tabicl: `tabicl-classifier-v2-20260212.ckpt`, tabpfn: the 3.5 weights).

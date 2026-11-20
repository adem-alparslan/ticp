from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

CACHE = ROOT / "cache"
DATA = ROOT / "data"

SEED = 0
SPLIT_SEED = 42
TARGET_SEED = 20240711
ROW_KEY = 0xF11B

_d = CFG["design"]
FOLDS = int(_d["folds"])
TARGETS = int(_d["targets_per_fold"])
CAP = int(_d["cap"])
ARMS = tuple(_d["arms"])
BUDGETS = tuple(int(b) for b in _d["budgets"])
MAX_BUDGET = max(BUDGETS)


MODELS = {m["name"]: dict(m) for m in CFG["models"]}
DATASETS = {d["name"]: dict(d) for d in CFG["datasets"]}


def did(name: str) -> int:
    return int(DATASETS[name]["did"])

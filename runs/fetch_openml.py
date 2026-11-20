import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as C

CACHE = C.CACHE


def datasets():
    return [(name, C.did(name)) for name in C.DATASETS]


def complete(did):
    folder = CACHE / "org" / "openml" / "www" / "datasets" / str(did)
    return (folder / "description.xml").exists() and ((folder / f"dataset_{did}.pq").exists() or (folder / "dataset.pkl.py3").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, download nothing")
    a = ap.parse_args()
    todo = [(n, d) for n, d in datasets() if not complete(d)]
    print(f"{len(datasets()) - len(todo)} of {len(datasets())} datasets complete in {CACHE}")
    for n, d in todo:
        print(f"  missing: {n} (did {d})")
    if a.check or not todo:
        sys.exit(1 if (a.check and todo) else 0)
    import openml
    openml.config.set_root_cache_directory(str(CACHE))
    for n, d in todo:
        print(f"downloading {n} (did {d}) ...", flush=True)
        ds = openml.datasets.get_dataset(d, download_data=True, download_features_meta_data=True)
        ds.get_data()
        print("   ", "ok" if complete(d) else "FAILED: data file not found after the download")
    sys.exit(0 if all(complete(d) for _, d in datasets()) else 1)


if __name__ == "__main__":
    main()

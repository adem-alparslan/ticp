"""Assemble the submission folder manuscript/ and manuscript.zip.

1. Compiles manuscript/ with paper/gen/ on the search path and records which generated files (macros,
   tables, figures) the document reads.
2. Copies exactly these files from paper/gen/ into manuscript/ (flat) and names any generated file in
   manuscript/ that the document no longer reads.
3. Compiles manuscript/ on its own, twice, and stops on a LaTeX error, a missing file or an undefined reference.
4. Writes manuscript.zip (sources and the generated files, no aux, log or pdf) and manuscript.pdf.

Run from the repository root after the generators: python make_manuscript.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "paper" / "gen"
MS = ROOT / "manuscript"
MAIN = "ticp_alparslan_v5.0"
SOURCES = [f"{MAIN}.tex", "anhang_v5.0.tex", "sn-jnl.cls", "sn-mathphys-num.bst"]


def pdflatex(folder, texinputs=None, recorder=False):
    env = dict(os.environ)
    if texinputs:
        env["TEXINPUTS"] = texinputs
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"] + (["-recorder"] if recorder else []) + [f"{MAIN}.tex"]
    for _ in range(2):
        subprocess.run(cmd, cwd=folder, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (Path(folder) / f"{MAIN}.log").read_text(errors="replace")


def problems(log):
    keys = ("! ", "not found", "Reference `", "Citation `", "There were undefined references")
    return [l for l in log.splitlines() if any(k in l for k in keys)]


def main():
    # 1. which generated files does the document read?
    with tempfile.TemporaryDirectory() as t:
        for s in SOURCES:
            shutil.copy2(MS / s, t)
        pdflatex(t, texinputs=f"{t}:{GEN}:", recorder=True)
        fls = (Path(t) / f"{MAIN}.fls").read_text()
    used = sorted({Path(l[6:]).resolve() for l in fls.splitlines()
                   if l.startswith("INPUT ") and Path(l[6:]).resolve().parent == GEN.resolve()})
    if not used:
        sys.exit("no generated file was read; check paper/gen/")

    # 2. copy them flat into manuscript/
    for f in used:
        shutil.copy2(f, MS / f.name)
    keep = set(SOURCES) | {f.name for f in used}
    stale = [p.name for p in MS.iterdir() if p.is_file() and p.suffix in (".tex", ".pdf") and p.name not in keep
             and not p.name.startswith(MAIN)]
    print(f"{len(used)} generated files copied from paper/gen/ to manuscript/")
    if stale:
        print("no longer read by the document (delete by hand):", ", ".join(sorted(stale)))

    # 3. compile manuscript/ on its own
    with tempfile.TemporaryDirectory() as t:
        for name in keep:
            shutil.copy2(MS / name, t)
        log = pdflatex(t)
        bad = problems(log)
        if bad:
            print("\n".join(bad[:20]))
            sys.exit("manuscript/ does not compile cleanly on its own")
        shutil.copy2(Path(t) / f"{MAIN}.pdf", ROOT / "manuscript.pdf")

    # 4. the zip for the submission
    with zipfile.ZipFile(ROOT / "manuscript.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(keep):
            z.write(MS / name, name)
    print(f"manuscript.zip: {len(keep)} files; manuscript.pdf written")


if __name__ == "__main__":
    main()

"""Summarize the port baseline run (scripts/bench_baseline.sh output directory).

Usage: python summarize.py <bench-dir>
Prints median ms/step [min-max] (n windows) per step-harness cell and the
median of each frozen suite backend per case (milliseconds).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def cells(path, keys):
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            groups[tuple(row[k] for k in keys)].append(float(row["ms_per_step"]))
    return groups


def show(title, path, keys):
    print(f"## {title}\n\n| {' | '.join(keys)} | median ms/step [min-max] (n) |")
    print("|" + "---|" * len(keys) + "---:|")
    for key, values in cells(path, keys).items():
        print(f"| {' | '.join(key)} | {statistics.median(values):.4g} [{min(values):.4g}-{max(values):.4g}] ({len(values)}) |")
    print()


def frozen(path):
    rows = list(csv.DictReader(open(path, newline="")))
    backends = sorted({r["backend"] for r in rows})
    print(f"## {path.stem}: median_ms per case\n\n| case | family | topology | batch | " + " | ".join(backends) + " |")
    print("|---|---|---|---:|" + "---:|" * len(backends))
    by_case = defaultdict(dict)
    meta = {}
    for r in rows:
        by_case[r["case"]][r["backend"]] = float(r["median_ms"])
        meta[r["case"]] = (r["family"], r["topology"], r["batch"])
    for case, values in by_case.items():
        cols = [f"{values[b]:.4g}" if b in values else "" for b in backends]
        print(f"| {case} | {' | '.join(meta[case])} | " + " | ".join(cols) + " |")
    print()


d = Path(sys.argv[1])
show("Full step, residual_train_bench (C9 matched protocol, status interval 1)", d / "residual-step.csv",
     ["precision", "branch", "case"])
show("C9 train_bench step mode", d / "c9-step.csv", ["precision", "protocol", "interval", "case"])
show("C6-C8 rational_train_bench", d / "rational-step.csv", ["precision", "policy", "case"])
for suite in ("m2", "m3", "m4"):
    frozen(d / f"frozen-{suite}.csv")

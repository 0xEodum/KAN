"""Per-step GPU kernel time from nsys cuda_gpu_kern_sum CSVs (profile.sh).

Normalizes by the number of candidate_kernel launches (one per training step)
and prints, per kernel family, launches per step and microseconds per step.
Usage: python kernels.py <kernels.csv> [...]
"""
import csv
import io
import re
import sys
from collections import defaultdict


def short(name):
    name = name.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    name = re.sub(r"\(.*", "", name)
    name = re.sub(r"<.*", "", name)
    name = name.replace("void ", "").replace("kan::cuda::", "").replace("(anonymous namespace)::", "")
    return name.strip()


def load(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    start = text.find('"Time (%)"')
    if start < 0:
        start = text.find("Time (%)")
    rows = list(csv.DictReader(io.StringIO(text[start:])))
    totals, counts = defaultdict(float), defaultdict(int)
    for row in rows:
        key = short(row["Name"])
        totals[key] += float(row["Total Time (ns)"])
        counts[key] += int(row["Instances"])
    return totals, counts


def main():
    for path in sys.argv[1:]:
        totals, counts = load(path)
        steps = counts.get("candidate_kernel", 1)
        print(f"## {path.split('/')[-1].replace('-kernels.csv', '')} ({steps} steps)")
        print("| kernel | launches/step | us/step |")
        print("|---|---:|---:|")
        for key in sorted(totals, key=lambda k: -totals[k]):
            print(f"| {key} | {counts[key] / steps:.2f} | {totals[key] / steps / 1000:.2f} |")
        total = sum(totals.values()) / steps / 1000
        launches = sum(counts.values()) / steps
        print(f"| **total** | {launches:.2f} | {total:.2f} |\n")


if __name__ == "__main__":
    main()

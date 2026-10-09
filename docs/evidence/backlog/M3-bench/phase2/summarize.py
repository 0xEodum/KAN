"""Medians of run_bench.sh samples: branch off vs baseline, branch on vs off.

Per (precision, case): median ms/step [min-max] (n) of base/off, final/off,
final/on; ratios off/base and on/off; and whether the off checksums equal
the baseline's (same trajectory).
"""
import csv
import statistics
import sys
from collections import defaultdict


def main():
    samples, checksums = defaultdict(list), defaultdict(set)
    order = []
    for row in csv.DictReader(open(sys.argv[1])):
        key = (row["precision"], row["case"])
        if key not in order:
            order.append(key)
        cell = (row["impl"], row["branch"])
        samples[key + cell].append(float(row["ms_per_step"]))
        if row["window"] == "2":
            checksums[key + cell].add(row["checksum"])

    def cell(key, impl, branch):
        v = samples.get(key + (impl, branch), [])
        if not v:
            return None, "-"
        m = statistics.median(v)
        return m, f"{m:.4g} [{min(v):.4g}-{max(v):.4g}] ({len(v)})"

    print("| precision | case | base off | phase 2 off | phase 2 on | off / base | on / off | checksum off = base |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for key in order:
        b, bt = cell(key, "base", "0")
        f, ft = cell(key, "final", "0")
        o, ot = cell(key, "final", "1")
        r1 = f"{f / b:.3f}" if b and f else "-"
        r2 = f"{o / f:.3f}" if f and o else "-"
        same = checksums.get(key + ("base", "0")) == checksums.get(key + ("final", "0"))
        print(f"| {key[0]} | {key[1]} | {bt} | {ft} | {ot} | {r1} | {r2} | {'yes' if same else 'NO'} |")


if __name__ == "__main__":
    main()

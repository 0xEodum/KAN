"""Median and spread of C6-C8 step samples (samples.csv from run_bench.sh)."""
import csv
import statistics
import sys
from collections import defaultdict

CASES = ["64x64x32x16-b1024", "256x256x256x10-b2048", "256x256x256x10-b8192"]


def main():
    samples, checksums = defaultdict(list), defaultdict(set)
    for row in csv.DictReader(open(sys.argv[1])):
        key = (row["precision"], row["policy"], row["case"], row["impl"])
        samples[key].append(float(row["ms_per_step"]))
        if row["checksum"] != "0":
            checksums[key[:-1]].add(row["checksum"])
    print("| precision | policy | case | baseline ms/step | final ms/step | final/baseline | checksums equal |")
    print("|---|---|---|---:|---:|---:|---|")
    for precision in ("f64", "f32"):
        for policy in ("guarded", "absolute", "smooth"):
            for case in CASES:
                cells, medians = [], {}
                for impl in ("base", "final"):
                    v = samples.get((precision, policy, case, impl), [])
                    if not v:
                        cells.append("-")
                        continue
                    medians[impl] = statistics.median(v)
                    cells.append(f"{medians[impl]:.3f} [{min(v):.3f}-{max(v):.3f}] ({len(v)})")
                if not medians:
                    continue
                ratio = f"{medians['final'] / medians['base']:.3f}" if len(medians) == 2 else "-"
                equal = len(checksums.get((precision, policy, case), set())) <= 1
                print(f"| {precision} | {policy} | {case} | " + " | ".join(cells) + f" | {ratio} | {'yes' if equal else 'NO'} |")


if __name__ == "__main__":
    main()

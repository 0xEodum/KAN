"""Compare frozen benchmark CSVs of two builds run in ABBA order.

Usage: python compare_bench.py <dir> <before-tag> <after-tag>
Expects <dir>/<suite>-<tag>-<n>.csv from m2/m3/m4_benchmark --backend all.
Pools the per-repeat samples of every run of one build, reports the pooled
median ratio after/before per (suite, backend) and checks that checksums agree.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(directory, tag):
    samples = defaultdict(list)
    checksums = {}
    for path in sorted(Path(directory).glob(f"*-{tag}-*.csv")):
        suite = path.name.split("-")[0]
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (suite, row["case"], row["family"], row["topology"], row["batch"], row["backend"])
                samples[key] += [float(v) for v in row["samples_ms"].split(";")]
                checksums[key] = (row["output_checksum"], row["gradient_checksum"], row["parameter_checksum"])
    return samples, checksums


def main(directory, before, after):
    base, base_sums = load(directory, before)
    new, new_sums = load(directory, after)
    groups = defaultdict(list)
    mismatched = []
    for key in sorted(base.keys() & new.keys()):
        ratio = statistics.median(new[key]) / statistics.median(base[key])
        groups[(key[0], key[5])].append(ratio)
        if base_sums[key] != new_sums[key]:
            mismatched.append(key)
    print("suite,backend,cases,geomean_ratio,min_ratio,max_ratio")
    for (suite, backend), ratios in sorted(groups.items()):
        geomean = statistics.geometric_mean(ratios)
        print(f"{suite},{backend},{len(ratios)},{geomean:.3f},{min(ratios):.3f},{max(ratios):.3f}")
    print(f"checksum mismatches: {len(mismatched)}")
    for key in mismatched:
        print("  ", key)


if __name__ == "__main__":
    main(*sys.argv[1:4])

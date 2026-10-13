"""Compare frozen benchmark CSVs of two builds.

Usage:
  python compare_bench.py <dir> <before-tag> <after-tag>
      ABBA runs: <dir>/<suite>-<tag>-<n>.csv from <suite>_benchmark --backend all.
  python compare_bench.py --frozen <before-dir> <after-dir>
      One pass each: <dir>/frozen-<suite>.csv from scripts/bench_baseline.sh.
Pools the per-repeat samples of every run of one build, reports the pooled
median ratio after/before per (suite, backend) and checks that checksums agree.
Suite and backend labels recorded before backlog R5 (m2/m3/m4, m1_host_full,
m<k>_resident_full, m<k>_transfer_full) are mapped to the current ones
(bench_labels.py), so either side may use either naming. Keys present on one
side only are listed and make the exit status non-zero; checksum mismatches are
listed but do not (they are expected across builds that change summation order,
e.g. C2, or between the parity and FMA builds).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import bench_labels


def read(path, suite, samples, checksums):
    suite = bench_labels.suite(suite)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (suite, row["case"], row["family"], row["topology"], row["batch"],
                   bench_labels.backend(row["backend"]))
            samples[key] += [float(v) for v in row["samples_ms"].split(";")]
            checksums[key] = (row["output_checksum"], row["gradient_checksum"], row["parameter_checksum"])


def load(directory, tag):
    samples, checksums = defaultdict(list), {}
    for path in sorted(Path(directory).glob(f"*-{tag}-*.csv")):
        suite = path.name.split("-")[0]
        if bench_labels.is_suite(suite):
            read(path, suite, samples, checksums)
    return samples, checksums


def load_frozen(directory):
    samples, checksums = defaultdict(list), {}
    for path in sorted(Path(directory).glob("frozen-*.csv")):
        suite = path.stem[len("frozen-"):]
        if bench_labels.is_suite(suite):
            read(path, suite, samples, checksums)
    return samples, checksums


def compare(base, base_sums, new, new_sums):
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
    unmatched = 0
    for side, keys in (("before", base.keys() - new.keys()), ("after", new.keys() - base.keys())):
        for key in sorted(keys):
            print(f"only {side}: {key}")
            unmatched += 1
    return 1 if unmatched or not groups else 0


def main(args):
    if len(args) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    if args[0] == "--frozen":
        return compare(*load_frozen(args[1]), *load_frozen(args[2]))
    return compare(*load(args[0], args[1]), *load(args[0], args[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

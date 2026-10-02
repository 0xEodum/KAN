"""List per-case pooled-median ratios above a threshold with each run's median.

Usage: python outliers.py <dir> <before-tag> <after-tag> [threshold]
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def runs(directory, tag):
    result = defaultdict(dict)
    for path in sorted(Path(directory).glob(f"*-{tag}-*.csv")):
        suite = path.name.split("-")[0]
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (suite, row["case"], row["family"], row["batch"], row["backend"])
                result[key][path.stem] = [float(v) for v in row["samples_ms"].split(";")]
    return result


def main(directory, before, after, threshold="1.1"):
    base, new = runs(directory, before), runs(directory, after)
    for key in sorted(base.keys() & new.keys()):
        pooled = lambda r: statistics.median([v for samples in r.values() for v in samples])
        ratio = pooled(new[key]) / pooled(base[key])
        if ratio > float(threshold):
            medians = {k: round(statistics.median(v), 3) for k, v in {**base[key], **new[key]}.items()}
            print(f"{key} ratio={ratio:.3f} {medians}")


if __name__ == "__main__":
    main(*sys.argv[1:])

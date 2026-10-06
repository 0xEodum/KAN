"""Median and spread of C3 step samples (samples.csv from run_bench.sh)."""
import csv
import statistics
import sys
from collections import defaultdict

CASES = ["64x64x32x16-b1024", "16x24x8-b1024", "256x256x256x10-b8192", "1024x1024x1024-b4096"]


def main():
    samples, checksums = defaultdict(list), defaultdict(set)
    for row in csv.DictReader(open(sys.argv[1])):
        key = (row["precision"], row["protocol"], row["interval"], row["case"], row["impl"])
        samples[key].append(float(row["ms_per_step"]))
        if row["checksum"] != "0":
            checksums[key[:-1]].add((row["impl"], row["checksum"]))
    for precision in ("f32", "tf32", "f64"):
        for protocol in ("matched", "realistic"):
            print(f"## {precision} {protocol}: median ms/step [min-max] (n), C3 / pre-C3")
            print("| case | N | pre-C3 | C3 | ratio | checksums equal |")
            print("|---|---:|---:|---:|---:|---|")
            for case in CASES:
                for interval in ("1", "64"):
                    cells = []
                    medians = {}
                    for impl in ("pre-C3", "C3"):
                        v = samples.get((precision, protocol, interval, case, impl), [])
                        if not v:
                            cells.append("-")
                            continue
                        medians[impl] = statistics.median(v)
                        cells.append(f"{medians[impl]:.3f} [{min(v):.3f}-{max(v):.3f}] ({len(v)})")
                    ratio = f"{medians['C3'] / medians['pre-C3']:.3f}" if len(medians) == 2 else "-"
                    sums = checksums.get((precision, protocol, interval, case), set())
                    equal = len({c for _, c in sums}) <= 1
                    print(f"| {case} | {interval} | " + " | ".join(cells) + f" | {ratio} | {'yes' if equal else 'NO'} |")
            print()


if __name__ == "__main__":
    main()

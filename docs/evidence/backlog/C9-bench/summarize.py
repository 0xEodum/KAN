"""Median and spread of C9 benchmark samples (samples.csv from run_bench.sh)."""
import csv
import statistics
import sys
from collections import defaultdict

CASES = ["64x64x32x16-b1024", "16x24x8-b1024", "256x256x256x10-b8192", "1024x1024x1024-b4096"]
COLUMNS = [("kan-preC9", "eager", "1", "KAN pre-C9 eager"), ("kan", "step", "1", "KAN step N=1"),
           ("kan", "step", "64", "KAN step N=64"), ("torch", "eager", "0", "torch eager"),
           ("torch", "graph", "0", "torch CUDA graph"), ("torch", "compile-cudagraphs", "0", "torch.compile cudagraphs")]


def main():
    samples = defaultdict(list)
    for row in csv.DictReader(open(sys.argv[1])):
        key = (row["impl"], row["mode"], row["interval"], row["precision"], row["protocol"], row["case"])
        samples[key].append(float(row["ms_per_step"]))
    for precision in ("f32", "tf32", "f64"):
        for protocol in ("matched", "realistic"):
            print(f"## {precision} {protocol}: median ms/step [min-max] (n samples)")
            print("| case | " + " | ".join(c[3] for c in COLUMNS) + " | best KAN / torch eager | best KAN / best torch |")
            print("|---|" + "---:|" * (len(COLUMNS) + 2))
            for case in CASES:
                cells, medians = [], {}
                for impl, mode, interval, label in COLUMNS:
                    v = samples.get((impl, mode, interval, precision, protocol, case), [])
                    if not v:
                        cells.append("-")
                        continue
                    m = statistics.median(v)
                    medians[label] = m
                    cells.append(f"{m:.3f} [{min(v):.3f}-{max(v):.3f}] ({len(v)})")
                kan = [medians[c[3]] for c in COLUMNS[1:3] if c[3] in medians]
                torch = [medians[c[3]] for c in COLUMNS[3:] if c[3] in medians]
                eager = medians.get("torch eager")
                ratio1 = f"{min(kan) / eager:.2f}" if kan and eager else "-"
                ratio2 = f"{min(kan) / min(torch):.2f}" if kan and torch else "-"
                print(f"| {case} | " + " | ".join(cells) + f" | {ratio1} | {ratio2} |")
            print()


if __name__ == "__main__":
    main()

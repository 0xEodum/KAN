"""Summarize Nsight Compute --csv --page details output: one row per kernel launch."""
import csv
import io
import sys

WANTED = ["Duration", "Memory Throughput", "DRAM Throughput", "Compute (SM) Throughput",
          "Registers Per Thread", "Achieved Occupancy", "Theoretical Occupancy", "Grid Size", "Waves Per SM"]

for path in sys.argv[1:]:
    lines = [l for l in open(path, encoding="utf-8", errors="replace") if l.startswith('"')]
    rows = list(csv.DictReader(io.StringIO("".join(lines))))
    kernels = {}
    for r in rows:
        key = (r["ID"], r["Kernel Name"].split("(")[0].split("::")[-1])
        kernels.setdefault(key, {})[r["Metric Name"]] = f'{r["Metric Value"]} {r["Metric Unit"]}'.strip()
    for (i, name), m in kernels.items():
        print(f"{path.split(chr(92))[-1]} #{i} {name}: " + "; ".join(f"{k}={m[k]}" for k in WANTED if k in m))

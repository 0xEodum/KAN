"""Summarize run_abba.sh output: per configuration the median step time of all
windows of each build, the ratio new/base and whether every run of both
builds printed the same parameter hash (bitwise-identical trajectories).
Usage: python summarize.py abba.csv"""
import csv, statistics, sys
from collections import defaultdict
times, hashes, order = defaultdict(list), defaultdict(set), []
for row in csv.DictReader(open(sys.argv[1])):
    key = (row["precision"], row["topology"], row["batch"], row["K"])
    if key not in order: order.append(key)
    times[key + (row["build"],)].append(float(row["step_ms"]))
    hashes[key].add(row["hash"])
print("| Precision | Topology | Batch | K | Base ms | C12 ms | C12/base | Hash equal |")
print("|---|---|---:|---:|---:|---:|---:|---|")
for key in order:
    b, n = statistics.median(times[key + ("base",)]), statistics.median(times[key + ("new",)])
    print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {b:.4g} | {n:.4g} | {n/b:.3f} | {'yes' if len(hashes[key]) == 1 else 'NO'} |")

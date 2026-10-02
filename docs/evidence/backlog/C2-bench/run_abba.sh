#!/usr/bin/env bash
# ABBA frozen benchmarks: base, after, after, base for m2/m3/m4 (--backend all).
# Usage: run_abba.sh <base-build> <after-build> <after-tag> <out-dir>
set -euo pipefail
base=$1 after=$2 tag=$3 out=$4
mkdir -p "$out"
run() { "./$1/$2_benchmark.exe" --backend all --warmups 2 --repeats 7 > "$out/$2-$3-$4.csv"; }
for suite in m2 m3 m4; do
  run "$base" $suite base 0; run "$after" $suite "$tag" 0; run "$after" $suite "$tag" 1; run "$base" $suite base 1
done
python docs/evidence/backlog/compare_bench.py "$out" base "$tag"

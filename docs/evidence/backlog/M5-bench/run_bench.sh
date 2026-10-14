#!/usr/bin/env bash
# Backlog M5 timing sweep: ABBA order (physicists, function, function,
# physicists) per configuration, 3 windows per run, rate 1e-12; CSV rows on stdout.
# Usage (repository root): run_bench.sh <hermite_bench exe> > bench.csv
set -euo pipefail
exe=$1
header=1
run() { # precision normalization topology batch terms
  if ((header)); then "$exe" "$@"; header=0; else "$exe" "$@" | tail -n +2; fi
}
abba() { local p=$1; shift; for n in physicists function function physicists; do run "$p" "$n" "$@" 3 1e-12; done; }
for terms in 8 16 32; do
  for precision in f32 tf32 f64; do abba $precision 256x256x256x10 2048 $terms; done
  abba f32 1024x1024x1024x10 2048 $terms
  abba cpu 64x64x64x10 256 $terms
done

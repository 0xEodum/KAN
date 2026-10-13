#!/usr/bin/env bash
# Backlog C5 crossover sweep: spline_bench (forced dense vs sparse paths) over
# precision x topology x term count, ABBA order (dense, sparse, sparse, dense),
# 3 timing windows per process, one process per run (PATHS overrides the
# order, e.g. PATHS="dense auto auto dense"). Prints CSV rows.
# Usage: run_sweep.sh <spline_bench exe> [configs file] > sweep.csv
set -uo pipefail
exe=$1
configs=${2:-/dev/stdin}
header=1
while read -r precision topology batch degree terms; do
  [[ -z ${precision:-} || $precision == \#* ]] && continue
  for path in ${PATHS:-dense sparse sparse dense}; do
    out=$("$exe" "$precision" "$path" "$topology" "$batch" "$degree" "$terms" 3) || { echo "FAILED $precision $path $topology $batch $degree $terms" >&2; continue; }
    if ((header)); then echo "$out" | head -1; header=0; fi
    echo "$out" | tail -n +2
  done
done < "$configs"

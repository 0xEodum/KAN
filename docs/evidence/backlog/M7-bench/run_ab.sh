#!/usr/bin/env bash
# Backlog M7 review: complete train_step of the wavelet networks before and
# after the review fixes, interleaved A B B A (one process per run, 3 timing
# windows each); A and B are wavelet_step executables linked against the two
# builds (scripts/link_harness.sh).
# Usage: run_ab.sh <exe A> <exe B> [configs] > ab.csv
# Config lines: <precision> <widths> <batch> <K>
set -uo pipefail
a=$1 b=$2 configs=${3:-/dev/stdin}
echo "build,precision,carrier,topology,batch,K,parameters,window,steps,step_ms,hash"
while read -r precision topology batch K; do
  [[ -z ${precision:-} || $precision == \#* ]] && continue
  for side in A B B A; do
    exe=$a; [[ $side == B ]] && exe=$b
    "$exe" "$precision" "$topology" "$batch" wavelet "$K" 3 | tail -n +2 | sed "s/^/$side,/" || echo "FAILED $side $precision $topology $batch $K" >&2
  done
done < "$configs"

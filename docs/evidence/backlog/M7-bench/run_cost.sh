#!/usr/bin/env bash
# Backlog M7 cost of trainability: full train_step of WaveletEdges networks
# with K wavelets per edge against fixed Mexican-hat dictionaries with 3K terms
# (the same 3K parameters per edge) and K terms (the same term count), in one
# build, interleaved A B C C B A (one process per run, 3 timing windows each).
# Usage: run_cost.sh <wavelet_step exe> [configs] > cost.csv
# Config lines: <precision> <widths> <batch> <K>
set -uo pipefail
exe=$1 configs=${2:-/dev/stdin}
echo "precision,carrier,topology,batch,K,parameters,window,steps,step_ms,hash"
while read -r precision topology batch K; do
  [[ -z ${precision:-} || $precision == \#* ]] && continue
  for run in "wavelet $K" "hat $((3*K))" "hat $K" "hat $K" "hat $((3*K))" "wavelet $K"; do
    set -- $run
    "$exe" "$precision" "$topology" "$batch" "$1" "$2" 3 | tail -n +2 || echo "FAILED $run $precision $topology $batch" >&2
  done
done < "$configs"

#!/usr/bin/env bash
# Backlog M7 launch-rule check: full train_step of wavelet networks with the
# executor's launch choice (fs = bs = 0) against forced samples per thread of
# the forward (fs) and backward (bs) kernels, slots and splits automatic.
# Usage: run_sweep.sh <wavelet_step exe built with -DM7_HOOK> [configs] > sweep.csv
# Config lines: <precision> <widths> <batch> <K>
set -uo pipefail
exe=$1 configs=${2:-/dev/stdin}
echo "fs,bs,precision,carrier,topology,batch,K,parameters,window,steps,step_ms,hash"
while read -r precision topology batch K; do
  [[ -z ${precision:-} || $precision == \#* ]] && continue
  for fb in "0 0" "1 0" "2 0" "4 0" "8 0" "0 1" "0 2" "0 4" "0 8"; do
    set -- $fb
    # FP64 instantiates forward S = 1 and backward S <= 4 only.
    [[ $precision == f64 && ( $1 -gt 1 || $2 == 8 ) ]] && continue
    "$exe" "$precision" "$topology" "$batch" wavelet "$K" 2 "$1" "$2" 0 0 | tail -n +2 | sed "s/^/$1,$2,/" || echo "FAILED $fb $precision $topology $batch $K" >&2
  done
done < "$configs"

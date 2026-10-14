#!/usr/bin/env bash
# Backlog C12 ABBA full-step comparison: rbf_step linked against the baseline
# (HEAD before C12) and the C12 build, order base, new, new, base, one process
# per run, 3 timing windows each. Prints CSV rows with a leading build column.
# Usage: run_abba.sh <base exe> <new exe> [configs file] > abba.csv
# Config lines: <precision> <widths> <batch> <K>
set -uo pipefail
base=$1 new=$2 configs=${3:-/dev/stdin}
echo "build,precision,path,topology,batch,K,window,steps,step_ms,hash"
while read -r precision topology batch K; do
  [[ -z ${precision:-} || $precision == \#* ]] && continue
  for tag in base new new base; do
    exe=$base; [[ $tag == new ]] && exe=$new
    "$exe" "$precision" "$topology" "$batch" "$K" 3 | tail -n +2 | sed "s/^/$tag,/" || echo "FAILED $tag $precision $topology $batch $K" >&2
  done
done < "$configs"

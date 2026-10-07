#!/usr/bin/env bash
# C6-C8 step benchmark: baseline vs final rational_train_bench, interleaved ABBA.
# Usage (Git Bash, repository root): run_bench.sh <baseline exe> <final exe> <out-dir> [reps]
# Repetition r runs the baseline first when r is odd and the final build first
# when r is even; each process runs every case with 3 timing windows, so each
# cell has n = 3*reps windows.
set -uo pipefail
base=$1 final=$2 out=$3 reps=${4:-3}
mkdir -p "$out"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$out/gpu-before.txt"
one() { # exe tag precision policy
  "$1" "$3" "$4" 3 | tail -n +2 | sed "s/^kan,/$2,/"
}
echo "rep,impl,precision,policy,case,window,steps,ms_per_step,checksum" > "$out/samples.csv"
for r in $(seq 1 "$reps"); do
  for p in f64 f32; do
    for pol in guarded absolute smooth; do
      if (( r % 2 )); then one "$base" base $p $pol; one "$final" final $p $pol
      else one "$final" final $p $pol; one "$base" base $p $pol; fi
    done
  done | sed "s/^/$r,/" >> "$out/samples.csv"
  echo "rep $r done $(date +%T)" >&2
done
nvidia-smi --query-gpu=name,driver_version,clocks.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-after.txt"
python docs/evidence/backlog/C6-C8-bench/summarize.py "$out/samples.csv" > "$out/summary.txt"

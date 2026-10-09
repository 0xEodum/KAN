#!/usr/bin/env bash
# M3 phase 2 step benchmark, interleaved ABBA (C9 protocol: matched train_step,
# status interval 1). Per repetition and precision the baseline (2437421,
# branch off), phase 2 (branch off) and phase 2 (branch on) processes run in
# forward order on odd and reverse order on even repetitions; each process runs
# every case with 3 timing windows, so each cell has n = 3*reps windows.
# TF32 runs the 1024-wide case only.
# Usage (Git Bash, repository root): run_bench.sh <base exe> <phase-2 exe> <out-dir> [reps]
set -uo pipefail
base=$1 final=$2 out=$3 reps=${4:-3}
mkdir -p "$out"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$out/gpu-before.txt"
one() { # exe tag precision branch [case]
  "$1" "$3" "$4" 3 ${5:-} | tail -n +2 | sed "s/^/$2,/"
}
echo "rep,impl,precision,branch,case,window,steps,ms_per_step,checksum" > "$out/samples.csv"
for r in $(seq 1 "$reps"); do
  for p in f64 f32 tf32; do
    c=""; [ "$p" = tf32 ] && c=1024x1024x1024-b4096
    if (( r % 2 )); then one "$base" base $p 0 $c; one "$final" final $p 0 $c; one "$final" final $p 1 $c
    else one "$final" final $p 1 $c; one "$final" final $p 0 $c; one "$base" base $p 0 $c; fi
  done | sed "s/^/$r,/" >> "$out/samples.csv"
  echo "rep $r done $(date +%T)" >&2
done
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-after.txt"
python docs/evidence/backlog/M3-bench/phase2/summarize.py "$out/samples.csv" > "$out/summary.md"

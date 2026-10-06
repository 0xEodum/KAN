#!/usr/bin/env bash
# C3 step benchmark: pre-C3 vs C3 train_bench (C9 harness), interleaved ABBA.
# Usage (Git Bash, repository root): run_bench.sh <pre-C3 bench exe> <C3 bench exe> <out-dir> [reps]
# Repetition r runs pre-C3 first when r is odd and C3 first when r is even;
# each process runs every topology with 3 timing windows (C9 protocol).
set -uo pipefail
base=$1 c3=$2 out=$3 reps=${4:-3}
mkdir -p "$out"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$out/gpu-before.txt"
one() { # exe tag precision protocol interval
  "$1" "$3" "$4" step "$5" 3 | tail -n +2 | sed "s/^kan,/$2,/"
}
echo "rep,impl,precision,protocol,mode,interval,case,window,steps,ms_per_step,checksum" > "$out/samples.csv"
for r in $(seq 1 "$reps"); do
  for p in f32 tf32 f64; do
    for proto in matched realistic; do
      for n in 1 64; do
        if (( r % 2 )); then one "$base" pre-C3 $p $proto $n; one "$c3" C3 $p $proto $n
        else one "$c3" C3 $p $proto $n; one "$base" pre-C3 $p $proto $n; fi
      done
    done
  done | sed "s/^/$r,/" >> "$out/samples.csv"
  echo "rep $r done $(date +%T)" >&2
done
nvidia-smi --query-gpu=name,driver_version,clocks.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-after.txt"
python docs/evidence/backlog/C3-bench/summarize.py "$out/samples.csv" > "$out/summary.txt"

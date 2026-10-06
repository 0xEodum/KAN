#!/usr/bin/env bash
# C9 matched/realistic benchmark against PyTorch, interleaved ABBA.
# Usage (Git Bash, repository root): run_bench.sh <pre-C9 bench exe> <C9 bench exe> <python> <out-dir> [reps]
# Each process runs all four topologies with 3 timing windows; repetition r
# runs KAN first when r is odd and PyTorch first when r is even, so neither
# side systematically follows the other's thermal/clock state.
set -uo pipefail
base=$1 c9=$2 py=$3 out=$4 reps=${5:-3}
torch=docs/evidence/backlog/C9-bench/torch_bench.py
mkdir -p "$out"; rm -f "$out/samples-raw.csv"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$out/gpu-before.txt"
kan() {
  local p=$1 proto=$2
  "$base" "$p" "$proto" eager 1 3 | tail -n +2 | sed 's/^kan,/kan-preC9,/'
  "$c9" "$p" "$proto" step 1 3 | tail -n +2
  "$c9" "$p" "$proto" step 64 3 | tail -n +2
}
pt() {
  local p=$1 proto=$2
  for m in eager graph compile-cudagraphs; do "$py" "$torch" "$p" "$proto" $m 3 2>/dev/null | tail -n +2; done
}
echo "impl,precision,protocol,mode,interval,case,window,steps,ms_per_step,checksum" > "$out/samples.csv"
for r in $(seq 1 "$reps"); do
  for p in f32 tf32 f64; do
    for proto in matched realistic; do
      if (( r % 2 )); then kan $p $proto; pt $p $proto; else pt $p $proto; kan $p $proto; fi
    done
  done | sed "s/^/$r,/" >> "$out/samples-raw.csv"
  echo "rep $r done $(date +%T)" >&2
done
cut -d, -f2- "$out/samples-raw.csv" >> "$out/samples.csv"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,utilization.gpu,power.draw --format=csv > "$out/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$out/gpu-after.txt"
"$py" docs/evidence/backlog/C9-bench/summarize.py "$out/samples.csv" > "$out/summary.txt"

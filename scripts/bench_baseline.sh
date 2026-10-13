#!/usr/bin/env bash
# One-pass GB10/Linux reference run of the frozen suites and the current full-step
# training harnesses (port step; no A/B comparison). Records the GPU state before
# and after, and nvidia-smi dmon samples during the run, so contention is visible.
# Usage (repository root): scripts/bench_baseline.sh <cuda-build-dir> <out-dir> [windows]
#   <cuda-build-dir> must be configured with -DKAN_BUILD_BENCHMARKS=ON.
# The harnesses are linked into <out-dir>/bin with scripts/link_harness.sh (delete it
# before committing an evidence directory). gpu-after.txt is taken right after the
# last benchmark, so it shows this run's own load.
set -uo pipefail
failed=0
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=$1 out=$2 windows=${3:-3}
ev="$root/docs/evidence/backlog"
mkdir -p "$out/bin"
harness() { "$root/scripts/link_harness.sh" "$build" "$ev/$1" "$out/bin/$2" "${@:3}" || exit 1; }
harness C9-bench/train_bench.cpp train_bench -DKAN_C9_API
harness M3-bench/phase2/residual_train_bench.cpp residual_train_bench
harness C6-C8-bench/rational_train_bench.cpp rational_train_bench
# Appends a command's CSV rows (header dropped) to a file; records failures.
rows() { local file=$1; shift; "$@" > "$out/.rows" || { echo "failed: $*" >&2; failed=1; }; tail -n +2 "$out/.rows" >> "$file"; }
gpu_state() {
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,utilization.gpu,power.draw,temperature.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
}
gpu_state > "$out/gpu-before.txt"
nvidia-smi dmon -s puc -d 1 > "$out/dmon.txt" 2>&1 &
dmon=$!
trap 'kill $dmon 2>/dev/null' EXIT
for suite in m2 m3 m4; do
  echo "frozen $suite $(date +%T)" >&2
  "$build/${suite}_benchmark" --warmups 2 --repeats 7 > "$out/frozen-$suite.csv" || { echo "failed: $suite" >&2; failed=1; }
done
echo "precision,branch,case,window,steps,ms_per_step,checksum" > "$out/residual-step.csv"
for p in f64 f32 tf32; do
  for branch in 0 1; do
    echo "residual $p $branch $(date +%T)" >&2
    rows "$out/residual-step.csv" "$out/bin/residual_train_bench" $p $branch "$windows"
  done
done
echo "impl,precision,protocol,mode,interval,case,window,steps,ms_per_step,checksum" > "$out/c9-step.csv"
for p in f64 f32 tf32; do
  for proto in matched realistic; do
    for interval in 1 64; do
      echo "c9 $p $proto $interval $(date +%T)" >&2
      rows "$out/c9-step.csv" "$out/bin/train_bench" $p $proto step $interval "$windows"
    done
  done
done
echo "impl,precision,policy,case,window,steps,ms_per_step,checksum" > "$out/rational-step.csv"
for p in f64 f32; do
  for pol in guarded absolute smooth; do
    echo "rational $p $pol $(date +%T)" >&2
    rows "$out/rational-step.csv" "$out/bin/rational_train_bench" $p $pol "$windows"
  done
done
gpu_state > "$out/gpu-after.txt"
rm -f "$out/.rows"
exit $failed

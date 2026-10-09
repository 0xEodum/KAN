#!/usr/bin/env bash
# M3 phase 2: Nsight Systems kernel summary of residual_train_bench (graph
# nodes traced individually), one case, branch off and on.
# Usage (Git Bash): profile.sh <bench.exe> <f64|f32|tf32> <case> <out-dir>
set -uo pipefail
exe=$1 p=$2 case=$3 out=$4
nsys="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
mkdir -p "$out"
for branch in 0 1; do
  rep="$out/$p-$case-branch$branch"
  "$nsys" profile --force-overwrite true --trace cuda --cuda-graph-trace node -o "$rep" "$exe" "$p" "$branch" 1 "$case" > /dev/null
  "$nsys" stats --force-export true --report cuda_gpu_kern_sum --format csv "$rep.nsys-rep" > "$rep-kernels.csv" 2>/dev/null
  rm -f "$rep.sqlite"
done

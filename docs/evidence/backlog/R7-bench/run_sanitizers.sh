#!/usr/bin/env bash
# R7: compute-sanitizer on the legacy API suite (Git Bash, from the repo root).
# Usage: docs/evidence/backlog/R7-bench/run_sanitizers.sh <build-dir>
set -uo pipefail
build=$1
cs="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/compute-sanitizer/compute-sanitizer.exe"
for tool in memcheck racecheck initcheck synccheck; do
  args=(--tool "$tool")
  [ "$tool" = memcheck ] && args+=(--leak-check full)
  echo "== $tool ${args[*]:2} cuda_test"
  "$cs" "${args[@]}" "$build/cuda_test.exe" 2>&1 | grep -E "passed|SUMMARY|FAIL"
done

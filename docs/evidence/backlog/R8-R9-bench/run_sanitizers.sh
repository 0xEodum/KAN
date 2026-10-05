#!/usr/bin/env bash
# R8/R9: compute-sanitizer on the parameter-upload suite and the suites that use
# the moved device query (Git Bash, from the repo root).
# Usage: docs/evidence/backlog/R8-R9-bench/run_sanitizers.sh <build-dir>
set -uo pipefail
build=$1
cs="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/compute-sanitizer/compute-sanitizer.exe"
for suite in resident_upload_test cuda_test resident_test; do
  echo "== memcheck --leak-check full $suite"
  "$cs" --tool memcheck --leak-check full "$build/$suite.exe" 2>&1 | grep -E "passed|SUMMARY|FAIL"
done
for tool in racecheck synccheck initcheck; do
  echo "== $tool resident_upload_test"
  "$cs" --tool "$tool" "$build/resident_upload_test.exe" 2>&1 | grep -E "passed|SUMMARY|FAIL"
done

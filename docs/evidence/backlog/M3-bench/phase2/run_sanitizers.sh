#!/usr/bin/env bash
# M3 phase 2 sanitizers: memcheck (leak check) on the residual suite and the
# resident suites whose code paths changed (training, upload, precision,
# contraction, rational, C3 fused expansion), racecheck/synccheck/initcheck
# on the residual suite.
# Usage (Git Bash, repository root): run_sanitizers.sh <build> [<build> ...] > sanitizers.txt
set -uo pipefail
cs="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/compute-sanitizer/compute-sanitizer.exe"
for b in "$@"; do
  for t in residual_resident_test resident_training_test resident_upload_test resident_precision_test \
           resident_contraction_test rational_resident_execution_test resident_fused_expansion_test; do
    echo "== $b memcheck $t"; "$cs" --tool memcheck --leak-check full "$b/$t.exe" 2>&1 | grep -E "passed|ERROR SUMMARY|LEAK SUMMARY"
  done
  for tool in racecheck synccheck initcheck; do
    echo "== $b $tool residual_resident_test"; "$cs" --tool $tool "$b/residual_resident_test.exe" 2>&1 | grep -E "passed|SUMMARY"
  done
done

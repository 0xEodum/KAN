#!/usr/bin/env bash
# C3 sanitizers (C9 list + resident_fused_expansion_test). Usage: run_sanitizers.sh <parity-build> <fma-build> > sanitizers.txt
set -uo pipefail
cs="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/compute-sanitizer/compute-sanitizer.exe"
for b in "$@"; do
  for t in resident_training_test resident_upload_test resident_precision_test resident_contraction_test resident_test \
           resident_review_test m3_resident_test m4_resident_test input_map_resident_test rational_policy_resident_test resident_fused_expansion_test; do
    echo "== $b memcheck $t"; "$cs" --tool memcheck --leak-check full "$b/$t.exe" 2>&1 | grep -E "passed|ERROR SUMMARY|LEAK SUMMARY"
  done
  for tool in racecheck synccheck initcheck; do
    echo "== $b $tool resident_training_test"; "$cs" --tool $tool "$b/resident_training_test.exe" 2>&1 | grep -E "passed|SUMMARY"
  done
done

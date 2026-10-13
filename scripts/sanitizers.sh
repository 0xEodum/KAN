#!/usr/bin/env bash
# Linux compute-sanitizer pass over a CUDA build: memcheck with leak check on
# every resident CTest executable, racecheck/synccheck/initcheck on the suites
# with shared-memory tiles and graph capture. Prints the pass lines, the
# sanitizer summaries and, per run, the exit status and duration; exits
# non-zero if any run failed (a test failure or a sanitizer error).
# Known: under initcheck resident_fused_expansion's own cudaMemGetInfo-based
# reservation check fails (the tool's allocations are counted; see
# docs/evidence/port-aarch64.md). That run is printed but does not set the exit
# status unless the sanitizer itself reports errors.
# Usage (repository root): scripts/sanitizers.sh <build-dir> [<build-dir> ...] > sanitizers.txt
set -uo pipefail
cs=${COMPUTE_SANITIZER:-$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)}
memcheck_tests="cuda_test resident_test resident_review_test m3_resident_test input_map_resident_test m4_resident_test
  rational_policy_resident_test rational_resident_execution_test resident_contraction_test resident_precision_test
  resident_upload_test resident_training_test resident_fused_expansion_test initializer_resident_test residual_resident_test"
deep_tests="resident_fused_expansion_test resident_training_test residual_resident_test"
failed=0
run() { # build tool test [options]
  local b=$1 tool=$2 t=$3 start=$SECONDS; shift 3
  echo "== $b $tool $t"
  "$cs" --tool "$tool" --error-exitcode 99 "$@" "$b/$t" 2>&1 | grep -E "^[0-9]+/[0-9]+ passed|FAIL|SUMMARY|rror"
  local status=${PIPESTATUS[0]}
  echo "   (exit $status, $((SECONDS-start)) s)"
  if [[ $tool == initcheck && $t == resident_fused_expansion_test && $status != 99 ]]; then status=0; fi
  ((status == 0)) || failed=1
}
for b in "$@"; do
  for t in $memcheck_tests; do run "$b" memcheck "$t" --leak-check full; done
  for tool in synccheck initcheck racecheck; do
    for t in $deep_tests; do run "$b" "$tool" "$t"; done
  done
done
exit $failed

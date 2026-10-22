#!/usr/bin/env bash
# Backlog M6 compute-sanitizer pass: every tool on spline_grid_resident, memcheck
# (leak check) on the B-spline suites whose dense layers now run basis_grid_kernel.
# Usage (repository root): sanitizers.sh <build-dir> ... > sanitizers.txt
set -uo pipefail
cs=${COMPUTE_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}
failed=0
run() { local b=$1 tool=$2 t=$3 start=$SECONDS; shift 3
  echo "== $b $tool $t $*"
  "$cs" --tool "$tool" --error-exitcode 99 "$@" "$b/$t" 2>&1 | grep -E "^[0-9]+/[0-9]+ passed|FAIL|SUMMARY|rror"
  local status=${PIPESTATUS[0]}; echo "   (exit $status, $((SECONDS-start)) s)"; ((status == 0)) || failed=1; }
for b in "$@"; do
  run "$b" memcheck spline_grid_resident_test --leak-check full
  for tool in racecheck synccheck initcheck; do run "$b" $tool spline_grid_resident_test; done
  for t in local_resident_test local_sparse_resident_test local_spline_gradient_resident_test; do run "$b" memcheck $t --leak-check full; done
done
exit $failed

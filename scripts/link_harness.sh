#!/usr/bin/env bash
# Linux equivalent of the evidence harness link scripts (cl_tool.cmd, build_golden.cmd,
# build_tool.cmd, build_train_bench.cmd): compiles one harness source against a
# Release CUDA build tree with the in-tree host flags (-O2, no FMA contraction).
# Usage: scripts/link_harness.sh <build-dir> <source.cpp> <output-exe> [extra g++ flags, e.g. -DKAN_C9_API]
# Headers come from the include directory of the source tree the build was
# configured from (CMAKE_HOME_DIRECTORY in its CMakeCache.txt), so a worktree's
# build uses that worktree's headers.
set -euo pipefail
build=$(cd "$1" && pwd) source=$2 output=$3; shift 3
home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$build/CMakeCache.txt")
[[ -n $home && -d $home/include ]] || { echo "no source tree recorded in $build/CMakeCache.txt" >&2; exit 1; }
cuda=${CUDA_HOME:-/usr/local/cuda}
"${CXX:-g++}" -std=c++20 -O2 -ffp-contract=off "$@" -I"$home/include" "$source" \
    "$build/libkan_cuda.a" "$build/libkan.a" -L"$cuda/lib64" -lcudart -lcublas -pthread -o "$output"

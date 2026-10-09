#!/usr/bin/env bash
# M3 phase 2 byte-identity: builds the golden (default, --layers), C3 FP32 all-family
# and C6-C8 rational dumps from this tree's harness sources against each given
# Release CUDA build directory (absolute Windows paths; headers from <build>\..\include)
# and prints the SHA-256 of every dump.
# Usage (Git Bash, repository root): dumps.sh <out-dir> <tag>=<abs-build-dir> ...
set -uo pipefail
out=$1; shift
mkdir -p "$out"
root=$(pwd -W)
tool="$root/docs/evidence/backlog/M3-bench/phase2/cl_tool.cmd"
for pair in "$@"; do
  tag=${pair%%=*} build=${pair#*=}
  for src in golden/golden.cpp C3-bench/f32_dump.cpp C6-C8-bench/rational_dump.cpp; do
    name=$(basename "$src" .cpp)
    cmd //c "$(cygpath -w "$tool") $build $(cygpath -w "$root/docs/evidence/backlog/$src") $(cygpath -w "$out/$name-$tag.exe")" > "$out/$name-$tag.build.log" 2>&1 \
      || { echo "build failed: $name $tag"; continue; }
  done
  "$out/golden-$tag.exe" > "$out/golden-$tag.txt"
  "$out/golden-$tag.exe" --layers > "$out/golden-layers-$tag.txt"
  "$out/f32_dump-$tag.exe" > "$out/f32-$tag.txt"
  "$out/rational_dump-$tag.exe" > "$out/rational-$tag.txt"
done
(cd "$out" && sha256sum *.txt)

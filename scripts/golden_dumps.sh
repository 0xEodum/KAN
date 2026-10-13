#!/usr/bin/env bash
# Linux equivalent of docs/evidence/backlog/M3-bench/phase2/dumps.sh: builds the
# golden (default, --layers), C3 FP32 all-family and C6-C8 rational dumps against
# each given Release CUDA build and prints the SHA-256 of every dump. A tag whose
# harness fails to link, or whose dump exits non-zero, is reported and its dumps
# are removed; the script then exits non-zero.
# Usage (repository root): scripts/golden_dumps.sh <out-dir> <tag>=<build-dir> ...
set -uo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=$1; shift
mkdir -p "$out"
failed=0 written=()
dump() { # exe output [args]
  local exe=$1 file=$2; shift 2
  if "$exe" "$@" > "$file"; then written+=("$(basename "$file")")
  else echo "dump failed: $file" >&2; rm -f "$file"; failed=1; fi
}
for pair in "$@"; do
  tag=${pair%%=*} build=${pair#*=} linked=1
  for src in golden/golden.cpp C3-bench/f32_dump.cpp C6-C8-bench/rational_dump.cpp; do
    name=$(basename "$src" .cpp)
    "$root/scripts/link_harness.sh" "$build" "$root/docs/evidence/backlog/$src" "$out/$name-$tag" > "$out/$name-$tag.build.log" 2>&1 \
      || { echo "build failed: $name $tag (see $out/$name-$tag.build.log)" >&2; linked=0; }
  done
  if ((!linked)); then failed=1; continue; fi
  dump "$out/golden-$tag" "$out/golden-$tag.txt"
  dump "$out/golden-$tag" "$out/golden-layers-$tag.txt" --layers
  dump "$out/f32_dump-$tag" "$out/f32-$tag.txt"
  dump "$out/rational_dump-$tag" "$out/rational-$tag.txt"
done
((${#written[@]})) && (cd "$out" && sha256sum "${written[@]}")
exit $failed

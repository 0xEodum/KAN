#!/usr/bin/env bash
# Full R8/R9 evidence run (Git Bash, from the repository root).
# Usage: run_all.sh <base-build> <r8r9-build> <tmp-dir>
#   tmp-dir must hold golden_base.exe (golden.cpp linked against the base
#   build) and upload_timing.exe (build_upload_timing.cmd against r8r9-build);
#   golden_new.exe is linked here.
set -uo pipefail
base=$1 after=$2 tmp=$3
out=docs/evidence/backlog/R8-R9-bench
wtmp=$(cygpath -w "$tmp")

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv > "$out/gpu-before.txt"

docs/evidence/backlog/R8-R9-bench/run_sanitizers.sh "$after" > "$out/sanitizers.txt" 2>&1

cmd //c "docs\\evidence\\backlog\\golden\\build_golden.cmd $after $wtmp\\golden_new.exe" > /dev/null 2>&1
for b in base new; do
  "$tmp/golden_$b.exe" > "$tmp/golden_$b.txt"; "$tmp/golden_$b.exe" --layers > "$tmp/golden_${b}_layers.txt"
done
{ cd "$tmp" && sha256sum golden_base.txt golden_new.txt golden_base_layers.txt golden_new_layers.txt; cd - > /dev/null
  cmp "$tmp/golden_base.txt" "$tmp/golden_new.txt" && cmp "$tmp/golden_base_layers.txt" "$tmp/golden_new_layers.txt" &&
    echo "both dumps byte-identical"; } > "$out/golden-diff.txt"

for i in 1 2; do "$tmp/upload_timing.exe" timing > "$out/timing-$i.csv"; done
docs/evidence/backlog/R8-R9-bench/profile_upload.sh "$tmp/upload_timing.exe" "$tmp"

rm -f "$out"/m?-*.csv
docs/evidence/backlog/C2-bench/run_abba.sh "$base" "$after" r8r9 "$out" > "$out/summary.txt" 2>&1

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv > "$out/gpu-after.txt"
echo done

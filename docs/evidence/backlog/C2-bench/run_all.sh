#!/usr/bin/env bash
# Full C2 evidence run against a baseline build (all paths relative to the repo).
# Usage: run_all.sh <base-build> <c2-build> <tmp-dir>
#   tmp-dir must already hold large_base.exe and golden_base.txt from the base build.
set -uo pipefail
base=$1 c2=$2 tmp=$3
out=docs/evidence/backlog/C2-bench
cs="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/compute-sanitizer/compute-sanitizer.exe"
nsys="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
wtmp=$(cygpath -w "$tmp")

{
  for t in resident_contraction_test resident_test resident_review_test m3_resident_test m4_resident_test \
           input_map_resident_test rational_policy_resident_test; do
    echo "== memcheck $t"; "$cs" --tool memcheck --leak-check full "$c2/$t.exe" 2>&1 | grep -E "passed|ERROR SUMMARY"
  done
  echo "== racecheck resident_contraction_test"
  "$cs" --tool racecheck "$c2/resident_contraction_test.exe" 2>&1 | grep -E "passed|RACECHECK SUMMARY"
} > "$out/sanitizers.txt"

cmd //c "docs\\evidence\\backlog\\golden\\build_golden.cmd $c2 $wtmp\\golden_c2.exe" > /dev/null 2>&1
"$tmp/golden_c2.exe" > "$tmp/golden_c2.txt"; "$tmp/golden_c2.exe" --layers > "$tmp/golden_c2_layers.txt"
{ sha256sum "$tmp/golden_base_layers.txt" "$tmp/golden_c2_layers.txt"
  python docs/evidence/backlog/golden/golden_diff.py "$(cygpath -w "$tmp/golden_base.txt")" "$(cygpath -w "$tmp/golden_c2.txt")"
} > "$out/golden-diff.txt"

rm -f "$out"/m?-*.csv "$out"/large-*.csv
docs/evidence/backlog/C2-bench/run_abba.sh "$base" "$c2" c2 "$out" > "$out/summary.txt" 2>&1

cmd //c "docs\\evidence\\backlog\\C2-bench\\build_large_step.cmd $c2 $wtmp\\large_c2.exe" > /dev/null 2>&1
for i in 1 2; do
  "$tmp/large_base.exe" 7 > "$out/large-base-$i.csv"; "$tmp/large_c2.exe" 7 > "$out/large-c2-$i.csv"
done

echo "build,case,kernel,time_pct,instances,median_us" > "$out/nsys-kernels.csv"
for t in base c2; do
  exe="$tmp/large_$t.exe"
  for c in cheb-64x64x32x16-b1024 cheb-256x256x256x10-b8192 trbf-256x256x256x10-b8192; do
    "$nsys" profile --trace=cuda --force-overwrite=true -o "$wtmp\\$t-$c" "$exe" 5 $c > /dev/null 2>&1
    "$nsys" stats --report cuda_gpu_kern_sum --format csv --output "$wtmp\\$t-$c" "$wtmp\\$t-$c.nsys-rep" > /dev/null 2>&1
    python -c "
import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f\"{sys.argv[2]},{sys.argv[3]},\\\"{r['Name'][:110]}\\\",{r['Time (%)']},{r['Instances']},{float(r['Med (ns)'])/1e3:.1f}\")
" "$tmp/$t-${c}_cuda_gpu_kern_sum.csv" $t $c >> "$out/nsys-kernels.csv"
  done
  for c in 1; do
    dir=$([ $t = base ] && echo "$base" || echo "$c2")
    "$nsys" profile --trace=cuda --force-overwrite=true -o "$wtmp\\$t-m2case$c" "$dir/m2_benchmark.exe" --backend resident --case $c --warmups 2 --repeats 50 > /dev/null 2>&1
    "$nsys" stats --report cuda_gpu_kern_sum --format csv --output "$wtmp\\$t-m2case$c" "$wtmp\\$t-m2case$c.nsys-rep" > /dev/null 2>&1
    python -c "
import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f\"{sys.argv[2]},{sys.argv[3]},\\\"{r['Name'][:110]}\\\",{r['Time (%)']},{r['Instances']},{float(r['Med (ns)'])/1e3:.1f}\")
" "$tmp/$t-m2case${c}_cuda_gpu_kern_sum.csv" $t m2-case$c >> "$out/nsys-kernels.csv"
  done
done
echo done

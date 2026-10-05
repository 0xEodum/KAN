#!/usr/bin/env bash
# Full C1 evidence run (paths relative to the repository root).
# Usage: run_all.sh <c2-build> <c1-parity-build> <c1-fma-build> <tmp-dir>
#   tmp-dir must hold golden_c2.txt / golden_c2_layers.txt of the C2 build
#   (golden.cpp compiled against the C2 headers) and large_c2.exe.
set -uo pipefail
base=$1 c1=$2 fma=$3 tmp=$4
out=docs/evidence/backlog/C1-bench
cs="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/compute-sanitizer/compute-sanitizer.exe"
nsys="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
wtmp=$(cygpath -w "$tmp")

{
  for b in "$c1" "$fma"; do
    for t in resident_precision_test resident_contraction_test resident_test resident_review_test m3_resident_test \
             m4_resident_test input_map_resident_test rational_policy_resident_test cuda_test; do
      echo "== $b memcheck $t"; "$cs" --tool memcheck --leak-check full "$b/$t.exe" 2>&1 | grep -E "passed|ERROR SUMMARY"
    done
    for t in resident_precision_test resident_contraction_test; do
      echo "== $b racecheck $t"; "$cs" --tool racecheck "$b/$t.exe" 2>&1 | grep -E "passed|RACECHECK SUMMARY"
    done
  done
} > "$out/sanitizers.txt"

for b in "$c1" "$fma"; do
  tag=$(basename "$b")
  cmd //c "docs\\evidence\\backlog\\golden\\build_golden.cmd $b $wtmp\\golden_$tag.exe" > /dev/null 2>&1
  "$tmp/golden_$tag.exe" > "$tmp/golden_$tag.txt"; "$tmp/golden_$tag.exe" --layers > "$tmp/golden_${tag}_layers.txt"
done
{ cd "$tmp" && sha256sum golden_c2.txt golden_c2_layers.txt golden_$(basename "$c1").txt golden_$(basename "$c1")_layers.txt \
      golden_$(basename "$fma").txt golden_$(basename "$fma")_layers.txt; cd - > /dev/null
  echo "== C2 vs C1 parity build (FP64 resident)"
  python docs/evidence/backlog/golden/golden_diff.py "$(cygpath -w "$tmp/golden_c2.txt")" "$(cygpath -w "$tmp/golden_$(basename "$c1").txt")"
  echo "== C2 vs C1 FMA build (FP64 resident)"
  python docs/evidence/backlog/golden/golden_diff.py "$(cygpath -w "$tmp/golden_c2.txt")" "$(cygpath -w "$tmp/golden_$(basename "$fma").txt")"
} > "$out/golden-diff.txt"

rm -f "$out"/m?-*.csv "$out"/large-*.csv
docs/evidence/backlog/C2-bench/run_abba.sh "$base" "$c1" c1 "$out" > "$out/summary.txt" 2>&1

for b in "$c1" "$fma"; do
  cmd //c "docs\\evidence\\backlog\\C1-bench\\build_large_step.cmd $b $wtmp\\large_$(basename "$b").exe" > /dev/null 2>&1
done
for i in 1 2; do
  "$tmp/large_c2.exe" 7 > "$out/large-c2-f64-$i.csv"
  for b in "$c1" "$fma"; do
    for p in f64 f32 tf32; do "$tmp/large_$(basename "$b").exe" 7 "" $p > "$out/large-$(basename "$b")-$p-$i.csv"; done
  done
done

python docs/evidence/review-2026-10-01/torch_reference.py > "$out/torch_reference.txt" 2>&1

echo "build,precision,case,kernel,time_pct,instances,median_us" > "$out/nsys-kernels.csv"
for p in f64 f32 tf32; do
  for c in cheb-64x64x32x16-b1024 cheb-256x256x256x10-b8192 cheb-1024x1024x1024-b4096 trbf-256x256x256x10-b8192; do
    rep="$wtmp\\fma-$p-$c"
    "$nsys" profile --trace=cuda --force-overwrite=true -o "$rep" "$tmp/large_$(basename "$fma").exe" 3 $c $p > /dev/null 2>&1
    "$nsys" stats --report cuda_gpu_kern_sum --format csv --output "$rep" "$rep.nsys-rep" > /dev/null 2>&1
    python -c "
import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f\"fma,{sys.argv[2]},{sys.argv[3]},\\\"{r['Name'][:110]}\\\",{r['Time (%)']},{r['Instances']},{float(r['Med (ns)'])/1e3:.1f}\")
" "$tmp/fma-$p-${c}_cuda_gpu_kern_sum.csv" $p $c >> "$out/nsys-kernels.csv"
  done
done
echo done

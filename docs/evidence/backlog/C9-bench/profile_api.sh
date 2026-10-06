#!/usr/bin/env bash
# C9 nsys evidence: CUDA API calls per step before (pre-C9 eager calls) and
# after (train_step), plus GPU kernel summaries.
# Usage (Git Bash): profile_api.sh <pre-C9 bench exe> <C9 bench exe> <tmp-dir> <out-csv>
set -uo pipefail
base=$1 c9=$2 tmp=$3 out=$4
nsys="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
wtmp=$(cygpath -w "$tmp")
echo "run,case,api,calls,calls_per_step,total_ms" > "$out"
profile() { # name exe args...
  local name=$1; shift
  "$nsys" profile --trace=cuda --cuda-graph-trace=node --force-overwrite=true -o "$wtmp\\$name" "$@" > /dev/null 2>&1
  "$nsys" stats --report cuda_api_sum --report cuda_gpu_kern_sum --format csv --output "$wtmp\\$name" "$tmp/$name.nsys-rep" > /dev/null 2>&1
}
for c in 64x64x32x16-b1024 1024x1024x1024-b4096; do
  steps=$([ $c = 64x64x32x16-b1024 ] && echo 220 || echo 22) # warmup + one window of the f32 bench
  profile before-matched-$c "$base" f32 matched eager 1 1 $c
  profile after-matched-n1-$c "$c9" f32 matched step 1 1 $c
  profile after-matched-n64-$c "$c9" f32 matched step 64 1 $c
  profile before-realistic-$c "$base" f32 realistic eager 1 1 $c
  profile after-realistic-n1-$c "$c9" f32 realistic step 1 1 $c
  for run in before-matched after-matched-n1 after-matched-n64 before-realistic after-realistic-n1; do
    python - "$tmp/$run-${c}_cuda_api_sum.csv" "$run" "$c" "$steps" >> "$out" <<'PY'
import csv, sys
path, run, case, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
for r in csv.DictReader(open(path)):
    name = r["Name"]
    if any(k in name for k in ("Synchronize", "MemcpyAsync", "MemsetAsync", "GraphLaunch", "LaunchKernel", "EventRecord", "StreamWaitEvent", "cuLaunchKernel")):
        calls = int(r["Num Calls"])
        print(f"{run},{case},{name},{calls},{calls/steps:.2f},{int(r['Total Time (ns)'])/1e6:.2f}")
PY
  done
done

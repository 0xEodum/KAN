#!/usr/bin/env bash
# R9: Nsight Systems CUDA API and memory-operation summaries of 10 uploads
# (captured between cudaProfilerStart/Stop, after construction and a warm-up).
# Usage (Git Bash): profile_upload.sh <upload_timing.exe> <tmp-dir>
# Writes nsys-api.csv and nsys-mem.csv next to this script.
set -euo pipefail
exe=$1; tmp=$2
nsys="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
out=$(cd "$(dirname "$0")" && pwd)
echo "case,precision,api,calls,total_ms,median_us" > "$out/nsys-api.csv"
echo "case,precision,operation,count,total_ms,median_us,total_mb" > "$out/nsys-mem.csv"
for c in 0 1; do
  for p in fp64 fp32; do
    report="$tmp/upload-$c-$p"
    "$nsys" profile --trace=cuda --capture-range=cudaProfilerApi --force-overwrite=true -o "$report" "$exe" profile $c $p > /dev/null
    "$nsys" stats --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output "$report" \
      "$report.nsys-rep" > /dev/null
    # Columns: Time (%),Total Time (ns),Num Calls,Avg,Med,...,Name
    tail -n +2 "${report}_cuda_api_sum.csv" | awk -F, -v c=$c -v p=$p '{gsub(/"/,""); printf "%s,%s,%s,%s,%.3f,%.1f\n", c, p, $NF, $3, $2/1e6, $5/1e3}' >> "$out/nsys-api.csv"
    paste -d, <(tail -n +2 "${report}_cuda_gpu_mem_time_sum.csv") <(tail -n +2 "${report}_cuda_gpu_mem_size_sum.csv") |
      awk -F, -v c=$c -v p=$p '{gsub(/"/,""); printf "%s,%s,%s,%s,%.3f,%.1f,%.1f\n", c, p, $9, $3, $2/1e6, $5/1e3, $10}' >> "$out/nsys-mem.csv"
  done
done

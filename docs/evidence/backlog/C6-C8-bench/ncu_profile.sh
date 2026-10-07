#!/usr/bin/env bash
# Nsight Compute counters of the rational kernels inside one captured training
# step (graph nodes profiled individually). Usage (Git Bash):
#   ncu_profile.sh <rational_train_bench.exe> <f64|f32> <policy> <case> <out.csv>
# Profiles the first step's rational kernels (arguments and forward per layer,
# input VJP, parameter VJP; 11 launches for three layers).
set -euo pipefail
exe=$1 precision=$2 policy=$3 case=$4 out=$5
ncu="/c/Program Files/NVIDIA Corporation/Nsight Compute 2025.4.1/ncu.bat"
metrics=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__grid_size,\
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed.sum,\
smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum
"$ncu" --graph-profiling node -k regex:rational --launch-count 11 --metrics "$metrics" --csv \
  "$exe" "$precision" "$policy" 1 "$case" > "$out"

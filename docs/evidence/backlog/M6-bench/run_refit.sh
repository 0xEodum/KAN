#!/usr/bin/env bash
# Backlog M6 refit cost: per configuration the M6 train_step (per-input grids),
# the device refit_grids and the host route (download, CPU refit, rebuild).
# Usage: run_refit.sh <grid_step of M6> <configs> [windows=5]
set -uo pipefail
b=$1 configs=$2 windows=${3:-5}
echo "mode,precision,grid,path,widths,batch,intervals,degree,window,value_ms,hash"
while read -r p w n G d path host; do
  [[ -z $p || $p == \#* ]] && continue
  "$b" $p $w $n $G $d input $path $windows step | sed "s/^/step,/"
  "$b" $p $w $n $G $d input $path $windows refit | sed "s/^/refit,/"
  [[ ${host:-1} == 1 ]] && "$b" $p $w $n $G $d input $path 3 host | sed "s/^/host,/"
done < "$configs"

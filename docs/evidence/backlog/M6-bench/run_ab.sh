#!/usr/bin/env bash
# Backlog M6 matched complete train_step: base (c06b2ca, shared grids) A, M6 B
# in the order A B B A per configuration, then M6 with replicated and
# per-input grids (B, same session). One CSV row per window (grid_step format)
# with the build tag first.
# Usage: run_ab.sh <grid_step of base> <grid_step of M6> <configs> [windows=5]
set -uo pipefail
a=$1 b=$2 configs=$3 windows=${4:-5}
echo "build,precision,grid,path,widths,batch,intervals,degree,window,value_ms,hash"
while read -r p w n G d path; do
  [[ -z $p || $p == \#* ]] && continue
  for exe in A B B A; do
    bin=$a; [[ $exe == B ]] && bin=$b
    "$bin" $p $w $n $G $d shared $path $windows | sed "s/^/$exe,/"
  done
  for g in replicated input; do "$b" $p $w $n $G $d $g $path $windows | sed "s/^/B,/"; done
done < "$configs"

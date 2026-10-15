#!/usr/bin/env bash
# Backlog C13: full train_step ABBA of B-spline networks, base (before C13) against the
# C13 build, with the C5 harness (docs/evidence/backlog/C5-bench/spline_bench.cpp: C9
# matched protocol, graph replay of forward+backward+SGD, status interval 1, 3 timing
# windows per process). Per configuration and path the order is base, new, new, base, one
# process per run; rows get a leading "build" column.
# Usage: run_abba.sh <base exe> <new exe> <configs file: precision path topology batch degree terms> > abba.csv
set -uo pipefail
base=$1 new=$2 configs=$3
echo "build,precision,path,topology,batch,degree,terms,sparse_layers,arena_mb,window,steps,step_ms,forward_ms,checksum"
while read -r precision path topology batch degree terms; do
  [[ -z ${precision:-} || $precision == \#* ]] && continue
  for build in base new new base; do
    exe=$base; [[ $build == new ]] && exe=$new
    out=$("$exe" "$precision" "$path" "$topology" "$batch" "$degree" "$terms" 3) || { echo "FAILED $build $precision $path $topology $batch $degree $terms" >&2; continue; }
    echo "$out" | tail -n +2 | sed "s/^/$build,/"
  done
done < "$configs"

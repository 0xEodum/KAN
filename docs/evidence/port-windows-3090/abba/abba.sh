#!/usr/bin/env bash
S=$(dirname "$0"); O=$S/abba; mkdir -p "$O"
P=$S/wt-pre/build-pre; N=/k/PycharmProjects/auxiliary_projects/KAN/build-3090
run() { "$1" --backend all --warmups 2 --repeats 7 > "$2"; }
for pair in "m2 basis" "m3 local" "m4 rational"; do set -- $pair
  run $P/$1_benchmark.exe $O/$1-pre-0.csv; run $N/$2_benchmark.exe $O/$2-now-0.csv
  run $N/$2_benchmark.exe $O/$2-now-1.csv; run $P/$1_benchmark.exe $O/$1-pre-1.csv
done; echo ABBA_DONE

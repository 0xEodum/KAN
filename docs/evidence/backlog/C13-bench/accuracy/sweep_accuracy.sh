#!/usr/bin/env bash
# Backlog C13: input-VJP accuracy sweep (dx_accuracy) over families, sizes, coefficient
# profiles and precisions. Usage: sweep_accuracy.sh <dx_accuracy exe> <out.csv> [parallel jobs=8]
set -euo pipefail
exe=$1 out=$2 jobs=${3:-8}
configs=$(mktemp)
for init in wave smooth decay fit vs; do
  for K in 6 12 24 48 96 192; do
    for p in f32 tf32; do for path in dense sparse; do echo "$p $path bspline:3:$K 64 64 1024 $init"; done; done
  done
  for K in 12 48 96; do for p in f32 tf32; do for path in dense sparse; do echo "$p $path bspline:1:$K 64 64 1024 $init"; done; done; done
  for fam in cheb:8 cheb:16 cheb:32 cheb:64 legendre:16 legendre:64 jacobi:16 jacobi:64 hermite:8 \
             hermitef:16 hermitef:32 hermitef:64 hermitef:128 hermitef:256 fourier:9:1 fourier:33:1 fourier:129:1 \
             gauss:8:1 gauss:32:1 gauss:128:1 gauss:256:1 gauss:32:2 mexhat:16:1 mexhat:64:1 mexhat:256:1; do
    for p in f32 tf32; do echo "$p dense $fam 64 64 1024 $init"; done
  done
done > "$configs"
echo "precision,path,family,inputs,outputs,batch,init,sparse,max_e,max_err,err_rel,tf32_tol,fp32_tol,amp,diff,ratio,edge,forward,c_max" > "$out"
xargs -P "$jobs" -L 1 "$exe" < "$configs" >> "$out"
rm -f "$configs"

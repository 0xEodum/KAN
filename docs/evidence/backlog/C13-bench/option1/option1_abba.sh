#!/usr/bin/env bash
# Backlog C13: cost of Option 1 (W = U*C GEMM in FP32 math under TF32) per family, ABBA
# (off, on, on, off), one process per run, 3 windows each. Usage: option1_abba.sh <fstep exe> > out.csv
exe=$1
echo "option,precision,family,topology,batch,window,steps,step_ms"
while read -r fam topo batch; do
  for opt in off on on off; do
    if [[ $opt == on ]]; then KAN_C13_W_FP32=1 "$exe" tf32 "$fam" "$topo" "$batch" 3 | sed "s/^/$opt,/"; else "$exe" tf32 "$fam" "$topo" "$batch" 3 | sed "s/^/$opt,/"; fi
  done
done <<'CFG'
cheb:8 256x256x256x10 8192
cheb:16 256x256x256x10 8192
legendre:16 256x256x256x10 8192
jacobi:16 256x256x256x10 8192
hermitef:32 256x256x256x10 8192
hermitef:128 256x256x256x10 8192
fourier:17:1 256x256x256x10 8192
gauss:16:1 256x256x256x10 8192
mexhat:16:1 256x256x256x10 8192
cheb:8 1024x1024x1024 4096
legendre:16 1024x1024x1024 4096
hermitef:32 1024x1024x1024 4096
cheb:8 64x64x32x16 1024
CFG

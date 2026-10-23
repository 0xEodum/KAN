#!/usr/bin/env bash
# 3090 decision check: auto vs forced alternatives, interleaved, one process per run.
S=$(dirname "$0"); B=$S/bin
{ echo "precision,path,topology,batch,degree,terms,sparse_layers,arena_mb,window,steps,step_ms,forward_ms,checksum"
while read -r p t b d k; do [[ -z $p ]] && continue
  for path in dense sparse auto auto sparse dense; do "$B/spline_bench.exe" $p $path $t $b $d $k 3 | tail -n +2 | tr -d '\r'; done
done < "$S/c5-configs.txt"; } > "$S/c5-3090.csv"
{ echo "precision,path,topology,batch,K,window,steps,step_ms,hash"
while read -r p t b k; do [[ -z $p ]] && continue
  for path in column tile auto auto tile column; do "$B/rbf_step.exe" $p $t $b $k 3 $path | tail -n +2 | tr -d '\r'; done
done < "$S/c12-configs.txt"; } > "$S/c12-3090.csv"
{ echo "fs,bs,precision,carrier,topology,batch,K,parameters,window,steps,step_ms,hash"
while read -r p t b k; do [[ -z $p ]] && continue
  for fb in "0 0" "1 0" "2 0" "4 0" "8 0" "0 1" "0 2" "0 4" "0 8" "0 0"; do set -- $fb
    [[ $p == f64 && ( $1 -gt 1 || $2 == 8 ) ]] && continue
    "$B/wavelet_step.exe" $p $t $b wavelet $k 2 $1 $2 0 0 | tail -n +2 | tr -d '\r' | sed "s/^/$1,$2,/"
  done
done < "$S/m7-configs.txt"; } > "$S/m7-3090.csv"
echo DONE

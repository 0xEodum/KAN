"""Summarize spline_bench sweep CSVs: median step and forward time per path
(pooled windows of the ABBA runs), dense/sparse ratios, arena sizes and the
relative checksum difference after three steps.
Usage: python summarize.py sweep.csv [more.csv ...]"""
import csv, statistics, sys
from collections import defaultdict
rows = defaultdict(list)
meta = {}
for name in sys.argv[1:]:
    for r in csv.DictReader(open(name)):
        key = (r['precision'], r['topology'], int(r['batch']), int(r['degree']), int(r['terms']))
        rows[(key, r['path'])].append((float(r['step_ms']), float(r['forward_ms'])))
        meta[(key, r['path'])] = (r['sparse_layers'], float(r['arena_mb']), float(r['checksum']))
keys = sorted({k for k, _ in rows}, key=lambda k: (k[0], k[1], k[3], k[4]))
print('precision,topology,batch,degree,terms,K/(p+1),dense_step_ms,sparse_step_ms,step_speedup,dense_fwd_ms,sparse_fwd_ms,fwd_speedup,dense_arena_mb,sparse_arena_mb,checksum_rel_diff')
for k in keys:
    d, s = rows.get((k, 'dense')), rows.get((k, 'sparse'))
    if not d or not s: continue
    ds, ss = statistics.median(x[0] for x in d), statistics.median(x[0] for x in s)
    df, sf = statistics.median(x[1] for x in d), statistics.median(x[1] for x in s)
    cd, cs = meta[(k, 'dense')][2], meta[(k, 'sparse')][2]
    rel = abs(cd - cs) / max(abs(cd), 1e-300)
    print(f'{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{k[4]/(k[3]+1):.2f},{ds:.3f},{ss:.3f},{ds/ss:.2f},{df:.3f},{sf:.3f},{df/sf:.2f},'
          f'{meta[(k,"dense")][1]:.0f},{meta[(k,"sparse")][1]:.0f},{rel:.1e}')

"""Backlog C13: summary of run_abba.sh output. Per configuration (precision, path, topology,
batch, degree, terms): pooled median train_step and eager forward ms per build (base = before
C13, new = C13; n = 2 processes x 3 windows each), ratio new/base, sparse layers, arena MB and
whether the 3-step checksums agree to 1e-6 relative.
Usage: python summarize.py abba.csv [more.csv ...] > summary.md"""
import csv, statistics, sys
rows = [r for path in sys.argv[1:] for r in csv.DictReader(open(path))]
keys = []
for r in rows:
    k = (r['precision'], r['path'], r['topology'], r['batch'], r['degree'], r['terms'])
    if k not in keys: keys.append(k)
print('| precision | path | topology | batch | degree | terms | sparse layers | arena MB base -> new | step ms base | step ms new | new/base | forward new/base | checksum rel. diff |')
print('|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|')
for k in keys:
    sel = {b: [r for r in rows if (r['precision'], r['path'], r['topology'], r['batch'], r['degree'], r['terms']) == k and r['build'] == b] for b in ('base', 'new')}
    if not sel['base'] or not sel['new']: continue
    med = {b: statistics.median(float(r['step_ms']) for r in sel[b]) for b in sel}
    fwd = {b: statistics.median(float(r['forward_ms']) for r in sel[b]) for b in sel}
    cb, cn = float(sel['base'][0]['checksum']), float(sel['new'][0]['checksum'])
    rel = abs(cn - cb) / max(abs(cb), 1e-300)
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {sel['new'][0]['sparse_layers']} | {sel['base'][0]['arena_mb']} -> {sel['new'][0]['arena_mb']} | "
          f"{med['base']:.3f} | {med['new']:.3f} | {med['new']/med['base']:.3f} | {fwd['new']/fwd['base']:.3f} | {rel:.1e} |")

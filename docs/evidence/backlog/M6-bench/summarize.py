"""Backlog M6 tables from the committed CSVs.
Usage: python3 summarize.py ab ab.csv      (base vs M6, shared vs per-input grids)
       python3 summarize.py refit refit.csv (step, device refit, host route)"""
import csv, sys
from collections import defaultdict
from statistics import median

def ab(path):
    runs = defaultdict(list)  # (config, build, grid) -> per-run medians
    cur = {}
    rows = list(csv.DictReader(open(path)))
    groups = defaultdict(list)
    for r in rows:
        key = (r['precision'], r['widths'], r['batch'], r['intervals'], r['degree'], r['path'])
        groups[(key, r['build'], r['grid'], r['hash'])].append(float(r['value_ms']))
    # split windows into runs of equal size per (key, build, grid)
    table = defaultdict(dict)
    hashes = defaultdict(set)
    for (key, build, grid, h), values in groups.items():
        table[key][(build, grid)] = median(values)
        hashes[(key, grid)].add(h)
    print('| precision | widths | batch | G | degree | path | base ms | M6 shared ms | M6/base | replicated ms | per-input ms | per-input/shared | parameters after 5 steps: base = M6 shared = replicated |')
    print('|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|')
    for key in table:
        t = table[key]
        a, b = t[('A', 'shared')], t[('B', 'shared')]
        rep, inp = t.get(('B', 'replicated')), t.get(('B', 'input'))
        same = len(hashes[(key, 'shared')]) == 1 and hashes[(key, 'shared')] == hashes[(key, 'replicated')]
        p, w, n, G, d, path = key
        print(f'| {p} | {w} | {n} | {G} | {d} | {path} | {a:.3f} | {b:.3f} | {b/a:.3f} | {rep:.3f} | {inp:.3f} | {inp/b:.3f} | {"yes" if same else "NO"} |')

def refit(path):
    table = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        key = (r['precision'], r['widths'], r['batch'], r['intervals'], r['degree'], r['path'])
        table[key].setdefault(r['mode'], []).append(float(r['value_ms']))
    print('| precision | widths | batch | G | degree | step ms | device refit ms | refit/step | host route ms | host/device |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for key, t in table.items():
        s, d = median(t['step']), median(t['refit'])
        h = median(t['host']) if 'host' in t else None
        p, w, n, G, deg, path = key
        hs = f'{h:.1f} | {h/d:.0f}' if h else '- | -'
        print(f'| {p} | {w} | {n} | {G} | {deg} | {s:.3f} | {d:.3f} | {d/s:.2f} | {hs} |')

{'ab': ab, 'refit': refit}[sys.argv[1]](sys.argv[2])

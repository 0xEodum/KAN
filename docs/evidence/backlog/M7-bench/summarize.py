#!/usr/bin/env python3
"""Backlog M7 evidence tables. sweep <csv> [<csv> ...]: launch-rule check
(median step of the default choice and forced samples per thread over every
window of every pass, as ratios to the default; with two or more passes also
the default's spread between passes, the noise floor). cost <csv>: wavelet K
against fixed dictionaries with 3K and K terms (medians). demo <csv>: held-out
MSE table of the demonstration. ab <csv>: before/after medians (run_ab.sh)."""
import csv, collections, statistics, sys


def sweep(*paths):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    passes = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, path in enumerate(paths):
        for r in csv.DictReader(open(path)):
            key = (r['precision'], r['topology'], r['batch'], r['K'])
            g[key][(r['fs'], r['bs'])].append(float(r['step_ms']))
            if (r['fs'], r['bs']) == ('0', '0'): passes[key][n].append(float(r['step_ms']))
    print('| precision | widths | batch | K | default ms | forward S=1/2/4/8 (ratio) | backward S=1/2/4/8 (ratio) | best/default | passes |')
    print('|---|---|---:|---:|---:|---|---|---:|---:|')
    worst, noise = (2, None), 0
    for key, d in g.items():
        med = {k: statistics.median(v) for k, v in d.items()}
        base = med[('0', '0')]
        f = ' '.join(f"{med[(s, '0')] / base:.2f}" if (s, '0') in med else '-' for s in '1248')
        b = ' '.join(f"{med[('0', s)] / base:.2f}" if ('0', s) in med else '-' for s in '1248')
        ratio = min(med.values()) / base
        p = [statistics.median(v) for v in passes[key].values()]
        spread = max(p) / min(p) - 1 if len(p) > 1 else 0
        noise = max(noise, spread)
        worst = min(worst, (ratio, key))
        print(f'| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {base:.3f} | {f} | {b} | {ratio:.3f} | {spread * 100:.1f}% |')
    print(f'\nworst best/default {worst[0]:.3f} ({" ".join(worst[1])}); largest default spread between passes {noise * 100:.1f}%')


def cost(path):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    order = []
    for r in csv.DictReader(open(path)):
        key = (r['precision'], r['topology'], r['batch'])
        if key not in order: order.append(key)
        g[key][(r['carrier'], int(r['K']), int(r['parameters']))].append(float(r['step_ms']))
    print('| precision | widths | batch | wavelet K | wavelet ms | fixed 3K ms (same parameters) | ratio | fixed K ms (same terms) | ratio |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for key in order:
        d = g[key]
        for (c, K, p), v in d.items():
            if c != 'wavelet': continue
            w = statistics.median(v)
            same_p = next(statistics.median(x) for (cc, KK, pp), x in d.items() if cc == 'hat' and KK == 3 * K)
            same_k = next(statistics.median(x) for (cc, KK, pp), x in d.items() if cc == 'hat' and KK == K)
            print(f'| {key[0]} | {key[1]} | {key[2]} | {K} | {w:.3f} | {same_p:.3f} | {w / same_p:.1f} | {same_k:.3f} | {w / same_k:.1f} |')


def ab(path):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        g[(r['precision'], r['topology'], r['batch'], r['K'])][r['build']].append(float(r['step_ms']))
    print('| precision | widths | batch | K | before ms | after ms | after/before |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for key, d in g.items():
        a, b = statistics.median(d['A']), statistics.median(d['B'])
        print(f'| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {a:.3f} | {b:.3f} | {b / a:.3f} |')


def demo(path):
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r['target'], r['model'], int(r['parameters_per_edge']))].append(float(r['heldout_mse']))
    print('| target | params/edge | wavelets (best rate) | fixed, gradient descent (best rate) | fixed least squares x1 (its optimum) | fixed least squares, best of x1/x2/x4/x8 width |')
    print('|---|---:|---:|---:|---:|---:|')
    for target in dict.fromkeys(t for t, _, _ in g):
        for p in sorted({p for t, _, p in g if t == target}):
            ls = {w: g[(target, f'fixed-lstsq-x{w}', p)][0] for w in (1, 2, 4, 8)}
            w = min(ls, key=ls.get)
            print(f"| {target} | {p} | {min(g[(target, 'wavelet', p)]):.2e} | {min(g[(target, 'fixed', p)]):.2e} | {ls[1]:.2e} | {ls[w]:.2e} (x{w}) |")


if __name__ == '__main__':
    {'sweep': sweep, 'cost': cost, 'ab': ab, 'demo': demo}[sys.argv[1]](*sys.argv[2:])

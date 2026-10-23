import csv, sys, statistics as st, collections as C
S = sys.argv[1]
def load(f):
    try: return list(csv.DictReader(open(f"{S}/{f}")))
    except FileNotFoundError: return []
med = lambda xs: st.median(xs) if xs else float('nan')
# C5
g = C.defaultdict(lambda: C.defaultdict(list)); lay = {}
for r in load("c5-3090.csv"):
    k = (r['precision'], r['topology'], r['batch'], r['degree'], r['terms'])
    g[k][r['path']].append(float(r['step_ms']))
    if r['path'] == 'auto': lay[k] = r['sparse_layers']
print("C5 B-spline: precision topology deg K | dense sparse auto ms | auto_layers | auto/best")
for k, v in g.items():
    d, s, a = med(v['dense']), med(v['sparse']), med(v['auto'])
    print(f"{k[0]:5} {k[1]:16} d{k[3]} K{k[4]:>3} | {d:8.3f} {s:8.3f} {a:8.3f} | {lay.get(k,'?'):>2} | {a/min(d,s):.2f}  (sparse/dense {s/d:.2f})")
# C12
g = C.defaultdict(lambda: C.defaultdict(list)); hs = C.defaultdict(set)
for r in load("c12-3090.csv"):
    k = (r['precision'], r['topology'], r['batch'], r['K'])
    g[k][r['path']].append(float(r['step_ms']))
print("\nC12 RBF: column tile auto | auto/best")
for k, v in g.items():
    c, t, a = med(v['column']), med(v['tile']), med(v['auto'])
    print(f"{' '.join(k):30} | {c:8.3f} {t:8.3f} {a:8.3f} | {a/min(c,t):.2f}")
# M7
g = C.defaultdict(lambda: C.defaultdict(list))
for r in load("m7-3090.csv"):
    k = (r['precision'], r['topology'], r['batch'], r['K'])
    g[k][(r['fs'], r['bs'])].append(float(r['step_ms']))
print("\nM7 wavelet: auto ms | best forced (fs,bs) ms | auto/best")
for k, v in g.items():
    a = med(v[('0','0')]); forced = {fb: med(x) for fb, x in v.items() if fb != ('0','0')}
    fb, b = min(forced.items(), key=lambda kv: kv[1])
    print(f"{' '.join(k):30} | {a:8.3f} | {fb} {b:8.3f} | {a/b:.2f}")

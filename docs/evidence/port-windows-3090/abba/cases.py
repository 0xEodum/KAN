import csv, sys, statistics as st, collections as C, glob, os
F = sys.argv[1]; m = {'m2':'basis','m3':'local','m4':'rational'}
def pool(tag):
    d = C.defaultdict(list); info = {}
    for f in glob.glob(f"{F}/*-{tag}-*.csv"):
        s = os.path.basename(f).split('-')[0]; s = m.get(s, s)
        for r in csv.DictReader(open(f)):
            b = r['backend'].replace('m2_resident_full','resident_full')
            k = (s, r['case'], b); d[k] += [float(x) for x in r['samples_ms'].split(';')]
            info[(s, r['case'])] = f"{r['family']} {r['topology']} b{r['batch']} K{r.get('terms', r.get('degrees'))} {r['precision']}"
    return d, info
a, ia = pool("pre"); b, ib = pool("now")
rows = []
for k in b:
    ka = k if k in a else None
    if not ka: continue
    r = st.median(b[k]) / st.median(a[k]); rows.append((r, k, st.median(a[k]), st.median(b[k])))
for r, k, x, y in sorted(rows, reverse=True)[:14]:
    print(f"{r:5.2f} {k[0]:8} {k[2]:14} case{k[1]:>3} {ib[(k[0],k[1])]:45} {x:9.3f} -> {y:9.3f} ms")

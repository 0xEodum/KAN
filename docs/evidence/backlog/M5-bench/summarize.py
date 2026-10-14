"""Median step time per configuration and the function/physicists ratio
(backlog M5). Usage: python summarize.py bench.csv"""
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
groups = {}
for r in rows:
    groups.setdefault((r['precision'], r['topology'], r['batch'], int(r['terms'])), {}).setdefault(
        r['normalization'], []).append(float(r['step_ms']))
print('precision,topology,batch,terms,physicists_ms,function_ms,ratio,n')
for key in sorted(groups, key=lambda k: (k[3], k[0], k[1])):
    g = groups[key]
    p, f = statistics.median(g['physicists']), statistics.median(g['function'])
    print(f'{key[0]},{key[1]},{key[2]},{key[3]},{p:.4f},{f:.4f},{f / p:.3f},{len(g["function"])}')

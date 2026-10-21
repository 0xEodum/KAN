"""Backlog M7 review: regret of the adopted launch cost model and of the
previous rules over the per-kernel probe (kf32.json, kf64.json; nsys kernel
times of 3080 forced geometries, kprobe.cpp). Regret = measured time of the
chosen (S, splits) / best measured; excess = summed extra time / summed best."""
from model2 import *
import oldrule
prm = (3, 20, 0.5, 50, 1)  # overhead, latency, wave tail, block cost, idle-slot cost (resident.cu)
cap = lambda p, i, o, K, b, prm, l, c: bwd_choice(p, i, o, K, b, prm, l, c, maxsplit=min(16, max(1, (i + o) // i)))
print('rule        kernel    mean regret  max regret  excess time')
for name, table, choose in [('model', F, fwd_choice), ('model', B, cap), ('previous', F, oldrule.old_f), ('previous', B, oldrule.old_b)]:
    m, x, e = summ(regret(table, choose, prm, 100))
    print(f"{name:10s}  {'forward' if table is F else 'backward':8s}  {m:11.3f}  {x:10.2f}  {e * 100:10.1f}%")

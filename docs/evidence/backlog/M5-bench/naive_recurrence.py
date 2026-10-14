"""Backlog M5: the naive recurrence seeded with pi^(-1/4) exp(-x^2/2) in the
working precision, at reference cases where the true psi_n is O(0.1-0.4)
(hermite_reference.py table); it returns 0 or a subnormal-seeded value."""
import math
import warnings
import numpy as np
warnings.filterwarnings("ignore")
def naive(x, terms, dtype):
    x = dtype(x)
    prev, cur = dtype(0), dtype(math.pi**-0.25) * np.exp(-x * x / dtype(2))
    out = [cur]
    for n in range(terms - 1):
        prev, cur = cur, dtype(math.sqrt(2 / (n + 1))) * x * cur - dtype(math.sqrt(n / (n + 1))) * prev
        out.append(cur)
    return out
cases = [(38.5, 1024, 741, np.float64, float.fromhex("0x1.00c9cfea2c79dp-2")), (41.0, 1024, 840, np.float64, float.fromhex("0x1.ed54fdf302453p-3")),
         (45.0, 1024, 1012, np.float64, float.fromhex("0x1.e5bca11304e6cp-3")), (13.5, 128, 90, np.float32, float.fromhex("0x1.0af306fcd4ca0p-2")),
         (14.75, 128, 100, np.float32, float.fromhex("0x1.4cc3bd85ed6f3p-5")), (20.0, 256, 199, np.float32, float.fromhex("0x1.04bb9f2c76509p-2"))]
for x, terms, n, dtype, ref in cases:
    v = naive(x, terms, dtype)[n]
    print(f"{dtype.__name__} x={x} n={n}: naive {float(v):.6e}, reference {ref:.6e}, relative error {abs(float(v) - ref) / ref:.2e}")

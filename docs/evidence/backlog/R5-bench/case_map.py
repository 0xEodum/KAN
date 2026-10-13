"""R5 before/after CTest mapping: per configuration, every pre-R5 test name maps to
its post-R5 name with the same status and the same number of executed cases.

Usage: python case_map.py <before.csv> <after.csv>
Inputs are `name,status,cases` rows extracted from `ctest -V` logs (cases.py).
Prints `old,new,before,after,same` and exits non-zero on any difference.
"""
import csv
import sys

RENAMED = {
    "m3_basis": "local_basis",
    "m3_layer": "local_layer",
    "m3_resident": "local_resident",
    "m3_python": "local_python",
    "m4_rational": "rational",
    "m4_layer": "rational_layer",
    "m4_allocation": "rational_allocation",
    "m4_resident": "rational_resident",
    "m4_python": "rational_python",
}


def rows(path):
    with open(path, newline="") as handle:
        return {r[0]: (r[1], r[2]) for r in csv.reader(handle)}


before, after = rows(sys.argv[1]), rows(sys.argv[2])
bad = 0
print("old,new,before,after,same")
for old in sorted(before):
    new = RENAMED.get(old, old)
    b, a = before[old], after.get(new)
    same = a is not None and a == b and b[0] == "Passed"
    bad += not same
    print(f"{old},{new},{b[0]} {b[1]},{a[0] + ' ' + a[1] if a else 'missing'},{'yes' if same else 'NO'}")
extra = set(after) - {RENAMED.get(n, n) for n in before}
for name in sorted(extra):
    print(f"-,{name},-,{after[name][0]} {after[name][1]},NO (new)")
print(f"tests before {len(before)}, after {len(after)}, differences {bad + len(extra)}")
sys.exit(1 if bad or extra else 0)

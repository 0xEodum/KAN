"""Per-test case counts from a `ctest -V` log: name,status,cases (C++ 'x/y passed',
Python 'Ran N tests' + skips, otherwise '-')."""
import re
import sys

names, status, cases = {}, {}, {}
for line in open(sys.argv[1], errors="replace"):
    if m := re.match(r"\s*Start\s+(\d+): (\S+)", line):
        names[m[1]] = m[2]
    elif m := re.match(r"(\d+): (\d+)/(\d+) passed", line):
        cases[m[1]] = f"{m[2]}/{m[3]}"
    elif m := re.match(r"(\d+): Ran (\d+) tests?", line):
        cases[m[1]] = f"ran {m[2]}" + cases.get(m[1], "")
    elif m := re.match(r"(\d+): OK \((.*)\)", line):
        cases[m[1]] = cases.get(m[1], "") + f" ({m[2]})"
    elif m := re.search(r"Test\s+#(\d+): (\S+) \.+\s*(\*\*\*)?\s*(\w+)", line):
        status[m[1]] = m[4]
for n in sorted(names, key=names.get):
    print(f"{names[n]},{status.get(n, '?')},{cases.get(n, '-').strip()}")

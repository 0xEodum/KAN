"""Compare two golden dumps value by value (backlog C2 onward).

Usage: python golden_diff.py <before.txt> <after.txt>
Lines must correspond one to one. Prints the count of identical lines and,
per dump section, the largest deviation of a hex-encoded double relative to
the largest magnitude on its line; structural differences (other tokens,
other exception messages) are listed verbatim.
"""
import re
import struct
import sys
from collections import defaultdict

HEX = re.compile("[0-9a-f]{16}")


def value(word):
    return struct.unpack(">d", bytes.fromhex(word))[0]


def main(before, after):
    a = open(before, encoding="ascii").read().splitlines()
    b = open(after, encoding="ascii").read().splitlines()
    if len(a) != len(b):
        sys.exit(f"line counts differ: {len(a)} vs {len(b)}")
    section, same, worst, structural = "?", 0, defaultdict(float), []
    for x, y in zip(a, b):
        if x.startswith("=="):
            section = " ".join(x.split()[1:3])
        if x == y:
            same += 1
            continue
        tx, ty = x.split(), y.split()
        if len(tx) != len(ty) or tx[0] != ty[0] or any((HEX.fullmatch(p) is None) != (HEX.fullmatch(q) is None) or
                                                      (HEX.fullmatch(p) is None and p != q) for p, q in zip(tx, ty)):
            structural.append((section, x, y))
            continue
        pairs = [(value(p), value(q)) for p, q in zip(tx[1:], ty[1:]) if HEX.fullmatch(p)]
        if any(p != p or q != q or abs(p) == float("inf") or abs(q) == float("inf") for p, q in pairs):
            structural.append((section, x, y))
            continue
        scale = max([abs(p) for p, _ in pairs] + [1e-300])
        for p, q in pairs:
            worst[section] = max(worst[section], abs(p - q) / scale)
    print(f"lines: {len(a)}, identical: {same}, numerically different: {len(worst) and len(a) - same - len(structural)}, "
          f"structural: {len(structural)}")
    for name, deviation in sorted(worst.items(), key=lambda item: -item[1]):
        print(f"  {name:45s} max deviation / line scale = {deviation:.3e}")
    for name, x, y in structural:
        print(f"STRUCTURAL [{name}]\n  - {x[:160]}\n  + {y[:160]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

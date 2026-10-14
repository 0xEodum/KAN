"""Independent high-precision reference for the normalized Hermite functions
(backlog M5).

psi_n(x) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi)) is evaluated from the
physicists' recurrence H_{n+1} = 2x H_n - 2n H_{n-1} in decimal arithmetic at
`--digits` significant digits (default 700), with the exact binary value of x,
pi from Machin's formula and exp/sqrt from the decimal module. It does not use
the production recurrence for psi_n, nor any float arithmetic before the final
rounding. The physicists' recurrence loses at most log10(max|term|/|result|)
digits to cancellation; `selftest` checks that the reference is unchanged at
twice the precision.

Modes (run from the repository root with the venv's Python):
  table   C++ initializer rows {x, n, value, derivative} (hex doubles) for the
          tabulated cases of tests/carriers/hermite_function_test.cpp
  check   max errors of kan.evaluate_basis (CPU, FP64) and of the resident FP32
          and FP64 executors against the reference over wide x grids; needs
          PYTHONPATH=<build>/python
  selftest  the reference at --digits and 2*--digits agrees to 1e-30
"""
import argparse
import decimal
import math
import struct
import sys
from decimal import Decimal


def machin_pi():
    def atan_inv(n):
        x = Decimal(1) / n
        x2, term, total, k = x * x, x, x, 1
        eps = Decimal(10) ** (-decimal.getcontext().prec - 2)
        while True:
            term *= -x2
            k += 2
            delta = term / k
            if abs(delta) < eps:
                return total
            total += delta
    return 16 * atan_inv(5) - 4 * atan_inv(239)


def reference(x, terms, digits=700):
    """[(psi_n(x), psi_n'(x))] for n < terms as Decimals; x is a float taken exactly."""
    with decimal.localcontext() as ctx:
        ctx.prec = digits
        ctx.Emin, ctx.Emax = -10**9, 10**9
        X = Decimal(x)  # exact binary value
        pi = machin_pi()
        weight = (-(X * X) / 2).exp()
        norm2 = pi.sqrt()  # 2^n n! sqrt(pi), n = 0
        h_prev, h = Decimal(0), Decimal(1)
        values = []
        for n in range(terms):
            values.append(h * weight / norm2.sqrt())
            h_prev, h = h, 2 * X * h - 2 * n * h_prev
            norm2 *= 2 * (n + 1)
        result = []
        for n in range(terms):
            # psi_n' = sqrt(2n) psi_{n-1} - x psi_n (exact identity of the definition)
            d = (Decimal(2 * n).sqrt() * values[n - 1] if n else Decimal(0)) - X * values[n]
            result.append((+values[n], +d))
        return result


def to_float(v):
    return float(v)


def to_float32(v):
    return struct.unpack('f', struct.pack('f', float(v)))[0] if abs(float(v)) < 3.4e38 else float(v)


# Tabulated cases: the FP64 subnormal/underflow window of exp(-x^2/2) at high
# degree, the turning point sqrt(2n+1), moderate inputs and the decaying tail.
TABLE = [
    (0.0, 6, [0, 1, 2, 4, 5]),
    (0.37, 6, [0, 1, 2, 3, 5]),
    (-1.25, 6, [0, 1, 3, 5]),
    (2.5, 40, [0, 7, 20, 39]),
    (-6.0, 64, [0, 17, 63]),
    (12.0, 128, [0, 40, 72, 127]),
    (13.5, 128, [0, 90, 127]),
    (14.75, 128, [0, 100, 127]),
    (20.0, 256, [0, 150, 199, 255]),
    (35.0, 1024, [0, 600, 1023]),
    (37.75, 1024, [0, 700, 1023]),
    (38.5, 1024, [0, 741, 1023]),
    (41.0, 1024, [0, 840, 1023]),
    (45.0, 1024, [0, 1012, 1023]),
    (-44.0, 1024, [1, 1000, 1022]),
    (60.0, 1024, [0, 1023]),
]


def table():
    for x, terms, ns in TABLE:
        ref = reference(x, terms)
        for n in ns:
            v, d = ref[n]
            print(f"    {{{x!r}, {terms}, {n}, {to_float(v).hex()}, {to_float(d).hex()}}},")


def selftest(digits):
    worst = Decimal(0)
    for x, terms, _ in TABLE:
        a, b = reference(x, terms, digits), reference(x, terms, 2 * digits)
        for (va, da), (vb, db) in zip(a, b):
            worst = max(worst, abs(va - vb), abs(da - db) / (1 + abs(db)))
    print(f"reference digits {digits} vs {2 * digits}: max difference {float(worst):.3e}")
    return worst < Decimal("1e-30")


def errors(name, grid, terms, evaluate, to_scalar, floor, normal):
    """Max absolute value/derivative errors (derivatives scaled by sqrt(2n+1),
    their sup norm); max relative value error where |psi| > floor (dominated by
    inputs near the zeros of psi_n); and max relative value error in the
    monotone tail x^2 > 2n+1 + 2(2n+1)^(1/3) (beyond the turning region, no
    zeros) for every |psi| >= the smallest normal: the scaled-weight path."""
    abs_v = abs_d = rel_v = rel_tail = 0.0
    where = tail_where = None
    for x in grid:
        ref = reference(x, terms, 700)
        values, derivatives = evaluate(x)
        for n in range(terms):
            v, d = to_scalar(ref[n][0]), to_scalar(ref[n][1])
            ev = abs(values[n] - v)
            ed = abs(derivatives[n] - d) / math.sqrt(2 * n + 1)
            if not (math.isfinite(values[n]) and math.isfinite(derivatives[n])):
                raise SystemExit(f"{name}: nonfinite at x={x} n={n}")
            if ev > abs_v:
                abs_v, where = ev, (x, n)
            abs_d = max(abs_d, ed)
            if abs(v) > floor:
                rel_v = max(rel_v, ev / abs(v))
            if abs(v) >= normal and x * x > 2 * n + 1 + 2 * (2 * n + 1)**(1 / 3) and ev / abs(v) > rel_tail:
                rel_tail, tail_where = ev / abs(v), (x, n)
    print(f"{name}: terms {terms}, {len(grid)} inputs: max |dpsi| {abs_v:.3e} (x={where[0]}, n={where[1]}), "
          f"max |dpsi'|/sqrt(2n+1) {abs_d:.3e}, max rel (|psi|>{floor:g}) {rel_v:.3e}, "
          f"max rel in the tail {rel_tail:.3e} (x={tail_where and tail_where[0]}, n={tail_where and tail_where[1]})")
    return abs_v, abs_d, rel_v, rel_tail


def check(fp64_terms, fp32_terms):
    import numpy as np
    import kan
    cfg = lambda k: kan.HermiteConfig(k, kan.HermiteNormalization.FUNCTION)
    # FP64 CPU: wide grid, the FP64 underflow window, the tail beyond the cutoff.
    grid64 = sorted(set([round(v, 3) for v in np.linspace(-60, 60, 241)] +
                        [round(v, 4) for v in np.linspace(35, 46, 45)] + [0.0, 1e-300, 1e10, -1e300]))
    def cpu(x):
        v, d = kan.evaluate_basis(cfg(fp64_terms), x)
        return list(v), list(d)
    errors("CPU FP64", grid64, fp64_terms, cpu, to_float, 1e-8, 2.0**-1022)
    if not (kan.cuda_enabled and kan.cuda_available()):
        print("CUDA not available: resident checks skipped")
        return
    # Resident: a 1 -> K basis layer with identity coefficients gives
    # y[b, o] = psi_o(x_b); K samples of one x with one-hot upstream rows give
    # dx[b] = psi_b'(x). One-hot contractions are exact in FP32 and FP64.
    def resident(terms, precision, grid):
        layer = kan.Layer(1, terms, cfg(terms))
        layer.set_parameters(np.eye(terms).reshape(terms, 1, terms), np.zeros(terms))
        net = kan.Network([layer])
        gpu = kan.ResidentNetwork(net, terms, precision)
        def evaluate(x):
            gpu.upload_input(np.full((terms, 1), x))
            gpu.forward()
            out = gpu.download_output()
            gpu.upload_output_gradient(np.eye(terms))
            gpu.backward()
            return list(out[0]), list(gpu.download_gradients().input.reshape(terms))
        return evaluate
    grid32 = sorted(set([round(v, 3) for v in np.linspace(-26, 26, 209)] +
                        [round(v, 4) for v in np.linspace(12, 15, 31)] + [0.0, 1e5, -3e38]))
    grid32 = sorted(set(to_float32(v) for v in grid32))  # the reference at the FP32 input itself
    errors("resident FP32", grid32, fp32_terms, resident(fp32_terms, kan.Precision.FLOAT32, grid32), to_float32, 1e-3,
           2.0**-126)
    errors("resident FP64", grid64[::4], fp64_terms, resident(fp64_terms, kan.Precision.FLOAT64, grid64), to_float, 1e-8,
           2.0**-1022)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["table", "check", "selftest"])
    parser.add_argument("--digits", type=int, default=700)
    parser.add_argument("--fp64-terms", type=int, default=1024)
    parser.add_argument("--fp32-terms", type=int, default=128)
    args = parser.parse_args()
    if args.mode == "table":
        table()
    elif args.mode == "selftest":
        sys.exit(0 if selftest(args.digits) else 1)
    else:
        check(args.fp64_terms, args.fp32_terms)

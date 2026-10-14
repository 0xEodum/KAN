"""Backlog M5: conditioning of physicists' Hermite polynomials against the
normalized Hermite functions (CPU bindings, FP64, deterministic).

1. cond(Phi^T Phi) of the design matrix Phi[b, k] = phi_k(x_b) for x from the
   initializer's reference measure N(0, 1/2) (4096 inverse-CDF quantiles) and
   from a wide uniform grid covering the support of every psi_n, n < K, and
   the term magnitudes max|phi| and min/max over k of RMS(phi_k).
2. Training: 2 -> 16 -> 1 network with a tanh map in front of the second KAN
   layer (both normalizations see the same inputs), VarianceScaling{1} under
   each normalization's own reference moments, full-batch SGD on the MSE of
   f(x1, x2) = sin(2 x1) cos(x2) + 0.3 x1 x2 for 512 inputs N(0, 1/2) per
   coordinate (fixed quasi-random), 1500 steps, learning-rate grid 10^-1..10^-6.
   Reported per (K, normalization): the best final MSE over the grid, the
   learning rates that diverged (nonfinite result or MSE > 1e6), and the
   gradient norm at initialization. The coefficient gradient of a term
   scales with its magnitude: for H_n it grows like sqrt(2^n n!).
Usage: PYTHONPATH=<build>/python python conditioning.py
"""
import math
import sys
from statistics import NormalDist
import numpy as np
import kan

FUNCTION, PHYSICISTS = kan.HermiteNormalization.FUNCTION, kan.HermiteNormalization.PHYSICISTS
NAMES = {FUNCTION: "function", PHYSICISTS: "physicists"}


def quantiles(count, sigma):
    dist = NormalDist(0, sigma)
    return np.array([dist.inv_cdf((i + 0.5) / count) for i in range(count)])


def design(terms, normalization, x):
    return np.array([kan.evaluate_basis(kan.HermiteConfig(terms, normalization), float(v))[0] for v in x])


def conditioning():
    print("1. Design matrices: N(0, 1/2) = 4096 quantiles (the initializer's reference measure);")
    print("   wide = 4096 equispaced points on [-L, L], L = sqrt(2K+1) + 6 (covers every psi_n, n < K:")
    print("   the measure under which psi_n are orthonormal, up to the factor 2L/4096)")
    print("measure,terms,normalization,cond_PhiTPhi,max_abs_phi,min_rms_phi,max_rms_phi")
    for terms in (8, 16, 32, 48):
        half = math.sqrt(2 * terms + 1) + 6
        measures = {"N(0,1/2)": quantiles(4096, math.sqrt(0.5)), "wide": np.linspace(-half, half, 4096)}
        for name, x in measures.items():
            for n in (PHYSICISTS, FUNCTION):
                phi = design(terms, n, x)
                s = np.linalg.svd(phi, compute_uv=False)
                rms = np.sqrt((phi**2).mean(axis=0))
                print(f"{name},{terms},{NAMES[n]},{(s[0] / s[-1])**2:.3e},{np.abs(phi).max():.3e},{rms.min():.3e},{rms.max():.3e}")


def objective(x):
    return np.sin(2 * x[:, 0]) * np.cos(x[:, 1]) + 0.3 * x[:, 0] * x[:, 1]


def network(terms, normalization):
    net = kan.Network([kan.Layer(2, 16, kan.HermiteConfig(terms, normalization)),
                       kan.InputMap(16, kan.TanhMap()),
                       kan.Layer(16, 1, kan.HermiteConfig(terms, normalization))])
    kan.initialize(net, kan.VarianceScaling(seed=11))
    return net


def train(terms, normalization, rate, x, y, steps):
    net = network(terms, normalization)
    first = None
    for step in range(steps):
        try:
            out = net.forward(x)
            residual = out - y
            mse = float((residual**2).mean())
            if not math.isfinite(mse) or mse > 1e6:
                return math.inf, first
            g = net.backward(x, 2 * residual / len(x))
            if first is None:
                first = math.sqrt(sum(float((l.coefficients**2).sum()) for l in g.layers if hasattr(l, "coefficients")))
            net.sgd(g, rate)
        except (OverflowError, ValueError):
            return math.inf, first
    out = net.forward(x)
    return float(((out - y)**2).mean()), first


def training(steps):
    count = 512
    u = np.array([[(i * 0.6180339887498949) % 1, (i * 0.7548776662466927) % 1] for i in range(count)])
    dist = NormalDist(0, math.sqrt(0.5))
    x = np.vectorize(lambda p: dist.inv_cdf(min(max(p, 1e-9), 1 - 1e-9)))(u)
    y = objective(x).reshape(-1, 1)
    print(f"\n2. Training 2 -> 16 -> 1 (tanh between layers), {steps} full-batch SGD steps, target variance {y.var():.4f}")
    print("terms,normalization,best_rate,best_mse,diverged_rates,initial_gradient_norm")
    rates = [10.0**-k for k in range(1, 7)]
    for terms in (8, 16, 32):
        for n in (PHYSICISTS, FUNCTION):
            results = [(rate, *train(terms, n, rate, x, y, steps)) for rate in rates]
            finite = [r for r in results if math.isfinite(r[1])]
            best = min(finite, key=lambda r: r[1]) if finite else (math.nan, math.inf, None)
            diverged = ";".join(f"{r[0]:g}" for r in results if not math.isfinite(r[1])) or "-"
            grad = next((r[2] for r in results if r[2] is not None), math.nan)
            print(f"{terms},{NAMES[n]},{best[0]:g},{best[1]:.3e},{diverged},{grad:.3e}")


if __name__ == "__main__":
    conditioning()
    training(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)

# M4 CPU rational edges GREEN

2026-10-01. CPU rational implementation follows `M4-plan.md` with no numerical
scope changes. Numerator and denominator Horner recurrences compute values,
derivatives and the absolute-term denominator bound. Checked intermediates
distinguish nonfinite overflow from finite cancellation rejection. Parameter
VJPs divide by Q before multiplying; they never form Q squared, preserving
finite small VJPs for very large denominators.

`scripts/build.ps1 -BuildDirectory build-m4-cpu -Test`: **10/10 passed**,
including both installed C++ consumer configurations and all M1/M3 CPU suites.
Full build/CTest transcript: `M4-cpu-green.log`.

Direct suites: **5/5 scalar cases** and **6/6 layer/network cases**. Independent
identities include the [1/1] exponential Padé rational, scaled quadratic/linear
quotient, constants and a Q=1e200 derivative stress case. Central finite
differences cover all numerator, denominator and input derivatives for unequal
orders and degree 16; all parameters/inputs/biases in a rational-only and mixed
basis/rational network are also independently differenced.

Tests exercise the relative guard boundary, exact/removable poles, cancellation
conditioning, zero upstream, configuration/data/shape errors, overflow, zero
batch, rational/basis setter separation, moved state, finite candidate SGD,
atomic failed setters and whole-network SGD, and numerator-only L2. Mixed
networks require no special Network implementation: existing transactional SGD
copies every layer and commits only after all updates succeed.

Deterministic 6000-step SGD learns `(0.4+0.7*x)/(1+0.35*x)` from 41 equally
spaced training samples. MSE falls from **0.204799** to **7.65998e-10**;
six independently placed holdout inputs have MSE **1.20136e-9**.
The experiment is executed by `m4_layer_test`, with thresholds 1e-7 on both
training and holdout and an improvement factor exceeding 10000.

This evidence closes the CPU implementation step only. Integrated CUDA/Python,
sanitizer, coverage, profiling and independent review gates remain root-owned.

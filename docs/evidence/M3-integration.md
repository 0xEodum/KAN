# M3 CPU, Python and installed integration evidence

2026-10-01 on `cpp-foundation`, Windows x64, MSVC19.50.35724.0,
CUDA13.1.115, RTX3090 sm86, Python3.13.10, NumPy2.5.2, pybind11 3.0.2.
CUDA compiler guard override is explicit, inherited from M1/M2, not vendor support.

## CPU and Python checkpoints

- CPU RED `b3c8ec2`: [record](M3-layer-red.md), executable 0/5.
- CPU GREEN `3113791`: `scripts/build.ps1 -BuildDirectory build-m3-cpu -Test`,
  **8/8 CTest**, layer integration **5/5**. Covers shared RBF center/log-width
  finite differences, SGD, zero batch, invalid shapes, width exponent
  overflow/underflow atomicity across layers, refinement at degrees0..4,
  repeated knots, endpoint/outside values and input gradients, stale gradient
  rejection, deterministic population/median adaptation and coefficient L2 VJP.
- Python RED `650dbe0`: [record](M3-python-red.md), missing API errors against
  the built M2 extension. Python GREEN `060c441`: strict config/parameter/gradient
  and grid interfaces, regularization tuples and optional resident L2.
- `scripts/build.ps1 -Python -BuildDirectory build-m3-python -Test`:
  **10/10 CPU-only CTest**, old Python **5/5**, M3 Python **3 pass +1 explicit
  GPU skip**. No CUDA dependency in this build.
- `scripts/build.ps1 -Asan -Configuration Debug -BuildDirectory build-asan-m3
  -Test`: **8/8 CPU CTest**, no reported memory errors.

The Python localized holdout fits `x^2-0.3*x+0.2` at 48 uniform inputs for 400
fixed SGD steps, then checks five distinct inputs against a fixed MSE <1e-6.
Refinement preserves those holdout predictions to 1e-13. RBF center/log-width
VJPs use independent objective finite differences and owned-snapshot mutation
checks. This gate validates localized learning, not convergence on arbitrary tasks.

## Integrated and installed pre-tuning checkpoint

At resident numerical source `e07bf09`, Python source `060c441`:
`scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler -Python -Benchmarks
-BuildDirectory build-m3-final -Test`: **14/14 CTest**, including the full M1/M2
regressions, new basis **10/10**, new layer **5/5**, new resident **7/7**,
old Python **5/5** and M3 Python **4/4** on hardware.

`cmake --install build-m3-final --prefix build-m3-install`, configure
`tests/consumer` into `build-m3-consumer` using Visual Studio18 2026,
`CMAKE_PREFIX_PATH=<workspace>/build-m3-install`, `KAN_CONSUMER_CUDA=ON`, then
`cmake --build ... --config Release` and `ctest ... -C Release`:
**1/1**. The external consumer actually uses installed spline refinement,
wavelet mathematics, nonlinear parameter setters/L2, mixed resident spline/RBF
forward/backward/regularization/SGD and learned snapshots. Installed Python
(`PYTHONPATH=build-m3-install/python`) passes M3 **4/4**, M2 **5/5**.

Final tuned execution and sanitizer replay must be recorded in M3 final evidence
before closure; these checkpoints alone do not close the performance gate.

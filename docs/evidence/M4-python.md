# M4 Python integration validation

2026-10-01. Strict NumPy bindings expose typed RationalConfig/Layer, numerator
and denominator parameters/VJPs, mixed CPU and resident snapshots, and independent
`evaluate_rational` scalar evaluation. All returned configurations/arrays own
their data. Gradient topology checks distinguish rational/basis layers and both
polynomial orders. Zero-order rational denominators retain `(outputs,inputs,0)`;
fixed basis denominators have `(0,)`.

`scripts/build.ps1 -Python -BuildDirectory build-m4-python -Test` passed **13/13**
CTest targets using MSVC19.50, Python3.13.10, NumPy2.5.2 and pybind11 3.0.2.
The dedicated M4 Python suite passes three CPU tests and explicitly skips its GPU
test in this build. Tests independently check the [1/1] Padé scalar identity,
all numerator/denominator/bias finite differences, shape/type rejection, owned
snapshots, scaled nonlinear holdout fitting, mixed networks and numerator-only L2.
This is the CPU GREEN after the missing-API RED in M4-python-red.md. Integrated
real-device parity and installed Python acceptance are recorded separately when
executed; they are not implied by this CPU-only result.

# M4 Python executable RED

2026-10-01, initial extension from unchanged M3 `b3777f8`.

`PYTHONPATH=build-m3-final/python .venv/Scripts/python.exe tests/m4_python_test.py --cuda`
ran all four new tests and failed with four `AttributeError: module 'kan' has no
attribute 'RationalConfig'` errors. This demonstrates the missing rational API
before any binding implementation. The suite covers independent [1/1] Padé
values/input VJPs, numerator/denominator/bias finite differences, owned snapshots,
strict shapes/types, deterministic holdout learning, mixed networks, regularization,
zero batch and actual resident parity/failure recovery. Raw console transcript is
local ignored `build-m3-final/m4-python-red.txt`.

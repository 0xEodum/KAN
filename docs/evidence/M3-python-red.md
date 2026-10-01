# M3 Python RED checkpoint

2026-10-01, `PYTHONPATH=build-m2-final/python .venv/Scripts/python.exe
tests/m3_python_test.py`: exit 1, three errors and one explicit CPU-mode skip.
The existing built extension lacks BSpline/MexicanHat enum members and new config
fields. The tests specify closed-form spline math, shared RBF finite differences,
owned gradient snapshots, exact refinement, L2 gradients, localized holdout training
and optional real-GPU parity with resident regularization.

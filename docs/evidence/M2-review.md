# M2 additional independent validation

2026-10-01: the coordinating agent added four separate GPU tests before the
resident implementation. `scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler
-BuildDirectory build-m2-review` compiled; `build-m2-review/resident_review_test.exe`
returned exit 1, **0/4**, all reporting the intentional resident stub exception.

These tests independently differentiate the GPU forward loss with respect to
inputs and representative parameters across all six families, check relative
accuracy of the RBF underflow derivative using a 100-digit Decimal oracle, verify
state preservation after failed uploads, cycle batch sizes without allocation,
and cover move assignment, moved CPU network rejection, and no-device construction.
The oracle computes `q = Decimal.from_float(28e-300) / Decimal.from_float(1e-300)`
then `-2*q*exp(-q*q)/width` at precision 100; its stored value is
`-1.82521578010036018009053376422967315034882158760868e-39`.

GREEN, sanitizer, install, and independent code-review results will be recorded
only after they have run. This checkpoint is not M2 closure.

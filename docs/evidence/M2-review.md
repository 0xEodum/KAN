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

## Baseline GREEN and sanitizer gates

At resident implementation `2123ed8`, `scripts/build.ps1 -Cuda
-AllowUnsupportedCudaCompiler -BuildDirectory build-m2-review -Test` passed
**9/9 CTest targets**. The independent resident suite passed **4/4**; the CUDA
suite passed **9/9** cases and the resident implementation suite **5/5** cases.

`compute-sanitizer --tool memcheck --error-exitcode 1` on both
`build-m2-review/resident_test.exe` and `resident_review_test.exe` passed,
each reporting **ERROR SUMMARY: 0 errors**. Separate processes with
`CUDA_VISIBLE_DEVICES=-1` verified `resident_review_test --expect-no-device`
exit 0 and default `resident_test` exit 1. These are explicit no-device tests,
not skipped GPU parity.

CPU-only NumPy builds passed **7/7 CTest targets** using both Ninja and
Visual Studio 18 2026 Release (multi-configuration output directories). The
Python suite separately validates basis identities/derivatives, layer/network
finite differences, owned snapshots, and invalid dtype/layout/rank/shape/data.

## Independent implementation review

The Python author independently reviewed the resident implementation against
the CPU numerical code: basis recurrences, Jacobi shifted sums and endpoints,
Gaussian log-space derivative and overflowed subtraction, contraction order,
workspace bounds, stream/arena ownership, exception cleanup, lifecycle flags,
and validation of all SGD candidates before any commit. No blocking finding.

The CUDA author independently reviewed the Python implementation for strict
dtype/layout/alignment checks, borrowed-array lifetime while the GIL is released,
owned result copies, snapshot semantics, and gradient topology checks. No blocking
finding. The constructor GIL-release and configuration-snapshot suggestions were
incorporated by the binding author. The coordinating agent additionally read
both implementations and authored the four separate GPU validation cases above.

Remaining final integration/optimization results are recorded in the M2 closure
evidence; this numerical baseline alone is not closure.

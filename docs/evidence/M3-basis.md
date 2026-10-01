# M3 CPU localized basis evidence

2026-10-01, branch `cpp-foundation`. Scope: `BasisConfig`, `BasisValues`, CPU
evaluation/validation, and independent numerical tests. Layer, resident GPU,
Python, refinement and regularization acceptance belong to the integrated M3
evidence. This lane alone does not close M3.

## Retained RED

Commit `e262ad9` adds the public fields/enumerators and eight numerical test cases
without their implementation. Direct MSVC C++20 build of
`src/basis.cpp tests/m3_basis_test.cpp`, with `include` and `tests` include paths,
produced an executable whose run exited 1, `0/8 passed`. The four spline tests
and wavelet tests fail with `unknown basis kind`; the trainable RBF test fails
with `Gaussian width must be finite and positive`; the extreme trainable test
observes 0 instead of `exp(-4)`. This is an executable behavioral RED, not a
compile/link failure. Later independent tail/normalization cases were added
after the implementation.

Replay the RED by building the basis source/header/test from `e262ad9`. The
local test executable was built in ignored `build-m3-basis`, using the same
MSVC developer environment imported by `scripts/build.ps1`, and:

```powershell
cl /nologo /std:c++20 /EHsc /W4 /Iinclude /Itests src/basis.cpp tests/m3_basis_test.cpp /Febuild-m3-basis/m3_basis_test.exe /Fobuild-m3-basis/
./build-m3-basis/m3_basis_test.exe
```

## Verified GREEN

The same compilation/run passed `10/10` cases after the implementation, and the
retained `tests/basis_test.cpp` passed `17/17`. No compiler warnings came from
either source. The Visual Studio environment script emitted a nonfatal
`vswhere.exe` PATH diagnostic while successfully loading and running MSVC.
The integrated CMake `m3_basis` target uses the same test source.

- Cubic clamped splines match independent Bernstein closed-form values and
  derivatives at both exact endpoints and interior points.
- Nonuniform/repeated knots and degree 16 satisfy nonnegativity, partition of
  unity, zero summed derivative, and input finite differences.
- Degree-zero, full interior multiplicity, right-hand knots, exact inward upper
  endpoints and zero domain extension are checked. Knot counts/order/finiteness,
  exact endpoint multiplicity, maximum degree and minimum size are rejected
  when invalid. A domain `[-DBL_MAX,DBL_MAX]` preserves half weights and
  subnormal slopes through scaled interval arithmetic; a subnormal interval
  producing unrepresentable slopes raises an explicit overflow error.
- Mexican hat values and derivatives match its defining closed form over
  multiple translations/scales. Independent Simpson quadrature with 12000
  intervals on `[-12,12]` gives unit squared energy and zero mean within
  `1e-11`. Finite differences verify input derivatives.
- Shared trainable RBF center/log-width derivatives match closed forms and
  independent parameter finite differences. Scalar width remains the fixed
  default; opt-in mode reads per-term exponentiated log widths. Nonlinear
  derivative vectors remain empty for fixed families.
- Nonfinite inputs, count mismatches, invalid scales, log-width exponent
  underflow/overflow and trainable mode on a different family are rejected.
  Extreme infinite normalized-distance tails return zeros. Representable
  derivatives survive underflowed values using logarithmic evaluation;
  unrepresentable results raise `std::overflow_error`.

## Independent extreme-tail oracles

Python standard-library Decimal with 100-digit precision supplies two numerical
oracles, not the production double-precision recurrence/evaluation. For
`w = 2**(-1074)`, Gaussian input `30*w` has value below double range but
derivative `-60*exp(-900)/w = -1.6570395742157519202673569125584e-66`.
Mexican hat scale `w`, center 0 and input `55*w` has value
`-1.5902576002161727838410480795978e-492` and derivative
`1.7691236392503480575250084983550e-167`. Both retained derivative ratios pass
within `2e-12`; the test additionally requires a nonzero computed derivative.

Mexican hat evaluates signed polynomial factors in log space, including its
normalization and inverse-scale derivative, to prevent `infinity * zero` and
lost tail derivatives. Trainable RBF center derivatives are negative input
derivatives; log-width derivatives are `2*q*q*exp(-q*q)` evaluated logarithmically.
Existing M1 fixed Gaussian numerical behavior and polynomial conventions are
retained. Original mathematical implementations follow the reference definitions
in `M3-plan.md`; no external implementation was copied.

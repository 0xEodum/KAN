# M4 review correction: representable derivatives after intermediate underflow

2026-10-01. Independent review identified finite nonlinear VJPs lost when an
intermediate power or P/Q underflows. The executable RED is committed as
`7947503`, with output in `M4-cpu-underflow-red.log`: the expected normalized
denominator VJP is one and the prior implementation returns zero.

The correction preserves the ordinary Horner/quotient path. Rare branches for
subnormal or zero intermediates reconstruct numerator/denominator derivatives
from logarithms of nonzero P, z and Q, applying the analytic sign. Denominator
VJPs use `log(abs(P))+k*log(abs(z))-2*log(abs(Q))`, which remains meaningful even
if P/Q itself has rounded to zero. Input VJPs similarly recover terms before
division by scale when the intermediate quotient/product has underflowed.
Nonfinite ordinary intermediates retain the declared overflow policy.

Independent identities verify z=1e-200, P=1e300 and Q=1 produce db2=-1e-100;
z=1e-20 with degree 16 recovers db16=-1e-20 without subnormal-power fidelity
loss; z=1e300, P=1e-300, b1=1e-270 produces db1=-1e-60 although P/Q rounds
to zero. A subnormal scale additionally exercises recovery of the input VJP.

`scripts/build.ps1 -BuildDirectory build-m4-cpu -Test`: **10/10 passed**.
`m4_rational_test`: **6/6 passed**. Transcript: `M4-cpu-underflow-green.log`.
No API, scope, guard or acceptance changes are made.

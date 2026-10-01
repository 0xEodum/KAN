# M3 resident CUDA evidence

2026-10-01 executable RED: Release MSVC/CUDA architecture 86 build using
`./scripts/build.ps1 -Cuda -Benchmarks -AllowUnsupportedCudaCompiler
-BuildDirectory build-m3-cuda`. `m3_resident_test.exe` returned exit 1, **0/5**.
[Exact output](m3/resident-red.txt) records unsupported localized bases, missing
regularization validation, and absent learned-width candidate validation.
The latter two execute the old resident implementation on real RTX 3090 hardware.
Only the optional backward argument was introduced for compilable RED tests;
the old implementation deliberately ignores it at this revision.

Tests cover independent spline hat values, mixed-family CPU/device VJPs and
repeated SGD, learned snapshots, zero-batch L2, invalid lambda, log-width
underflow rejection before any network mutation, reusable failed gradients,
and explicit CPU snapshot/refine/device reconstruction.

## Resident implementation GREEN

The device arena adds immutable spline knots/wavelet scales, persistent RBF
log-width derivative workspaces, and centers/log widths inside the two existing
parameter/candidate regions. Every SGD candidate is checked for finite parameters
and finite positive exponentiated widths before the single whole-network region
swap. Snapshots download current nonlinear parameters. New basis evaluation and
shared-parameter VJPs execute on device; L2 is added only to coefficient gradients.
Grid changes remain explicit snapshot/refine/reconstruction setup operations.

Release real-hardware tests now pass **7/7**, including independent linear spline
hat values, degree 0/16, repeated full-multiplicity interior knots, domain outside
zeros, huge knot-domain ratios, MexicanHat subnormal-scale representable tail
derivatives, mixed topologies, learned snapshots, zero and nonzero batch L2,
invalid lambda, candidate underflow atomicity and repeated SGD. Existing CUDA,
resident and resident-review targets plus M3 resident pass **4/4** in
[resident-green.txt](m3/resident-green.txt). Compute Sanitizer
`--tool memcheck --error-exitcode 9 build-m3-cuda/m3_resident_test.exe`
returned exit 0 with **0 errors**, retained in
[resident-memcheck.txt](m3/resident-memcheck.txt). Workspace allocations remain two.

Profiling and complete-call performance acceptance are recorded separately in
[M3-benchmark.md](M3-benchmark.md); the initial scalar shared-parameter reduction
is a numerical baseline, not an accepted performance claim.

## Tiled shared-parameter reduction review closure

Nsight Systems measured the initial RBF shared-parameter VJP at 99.8% of kernel
time for frozen case 11. It used just one thread per basis term. Tiled GPU
partial reductions and a second fixed-order tile accumulation now expose batch
and edge work across blocks; construction reserves all scratch. Scratch tile
capacity is bounded by the configured maximum batch and edge count, avoiding
64-fold oversized storage for capacity-zero/single-edge models.

Independent review found the first finish kernel omitted terms beyond the capped
launch size. The retained [review RED](M3-reduction-red.md) reproduces an actual
nonzero tail with 8,388,481 trainable terms. A grid-stride finish loop now covers
every parameter. The same manual test passes on the RTX 3090 with two unchanged
workspace allocations, exact output in [large-basis-green.txt](m3/large-basis-green.txt).
The default numerical suite remains **7/7**; Compute Sanitizer memcheck remains
**0 errors** in [resident-tuned-memcheck.txt](m3/resident-tuned-memcheck.txt).

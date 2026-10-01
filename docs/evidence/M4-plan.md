# M4 rational edges: scope and acceptance plan

2026-10-01, branch `cpp-foundation`, starting head `b3777f8`, clean tree.
The current M3 integrated build was replayed: 14/14 CTest targets pass on the
RTX 3090. M4 alone is in scope; no BACKLOG.md exists.

## Numerical contract and defaults

Rational edges are a distinct Layer constructor with `RationalConfig`, not a
BasisKind or a fixed linear expansion. For each output/input edge:

`z=(x-center)/scale`, `P=sum(a[k]*z^k,k=0..m)`,
`Q=1+sum(b[k-1]*z^k,k=1..n)`, `r=P/Q`.

The constant denominator is fixed at one to remove common-scale ambiguity.
Degrees m,n are independently 0..16 (defaults 3,2); center=0, scale=1, scale
must be finite and positive. Numerator and denominator coefficients are owned
per edge in output/input/term order; bias remains per output. Zero initialization
means P=0,Q=1. No Taylor-series fitting algorithm is claimed: the parameterization
admits Padé coefficients supplied by callers and learns rational functions.

Use Horner evaluation and analytic quotient-rule input and parameter VJPs.
Reject a sample with `abs(Q)<=epsilon*(1+sum(abs(b[k-1]*z^k)))`, default epsilon
1e-8, finite 0<epsilon<1. This is a relative cancellation guard, not clipping,
pole removal or a proof of pole freedom between samples. Exact removable poles
also fail. Nonfinite intermediates/results raise overflow_error; finite unsafe
denominators raise domain_error. The guard is checked even for zero upstream/P.
No hidden pole penalties. CPU coefficient L2 and optional resident backward L2
penalize numerator coefficients only; denominator/bias gradients remain explicit
and unpenalized. SGD checks finite candidate parameters atomically across the
network; guard safety is evaluated when the candidate is subsequently executed.

Layer API: `Layer(inputs,outputs,RationalConfig)`, `is_rational()`,
`rational_config()`, `denominators()`, `set_rational_parameters(a,b,bias)`.
`coefficients()` exposes numerator parameters for rational layers. `basis()`
rejects rational layers; `rational_config()` rejects basis layers. LayerGradients
adds `denominators` (empty for basis layers). Existing basis-only setters reject
rational layers. Mixed Network composition, whole-network SGD, strict NumPy
snapshots and resident CUDA must preserve the distinction. Resident execution
reserves all storage at construction, owns its stream, uses no host math fallback,
and preserves existing state invalidation and atomic update behavior.

## Exit gates

1. Commit/push executable RED before implementation and verified GREEN for CPU,
   resident and Python interfaces. Independent scalar identities (including the
   [1/1] Padé exponential), finite differences for every parameter and input,
   mixed networks, deterministic learning with independent holdout.
2. Verify degrees 0/0 and unequal orders, scaling, relative guard boundary,
   exact/removable poles, cancellation conditioning, overflow, invalid shapes,
   zero batch, atomic setters/SGD, unchanged resident allocation counts and
   resident failure invalidation/recovery. Retain full-call CPU/GPU parity.
3. Full regression suite, CPU-only Python, installed C++/Python consumption,
   AddressSanitizer, GCC CPU line coverage >=80%, actual GPU Compute Sanitizer.
4. Freeze rational benchmark fixtures before CUDA tuning. Profile complete calls,
   identify and resolve a measured bottleneck, and accept tuning only with balanced
   matched full-call measurements verifying all VJPs/parameters/trajectories.
   Preserve M2/M3 regressions and report hardware/profiler limitations accurately.
5. Independent source/evidence acceptance review, resolution of blocking findings,
   incremental conventional commits pushed to the tracked branch. Only then mark
   M4 DONE and hand off M5 without implementing M5/M6.

Mathematical reference: [NIST DLMF rational and Padé approximation definitions](https://dlmf.nist.gov/3.11).
Implementation is original; no external interfaces or code are ported.

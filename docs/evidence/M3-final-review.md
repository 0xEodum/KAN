# M3 independent final review

2026-10-01, `cpp-foundation`. This reviewer authored none of the production
implementation or retained numerical tests. Read the complete current roadmap,
project instructions and M3 acceptance plan before reviewing. This document
supersedes no earlier review; it independently covers the CPU basis lane as well
as the integrated CPU/CUDA/Python implementation and final evidence.

## Implementation inspection

Read the public basis/layer/network/resident interfaces, `src/basis.cpp`,
`src/layer.cpp`, `src/network.cpp`, `src/resident.cu`, Python bindings, M3 tests,
installed consumer and frozen M3 benchmark source.

- Spline validation enforces finite ordered knots, exact clamped endpoint
  multiplicity, degree 0..16 and a positive domain. CPU recurrence and bounded
  device local recurrence implement the declared right-hand interior and inward
  upper endpoint convention, including degree zero/full repeated knots.
- Mexican-hat normalization and its signed analytic input derivative agree with
  the stated continuous-wavelet formula. Logarithmic extreme-tail evaluation
  preserves representable derivatives. Gaussian center VJPs have the negative
  input-derivative sign; log-width VJPs use `2*q*q*exp(-q*q)` and sum all edges.
- Boehm coefficient copy/blend ranges and median/midpoint adaptation agree with
  the plan. Candidate vectors are built before swaps, including full-network
  SGD; refinement changes shapes and stale parameter gradients reject.
- Coefficient L2 excludes bias and nonlinear parameters; zero-batch parameter
  VJPs retain valid shapes. Python arrays/configurations are owned snapshots,
  with strict tensor dtype/rank/layout checks and preserved basis tuple API.
- Device arena offsets include both nonlinear parameter regions, immutable
  grid data, derivative arrays and bounded persistent partial scratch. Actual
  batch tile counts compactly index overwritten partials within the reserved
  maximum. Zero batch reads no absent activations and initializes all VJPs.
- All candidates and exponentiated widths validate before the whole-network
  region swap. Successful updates invalidate saved forward/backward state;
  rejected candidates preserve active parameters and reusable gradients.
- The corrected nonlinear finish kernel uses a grid-stride loop through all
  `2*terms` entries under the capped launch helper. This resolves the earlier
  review's retained large-tail defect. The actual 8,388,481-term GREEN output
  and zero-error tuned memcheck record were inspected.

No new blocking source finding was identified.

## Independent executable probes

After the CUDA author released the matched measurement window, executed a
separate inline Python prober with `.venv/Scripts/python.exe -`, importing
`build-m3-final/python`. Exit 0, three PASS records on the RTX3090:

1. Every degree 0..16 was checked against independent Bernstein closed forms,
   using only clamped endpoint knots at -2 and 3 and inputs -2, -1.73, 0.137,
   2.91 and 3. Values use `binomial(p,k)*q^k*(1-q)^(p-k)`, `q=(x+2)/5`;
   slopes use `p/5*(B[p-1,k-1]-B[p-1,k])`, with absent terms zero.
   All values/slopes pass relative 2e-12 plus absolute 2e-14, including degree
   zero and exact endpoint conventions. This independently reviews the CPU
   basis author's own implementation.
2. A new mixed chain, MexicanHat 2->3, trainable RBF 3->2, B-spline 2->1,
   passed GPU/CPU forward, every intermediate layer input VJP, coefficient,
   bias, center and log-width VJP parity with L2 0.13. Five-term configurations
   use centers [-1,-0.4,0.1,0.6,1.2], scales [0.3,0.5,0.7,0.9,1.1], log widths
   [-0.8,-0.4,0,0.2,0.4] and degree-two knots [-2,-2,-2,-0.4,0.7,2,2,2].
   Coefficients are `0.08*sin(flat_index+layer_index)` and biases linearly
   range 0.03..0.09. Input and upstream are deterministic sine/cosine arrays.
   Changing batches 1025,1,0,17,1025 within one capacity-1025 executor exercises
   large/small/zero compact partial tile layouts. Each step applies SGD 0.001,
   verifies output/gradient downloads invalidate afterward and rebuilds the CPU
   reference from the learned snapshot. Exactly two allocations remain.
3. The resulting mixed network input VJP passed independently differentiated
   scalar objectives at inputs [[0.173,-0.327],[0.611,0.419]] and upstream
   [[0.3],[-0.2]], central difference 1e-6 and relative 1e-6/absolute 1e-10.

## Matched benchmark/evidence audit

Read the benchmark source, all final/diagnostic CSVs, balanced summary, Nsight
exports, Compute Sanitizer output, manifest and author evidence. The benchmark
source is unchanged from `e07bf09`. Resident measurement includes synchronized
forward/backward/SGD public calls and small status transfers; transfer mode also
includes tensor upload/download. The CPU reference recomputes backward
activations. Untimed resident trajectory replay checks final parameters against
the actual timed trajectory. The benchmark compares network input and all
parameter VJPs; intermediate input VJPs are independently checked above.

A separate standard-library CSV audit verifies all 36 final M3 rows are the
complete twelve-case/three-backend product, each with two warmups and seven
finite positive samples, exact recomputed medians, passing numerical comparison
and unchanged two allocations. It independently recomputes all pooled balanced
medians/IQRs from 14 actual complete-call samples per version/mode. The M2
regression has 76 rows: 24 CPU, 24 resident, 24 transfer-inclusive, plus four
Chebyshev M1 host calls; seven samples, recomputed medians and parity all pass.

The accepted final case-11 comparison uses balanced preopt/final/final/preopt
runs, identical fixtures and final corrected reduction source. Median complete
resident calls fall from 1222.5449 to 4.5011 ms; transfer-inclusive calls from
1220.2137 to 6.6203 ms. The final Nsight trace distributes kernel time across
the partial/finish reduction (38.0%), input VJP, coefficient VJP and evaluation,
resolving the measured 99.8% serial nonlinear reduction bottleneck. These
kernel diagnostics support the mechanism, not the complete-call speedup ratio.

Raw sweep cases 7 and 11 retain substantial timing dispersion; the evidence
explicitly reports their IQRs and forbids broad throughput conclusions. The
retained `ERR_NVGPUCTRPERM` output prevents an occupancy/saturation claim.
Hardware/toolchain/compiler-guard limitations, active WDDM display and unlocked
clocks are reported. Performance acceptance is limited to the matched measured
workload and does not imply full GPU utilization.

The first manifest audit found one local ignored preopt `.nsys-rep` byte-size/
hash mismatch while every retained artifact matched. It was reported to the
author, who settled the re-export and regenerated provenance. A second
independent audit passes all 18 retained artifact and five local binary/trace
hashes and byte sizes. Committed Git-blob verification is recorded below.

## Complete-stage acceptance audit

Read the final `M3.md` candidate against every exit gate in `M3-plan.md`, its
retained RED/GREEN revisions, mathematical contract, integration evidence,
review findings and performance artifacts. Inspected actual final CTest logs:
`build-m3-final` has 14 passes/no failures, `build-asan-m3` eight passes/no
failures, `build-coverage-m3` eight passes/no failures, CPU-only
`build-m3-python` ten passes/no failures, and installed consumer one pass/no
failures with its explicit installed CPU/CUDA/M3 success messages.

The live GCC coverage summary agrees with recorded 449/456 lines (98.5%),
541/597 branches (90.6%) and 28/28 functions (100%). Coverage excludes CUDA
and the stated exception/unreachable machinery. Retained actual-device tuned
Compute Sanitizer output reports all seven M3 cases pass and zero errors;
the final stage record also reports separate zero-error original resident and
resident-review replay. ASAN/GCC/final builds include the explicit aggregate
initializers at `ee03df9`; the CUDA measured source remains `f36d22d`.

Independently reran installed Python `tests/m3_python_test.py --cuda` using
`PYTHONPATH=build-m3-install/python`: four tests pass on hardware. The installed
extension exactly matches the final built extension SHA256
`D205BF7DE33219792406EB9E1FB0F2D36E3B04D0B10CAFEB65958FD29E1A1178`.
This installed replay supplements the independent mixed-chain probes above.

Localized training has an independently held-out error criterion; nonlinear
derivatives have scalar-objective finite differences; rejected candidates,
zero batch, refinement preservation, shape rejection and persistent storage
have executable checks. The initial review's large-tail blocker has retained
actual RED and corrected actual GREEN, with its memory-capacity refinement
separately documented. Rational/quantum/distribution work remains outside M3.

At `52064cb`, independently read all 18 committed artifact blobs through
`git show` and verified their raw lengths and SHA256 against the committed
manifest: all pass. This verifies repository bytes, including the scoped
evidence attribute, rather than only working-tree copies. The parent's actual
standalone replay after lowering the manual regression's memory guard to 2 GiB
also passes the same 8,388,481-term tail and reports two unchanged allocations;
the guard/comment update is committed separately at `52064cb`.

**Verdict: PASS for M3 acceptance.** All five exit gates in the declared plan
have implementation and executable/evidence coverage; the recorded source and
provenance findings are resolved. No blocking finding remains. M3 can be marked
DONE and the roadmap can hand off M4 after committing its final stage evidence
and closure update. This verdict preserves the explicit performance, hardware,
toolchain and mathematical scope limitations above.

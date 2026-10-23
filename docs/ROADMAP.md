# Implementation roadmap

## M1: numerical foundation — DONE

Independent basis/layer/CUDA development with tests before implementation.
Acceptance: six basis families, analytic derivatives, finite-difference layer and
network gradients, arbitrary compatible topology, deterministic training example,
optional Chebyshev CUDA forward/backward parity on real hardware, CPU coverage >=80%,
build/use documentation, independent review and incremental RED/GREEN commits.
Completed 2026-09-30 with all acceptance gates passed and all five independent
review findings resolved. See [M1 evidence](evidence/M1.md) for exact validation.

## M2: persistent GPU execution — DONE

Resident tensors/parameters, stream ownership, reusable workspaces, GPU optimizer,
all M1 basis families, Python bindings. Frozen full-call CPU/GPU benchmarks must
precede optimization claims; kernel timing alone is insufficient.
Completed 2026-10-01 with all acceptance gates passed: six-family resident mixed
networks, reusable storage, atomic GPU SGD, optional NumPy bindings, real-hardware
parity/sanitizer/installation checks, matched full-call profiling and independent
implementation/evidence review. See [M2 evidence](evidence/M2.md).

## M3: localized and adaptive bases — DONE

B-splines with knot/domain contracts, wavelets with scale/translation conventions,
learnable RBF centers/widths, adaptive grids and regularization.
Completed 2026-10-01: explicit clamped splines and normalized Mexican-hat wavelets,
shared trainable RBF centers/log widths, exact sample-driven knot refinement,
coefficient L2, persistent CUDA and NumPy interfaces. All gates passed: integrated
14/14, sanitizers, 98.5% CPU coverage, installed consumers, matched full-call
profiling and independent acceptance review. The reviewed large-launch reduction
defect is resolved. See [M3 evidence](evidence/M3.md) and [scope/defaults](evidence/M3-plan.md).

## M4: rational edges — DONE

Padé/rational parameterization, singularity policy, conditioning and nonlinear
parameter derivatives. Do not disguise rational edges as fixed linear basis terms.
Completed 2026-10-01: distinct trainable numerator/denominator edges, explicit
scaling and relative pole guard, analytic nonlinear VJPs, mixed CPU/resident CUDA
and NumPy execution. All gates passed: integrated19/19, sanitizers,98.1% CPU
coverage, installed consumers, matched complete-call CPU/GPU profiling and
independent source/evidence acceptance. Reviewed extreme derivative underflow is
resolved; CPU per-edge allocations removed and GPU parameter reduction tuned.
See [M4 evidence](evidence/M4.md) and [scope/defaults](evidence/M4-plan.md).

## B: M1–M4 review backlog — DONE

Close every item in [BACKLOG.md](BACKLOG.md) (carrier types R, mathematics and
trainability M, CUDA efficiency C) before M5 starts. Items are taken in small
batches (at most three per pass), dependency-free P0 items first. Each item
needs a verified fix, its evidence and a status journal entry; contract changes
are recorded in `docs/evidence`. See [stage decision](evidence/backlog-gate.md).
Completed 2026-10-22: all 29 items closed in nine passes (R1–R9, M1–M7, C1–C13;
C11 closed by the owner, C13 added by owner decision), each with a verified fix,
its evidence and a journal entry. The last item, R6, formats the sources with a
pinned clang-format and a CTest check; library and kernel code are
instruction-identical. See the [stage closure](evidence/backlog-closure.md).

## M5: experimental quantum carriers — NEXT (depends on B, DONE)

Typed PQC/Fock carrier interfaces and simulator adapters, physical normalization,
measurement/gradient semantics and independently verified QUBO/quadratization.
No string hashing or symbolic parsing in numerical kernels.
Acceptance (scope amended 2026-10-23): typed `PqcEdges` (DARUAN-style re-uploading circuits)
through the carrier protocol on CPU and resident CUDA, with unit-norm states and Born-rule
checks; exact gradients (adjoint, parameter-shift) verified against finite differences and
each other; a CPU QUBO/quadratization module verified by exhaustive enumeration and
separately written references; Python bindings; a time-to-quality study (measurement, not a
gate) against the classical carrier with the matching spectrum; existing results, golden
dumps and kernels unchanged; integrated tests, sanitizers, CPU coverage >=80%, installed
consumers, matched CPU/GPU profiling (removal of pathological bottlenecks, not exhaustive
tuning) and independent acceptance review. A device-adaptation contract is recorded before
the resident phase and the new kernels follow it; GPU phases are runtime-validated on GB10
(sm_121) and RTX 3090 (sm_86). `FockEdges` (truncated single-mode CV circuits with a
norm-deficit and guard-band rejection) proceeds only after a go/no-go gate (numerical
preflight and a CPU prototype against normalized Hermite functions; owner decision).
The cuStateVec comparison and finite shots are deferred outside M5 acceptance unless the
owner reinstates them. See the [scope/defaults and plan](evidence/M5-plan.md) and the
[2026-10-23 decision](evidence/roadmap-2026-10-23.md).
Phase 1 (PQC carrier on the CPU: exact expectations, adjoint and parameter-shift gradients,
D8 initializers, D13 stubs) completed 2026-10-23: exit criteria, isolation gates, review fixes
R3/R5 and a two-pass independent review with nothing blocking, on GB10 and under MSVC; see the
[phase evidence](evidence/M5-pqc-cpu.md). Next: the device-adaptation contract, then phase 2a
(PQC on the resident executor).

## M6: external baseline — PLANNED (depends on M5)

Measurement only, no optimization obligation: a pre-registered matched comparison of
complete training steps, accuracy and time-to-quality with existing KAN implementations
(efficient-kan, FastKAN, FasterKAN and others) on the families they share with this library,
on the reference machine (GB10) and the RTX 3090. Results set the priorities of M7 and
cross-check the matched families numerically. Acceptance: protocol written before the runs,
library versions and hardware recorded, numerical agreement of matched configurations,
results and failures reported as measured.

## M7: device adaptation — PLANNED (depends on M6)

Every device-dependent choice (sparse B-spline selection, wavelet launch model, contraction
crossovers, tiles, workspaces; the [port inventory](evidence/port-aarch64.md#adaptability-inventory-ga102-tuned-constants-in-srcresidentcu))
follows the device-adaptation contract: decisions from device attributes, occupancy and/or
on-device calibration, validated on at least two architectures, conservative on unvalidated
devices, deterministic per device. CPU SIMD for the hot loops (Google Highway considered)
under the unfused, fixed-order CPU contract with ISA-independent results stated.
Acceptance: on GB10, RTX 3090, x86-64 and aarch64 CPUs no automatic choice slower than the
best forced alternative beyond recorded noise; the recorded C5 sm_86 regression resolved;
CI compiles the CUDA targets on Linux and Windows.

## M8: heterogeneous networks — PLANNED (depends on M7)

Different carriers per layer are already supported (mixed networks on CPU and resident CUDA);
this stage adds mixed carriers within a layer (default design: a layer as a sum of
sub-blocks with different carriers over the same inputs, reusing the existing kernels) and
pre-registered experiments comparing homogeneous, per-layer and within-layer mixtures.

## M9: comparison with existing implementations — PLANNED (depends on M8)

The M6 protocol repeated on the final code and extended to heterogeneous networks:
performance, accuracy and time-to-quality, with the same reporting rules.

## M10: research and distribution — PLANNED (depends on M9)

Versioned model format, packaging, reproducible matched benchmarks and expanded
examples. License selection belongs to the repository owner.

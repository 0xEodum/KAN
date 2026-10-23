# M5 Owner Questions — Recommendations and Recorded Direction

**Project:** [0xEodum/KAN](https://github.com/0xEodum/KAN)  
**Branch:** `cpp-foundation`  
**Source:** [`docs/evidence/M5-plan.md` — Owner questions](https://github.com/0xEodum/KAN/blob/cpp-foundation/docs/evidence/M5-plan.md#owner-questions)  
**Date:** 2026-10-23  
**Status:** Recommendations for Q2–Q11; **Q1 decision confirmed by the owner**. This document does not change the repository's ROADMAP or CONTRACT by itself.

## Agreed milestone philosophy

**M5 is primarily experimental.** Conventional KAN carriers (polynomials, Fourier, splines, wavelets, and related families) prioritize high execution performance. Experimental PQC and Fock carriers prioritize a physically and mathematically sound implementation, correct gradients, usable training, and reproducible comparisons. Performance remains a practical constraint, but peak single-step speed is secondary.

The important comparative metric is **wall-clock time to reach the same independently measured result**, not the time of an isolated forward pass or optimization step:

\[
T_{\mathrm{target}} = \min\{t : \mathcal{L}_{\mathrm{validation}}(t)\leq\varepsilon\}.
\]

Experiments should define quality targets in advance, evaluate them on held-out data, measure complete end-to-end costs, use repeated seeds, and report unsuccessful runs as such. There is no blanket acceptable slowdown such as `99×` or `10×`: quality, total training time, reliability, parameter count, and memory must be considered together. Results on classical quantum simulators should not be described as quantum hardware advantage.

## Decision overview

| Question | Recommended direction | Decision status |
|---|---|---|
| Q1 — Quantum libraries | Permit cuQuantum/cuStateVec and third-party simulators in evidence/tests; adopt a shipping backend only after comparison | **Owner confirmed** |
| Q2 — Size limits | Keep current caps initially; revise from measured feasibility | Recommended |
| Q3 — Initialization | Keep analytic heuristics, validate their actual moments empirically | Recommended |
| Q4 — Fock truncation | Keep true truncation and no renormalization; add norm-deficit rejection to guard band | **Strong correction recommended** |
| Q5 — Shots | Deterministic keyed sampling with explicit user-controlled resampling | Recommended |
| Q6 — QUBO extensions | Keep the minimal CPU-only scope and offset binary | Recommended |
| Q7 — RTX 3090 | Compile-only `sm_86` evidence if no runtime device is available | Recommended |
| Q8 — Python names | Accept proposed names; keep naming consistent | Recommended |
| Q9 — Tape API | Keep tape and adapter boundary private | Recommended |
| Q10 — QUBO location | Keep `include/kan/qubo.hpp` and `src/qubo/` | Recommended |
| Q11 — Fock resident | Include a correct CUDA implementation in M5; tune proportionally to research value | Recommended |

## Q1 — cuQuantum, cuStateVec, and independent simulators

**Confirmed direction:** allow the use of CUDA quantum ecosystem packages and third-party quantum simulators for development, independent checking, and matched comparisons. The project already uses cuBLAS and has evaluated CUTLASS; using suitable ecosystem components is consistent with its existing engineering approach.

Separate three roles:

1. **Core implementation:** a self-contained CPU reference and a resident CUDA implementation without mandatory cuQuantum, Qiskit, PennyLane, or Strawberry Fields dependencies for ordinary library consumers.
2. **Test/evidence environment:** cuQuantum, Qiskit, PennyLane, and Strawberry Fields may be installed as *optional* independent cross-checks. Strawberry Fields can be particularly useful for continuous-variable/Fock comparisons where compatible observable and truncation conventions exist.
3. **Optional production CUDA backend:** `cuStateVec` may be adopted behind an optional build flag only **after** a matched comparison demonstrates a meaningful benefit and the dependency/compatibility burden is acceptable.

Comparison should include complete forward/backward/update behavior, numerical agreement, determinism, peak memory, integration cost, and — where applicable — time-to-quality. A small noisy speed difference does not by itself justify adopting an extra runtime dependency. A custom kernel is not inherently preferable to a well-supported library primitive, nor vice versa.

**Owner decision recorded:** “Permit quantum CUDA packages; optional use is welcome and could accelerate development. Choose between library and custom implementations after comparison. Do not adopt automatically.”

## Q2 — PQC/Fock size and repetition limits

**Recommendation:** retain the D9 limits for initial implementation: PQC qubits `1..5`, repetitions `1..32`, block length `1..3`; Fock cutoff `8..32`, repetitions `1..16`, and `r·(B+1) <= 16` for the initially bounded working set.

These are engineering limits, not assertions that maximum-sized models are practical. Distinguish *supported* from *recommended* research configurations, estimate total state, parameter, gradient, and scratch-memory requirements before execution, and raise limits only after profiling and validation. Revisit Fock workspaces in Phase 4 as planned.

## Q3 — Initialization distributions and moments

**Recommendation:** retain `[-π, π)` angle draws and the current VarianceScaling conventions as explicit initial approximations, while reporting measured behavior.

The PQC Haar moment `1/(2^n+1)` is exact under its assumed Haar ensemble but is not automatically the moment of a finite-depth, structured ansatz with restricted gates. The Fock displacement-only formulas are likewise approximations once nontrivial squeezing/Kerr operations are present.

Keep constructors and initializers deterministic, measure output second moments for shallow/deep circuits across multiple seeds and readouts, and document systematic departures. Do not require expensive per-instance Monte Carlo calibration at initialization unless evidence shows it is necessary.

## Q4 — Fock cutoff, guard band, norm loss, precision

**Recommendation:** retain true truncated matrix elements and **no automatic renormalization**, but change the acceptance policy before Phase 3.

The current guard-band test can accept pathological states with nearly zero population in the last retained levels because almost all weight has leaked beyond the cutoff. Add a complementary accumulated norm-deficit check, reject significant norm gain, and compare accepted values/derivatives with a higher-cutoff reference. Both checks should be interpreted as operational safeguards, not universal formal observable-error bounds.

FP64 should establish the reference contract. For FP32, either define a demonstrably sound precision-dependent guard strategy (possibly with higher-precision validation) or explicitly constrain FP32 Fock execution until adequate evidence exists. The `1e-10` threshold cannot simply be transplanted to an ordinary FP32 norm test.

No silent clipping or normalization. Specify clear errors and rollback behavior on both CPU and resident execution paths.

## Q5 — Sampling keys and finite-shot training

**Recommendation:** retain deterministic logical-index-based sampling and make resampling an **explicit state update**. A caller-controlled seed, epoch, or nonce is preferable to an implicit counter incremented inside `train_step`.

This permits both reproducible fixed-sample experiments and freshly resampled training without silently changing eager-versus-graph or CPU-versus-GPU semantics. Keep a key formula precise enough to reproduce independently. Test both training regimes, statistical gradient behavior, and variance/covariance of estimates sharing the same bitstring samples.

GPU shots remain within M5 if they can meet reproducibility and functional acceptance tests; maximizing their throughput is not a first-order milestone requirement.

## Q6 — QUBO scope and encoding

**Recommendation:** keep the initial module CPU-only: canonical pseudo-Boolean polynomials, offset-binary coefficient encoding, Freedman reduction of negative high-order terms, greedy Rosenberg for positive terms, exhaustive checks, and the one-layer KAN compiler.

Do not add annealer export, dimod integration, alternative solvers, Ishikawa reduction, or two's-complement coding without a concrete experimental need. Make coefficient-accumulation order and constant-energy semantics explicit as part of deterministic canonicalization. Independently verify objective values and argmin preservation.

## Q7 — RTX 3090 / `sm_86` validation

**Recommendation:** if an RTX 3090 is unavailable, require a successful `sm_86` compile/ptxas record but do not represent it as runtime validation. Record runtime results on available `sm_121` hardware and list the unverified target as a documented limitation. Re-run runtime parity/performance/sanitizers on `sm_86` when hardware becomes available.

## Q8 — Python API names

**Recommendation:** accept the proposed names: `kan.PqcConfig`, `kan.Rotation`, `kan.FockConfig`, and `kan.qubo.*`, together with consistent gate/readout/gradient enums and documented shape conventions. Keep bindings as thin reflections of the C++ numerical contracts; test snapshot, setter, error, and resident behavior before publicizing the surface as stable.

## Q9 — Public tape/adapter API

**Recommendation:** keep the opcode tape, simulator adapter boundary, and launch-level internals private during M5. Their representation will likely evolve with fused gates, checkpointing, alternative backends, and optimization. Make them public later only if a concrete extension use case warrants a supported compatibility commitment.

## Q10 — QUBO code location

**Recommendation:** preserve `include/kan/qubo.hpp` and `src/qubo/`. QUBO is a CPU-side discrete optimization/compilation module, not an edge carrier or CUDA simulator. It should remain architecturally independent of the quantum tape.

## Q11 — Fock on the resident CUDA executor

**Recommendation:** retain Phase 4 in M5. A CPU-only Fock implementation could lead to overly pessimistic conclusions about practical research cost. The CUDA acceptance target should be correct forward/backward/SGD execution, parity against the CPU reference, memory and sanitizer safety, determinism under the established contract, and measured end-to-end cost.

Use profiling to remove clearly pathological bottlenecks and choose an adequate working-set strategy. Do not turn M5 into an indefinite project of squeezing marginal percentage gains from custom kernels. The current `AGENTS.md` rule 6 and the plan's profiling/optimization gates should be reconciled with this research-first priority **through an explicit documented scope/acceptance clarification**, not by silently skipping existing gates.

## Proposed M5 acceptance addition — Time to quality

Add a small pre-registered comparative experiment set, independent of the unit tests:

- **Tasks:** smooth, oscillatory, compositional, and localized functions, with data splits and quality thresholds chosen before benchmarking.
- **Comparators:** relevant classical KAN carriers, for example Fourier on periodic problems and splines/wavelets on local features; match model complexity and tuning budgets as far as practicable.
- **Primary outcome:** wall-clock time to a shared held-out quality target, accounting for full training and finite-shot backward costs.
- **Secondary outcomes:** fraction of seeds reaching the target, quality-versus-time curves, total parameters, peak memory, and final held-out error.
- **Reporting:** multiple seeds; record censored/nonconverged runs, hardware, compiler, library versions, precision, and complete training configuration.

A quantum-inspired parameterization that is slower per step but reaches the target much earlier may be useful. One that is fast per step but never reaches the target is not. No claim of quantum computational advantage follows from speed measurements on classical simulators.

## Immediate next actions

1. Mark **Q1 as owner-approved** in the project decision record; the actual cuStateVec backend remains conditional on Phase 2b evidence and a subsequent adoption decision.
2. Amend D10/Q4 and the Phase 3/4 RED tests to cover catastrophic Fock truncation and norm loss.
3. Clarify Q5's explicit resampling semantics and Q3's heuristic initializer moments.
4. Retain the remaining Q2/Q6–Q10 defaults unless new evidence motivates a change; retain the Fock resident phase under Q11.
5. Add a controlled **time-to-quality** study and align the profiling policy with the experimental scope through the project's documented change process.

**Bottom line:** maximize correctness and experimental usefulness first, measure practical cost honestly, and select library primitives versus custom implementations from evidence rather than ideology.

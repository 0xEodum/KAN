# M5 Independent Plan Review

**Project:** [0xEodum/KAN](https://github.com/0xEodum/KAN)  
**Branch:** `cpp-foundation`  
**Document reviewed:** [`docs/evidence/M5-plan.md`](https://github.com/0xEodum/KAN/blob/cpp-foundation/docs/evidence/M5-plan.md)  
**Reviewed plan blob:** `13ef2b21bb76bbe1566a23d9daceb02681451a76`  
**Review date:** 2026-10-23  
**Status:** Advisory, read-only plan review; not an implementation audit or an executed test report.

## Executive assessment

M5 is substantially more than adding two new basis families. It introduces a small quantum-simulation infrastructure within KAN: typed parametrized quantum-circuit (PQC) and single-mode Fock carriers, CPU and resident CUDA adapters, physical and numerical contracts, differentiation, sampling, and a separate pseudo-Boolean/QUBO subsystem.

The direction is sound. In particular, the plan:

- Integrates both carriers through the existing `Layer`/`Network` carrier protocol instead of duplicating the architecture.
- Uses typed operations and a compiled integer opcode tape rather than string conversion, AST hashing, or parsing in numerical kernels.
- Defines explicit state, readout, gradient, and parameter layouts.
- Requires independently written numerical references, RED/GREEN evidence, sanitizer runs, and isolation gates protecting existing numerical results and CUDA code.
- Compares complete execution/training steps rather than celebrating an isolated fast kernel.

**Recommendation:** retain the architecture and overall phase structure, but revise the numerical acceptance criteria below before beginning the corresponding implementations. The most serious issue is the sufficiency of the proposed Fock truncation guard.

## Finding R1 — Fock guard-band population is not sufficient to establish safe truncation

**Severity:** P0 — specification-level correctness risk.  
**Plan references:** D10, *Fock carrier / Truncation and leakage*, Phase 3 tests, Q4.

The plan accepts a truncated state if the population of the highest `guard` Fock levels is below `max_leakage`, with defaults `cutoff = 32`, `guard = 2`, and `max_leakage = 1e-10`.

This check is an **indicator**, not a reliable acceptance criterion in isolation. A state can have negligible population in the retained boundary levels precisely because almost all probability mass lies **beyond** the cutoff.

For example, after a real displacement `D(alpha)|0>`, the exact photon-number distribution is

\[
P(m) = e^{-\alpha^2}\frac{\alpha^{2m}}{m!}.
\]

At `alpha = 10` and cutoff 32:

- The population in levels 30 and 31 is approximately `5.9265e-16`.
- The norm squared of the projected state on levels 0 through 31 is approximately `6.5159e-16`.
- Consequently, almost all norm is outside the truncated space, yet the specified guard-band threshold passes.

These figures describe the exact projected coherent state; they illustrate the logical gap in the acceptance predicate independently of the chosen recurrence implementation.

**Proposed change:** keep the boundary-population check, add an explicit accumulated norm-deficit check after truncated nonunitary maps, and benchmark observable errors against a larger-cutoff reference. Do **not** silently renormalize. A norm-deficit check is not, by itself, a universal rigorous observable-error bound; it is an essential additional rejection criterion alongside empirical larger-cutoff comparisons.

Suggested basic predicate, with precision-appropriate tolerances:

\[
\text{accept only if}\quad
P_{\mathrm{guard}}\leq\varepsilon_{\mathrm{guard}}
\quad\text{and}\quad
1-\lVert\psi\rVert^2\leq\varepsilon_{\mathrm{norm}}.
\]

For true projections of unitary operations, a detected norm gain beyond roundoff also warrants rejection. FP32 cannot reliably resolve norm losses on the order of `1e-10`, so FP32 needs a separately defined strategy and tolerance, rather than a copied FP64 threshold.

**Additional RED cases:** large displacements such as `alpha = 8` and `10`; multiple displacement/squeezing gates; states close to both boundaries; and an explicit cutoff-convergence table for accepted states.

## Finding R2 — Canonical QUBO representation does not automatically imply bitwise-deterministic accumulation

**Severity:** P1 — reproducibility-contract ambiguity.  
**Plan references:** *QUBO and quadratization*, Phase 5 test 1.

The plan promises bitwise-identical canonical polynomials under permutations of the input monomials. Ordering the **resulting monomials** by degree and lexicographically is not enough if equal monomials are merged by ordinary floating-point addition in input order.

For example, in binary64 arithmetic, the sequences `1e16 + 1 - 1e16` and `1e16 - 1e16 + 1` need not produce the same result.

**Proposed change:** specify a deterministic reduction order for all contributions to the same canonical monomial, including the behavior of signed zero and cancellation. One possible implementation is to sort contributions by a precisely defined representation before summing; stronger cross-platform guarantees may call for a suitably exact accumulator. The required guarantee and its cost should be documented.

Also explicitly define whether `energy(Qubo, assignment)` already includes `Qubo::constant`. The Phase 5 assertion `energy(z) + constant == L(decode(z))` otherwise risks double-counting or an API mismatch.

## Finding R3 — “Within 4 ULP of zero” is a fragile parameter-shift acceptance test

**Severity:** P1 — numerical test design.  
**Plan references:** PQC symmetric-point discussion and Phase 1 gradient tests.

The chosen ZYZ structure contains angles that are analytically inert for certain initial states or Z/parity readouts. Requiring the adjoint result to be exactly zero can be valid when it follows from the implementation's algebraic path. However, the parameter-shift result is a difference of two independently computed expectations.

An acceptance condition defined as “within 4 ULP of zero” is not a useful error scale for that subtraction: roundoff is determined by the magnitude of the expectations being subtracted, not the ULP spacing at zero.

**Proposed change:** retain exact-zero assertions where the algorithm itself guarantees them; for parameter-shift use an absolute bound justified by machine epsilon, gate count, and expectation magnitude. Preserve independent finite-difference and adjoint-vs-shift comparisons away from symmetry points.

## Finding R4 — Fixed shot keys improve reproducibility but change training-noise behavior

**Severity:** P1 — experimental semantics.  
**Plan references:** D11, *Measurement semantics*, Phase 6, Q5.

With a fixed `shot_seed`, repeated evaluation of a given circuit at a given logical index reuses the same pseudo-random quantiles. That is valuable for replayability, graph/eager parity, debugging, and common-random-number comparisons, but it is not the same training process as drawing fresh Monte Carlo samples at every optimization step.

A model can adapt to a fixed finite-shot approximation. Therefore a successful regression under fixed samples does not alone establish the behavior of genuinely resampled shot-based optimization.

**Proposed change:** keep the deterministic key formula and make resampling an explicit, uploadable state change (for example, an epoch/nonce or a new seed). Avoid an implicit, backend-specific training-step counter. In Phase 6, compare fixed-sample and explicitly resampled training trajectories and report gradient variance and time-to-quality.

For multi-qubit `PauliZ` readout, also test covariance between different `Z_q` estimators: measurements of different terms from the same sampled bitstrings are not statistically independent.

## Finding R5 — Repeated CPU tape compilation may add avoidable work in the hot path

**Severity:** P2 — execution cost/architecture.  
**Plan references:** D3 and Phase 1 allocation/performance tests.

The plan says the CPU tape is compiled “per call.” If every edge/sample call to `evaluate_pqc` rebuilds the same structural tape, a layer may repeatedly perform identical work across `batch × outputs × inputs` evaluations.

**Proposed change:** compile the tape once per `Layer::forward`/`Layer::backward` invocation (or another appropriately scoped operation) and pass an immutable view to the edge evaluator. A standalone public `evaluate_pqc` call can still build its own bounded local tape. Measure this preparation cost separately from circuit evaluation and VJP cost.

A further useful distinction is between *formally supported maximum configurations* and *practically recommended research configurations*. With 5 qubits, 32 repetitions, and a 3-axis block, an individual edge can have hundreds of nonlinear angles/encoding parameters; multiplying that by a large dense layer creates substantial parameter and gradient storage even though the state vector is small.

## Finding R6 — Front-load a small Fock mathematical feasibility check

**Severity:** P2 — sequencing/risk reduction.  
**Plan references:** Phases 1–4.

The overall ordering (PQC CPU → PQC resident → cuStateVec comparison → Fock CPU → Fock resident) is defensible and should be retained. Nevertheless, the Fock truncation contract is risky enough to justify a short, independent mathematical feasibility check *before* committing to the full Phase 3/4 integration.

This check should exercise recurrence-derived displacement/squeezing elements and derivatives, extreme inputs, norm deficits, guard-band behavior, and convergence under increasing cutoff. It need not change the official milestone ordering or produce shipping code.

The existing cuStateVec comparison remains worthwhile. Measure peak memory as well as matched full-step time, especially because per-state batching and ancillary buffers can dominate GPU feasibility.

## Performance interpretation for an experimental milestone

For experimental PQC/Fock carriers, **mathematical correctness, trainability, integration, and scientifically usable cost** should precede exhaustive low-level tuning. A slow individual optimization step is not proof of an impractical method; a faster step is not proof of a better one.

A decisive future comparison is **time to an independently evaluated quality target**:

\[
T_{\mathrm{target}} = \min\{t:\mathcal{L}_{\mathrm{validation}}(t)\leq\varepsilon\}.
\]

This should be measured alongside success rate, parameter count, peak memory, and full optimization cost; it should not be reduced to an arbitrary acceptable slowdown factor. The plan must still honor its current profiling/isolation requirements until any revised performance policy is explicitly recorded.

## Suggested edits before implementation

1. **D10 / Q4 / Phase 3:** add a norm-deficit rejection criterion, FP32-specific handling, and catastrophic-tail RED tests.
2. **QUBO / Phase 5:** define deterministic coefficient summation and the `energy()` constant convention.
3. **PQC / Phase 1:** replace the near-zero parameter-shift ULP condition with a justified absolute tolerance.
4. **D11 / Q5 / Phase 6:** define explicit resampling and compare fixed-versus-fresh sampling.
5. **D3 / Phase 1:** document the compilation lifetime of the CPU tape and benchmark its overhead.
6. **Phase 3 preflight:** add the focused Fock numerical feasibility check.

**Conclusion:** the M5 design is a coherent foundation for experimental quantum KAN carriers. The changes above strengthen its scientific and numerical validity without turning M5 into an open-ended optimization project.

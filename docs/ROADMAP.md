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

## M5: experimental quantum carriers — NEXT

Typed PQC/Fock carrier interfaces and simulator adapters, physical normalization,
measurement/gradient semantics and independently verified QUBO/quadratization.
No string hashing or symbolic parsing in numerical kernels.

## M6: research and distribution — PLANNED

Versioned model format, packaging, reproducible matched benchmarks and expanded
examples. License selection belongs to the repository owner.

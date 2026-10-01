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

## M3: localized and adaptive bases — NEXT

B-splines with knot/domain contracts, wavelets with scale/translation conventions,
learnable RBF centers/widths, adaptive grids and regularization.

## M4: rational edges — PLANNED

Padé/rational parameterization, singularity policy, conditioning and nonlinear
parameter derivatives. Do not disguise rational edges as fixed linear basis terms.

## M5: experimental quantum carriers — PLANNED

Typed PQC/Fock carrier interfaces and simulator adapters, physical normalization,
measurement/gradient semantics and independently verified QUBO/quadratization.
No string hashing or symbolic parsing in numerical kernels.

## M6: research and distribution — PLANNED

Versioned model format, packaging, reproducible matched benchmarks and expanded
examples. License selection belongs to the repository owner.

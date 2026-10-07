# Decision: permit CUDA ecosystem libraries

Date: 2026-10-07. Branch: `cpp-foundation`. Workspace baseline: `6d59ffd`.
Stage: B (review backlog); M5 remains PLANNED and depends on closure of B.

## Owner decision

The owner approved an explicit general permission to use CUDA ecosystem libraries
and components, rather than granting permission separately for each library.
[AGENTS.md](../../AGENTS.md), rule 8, permits cuBLAS/cuBLASLt, CUTLASS/CuTe,
cuTENSOR, cuDNN, cuSPARSE, cuSOLVER, cuFFT, CUB/Thrust and other CUDA components.
Prefer suitable library primitives and their extension. Use custom CUDA kernels
when library facilities do not cover the operation or contract, or when profiling
and matched measurements demonstrate an advantage for the custom implementation.

Implementation acceptance still requires:

- Compliance with numerical, precision and reproducibility contracts.
- Measurements of complete calls or training steps, rather than kernel timing alone.
- Evidence of adopted dependency versions, GPU/toolchain requirements and installation
  validation; the CPU-only build remains independent of CUDA.

This is permission to select a library, not a requirement to add every listed
dependency. It makes no claim about coverage or guaranteed acceleration. Adopting
a concrete dependency is still recorded and validated under these rules, without
a separate approval requirement merely because it is a CUDA library.

## Rationale and scope

[C3 evidence](backlog/C3.md) records that stage-2 hand-written fused GEMM prototypes
lost to the existing library path on the principal shapes. Their GEMM and fusion
costs exceeded the memory-traffic savings. Library primitives and customizable
implementations are therefore permitted tools for subsequent optimization; their
actual benefit must be measured.

C3 remains closed at stage 1. Permission to use CUTLASS does not reopen C3, accept
the rejected prototypes, or add an implementation task. No dependency, runtime
behavior, precision default or stage ordering changes in this documentation pass.

The owner also requested a complete English translation of [BACKLOG.md](../BACKLOG.md)
and an updated quick context. Historical findings, measurements and status-journal
entries are retained; current context is distinguished from the original review.

## Verification of the policy step

- Read ROADMAP completely; verified stage B is NEXT and M5 depends on B.
- Verified the clean `cpp-foundation` baseline and tracked upstream
  `origin/cpp-foundation` before editing.
- Read C3 evidence and checked CMake: the current CUDA target links cudart/cuBLAS;
  CUDA remains optional and the CPU target is separate.
- Checked the documentation patch with `git diff --check` and reviewed its scope.
  No code, build dependencies or benchmark artifacts were changed; runtime tests
  and profiling are not needed for this documentation-only policy change.

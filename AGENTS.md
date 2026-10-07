# Project Agent Instructions

Before doing any work in this workspace:

1. Read `docs\ROADMAP.md` completely. It is the source of truth for scope, ordering,
   architectural invariants, acceptance criteria, and the current handoff.
2. Verify the workspace and recorded evidence instead of assuming backlog facts
   are current.
3. Work only on the single `NEXT` stage, or on the nearest `READY` stage
   when no stage is in progress. Respect all declared dependencies.
   Until every item in `docs\BACKLOG.md` is closed, the backlog stage is the
   `NEXT` stage: M5 must not start while any backlog item remains open.
   Close an item only after its fix is verified, and record the closure in
   the backlog status journal with a link to its evidence.
4. Do not mark a stage `DONE` without satisfying its tasks, exit criteria, and
   evidence requirements.
5. Record scope, default, dependency, or acceptance-criteria changes in the
   `docs\evidence` directory.
6. When working on CUDA code, carry out profiling to identify bottlenecks and resolve them. 
   The GPU should be utilised to its full potential.
7. Commit work incrementally as it is completed: each finished, verified step
   (RED test, fix, evidence, backlog update) gets its own conventional commit
   (`feat:`, `fix:`, `test:`, `docs:` …) and is pushed to the tracked branch.
   Do not leave completed work uncommitted at the end of a session. Commit only
   files that belong to the step; regenerated `artifacts/` are committed only
   when recorded as evidence per `BACKLOG.md`.
8. CUDA ecosystem libraries and components are permitted, including
   cuBLAS/cuBLASLt, CUTLASS/CuTe, cuTENSOR, cuDNN, cuSPARSE, cuSOLVER, cuFFT,
   CUB/Thrust and others. Prefer suitable library primitives and their extension.
   Use custom CUDA kernels when library facilities do not cover the required
   operation or contract, or when profiling and matched measurements demonstrate
   an advantage for the custom implementation. Evaluate implementations against
   the numerical, precision and reproducibility contracts and measure complete
   calls or training steps, not only individual kernels. Record adopted dependency
   versions, GPU/toolchain requirements and installation validation in evidence;
   keep the CPU-only build free of CUDA dependencies. This policy grants permission
   to use libraries; it does not require adding them or guarantee a performance gain.

# M1–M4 review backlog

Original review: 2026-10-01, branch `cpp-foundation`, HEAD `7b58b61` (M4 was closed;
M5 was NEXT at the time). Current context verified on 2026-10-07 against `6d59ffd`
(C3 closure) and the subsequent CUDA library policy decision; status updated for the
C6–C8 closure (`f39004d`).
[ROADMAP.md](ROADMAP.md) is the source of truth for stages; this file records review
findings, their rationale and the proposed work order. Changes to ROADMAP (a new
stage or a change to M5 scope) require a separate record in `docs/evidence`, as
specified by `AGENTS.md`.

## Quick context

- **Current stage:** B is NEXT. All remaining backlog items must close before
  roadmap M5 starts; M5 is PLANNED and depends on B. Work in passes of at most three
  items, honoring dependencies and taking dependency-free P0 items first. The
  original R → C1/C2 → M5 recommendation is historical; it does not bypass the
  [backlog gate](evidence/backlog-gate.md). Backlog IDs M1–M7 are review items,
  distinct from roadmap milestones with the same names.
- **Status:** 28 items, 19 closed and 9 open. R1–R3 and R7–R9 are closed; R4–R6
  remain open. M1, M2 and M4 are closed; M3 and M5–M7 remain open. C1–C4 and C6–C11
  are closed (C3 at stage 1); C5 and C12 remain open. All P0 items are closed.
  C12 is still open for FP64; its FP32 portion was addressed in C1.
- **Architecture:** typed per-family `BasisConfig`; `Carrier` separates fixed-basis
  edges, trainable RBF edges and rational edges. The shared CPU linear engine is
  in `src/carriers/linear_engine.hpp`, CPU carrier loops in `src/carriers/`, and
  shared host/device formulas in `src/detail/`. Input maps are in `src/input_map.cpp`;
  initializers in `src/initializers.cpp` and `src/init/`; topology in `src/network.cpp`.
  Public headers are in `include/kan/`. See [CONTRACT.md](CONTRACT.md) for numerical
  semantics and extension boundaries; R4's directory reorganization is pending.
- **CUDA:** `src/resident.cu` implements persistent mixed-network execution with
  reusable storage, cuBLAS contractions, specialized small kernels, explicit
  `upload_parameters`, and graph-based `train_step`. `src/cuda.cu` is a deprecated,
  kernel-free M1 adapter over `ResidentNetwork`; the supported device query is in
  `kan/cuda_runtime.hpp` / `src/cuda_runtime.cpp`. CUDA is an optional `kan::cuda`
  target, currently linked to cudart/cuBLAS (CUDA >= 12); the CPU target has no
  CUDA dependency. FP64 remains the default; FP32 and TF32 are opt-in.
  `KAN_CUDA_FMA=OFF` remains the default parity build; ON selects the performance
  build. `status_interval` currently defaults to 1; C9's owner question about that
  default remains recorded in the journal.
- **Libraries and custom kernels:** CUDA ecosystem libraries and components are
  permitted, including cuBLAS/cuBLASLt, CUTLASS/CuTe, cuTENSOR, cuDNN, cuSPARSE,
  cuSOLVER, cuFFT, CUB/Thrust and others. Prefer suitable library primitives and
  their extension. Use custom CUDA kernels when library facilities do not cover
  the operation or contract, or when profiling and matched measurements demonstrate
  an advantage. Preserve numerical, precision and reproducibility contracts;
  measure complete calls or training steps, document adopted dependency versions,
  GPU/toolchain requirements and installation validation, and keep CPU-only builds
  independent of CUDA. See `AGENTS.md` rule 8 and the
  [policy decision](evidence/cuda-library-policy.md). Permission does not add a
  dependency or promise acceleration. C3 remains closed; further GEMM fusion is a
  possible follow-up, not an open item created by this policy.
- **Evidence:** the latest recorded C3 validation reports 41/41 CTest in both
  builds, byte-identical baseline comparisons, clean sanitizers and no FP64 frozen
  regression ([C3](evidence/backlog/C3.md)); these are recorded results, not new
  test runs in this documentation pass. Current training harnesses and profiles
  are in [C9-bench](evidence/backlog/C9-bench/) and
  [C3-bench](evidence/backlog/C3-bench/); each closure below links its own evidence.
  [review-2026-10-01](evidence/review-2026-10-01/) contains the original diagnostic
  PyTorch/resident comparison, not a benchmark of the current implementation.

Priorities: **P0** — blocks the next milestone or offers a multiplicative speedup;
**P1** — important; **P2** — desirable.

The tables below retain the recorded findings and proposed tasks. The
"Where / why" column describes the original finding (or a later recorded addition),
not the current implementation of closed items; source line numbers are historical.
Use the linked evidence and CONTRACT for accepted behavior.

---

## R. Code organization

Key point: `carriers/…` directories are a useful direction, but the problem is in
the types, not the folders. Group by properties relevant to execution rather than
"orthogonal/harmonic" (Fourier is also orthogonal; B-splines are not).

| ID | P | Task | Where / why (recorded finding) |
|---|---|---|---|
| R1 | P0 | (DONE, [evidence](evidence/backlog/R1.md)) Replace the flat `BasisConfig` with `std::variant<ChebyshevConfig, …, BSplineConfig>` (pybind11 supports variants) | `include/kan/basis.hpp:10-23` — all fields of all families in one structure |
| R2 | P0 | (DONE, [evidence](evidence/backlog/R2.md)) Remove the `rational_` flag and branches from `Layer`: separate carriers **linear in their parameters** (shared "expansion + GEMM" engine) from **nonlinear** carriers (rational, trainable RBF, future PQC) with their own VJPs. Move family-specific methods (`insert_knot`, `adapt_grid`, `set_rbf_parameters`) out of the common class | `src/layer.cpp:88`, `:127`, sgd/validate; otherwise M5 adds a third branch |
| R3 | P0 | (DONE, [evidence](evidence/backlog/R3.md)) Single source for each family's formulas: `__host__ __device__` header functions shared by CPU and CUDA | Duplicated at review time: `src/basis.cpp` ↔ `src/resident.cu:48-172`, `src/rational.cpp` ↔ `src/resident.cu:258-327` (Jacobi, log-space Gaussian/Mexican hat, Cox–de Boor, Horner) |
| R4 | P1 | Directory layout: `carriers/{polynomial,trigonometric,local,rational,quantum}`, `backends/{cpu,cuda}`, `core/` (Layer, Network, errors, shapes) | Review proposal; `local/` = B-spline, RBF, Mexican hat |
| R5 | P1 | Rename tests and benchmarks by feature, mirroring `carriers/` | `m3_layer_test`, `m4_resident_test`, etc. are named after milestones |
| R6 | P2 | Add `.clang-format` and format M3/M4 code accordingly | Dense lines with multiple statements, e.g. `src/resident.cu:520-523`, `src/rational.cpp` |
| R7 | P2 | (DONE, [evidence](evidence/backlog/R7.md)) Deprecate the M1 API in `src/cuda.cu` or route it through the resident executor | Chebyshev only, malloc per call, `coefficient_gradient_kernel` at O(K²·B) (`src/cuda.cu:140`) |
| R8 | P1 | (DONE, [evidence](evidence/backlog/R8-R9.md)) Shared device-query header `kan/cuda_runtime.hpp`: both `kan/cuda.hpp` and `kan/resident.hpp` include `kan::cuda::available()` (same name, not deprecated); implementation separate from the legacy adapter | Owner decision following R7 ([decision](evidence/backlog/R8-R9-decision.md)): the deprecation boundary was incorrect — the device query is also needed by the modern API |
| R9 | P1 | (DONE, [evidence](evidence/backlog/R8-R9.md)) `ResidentNetwork::upload_parameters(...)` — explicit parameter upload (weights after CPU training, model restoration) without changing existing signatures; no automatic per-thread executor cache | Owner decision following R7 ([decision](evidence/backlog/R8-R9-decision.md)): the main path is one `ResidentNetwork` reused across many calls |

---

## M. Mathematics and trainability

Verified in the review: the Jacobi recurrence (against `scipy.special.eval_jacobi`,
maximum relative error 6e-15, including |x|>1 and α+β=−1), Cox–de Boor derivative,
Boehm knot insertion, Mexican hat (Ricker) derivative and normalization,
d/dlog-width RBF = 2q²e^{−q²}, and rational VJP
(dr/da = z^k/Q, dr/db = −r·z^k/Q, Q = 1+Σb·z^k).

For families linear in their coefficients, a KAN layer is equivalent to an
expansion Φ: ℝ^I→ℝ^{I·K} followed by a dense linear layer. This follows from the
definition and is the basis of C2.

| ID | P | Task | Where / why (recorded finding) |
|---|---|---|---|
| M1 | P0 | (DONE, [evidence](evidence/backlog/M1.md)) Explicit typed input map (affine / tanh / LayerNorm) as a separate layer | The contract forbids implicit normalization, but no tool existed. For \|x\|≫1, polynomials grow as (2\|x\|)^n → exploding gradients; local bases outside [t_p, t_K] give zero and zero gradient (a "dead" edge) |
| M2 | P0 | (DONE, [evidence](evidence/backlog/M2.md)) Rational: safe pole-free denominator — PAU (Molina et al. 2019) `Q = 1+\|Σ b_k z^k\|` or smooth `Q = 1+(Σ…)²` — as an optional singularity policy | The guard throws `domain_error` (`src/rational.cpp:41`, GPU status 2): one SGD step into a pole aborts training without recovery |
| M3 | P1 | Optional residual branch `w_b·silu(x)` (as in the original KAN) | Only gradient path outside the grid for B-spline/RBF |
| M4 | P1 | (DONE, [evidence](evidence/backlog/M4.md)) Initializers (family-specific variance preservation, noise initialization as in pykan) | The contract acknowledges that zero initialization does not train multilayer networks |
| M5 | P1 | Normalized Hermite functions `H_n(x)e^{−x²/2}/√(2^n n! √π)` as an option | Physicists' H_n grow as ~2^n·n!, with poor conditioning |
| M6 | P2 | Per-input grid (as in pykan), rather than one grid per layer; refit the grid by quantiles; `adapt_grid` with samples propagated through preceding layers | Knots/centers/scales are shared across a layer; `adapt_grid` inserts one knot at a time |
| M7 | P2 | Trainable per-edge scale/translation for Mexican hat (Wav-KAN) | The implementation uses a dictionary of fixed wavelets |

---

## C. CUDA

**Historical diagnostic baseline (2026-10-01), not current performance.**
RTX 3090 (FP64:FP32 = 1:64), Chebyshev K=7, complete forward+backward+SGD step.
Single runs under WDDM, outside the project's frozen protocol — diagnostic
measurements, not accepted performance evidence.

| Topology, batch | resident FP64 | PyTorch FP64 | PyTorch FP32 |
|---|---:|---:|---:|
| 64→64→32→16, 1024 | **2.46 ms** | 3.11 | 3.60 |
| 256→256→256→10, 8192 | 310 ms | 87 | 9.5 |
| 1024→1024→1024, 4096 | 7039 ms | 613 | 22.9 |

Interpretation of that baseline: on the small network, resident beat eager
PyTorch (kernel launches dominated). On large networks, ~360 GFLOP/step yielded
≈51 GFLOPS — about 9% of peak FP64 (~0.56 TFLOPS; PyTorch FP64 reached the peak).
FP32 offered a further ~27× on GeForce. C1/C2/C9/C3 have since changed the executor;
their evidence, rather than this table, describes accepted subsequent results.

| ID | P | Task | Where / why (recorded finding) |
|---|---|---|---|
| C1 | P0 | (DONE, [evidence](evidence/backlog/C1.md)) Precision policy: template on `Scalar`; FP32 (optionally TF32/BF16) for training, FP64 as the parity reference | Largest speedup factor on GeForce |
| C2 | P0 | (DONE, [evidence](evidence/backlog/C2.md)) Express the contraction as GEMM (cuBLAS/cuBLASLt, bias via epilogue): `Y = Φ·Cᵀ + b`, `dC = Uᵀ·Φ`, `dX = Σ_k (U·C)⊙Φ'` | `src/resident.cu:174-212` — naive untiled GEMMs, uncoalesced coefficient access |
| C3 | P1 | (DONE at stage 1 by owner decision, [evidence](evidence/backlog/C3.md)) Fused kernel: evaluate the basis in shared memory while loading the X tile; recompute Φ' in backward rather than storing it | V and D tensors of size B·I·K were written to global memory (for 1024-wide, B=4096 — ~235 MB each per layer) |
| C4 | P1 | (DONE with R3, [evidence](evidence/backlog/R3.md)) Template `basis_kernel` by family | `src/resident.cu:48`, scratch `double lower[18], next[18]` (`:67`) dictates register/local memory requirements for all families |
| C5 | P1 | Sparse B-spline path: store `(span, p+1 values)` | Only p+1 values are nonzero, but all K are stored and multiplied; K grows after `adapt_grid` |
| C6 | P1 | (DONE, [evidence](evidence/backlog/C6-C8.md)) Rational forward: make sample the fastest-varying index | `rational_forward_kernel`, `index%outputs` (`:283-286`) → cache writes with stride I·capacity |
| C7 | P1 | (DONE, [evidence](evidence/backlog/C6-C8.md)) Rational forward: remove evaluation of all VJPs solely for finiteness checks | `:317-322`, extra FP64 divisions (very expensive on GA102) |
| C8 | P1 | (DONE, [evidence](evidence/backlog/C6-C8.md)) Rational parameter VJP: one warp per edge, collecting all m+n+1 sums together | `rational_parameter_kernel` (`:337`): one warp per parameter, each rereads z, P, Q and recomputes powers |
| C9 | P1 | (DONE, [evidence](evidence/backlog/C9.md)) Check status once per step / once every N steps; capture the step in a CUDA Graph | `result()` (`:482`) — status memcpy + sync after forward, backward and SGD (3 times per step) |
| C10 | P2 | (DONE, [evidence](evidence/backlog/C1.md)) Use `--fmad=false` only in the parity build; enable FMA in the performance build | FMA was disabled everywhere |
| C11 | P0 (process) | (DONE) Enable Nsight Compute counters: NVIDIA Control Panel → Developer → Manage GPU Performance Counters → "Allow access to all users" (or run ncu as administrator) | `ERR_NVGPUCTRPERM` in all three milestones — optimization proceeded without occupancy/bandwidth counters |
| C12 | P2 | Nonlinear RBF reduction: coalesced access | `nonlinear_partial_kernel`: `dx`/`dw`/`c` reads with stride K |

---

## Reproducing measurements

The commands below reproduce the **original review diagnostic**. For subsequent
accepted benchmarks, use each item's evidence and its baseline/build instructions
(in particular [C9](evidence/backlog/C9.md) and [C3](evidence/backlog/C3.md)).

```powershell
# PyTorch reference (requires torch with CUDA)
python docs/evidence/review-2026-10-01/torch_reference.py

# Resident executor: requires a Release CUDA build (scripts/build.ps1 -Cuda ... -BuildDirectory build-m4-cuda)
cmd /c docs\evidence\review-2026-10-01\build_resident_bench.cmd build-m4-cuda
& "$env:TEMP\resident_bench.exe"
```

## Status journal

Entries retain the decisions, validation results and measurements recorded at
the time; later entries and linked evidence describe subsequent changes.

| Date | Change |
|---|---|
| 2026-10-01 | Backlog created from the M1–M4 review; all items open |
| 2026-10-01 | C11 closed by the owner (Nsight Compute counters available) |
| 2026-10-01 | Backlog became stage B in ROADMAP; M5 starts only after all items close ([decision](evidence/backlog-gate.md)). First pass: R3, R1 |
| 2026-10-01 | R3 closed: basis and rational formulas share `KAN_HOST_DEVICE` templates in `src/detail/`; CPU+CUDA golden dump byte-identical, 19/19 CTest, GCC 11/11. Profiling identified and resolved three regressions ([evidence](evidence/backlog/R3.md)) |
| 2026-10-01 | C4 closed with R3: `basis_kernel` instantiated per family (66 → 36–62 registers, spline scratch only for B-splines), kernel time −0.2…−12.6% ([evidence](evidence/backlog/R3.md)). Pass 1: R3, C4, R1 |
| 2026-10-01 | R1 closed: `BasisConfig` = `std::variant` of typed per-family configurations, local-family size inferred, separate `TrainableRbfConfig`, `BasisKind` moved to `detail`; golden byte-identical, 20/20 CTest, GCC 12/12 ([evidence](evidence/backlog/R1.md)). Pass 1 completed: R3, C4, R1 |
| 2026-10-02 | R2 closed: `Layer` holds `kan::Carrier = std::variant<BasisEdges, TrainableRbfEdges, RationalEdges>`, shared linear "expansion Φ + contraction" engine (`src/carriers/linear_engine.hpp`), per-carrier nonlinear VJPs, family operations as free functions in `kan/families.hpp`, resident plan per carrier; golden byte-identical (+ new `--layers` dump), 21/21 CTest, GCC 13/13, coverage 98.5%, CPU 1.1–2.2× faster ([evidence](evidence/backlog/R2.md)). Pass 2: R2, then M1 and M2 |
| 2026-10-02 | M2 closed: `RationalConfig::denominator_policy` = `Guarded` (default, previous behavior) / `Absolute` (PAU, 1+\|S\|) / `Smooth` (1+S²); shared host/device formulas, kernels instantiated per policy; an SGD step into a pole no longer aborts training with safe policies; golden (both modes) byte-identical, 23/23 CTest, GCC 14/14, coverage 98.5% ([evidence](evidence/backlog/M2.md)). Remaining parameter-VJP cost → C8, nonzero denominator initialization → M4 |
| 2026-10-02 | M1 closed: explicit typed `kan::InputMap` layer with `AffineMap` (fixed, helpers `affine_from_range`/`affine_from_moments`), `TanhMap`, `LayerNormMap` (trainable gain/bias); `Network` = sequence of `std::variant<Layer, InputMap>`; shared host/device formulas, CUDA map kernels ≈1% of the step; demonstration: inputs in [100, 500] without a map cause overflow (Chebyshev) or zero gradient (B-spline), with a map loss is 8e-20 ([evidence](evidence/backlog/M1.md)) |
| 2026-10-02 | Pass 2 completed: R2, M2, M1. M1 and M2 ran in parallel in worktrees and were merged in `39e115c` (M2 tests adapted to heterogeneous `Network::layers()`); combined tree: MSVC+CUDA+Python 26/26, GCC 15/15, golden (both modes) byte-identical to `5dc6819` |
| 2026-10-02 | Pass 3: C2, then C1 (+ C10). The owner accepted tolerance-based rather than bitwise CPU/GPU parity and a cuBLAS dependency |
| 2026-10-02 | C2 closed: resident contractions use cuBLAS DGEMM (`Y = ΦCᵀ`, `dC = UᵀΦ + λC`, `W = UC`) plus two small kernels where cuBLAS hits single-tile latency (forward ≤ 2²³ FMA — warp per output; parameter VJP ≤ 2¹⁵ — batch tiles); RBF reduction over `W` (B·I·K rather than B·I·O·K). Step: 256-wide 286 → 96 ms, 1024-wide 6.8 → 0.69 s (≈94% of FP64 peak), trainable RBF 920 → 113 ms; frozen resident m2 ×0.61, m3 ×0.65, no basis case slower by >2%. 27/27 CTest, GCC 15/15, clean sanitizers, CPU golden byte-identical, resident deviation ≤ 3.1e-14; CUDA ≥ 12, Python registers the cuBLAS DLL directory ([evidence](evidence/backlog/C2.md)) |
| 2026-10-05 | Pass 4 began with R7 (P2), outside strict priority order: it was the only open item that did not conflict with parallel C1 (+ C10) work on `src/resident.cu`/`src/detail/` (owner-approved) |
| 2026-10-05 | R7 closed: `kan::cuda::forward`/`backward` are `[[deprecated]]` (message points to `ResidentNetwork`), signatures unchanged; `src/cuda.cu` (239 → 60 lines) is a kernel-free adapter constructing a single-layer `ResidentNetwork` per call; all resident carriers accepted (contract change), CPU tolerance `\|a-e\| ≤ 1e-12\|e\| + 1e-13 max\|e\|`; `available()` not deprecated; frozen benchmarks deliberately call the legacy API with local warning suppression. 29/29 CTest (+2 deprecation compile probes), GCC 15/15, memcheck: 0 leaked bytes, golden (both modes) byte-identical; `cudaMalloc` calls unchanged (5 per call), large Chebyshev backward 178 → 24 ms, small forward 0.46 → 1.1 ms due to executor construction (diagnostics on a busy GPU) ([evidence](evidence/backlog/R7.md)) |
| 2026-10-05 | C1 and C10 closed: `kan::cuda::Precision` = `Float64` (default, bitwise C2 executor: golden identical in both modes) / `Float32` / `TensorFloat32` (FP32 + TF32 tensor GEMMs, opt-in); kernels, plans and executor templated on scalar, shared `src/detail/` formulas templated on `Scalar`; host API remains `double`, FP32 rejects unrepresentable data/configurations (`invalid_argument`), FP32 guarded-pole threshold `max(epsilon, n·2⁻²³)`; FP32 tolerance `2e-4\|e\| + 5e-5 max\|e\|`, TF32 `1e-2`; Python `kan.Precision`. C10: CMake `KAN_CUDA_FMA` (default OFF = parity build with `--fmad=false`; ON = performance build, `build.ps1 -CudaFma`). Profiling: uncoalesced Φ/Φ'/W rows (shared-memory tiles, FP32 only), tiled RBF reduction (FP32 portion of C12), FP32 small-kernel thresholds, pinned uploads. FP32 step: 256-wide 99 → 3.85 ms (34% of FP32 peak, PyTorch FP32 9.0 ms), 1024-wide 0.70 s → 27.8 ms (36%; without upstream upload 17.6 ms, 58%; PyTorch 22.3 ms), trainable RBF 116 → 5.5 ms; no FP64 frozen regression. 31/31 CTest in both builds, GCC 15/15, clean sanitizers ([evidence](evidence/backlog/C1.md)). C12 remains open for FP64 |
| 2026-10-05 | R8 (shared `kan/cuda_runtime.hpp` header for `available()`) and R9 (`ResidentNetwork::upload_parameters`) added by owner decision on R7's open questions; automatic per-thread executor cache rejected ([decision](evidence/backlog/R8-R9-decision.md)) |
| 2026-10-05 | R8 closed: `kan::cuda::available()` (same name, not deprecated) declared in new CUDA-independent `kan/cuda_runtime.hpp`, included by `kan/resident.hpp` and `kan/cuda.hpp`; implementation moved from legacy `src/cuda.cu` to `src/cuda_runtime.cpp`; `src/resident.cu` and Python bindings no longer include `kan/cuda.hpp`. New compile probe `cuda_deprecation_resident_only` (only `kan/resident.hpp`, deprecation as error) passes, R7 probes unchanged, Python `cuda_available()` works ([evidence](evidence/backlog/R8-R9.md)) |
| 2026-10-05 | R9 closed: `ResidentNetwork::upload_parameters(network)` (Python `upload_parameters`) uploads all trainable state (coefficients, bias, trainable RBF centers/log widths, rational denominators, LayerNorm gain/bias) without device allocation; strict structure validation (layer kinds and sizes, carriers, fixed configuration via `operator==`, including spline knots: knots are structure, so `insert_knot`/`adapt_grid` requires a new executor) and value validation under constructor rules (FP32: representability) before any mutation, otherwise `invalid_argument` with executor unchanged; success invalidates output/gradients, retaining input and upstream. One copy (FP64 > 1 MiB — per tensor without staging: 62 → 45 ms). Cost: 0.16 ms versus 2.2 ms construction (small network), 45 ms versus 116 ms (1024-wide, PCIe-bound at 3.2 GB/s). 34/34 CTest, GCC 15/15, memcheck: 0 leaked bytes, golden (both modes) byte-identical ([evidence](evidence/backlog/R8-R9.md)) |
| 2026-10-06 | Owner decisions on C1: default build remains parity (`KAN_CUDA_FMA=OFF`), FP32 pole threshold `max(epsilon, n·2⁻²³)` accepted ([evidence](evidence/backlog/C1.md#decisions-for-the-owner)). Pass 5: C9 (main tree) and M4 (worktree) in parallel |
| 2026-10-06 | M4 closed: `kan/initializers.hpp` — `kan::initialize(Layer\|Network, Initializer)`, `Initializer = std::variant<VarianceScaling, NoiseInit>`, `DenominatorInit`, `Distribution` (Uniform/Normal), `reference_moments`, `layer_seed`; constructors remain zero-initializing (opt-in). VarianceScaling: σ_k² = gain²·Var_ref/(I·K·m_k) under each family's reference measure (closed forms for Chebyshev/Legendre/Jacobi/Hermite/Fourier, deterministic Gauss–Legendre quadrature for B-spline/RBF/Mexican hat, rational moments under each edge's own denominator); NoiseInit follows pykan `U(-a/2,a/2)`, `a = scale/(G·√in)` (without the base SiLU branch — M3); nonzero rational denominators with \|S(z)\| ≤ bound for \|z\| ≤ radius — pole-free for Guarded/Absolute/Smooth. SplitMix64 + portable polar transform: parameters bitwise identical on MSVC and GCC (9 pinned digests). Demonstration: 5-layer network on x₁x₂ — zero and noise initializations remain at a saddle (MSE 0.1205), VarianceScaling reaches 2e-22 (Chebyshev), 7e-15 (B-spline), 0.9–2.1e-3 (rational, all policies); resident via `ResidentNetwork`/`upload_parameters` without executor changes. 38/38 CTest, GCC 17/17, coverage 98.5%, golden (both modes) byte-identical ([evidence](evidence/backlog/M4.md)) |
| 2026-10-06 | C9 closed (expanded scope: training without per-step host exchange, [evidence](evidence/backlog/C9.md)): `ResidentNetwork::train_step(rate, l2, loss)` — forward + loss gradient + backward + SGD in one CUDA Graph (with `Loss::OutputGradient`, bitwise equal to the eager sequence), `Loss::MeanSquaredError` and `upload_target`/`download_loss` — device MSE, `train_step(input, target, batch, ...)` — new host batch through double pinned buffers and a separate copy stream, overlapped with computation; status checked every `status_interval()` steps (default 1) and by any synchronous call: phase-specific sticky status words and a device commit/rollback kernel attribute errors to the first failed step (`trained_steps()`), preserving parameters from the last successful step; graph rebuilt when batch, active parameter region, input/target buffer, loss or L2 changes, learning rate updated in the graph node without rebuilding; the step skips the network input gradient (as in PyTorch). Synchronizations per step: 3 → 1 (N=1) / 0.06 (N=64). FP32 versus PyTorch (ABBA, median n=9, quiet GPU), matched: 0.18/0.09/3.3/15.6 ms versus eager 3.2/2.0/9.0/21.9 and CUDA Graph 0.35/0.18/8.9/22.0; realistic loop (new host batch every step): 0.38/0.17/7.0/22.7 ms (N=64: 0.20/0.08/4.0/16.2) versus eager 3.4/2.3/12.2/32.4 and CUDA Graph 0.52/0.27/11.7/32.4 — faster on all topologies; TF32 and FP64 also no slower (FP64 N=1 in the realistic loop is within noise). torch.compile/inductor unavailable (no Triton), cudagraphs backend measured. 40/40 CTest in both builds (tree with M4 merged), GCC 17/17, clean sanitizers, golden (both modes) byte-identical, no frozen m2/m3/m4 regression. Open owner question: default `status_interval` |
| 2026-10-06 | C3 closed at stage 1 (owner decision, [evidence](evidence/backlog/C3.md)): FP32/TF32 `BasisEdges` layers no longer store Φ' between forward and backward — `basis_kernel` writes only Φ, backward finish recomputes Φ' from the layer input using the same formula in a staged tile (up to 85 terms; FP64, trainable RBF and long rows retain the previous path); one less `capacity·inputs·terms` region per layer (58.7 MB per 256-wide layer at capacity 8192). Results bitwise unchanged in both builds (golden, FP32 dump of all families), 41/41 CTest in both builds, clean sanitizers, no FP64 frozen regression. FP32 step (ABBA, median n=9): 256-wide 3.12 → 2.91 ms (−6.8…−7.9%), 1024-wide 15.02 → 14.73 ms (−1.4…−1.9%); TF32 256-wide −8…−10%. Stage 2 (fusion of expansion with GEMM: basis in the forward A tile, dx epilogue in W-GEMM) prototyped but not adopted: hand-written SIMT SGEMM core 10–30% slower than cuBLAS/CUTLASS on these shapes, exceeding memory savings (fused forward 0.63 ms versus 0.52, fused dx 1.27–1.34× slower); the 20–25% estimate assumed a cuBLAS-class core. Further fusion was recorded as a possible follow-up with a CUTLASS dependency (owner decision), not an open item |
| 2026-10-07 | Owner-approved CUDA library policy added to `AGENTS.md` rule 8 and this backlog: library primitives and their extension preferred when suitable; custom kernels chosen for missing operations/contracts or demonstrated advantage. Complete English translation and quick-context refresh distinguish current stage B/architecture from historical findings and measurements. No item status, runtime dependency or stage order changed; C3 remains closed at stage 1 ([decision and verification](evidence/cuda-library-policy.md)) |
| 2026-10-07 | Pass 6: C6, C7, C8 (all in the resident rational path, dependency-free P1; no P0 items remained) |
| 2026-10-07 | C6, C7 and C8 closed ([evidence](evidence/backlog/C6-C8.md)). C6: sample is the fastest index in the rational forward and input-VJP kernels (coalesced edge-major cache writes; FP32 store sectors/request 32 → 4). C7: forward still detects parameter-VJP overflow (resident `forward()` keeps raising where the CPU does; C9 attribution unchanged), but a sufficient bound (largest power, ≤ 1 division, two products against max/4) skips the full check, which runs only near the overflow threshold — reported status identical (host fuzz 8.33M samples). C8: one warp per edge accumulates all m+n+1 sums and the bias in one pass with each lane's sample order and shuffle tree preserved. Additional bitwise-neutral profiling fixes: integer exponent-bit finiteness tests in FP64 device code (`src/detail/host_device.hpp`, also benefits FP64 basis kernels), z cached per input/sample, one guard per Horner chain; CUB `WarpReduce`/cuBLAS evaluated and not adopted (summation order / no GEMM form). Rational kernels per step (256→256→256→10, B=2048): FP64 494 → 191 ms (now FP64-division bound), FP32 170 → 16.1 ms; full `train_step` (ABBA n=9, three policies): FP64 2.3–2.9×, FP32 3.2–13.9× faster. Results byte-identical to `7abbb7d` in both builds (golden, C3 FP32 dump, new rational dump); 42/42 CTest in both builds (re-run by the coordinator at `f39004d`), GCC 17/17, clean sanitizers, no frozen m2/m3/m4 regression; `CONTRACT.md` records the rational layout (one extra `capacity·inputs` arena region per rational layer, allocation count unchanged). Open owner questions: optional reciprocal-multiply for FP64 rational quotients (≈30–40% of those kernels, ≤ 1 ulp, breaks bitwise CPU parity); applying the C7 bound to the CPU forward |

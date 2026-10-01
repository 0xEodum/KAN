# M2 execution contract and acceptance plan

Date: 2026-10-01. Scope source: `docs/ROADMAP.md`, M2 NEXT. M1 remains
DONE; an initial live replay of `ctest --test-dir build-cuda --output-on-failure`
passed all seven targets on the RTX 3090. The starting working tree was clean,
branch `cpp-foundation`, local head `dfc5000` (the supplied agent instructions).
No BACKLOG.md exists; the roadmap is the stage authority.

## Resolved scope and defaults

- Add an optional CUDA `ResidentNetwork`, supporting all six M1 basis families
  and arbitrary compatible mixed-family networks, preserving double precision,
  finite-input/result checks, explicit domain conventions, and fixed-order
  reductions. Existing CPU and synchronous M1 CUDA APIs remain available.
- Each executor owns a nonblocking stream, parameters, input/upstream tensors,
  intermediate activations, basis values/derivatives, parameter/input gradients,
  and candidate SGD parameters. Capacity is fixed at construction; larger batches
  fail explicitly. Numerical calls reuse all GPU storage. Explicit uploads and
  downloads are synchronous, making host-buffer lifetime unambiguous.
- Forward retains its activations for backward. Backward requires a current
  successful forward and matching uploaded upstream gradients. Parameter updates
  invalidate forward/gradients; input uploads invalidate forward/gradients;
  replacing upstream gradients invalidates gradients. Invalid SGD leaves every
  layer's parameters unchanged. A new forward/backward permits repeated training
  using the same resident input/upstream tensors.
- Operations validate errors through a small device status transfer; this is
  included in full-call timing. No claim of fully asynchronous host submission
  or CUDA graph capture is made. Different executors are independent; callers
  serialize access to the same executor. Moved-from computation fails explicitly.
- Python bindings are optional, use pybind11/NumPy, and expose CPU mathematics
  and optional resident CUDA execution. Inputs are contiguous float64 arrays with
  explicit shape checks; no implicit dtype conversion. CMake build/install and
  import instructions are in scope. Distribution wheels/model serialization are M6.
- Profiling and frozen matched full-call CPU/M1 GPU/M2 GPU benchmarks precede
  optimization claims. Report setup, resident execution, and transfer-inclusive
  execution separately; GPU synchronization and validation belong to timing.

## Acceptance gates

1. Runnable committed RED tests, followed by verified GREEN commits, for GPU
   residency, all bases, mixed networks, SGD, invalid lifecycle/shapes/nonfinite
   data, zero batch, overflow/atomicity, moves, and independent concurrent use.
2. CPU forward/VJP/SGD parity plus independent derivative checks and deterministic
   training from Python/C++; demonstrate unchanged allocation counts on repeated
   calls up to capacity and explicit upload/download semantics.
3. Full existing regression suite; real RTX 3090 GPU tests; Compute Sanitizer
   memcheck; CPU-only build/import and installed C++ GPU consumer.
4. Optional Python build, actual imports and NumPy interface tests; CPU build
   retains no Python or CUDA runtime dependency.
5. Frozen benchmark inputs, reproducible commands/toolchain, verified outputs,
   full-call timings and profiler evidence, with hardware-scoped conclusions.
6. Independent implementation/evidence review and resolution of blocking findings;
   final evidence and roadmap update only after all gates pass. Incremental
   conventional commits are pushed to the tracked branch without rewriting history.

Local GPU/compiler configuration inherits M1's explicit
`--allow-unsupported-compiler` opt-in; vendor support is not inferred from passes.

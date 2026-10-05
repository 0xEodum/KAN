# Scope decision: R8 (device-query header) and R9 (resident parameter upload)

Date: 2026-10-05. Branch `cpp-foundation`. Stage B (backlog), pass 4.

## Change

The repository owner reviewed the two open questions from [R7](R7.md#decisions) and added
two backlog items. Both must close before M5, like every other backlog item.

- **R8 (P1): shared device-query header.** `kan::cuda::available()` stays non-deprecated.
  Its declaration moves to a small shared header `kan/cuda_runtime.hpp`, which both
  `kan/cuda.hpp` (legacy adapter) and `kan/resident.hpp` include. Its implementation moves
  out of the legacy adapter `src/cuda.cu` into its own source file. The name
  `kan::cuda::available()` is kept, so no user has to migrate.
- **R9 (P1): `ResidentNetwork::upload_parameters(...)`.** This adds an explicit parameter
  upload to the resident API, for loading weights after CPU training or restoring a model.
  It adds a method and leaves every existing signature unchanged.

## Rationale (owner)

- `available()` is not part of the obsolete per-call execution method. The modern API also
  needs a device query. R7's deprecation warnings inside `src/resident.cu` showed that the
  deprecation boundary was in the wrong place. They were not the reason for keeping the
  function.
- The small-call slowdown of the legacy API (R7: 0.46 → 1.07 ms, measured on a busy GPU)
  is real. The fix is still not an automatic per-thread executor cache. A cache would hide
  ownership and release timing, and transfers and synchronization would remain anyway.
  The intended path is to construct a `ResidentNetwork` once and reuse it. A parameter
  upload makes that path complete.

## Not in scope

- An automatic per-thread executor cache for `kan::cuda::forward/backward`. It was rejected.
  The legacy API keeps building an executor on every call.

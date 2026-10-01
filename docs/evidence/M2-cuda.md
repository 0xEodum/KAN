# M2 resident CUDA execution evidence

Scope: move-only `kan::cuda::ResidentNetwork`, all six M1 basis families,
arbitrary compatible mixed networks, per-instance stream, construction-time
workspace reservation, device SGD with whole-network candidate validation.
Existing M1 CUDA layer functions remain available separately.

## Executable RED

2026-10-01: `scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler
-BuildDirectory build-m2-cuda` built the stub API and `resident_test` with
CUDA 13.1 / MSVC 19.50. `build-m2-cuda/resident_test.exe` failed all five
tests with `M2 resident CUDA not implemented` on real GPU hardware.
The tests cover all-family CPU parity, repeated device training, fixed
allocation count, mixed topology, lifecycle validation, move ownership,
empty batches, concurrent instances, overflow, atomic failed network SGD,
Jacobi endpoints and Gaussian underflow-tail derivatives.

## Acceptance semantics

Construction reserves a maximum batch capacity. Execution may synchronize
only a scalar numerical error flag. Downloads explicitly request full host
tensors. Uploading input invalidates forward/upstream/backward state; uploading
upstream invalidates backward state but preserves valid forward activations.
SGD invalidates forward/backward state while retaining uploaded input/upstream,
allowing subsequent forward/backward/SGD iterations without uploads.
Failed validation or SGD candidates preserve the prior usable state.

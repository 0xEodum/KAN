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

## Baseline GREEN

2026-10-01: same CUDA build command succeeded. On the real GPU,
`resident_test` passed 5/5, independently authored `resident_review_test`
passed 4/4 (all-family finite differences, failed-upload preservation,
100-digit Decimal Gaussian derivative oracle, move assignment), and the
unchanged M1 `cuda_test` passed 9/9.

The baseline implementation uses exactly two construction-time device
allocations: one contiguous double arena and one scalar status buffer. Each
instance owns one nonblocking CUDA stream. Basis values/derivatives are
computed once per activation, then reused by all contractions. Parameter
gradient reductions have deterministic serial batch summation. All operation
results complete before return; only scalar error status crosses the device
boundary during forward/backward/SGD. SGD computes candidates in a separate
resident region, validates the entire network, then copies device-to-device.

Performance acceptance is tracked in the separate matched benchmark evidence;
this numerical baseline makes no speedup claim by itself.

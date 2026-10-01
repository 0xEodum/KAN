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

## Profile-guided SGD commit improvement

The resident baseline profile identified input-gradient contractions (36.6%)
and parameter-gradient contractions (33.6%) as the largest GPU kernel costs,
with cached basis evaluation now 5.5%. This resolves the M1 repeated-basis
coefficient bottleneck (71.9% in its profile) without changing reduction order.
The remaining API trace includes synchronous error checks required by the
numerical contract; hardware-counter access was denied, so no claim of maximum
GPU utilization is made.

After the frozen baseline, SGD's device-to-device parameter commit copy and
additional synchronization were replaced by swapping the active/candidate arena
offsets. Candidate execution and whole-network validation already finish before
the swap, preserving atomicity and synchronous semantics with no allocation.
Matched balanced ABBA measurements found modest steady full-call improvements
of 6.34% for case 0 and 3.35% for case 1. The larger case's 1.41% change was
comparable to noise; transfer-inclusive timings do not establish a general gain.
See [frozen benchmark evidence](M2-benchmark.md) for raw samples, protocol,
profiling, and the separate baseline CPU/M1/resident comparisons.

After this change, `resident_test` passed 5/5, independent `resident_review_test`
4/4 and unchanged M1 `cuda_test` 9/9. The atomic-update test was strengthened:
an earlier layer now has a real nonzero candidate update, which must remain
hidden when a later layer overflows; a subsequent valid retry updates it.

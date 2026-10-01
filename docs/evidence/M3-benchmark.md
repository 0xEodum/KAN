# M3 frozen matched benchmark protocol

Frozen before CUDA implementation/tuning on 2026-10-01. `m3_benchmark.cpp`
fixes three families (cubic clamped B-spline, normalized MexicanHat, trainable
Gaussian RBF), seven terms, topologies 16 -> 24 -> 8 and 64 -> 64 -> 32 -> 16,
batch 32/1024, double arithmetic, deterministic nonzero parameters, sine inputs
and upstream, and learning rate 0.001. Explicit knots, scales and log widths are
in source. Input/upstream generation occurs outside measurement.

Each backend starts identically, performs two warmups then seven measured
forward/backward/SGD steps on the evolving parameters. CPU backward recomputes
activations; resident backward reuses the preceding forward state. Resident
full calls include scalar numerical validation/synchronization and exclude full
tensor traffic. Transfer full calls additionally upload input/upstream and
download output/all gradients before SGD each step. Setup and final parameter
snapshots are measured separately/outside steady timing. Timings use host
steady_clock and synchronized complete public calls, never kernel-only ratios.

All outputs, input/coefficient/bias/center/log-width VJPs and learned parameter
snapshots are compared elementwise to CPU with 2e-10 absolute-plus-relative
tolerance. Resident verification uses an untimed identical trajectory replay
whose final parameters must equal the timed trajectory. Device allocation count
must stay constant. Each CSV retains all samples, checksums, median and IQR.

Baseline/profiling/tuning results are appended only after real execution.

## Hardware, measured bottleneck and accepted implementation

Release uses MSVC 19.50.35724.0, CUDA 13.1.115, compute architecture 86,
`--fmad=false`, explicit `--allow-unsupported-compiler` (this newer MSVC exceeds
nvcc's default supported range). Hardware verified live: Intel i5-12400,
6 cores/12 logical processors, NVIDIA RTX 3090 24,576 MiB, driver 591.86.
Windows WDDM has active display processes; clocks are unlocked and access is
not exclusive. Parent/reviewer builds and tests stopped during all accepted
unprofiled measurement runs. Timing observations apply to this environment.

Nsight Systems 2025.5.2 profiled frozen case 11, which uses trainable RBFs in
all three layers of topology 64 -> 64 -> 32 -> 16, batch 1024. Each trace includes
three executors: timed resident, untimed identical trajectory replay, and timed
transfer-inclusive trajectory, each nine steps. The initial implementation used
seven active threads, one per term, to scan every batch/edge contribution.
Its shared nonlinear VJP accounts for **99.8%** of GPU kernel time. Timeline
evidence records 34.053 seconds across 81 calls and directly identifies that
serialized reduction as the tuning target.
Long cudaMemcpyAsync API durations also include waiting for kernel completion
during status/result downloads; they do not establish a bandwidth bottleneck.

The replacement assigns up to 64 tiles per term, 256 threads per tile, using
persistent partial-sum storage and a second tile reduction. Case 11 exposes
448 partial blocks per layer. Its final trace records 47.491 ms across 81 partial
calls plus 0.353 ms across 81 finish calls; together **38.0%** of kernel time.
The remaining time is input VJP 18.4%, coefficient/bias VJP 17.0%, basis 14.8%,
forward 11.7% and candidate/width validation 0.2%. The profile confirms removal
of the overwhelming serial bottleneck; its durations are diagnostic, excluded
from accepted speedup calculations. Reduction order changes are checked against
CPU VJPs and the entire evolving SGD trajectory. Independent review's capped
launch-tail bug was reproduced and fixed before final acceptance; see
[review RED](M3-reduction-red.md) and [CUDA evidence](M3-cuda.md).

Both profiles have six allocations/frees and three stream constructions for
their three executors. Final launch count is 594 versus 513: the added finish
kernel contributes one launch per layer/step, without execution allocations.
Both have 337 cudaMemcpyAsync calls. The final memory summary contains no D2D
copy; SGD commits by swapping already reserved arena offsets.

Nsight Compute 2025.4.1 was retried with `--set basic --launch-count 1` and
returned **ERR_NVGPUCTRPERM**, retained in [ncu-preopt.txt](m3/ncu-preopt.txt).
Hardware occupancy/bandwidth counters remain unavailable. Neither profile proves
full device utilization, and no saturation or universal performance claim is made.

## Matched complete-call acceptance

Source baseline `e07bf09`, final reduction `f36d22d`, unchanged mathematical
fixtures, two warmups/seven samples per run. Four balanced runs in
`preopt, final, final, preopt` order start from identical parameters each time.
[balanced-final-rbf.csv](m3/balanced-final-rbf.csv) retains all eight rows;
[balanced-final-summary.json](m3/balanced-final-summary.json) pools 14 measured
samples per version/mode, using the same interpolated quantiles as the executable.

| Case 11 complete calls | Baseline median (IQR), ms | Final median (IQR), ms | Median reduction |
| --- | ---: | ---: | ---: |
| Resident | 1222.545 (14.883) | 4.501 (0.787) | 99.63% |
| Transfer inclusive | 1220.214 (14.764) | 6.620 (1.008) | 99.46% |

Every elementwise comparison passes, including learned centers/log widths and
pre-update VJPs. Both final runs beat both baseline runs in both modes by a large
margin. This is accepted evidence for the scoped shared-RBF reduction change on
this frozen workload. Resident timings exclude full tensor transfers; transfer
timings include them. CPU uses serial reference code and recomputes forward in
backward, so CPU comparisons below are public-workload observations.

[final-m3.csv](m3/final-m3.csv) contains all **36** final rows: twelve fixtures,
three backends, maximum absolute CPU/device discrepancy **8.674e-18**, and two
unchanged device allocations for each resident executor. Representative large
topology/batch-1024 full-call observations (IQR in parentheses):

| Family | CPU, ms | Resident, ms | Transfer inclusive, ms |
| --- | ---: | ---: | ---: |
| Cubic B-spline | 351.384 (10.557) | 2.518 (0.181) | 4.294 (0.228) |
| MexicanHat | 394.289 (29.251) | 21.183 (22.383) | 5.431 (0.324) |
| Trainable RBF | 423.854 (16.680) | 5.426 (32.480) | 6.816 (0.848) |

The single sweep shows substantial WDDM/order variation, including a slower
MexicanHat resident observation than its transfer-inclusive counterpart. These
rows must not support general throughput/speedup conclusions. The separately
balanced RBF comparison above is the tuning acceptance evidence. Setup is one
observation per row and is not included in steady timing.

Existing M2 workloads also replayed unchanged: [final-m2-regression.csv](m3/final-m2-regression.csv)
contains **76** rows (24 CPU, four M1 Chebyshev, 48 resident/transfer), all numerical
checks pass, maximum discrepancy **5.204e-18**, every resident allocation count
remains two. [balanced-rbf.csv](m3/balanced-rbf.csv) is the earlier diagnostic
comparison of the intermediate tiled implementation before the launch-tail fix;
it does not substitute for final-source acceptance.

## Reproduction and retained provenance

Build each source revision in Release with `KAN_ENABLE_CUDA=ON`,
`KAN_BUILD_BENCHMARKS=ON`, architecture 86 and the explicit compiler override.
Preserve the preopt executable before compiling the final source. Run each final
benchmark `--backend all --warmups 2 --repeats 7`. For the balanced comparison run
the two executables with `--backend resident --case 11 --warmups 2 --repeats 7`
in the order above. Each command performs independent CPU/trajectory verification.

Profiler command (replace executable/prefix for each source revision):

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true
  -o build-m3-cuda/m3-final-profile build-m3-cuda/m3_benchmark.exe
  --backend resident --case 11 --warmups 2 --repeats 7
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum
  --format csv --output docs/evidence/m3/final-profile --force-export=true
  --force-overwrite=true
  build-m3-cuda/m3-final-profile.nsys-rep
```

Small exported CSVs and exact sanitizer diagnostics are committed. Binary traces
and executables stay in the ignored build directory and are reproducible.
[manifest.json](m3/manifest.json) records source/tool/hardware metadata, SHA256 and
byte size for every retained artifact and local benchmark/profile binary.
The scoped `.gitattributes` preserves committed evidence bytes for stable hashes.

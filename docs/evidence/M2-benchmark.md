# M2 frozen benchmark protocol

Protocol frozen before any M2 optimization claim, 2026-10-01. The executable
`benchmarks/m2_benchmark.cpp` fixes all six M1 basis families, seven terms,
topologies 16 -> 24 -> 8 and 64 -> 64 -> 32 -> 16, batch sizes 32 and 1024,
double storage/arithmetic, deterministic sine inputs, nonzero deterministic
coefficients, cosine biases, fixed upstream VJP and learning rate 0.001.
Jacobi alpha/beta 0.25/0.5, Fourier frequency 1.25, RBF width 0.65 and seven
uniform centers in [-1,1] are fixed in source. There is no RNG or data generation
inside the measured calls. The upstream includes 1/batch scaling.

Each backend starts from identical parameters, executes two warmup training
steps then seven measured steps, and retains the same evolving parameter trajectory.
The final pre-update output and gradients, and final post-update parameters,
are independently compared elementwise with CPU (2e-10 absolute-plus-relative
tolerance). Weighted checksums and maximum absolute discrepancy accompany each row.
Every measured sample is retained; median and interpolated interquartile range
are reported in milliseconds. Timer is steady_clock on the host and all GPU work
is synchronized before stopping it. No event or kernel-only time is a speedup metric.

- `cpu_full`: CPU Network forward, Network backward (which recomputes forward),
  and Network SGD, including their validation and host allocations.
- `m1_host_full`: the same public forward/backward/SGD workload using the M1
  Chebyshev host layer CUDA calls and CPU network SGD. Includes repeated device
  allocation, uploads/downloads, stream creation and synchronization. The explicit
  wrapper recomputes activations in backward to match CPU Network behavior.
- `m2_resident_full`: uploaded input/upstream and resident parameters; public
  forward, backward, SGD and synchronization including scalar status traffic and
  validation. Caches activations per the resident API. Full tensor downloads occur
  after timing. This row must not be described as a host transfer inclusive result.
  Because SGD invalidates downloadable pre-update state, an untimed identical
  trajectory replay captures final output/gradients before its final SGD; replay
  final parameters must match the timed object's parameters exactly.
- `m2_transfer_full`: the same M2 calls plus input/upstream uploads each step and
  output/all-gradient downloads, timed. Parameter download for independent final
  verification is outside timing, as CPU/M1 already hold parameters on the host.
  These downloads occur before SGD, respecting the API's state validity contract.

Setup is measured separately once per row, including network construction and,
for M2, resident allocation, parameter/input/upstream uploads and synchronization.
Workspace allocation count is recorded and must remain constant. Setup times are
single observations, not robust statistics. Legacy comparisons are Chebyshev only;
unsupported M1 families are omitted, never silently run on CPU. CPU is the serial
reference, not a claim to compare against an optimized BLAS or other KAN library.

Reproduction: build Release CUDA with `KAN_BUILD_BENCHMARKS=ON`; run
`./scripts/benchmark-m2.ps1`. Baseline can be built with `KAN_BENCH_RESIDENT=OFF`.
Profiling invocations and matched before/after evidence will be recorded below.

## Baseline recorded before resident optimization

Frozen protocol commit: `a784993`. Release build used MSVC 19.50.35724.0,
CUDA 13.1.115, compute architecture 86, `/O2 /Ob2 /DNDEBUG`, CUDA `--fmad=false`,
and explicit `--allow-unsupported-compiler` because CUDA rejects this newer MSVC
by default. Hardware: Intel Core i5-12400 (6 cores/12 logical), NVIDIA RTX 3090
(24 GiB), driver 591.86, Windows WDDM with active display processes. No locked
clocks or exclusive GPU access; these are this machine's measured observations.

`./scripts/benchmark-m2.ps1 -Output docs/evidence/m2/baseline-host.csv`
returned exit 0: 24 CPU rows and four Chebyshev M1 rows, all M1 numerical
comparisons passed with maximum absolute discrepancy zero on this input set.
Chebyshev median full-call milliseconds:

| Topology | Batch | CPU | M1 host CUDA |
| --- | ---: | ---: | ---: |
| 16 -> 24 -> 8 | 32 | 0.797 | 3.485 |
| 16 -> 24 -> 8 | 1024 | 25.909 | 14.109 |
| 64 -> 64 -> 32 -> 16 | 32 | 7.010 | 7.561 |
| 64 -> 64 -> 32 -> 16 | 1024 | 219.001 | 34.212 |

This already demonstrates that CUDA is slower for the small case. Raw samples,
IQRs and checksums are in [baseline-host.csv](m2/baseline-host.csv).

Nsight Systems 2025.5.2 profile of case 3:

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true
  -o build-m2-bench/m1-profile build-m2-bench/m2_benchmark.exe
  --backend legacy --case 3 --warmups 2 --repeats 7
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum
  --format csv --output docs/evidence/m2/m1-profile --force-export=true
  build-m2-bench/m1-profile.nsys-rep
```

The nine-step trace includes warmups and first CUDA initialization, unlike the
unprofiled steady timing rows. It recorded 378 cudaMalloc/378 cudaFree calls,
81 stream constructions, 378 cudaMemcpyAsync calls and 135 kernel launches.
CUDA API time was 71.4% cudaMemcpyAsync and 23.5% cudaMalloc; the latter includes
85.9 ms first-use initialization and is not per-call steady cost. The coefficient
gradient kernel accounted for 71.9% of kernel time, 169.758 ms across 27 calls
(median 5.157 ms), with each thread scanning the batch and recomputing recurrence.
Forward accounted for 17.7%, input gradient 9.6%, bias reduction 0.8%.
This supports persistent buffers/streams, basis reuse and a targeted investigation
of the coefficient reduction, without claiming its actual attainable speedup.
The small exported profile CSVs are retained under `m2/m1-profile_*`; binary traces
and SQLite exports remain reproducible local build artifacts.

Nsight Compute 2025.4.1 `--set basic --launch-count 1 ... --backend legacy --case 3
--warmups 0 --repeats 1` reported `ERR_NVGPUCTRPERM`; the exact diagnostic is
[ncu-baseline.txt](m2/ncu-baseline.txt). No occupancy, bandwidth utilization or
hardware-counter bottleneck claim is supported. Profiler-instrumented wall times
are deliberately excluded from matched benchmark results.

## Resident baseline and profiling

Resident implementation baseline `2123ed8`, benchmark API-state correction
`a464876`. [resident-preopt.csv](m2/resident-preopt.csv) retains all 76 rows:
24 CPU, 4 M1 and 48 resident/transfer-inclusive observations. All numerical checks
passed, maximum discrepancy 5.2042e-18, and every resident object retained two
device workspace allocations. Concurrent CPU build/preparation affected this
first resident sweep (case 3 CPU IQR 122.94 ms). Some WDDM resident/transfer rows
also show large order-sensitive variance. This sweep is diagnostic evidence;
its ratios are not accepted optimization performance claims. A matched isolated
before/after rerun is required below.

Nsight Systems used the same profile/stats commands with output prefix
`resident-preopt-profile`, `--backend resident --case 3 --warmups 2 --repeats 7`.
The trace includes three resident objects: timed resident trajectory, untimed
verification replay, timed transfer-inclusive trajectory, each nine steps.
Exactly six cudaMalloc and six cudaFree calls and three stream constructors
support allocation at construction only; source and recorded constant workspace
count independently confirm no execution allocations. The 351 kernel launches
are 13 per step (three layers, basis/forward/input/parameter plus SGD).

GPU kernel time was input VJP 36.6% (21.476 ms/81 calls), parameter VJP 33.6%
(19.711 ms/81), forward 24.2% (14.176 ms/81), basis 5.5% (3.237 ms/81), SGD
candidate validation 0.1% (0.060 ms/27). Unlike M1, coefficient-gradient recurrence
is no longer the overwhelming kernel bottleneck. cudaMemcpyAsync API time was
71.259 ms across 268 calls, including synchronous scalar status checks and
untimed setup/verification; stream synchronization 6.407 ms, launches 3.172 ms.
The first stream construction includes 82.667 ms CUDA initialization and is
excluded by the benchmark warmups. These timeline data cannot establish hardware
utilization without unavailable Nsight Compute counters. Baseline summaries are
retained as [resident-preopt-profile CSVs](m2/resident-preopt-profile_cuda_gpu_kern_sum.csv).

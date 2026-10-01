# M4 frozen matched rational benchmark protocol

Frozen 2026-10-01 before resident implementation or tuning. The original
`benchmarks/m4_benchmark.cpp` defines degree pairs 1/1, 3/2, 6/4, center 0.1,
scale 1.2, epsilon 1e-8, topologies 16 -> 24 -> 8 and 64 -> 64 -> 32 -> 16,
batch 32/1024 and double precision. It fixes deterministic sine numerators,
cosine denominators/bias, sine input and upstream and SGD rate 0.001.

Each executor starts from identical parameters and performs two warmups then
seven measured evolving forward/backward/SGD steps. CPU backward recomputes
activations. Resident complete calls include scalar status validation and stream
synchronization. Transfer complete calls additionally upload inputs/upstream and
download output and every gradient before SGD. Setup is separate. Complete-call
timing uses host steady_clock; no kernel-only speedup is accepted.

Every output, input VJP, numerator/denominator/bias VJP and final parameter is
verified elementwise against CPU at 2e-10 absolute-plus-relative tolerance.
Steady resident verification uses an untimed identical trajectory replay whose
parameters must exactly match the measured executor. Allocation counts must
remain two. CSV retains every sample, interpolated median/IQR and checksums.
Fixture changes require explicit evidence; tuning uses balanced baseline/final/
final/baseline comparisons of the same frozen case. Profiling durations are
diagnostic only. Hardware/profiler constraints and raw artifacts follow below.

## Measured bottleneck and device implementation

Live hardware: Intel i5-12400 (6 cores,12 logical processors), NVIDIA RTX 3090
24576 MiB, driver 591.86. Release MSVC 19.50.35724.0, CUDA 13.1.115, architecture 86,
`--fmad=false` and explicit unsupported-host-compiler override. Windows WDDM
display processes and unlocked clocks remain active. Measurements describe
this environment; access is not exclusive at the operating-system level.

Nsight Systems 2025.5.2 profiles frozen case 11 (degree 6/4,64x64x32x16,batch 1024).
Each trace covers three executors: timed resident, untimed identical replay,
and timed transfer calls, each nine steps. The original nonlinear parameter VJP
assigns one thread per parameter to scan all samples. Its edge-major P/Q caches
expose contiguous samples, but neighboring threads scan different parameters.
The profile measures that parameter reduction at 63.2% of kernel time (470.591ms
across 81 calls), versus36.4% forward (270.846ms).

The tuned kernel assigns one full warp per parameter, each lane scans every
32nd sample, and uses a shuffle reduction. Its grid-stride parameter loop
preserves tails beyond the capped 65535 block launch. It adds no kernels, device
allocations or scratch buffers. Matched numerical fixtures and state behavior
are unchanged; all nonlinear VJPs and evolving final parameters are verified.
Final profile parameter time is 240.786 ms across 81 calls (47.0%), with forward
268.497ms (52.4%), input VJP 2.634 ms (0.5%) and candidate 0.093 ms. This confirms
removal of roughly half the measured parameter-reduction cost; these profiled
durations are diagnostic, excluded from accepted speedup calculations.

Both traces have 270 kernel launches,6 malloc/frees and3 streams for three
executors, and 289 cudaMemcpyAsync calls. Their device-memory summaries contain
no D2D copy because SGD swaps permanent parameter regions. Long status-copy API
durations include waiting for preceding kernels; they do not prove bandwidth
saturation. Nsight Compute 2025.4.1 `--set basic --launch-count 1` again returned
ERR_NVGPUCTRPERM, retained in `m4/ncu-preopt.txt`. Hardware occupancy/bandwidth
counters are unavailable. Neither trace proves full device saturation.

CPU, parent and reviewer builds/tests were paused during the accepted balanced
measurement window. All numerical data generation and setup are outside timing.
The benchmark checks final pre-update output/VJPs and final learned parameters
after the evolving trajectory, with an exact untimed GPU replay. The resident
tests independently check each step of representative mixed trajectories; the
benchmark does not download every intermediate step in steady mode.

## Matched complete-call acceptance

Baseline rational source `c5a54e3`, tuned warp reduction `0071628`, unchanged CPU
source `4b808d2` and frozen fixture source `73e3f47`. Four runs in preopt/final/
final/preopt order each use two warmups/seven measured evolving steps. Every
final output/VJP/parameter comparison passes, baseline maximum error 0 and tuned
maximum absolute error 3.469e-18. Allocation count remains two. Both tuned runs
beat both baseline runs in each mode. `m4/balanced-final.csv` retains all eight
rows; `m4/balanced-final-summary.json` retains all 14 samples per source/mode.

| Case 11 complete call | Baseline median (IQR), ms | Tuned median (IQR), ms | Median reduction |
| --- | ---: | ---: | ---: |
| Resident | 29.951(3.298) | 20.780(0.351) | 30.62% |
| Transfer inclusive | 28.804(0.608) | 20.885(0.871) | 27.49% |

The balanced comparison accepts the scoped parameter-reduction optimization
on this hardware and workload. Resident timing excludes full tensor traffic;
transfer timing includes it. These results do not establish device saturation,
universal speedups or performance of every degree/topology. Initial single-run
CSV files are diagnostic and do not substitute for the balanced comparison.

Final integration CPU `c80b7d3` removes per-edge heap storage through a private
fixed-array evaluator while preserving public owning results. Its independently
matched CPU performance/allocation evidence is recorded in M4-cpu-performance.md.
`m4/final-m4.csv` retains all 36 final sweep rows (12 fixtures,3 backends), every
final output/VJP/parameter check passes, maximum absolute discrepancy 5.204e-18,
and every resident executor keeps two device allocations. This sweep overlapped
CPU-lane timing after an early release assumption; its times are diagnostic
only, excluded from performance acceptance. The CPU lane discarded/replayed its
overlapping comparison. The earlier balanced GPU acceptance above was isolated
and remains valid. Final sweep results establish numerical/allocation regression.

Unchanged M2/M3 fixtures also replayed with the tuned resident implementation:
`m4/final-m2-regression.csv` has 76 rows, maximum discrepancy 5.204e-18;
`m4/final-m3-regression.csv` has 36 rows, maximum discrepancy 8.674e-18. Every
resident/transfer executor retains two allocations. Their timings are diagnostic
under concurrent CPU activity, and support no additional performance claim.

## Reproduction and retained provenance

Build both revisions through `scripts/build.ps1 -Cuda -Benchmarks
-AllowUnsupportedCudaCompiler -BuildDirectory build-m4-cuda`. Preserve the
preopt executable before rebuilding the tuned source. Frozen sweep command:
`m4_benchmark.exe --backend all --warmups 2 --repeats 7`. Balanced commands use
`--backend resident --case 11 --warmups 2 --repeats 7`, in preopt/final/final/
preopt order, each starting independently. `m4/summarize_matched.py` pools 14
samples per source/mode using the executable's interpolated quantiles.

Profiler commands:

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true
  -o build-m4-cuda/m4-final-profile build-m4-cuda/m4_benchmark.exe
  --backend resident --case 11 --warmups 2 --repeats 7
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum
  --format csv --output docs/evidence/m4/final-profile --force-export=true
  --force-overwrite=true build-m4-cuda/m4-final-profile.nsys-rep
```

Small raw CSV/logs are retained; binary traces/executables remain in ignored
build storage. The evidence directory's `.gitattributes` preserves exact bytes.
`m4/manifest.json` records hashes/sizes, source revisions, fixture source hash,
toolchain/hardware and local binaries/traces. Sanitizer output is retained.
The exact baseline executable remains preserved. The final integration rebuild
relinked the tuned executable with the later CPU refinement, so its current hash
identifies that integration build, not the earlier balanced executable. The
tuned GPU library remains unchanged; the manifest labels both roles and records
the exact tuned/baseline source blobs used for the isolated comparison.

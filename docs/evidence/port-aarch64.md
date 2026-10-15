# Platform port: Linux aarch64 / NVIDIA GB10 (step 0, not a backlog item)

Date: 2026-10-12. Branch `cpp-foundation`, base `37a07f8` (M3 closed). Purpose: build and
test the project on the new development machine in the CPU-only and both CUDA
configurations, fix what the port exposes, and record local reference data (golden dumps,
benchmarks) that later backlog items compare against. Nothing was retuned; no backlog item
is opened or closed by this step.

All earlier evidence was produced on Windows x86-64 (MSVC + CUDA, i5-12400 + RTX 3090,
sm_86); GCC was used there for CPU-only builds only. Linux commands below are the reference
for every later sub-agent on this machine; `scripts/*.ps1` and the `.cmd` harness scripts in
`docs/evidence` stay the Windows equivalents.

## Machine and toolchain

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, Linux 6.17.0-1022-nvidia aarch64, glibc 2.39 |
| CPU / memory | 20 cores, 121 GB unified LPDDR5x (shared with the GPU) |
| GPU | NVIDIA GB10, compute capability 12.1, 48 SMs, 2.4 GHz (max 3.0 GHz reported), 24 MiB L2, 100 KiB shared memory per SM (99 KiB opt-in per block), integrated (`cudaDeviceProp::integrated = 1`), driver 580.159.03 |
| CUDA | 13.0 (nvcc V13.0.88) at `/usr/local/cuda`, cuBLAS 13.1.0.3, compute-sanitizer 2025.3.1, Nsight Systems 2025.3.2, ncu available |
| Host compiler | GCC 13.3.0 (also the nvcc host compiler) |
| Build | CMake 3.28.3, Unix Makefiles (no Ninja installed) |
| Python | 3.12.3 venv `.venv`, numpy 2.5.2, pybind11 3.0.2; PyTorch not installed (no torch comparison in this step) |

Measured device throughput ([device-peak.txt](port-aarch64/device-peak.txt) from [device_peak.cu](port-aarch64/device_peak.cu), `nvcc -arch=native -O2 device_peak.cu -lcublas`, 3 runs): DGEMM 4096 0.401 TFLOP/s,
SGEMM 8192 20.8–21.0 TFLOP/s, TF32 8192 30–40 TFLOP/s, device-to-device copy 243–246 GB/s
(read + write). For comparison the RTX 3090 has about 0.56 TFLOP/s FP64 peak (C2 reached
≈94% of it), 35.6 TFLOP/s FP32 peak and 936 GB/s. GB10's FP64:FP32 ratio in practice is
about 1:52, similar to GA102's 1:64; its absolute FP64 rate is about 0.7x, and its memory
bandwidth about 0.26x of the 3090's (measured GB10 copy rate against the 3090's 936 GB/s
specification).

## Commands (reuse these)

```bash
# Configure + build (+ test). scripts/build.sh is the Linux equivalent of scripts/build.ps1.
scripts/build.sh --python --test                                      # CPU-only, build-cpu
scripts/build.sh --cuda --python --benchmarks --test                  # parity build, build-cuda
scripts/build.sh --cuda-fma --python --benchmarks --test              # performance build, build-cuda-fma
scripts/build.sh --asan --config RelWithDebInfo --build-dir build-asan  # ASan, CPU-only, no Python
# Equivalent raw CMake (as run for this record):
cmake -S . -B build-cpu  -DCMAKE_BUILD_TYPE=Release -DKAN_BUILD_PYTHON=ON -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
      -DKAN_BUILD_PYTHON=ON -DKAN_BUILD_BENCHMARKS=ON -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake -S . -B build-cuda-fma -DCMAKE_BUILD_TYPE=Release -DKAN_ENABLE_CUDA=ON -DKAN_CUDA_FMA=ON \
      -DKAN_BUILD_PYTHON=ON -DKAN_BUILD_BENCHMARKS=ON -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build <dir> -j20 && ctest --test-dir <dir> --output-on-failure

# compute-sanitizer: memcheck (leak check) on every resident suite, racecheck/synccheck/initcheck
# on residual_resident, resident_training, resident_fused_expansion.
scripts/sanitizers.sh build-cuda build-cuda-fma > sanitizers.txt

# Link an evidence harness against a CUDA build (Linux cl_tool.cmd / build_golden.cmd equivalent;
# always -O2 -ffp-contract=off, as the in-tree targets).
scripts/link_harness.sh build-cuda docs/evidence/backlog/golden/golden.cpp /tmp/golden
# Golden, C3 FP32 and C6-C8 rational dumps + SHA-256 (Linux dumps.sh equivalent).
scripts/golden_dumps.sh <out-dir> parity=build-cuda fma=build-cuda-fma
# One-pass reference run of the frozen suites and the full-step harnesses.
scripts/bench_baseline.sh build-cuda <out-dir> 3
python docs/evidence/port-aarch64/summarize.py <out-dir>
```

`CMAKE_CUDA_ARCHITECTURES` now defaults to `native` (the visible GPU, which must be present at
configure time) when neither `-DCMAKE_CUDA_ARCHITECTURES` nor the `CUDAARCHS` environment
variable is given; `scripts/build.ps1` keeps passing `86` explicitly.

## Port defects and fixes (RED first)

### 1. Host FMA contraction outside the `kan` target (`residual` test)

RED ([red-cpu.log](port-aarch64/red-cpu.log)): CPU build 28/29,
`FAIL sgd_updates_the_branch_and_validates_its_gradient: l.residual()->weights[j] ==
before.residual()->weights[j] - 0.1 * g.residual[j]`.

Root cause: only `kan` had `-ffp-contract=off` (PRIVATE). CMake's default `CXX_EXTENSIONS`
selects `-std=gnu++20`, under which GCC defaults to `-ffp-contract=fast`; aarch64 has FMA in the
base ISA, so GCC contracted the test's `before - 0.1*g` into one `fmsub` while the library's SGD
rounds the product first. On x86-64 without `-march` GCC has no FMA instruction to contract into,
and MSVC does not contract under its default `/fp:precise`, so the Windows/x86 runs never saw it.
The test's exact comparison is the contract ("CPU arithmetic is unfused"), so it stays exact.

Fix (`CMakeLists.txt`): directory-scope `add_compile_options` for non-MSVC compilers —
`-ffp-contract=off` for C++ and `-Xcompiler=-ffp-contract=off` for the nvcc host pass — placed
before every target, so the library, tests, examples, benchmarks, the `_kan` module and the
`kan_cuda` host code (`src/cuda_runtime.cpp` and the host side of the `.cu` files, previously
without the flag) all compile unfused. Directory options are not exported: the installed
`KANTargets.cmake` contains no `-ffp-contract` and the installed-consumer test project
(`package_*` CTest) compiles without it. The duplicate flag on `kan` itself was removed. MSVC
is unchanged. All 55 compile commands of each CUDA build and all 30 of the CPU build carry the
flag (checked in `compile_commands.json`); device code is still governed by `--fmad`.

Other exact floating-point comparisons in the tests that rely on unfused host arithmetic
(`residual_test.cpp:316,356`, the `before - rate*g` expectations in
`residual_resident_test.cpp:327` and the FP64 MSE gradients in `resident_training_test.cpp:133`,
`residual_resident_test.cpp:268`) are covered by the same fix; none were loosened.

### 2. GCC name lookup in a nested generic lambda (`src/resident.cu`)

RED ([red-cuda-build.log](port-aarch64/red-cuda-build.log)): `src/resident.cu:2146:53: error:
'capacity' was not declared in this scope` for `Engine<double>` and `Engine<float>`.

Root cause: `capacity` is a member of the dependent base `Context<T>`, made visible by
`using Context<T>::capacity;`. Inside a generic lambda nested in `std::visit` with
`if constexpr`, GCC 13 does not find the using-declared member during instantiation (MSVC does;
the file had never been compiled with GCC as nvcc host compiler). Fix: `this->capacity`, as the
same statement already does for `this->has_residual_workspace`. It was the only error; the CUDA
13 toolchain needed no API changes (CUDA 13 removed nothing the code uses).

### 3. TF32 bias-VJP tolerance under cancellation (`resident_precision`, both CUDA builds)

RED: the first parity CTest after fix 2 (fix 1 not yet applied,
[red-ctest-cuda.log](port-aarch64/red-ctest-cuda.log)) is 45/47 (`residual` and
`resident_precision`); after fixes 1–2 both builds are 46/47
([parity](port-aarch64/red-ctest-cuda-parity-after-fix12.log),
[FMA](port-aarch64/red-ctest-cuda-fma-after-fix12.log)) with
`FAIL tensor_float32_contractions_match_fp64_cpu: index 21 of 80 actual=-0.0161088146
expected=-0.0174858991 scale=0.0964217672` (FMA: actual -0.0161158815). The same message
repeated in three direct runs (not logged).

Diagnosis:
Artifacts in [port-aarch64/tf32/](port-aarch64/tf32/) (probe sources and outputs).
- `NVIDIA_TF32_OVERRIDE=0` (TF32 disabled in cuBLAS): the diagnostic's TF32 error drops to the
  FP32 level (`tf32_diag-override0.txt`), and the suite's only failure is its own
  `REQUIRE(a != b)` (TF32 must differ from FP32; `resident_precision-override0.txt`). TF32 is
  the only cause.
- cuBLAS 13.1 rounds TF32 operands to nearest here (`tf32probe.cu`: `1 + 3*2^-12` becomes
  `1 + 2^-10`, not `1`), and the diagnostic network's step uses four
  `cutlass_80_tensorop_s1688gemm_*` GEMMs (`nsys-gemm-kernels.txt`). The RTX 3090 runs used
  cuBLAS 13.1 as well (C3 evidence), but their kernel selection and TF32 operand handling for
  these shapes were not recorded, so why the same check passed there is unknown.
- The failing tensor is layer 0's bias gradient (80 outputs): `db[o] = sum_b U0[b,o]`, a 1000-
  sample cancelling sum (result 0.0175 against |U0| summing to about 17) of layer 1's input VJP,
  whose `W = U*C` contraction carries TF32 operand rounding. Measured (`tf32_diag.txt`)
  `max |err| / sum_b |U0[b,o]| = 8.2e-5` and 8.3e-5 in the two steps (FP32: 8.8e-8), i.e. 0.17 x 2^-11, consistent with TF32
  operand rounding, not a defect; against the old bound it was 1.21x (first step) and 0.66x
  (second). The layer 0 coefficient gradient is at 0.88x of the unchanged bound (passes).

Fix (`tests/resident_precision_test.cpp`): following the M3 weight-VJP precedent
(`2e-3*sum_b |U||S|`, CONTRACT "Order"), the TF32 bias-gradient check of every layer except the
last adds `2^-11*sum_b |U[b,o]|` (TF32 unit roundoff times the summands), where U is the next
stage's input VJP of the FP64 reference. The last layer gets nothing: its U is the uploaded
output gradient, which no TF32 contraction touches. `sum_b |U|` is an empirical scale (the
rounding of U really scales with `sum_k |dy||C|` of the next layer), with 5.9x headroom over
the measurement (max 8.3e-5); in absolute terms layer 0's allowance is about 8e-3 against a failing entry of
0.0175 (the old bound there was 1.1e-3), so the check still catches errors of the order of the
value itself, not smaller ones. The reviewer verified 2.5e-4 (3x headroom) as also passing in
both builds; the constant is an owner decision. `tf32_relative`, `tf32_floor`, all FP32 and FP64 tolerances and every other
check are unchanged; the allowance is zero outside TF32 calls. (A first version used 2e-3 and
also covered the last layer, where it made the check vacuous; the independent review caught
it, see below.) `docs/CONTRACT.md` (precision section) records the
term. **Decision for the coordinator/owner:** this is a test-tolerance contract change for the
opt-in TF32 mode, required by GB10/cuBLAS 13.1 kernel selection.

**Owner decision (2026-10-13):** the owner accepted `2^-11` (over the tighter 2.5e-4 alternative);
the allowance stays as committed in `6dc1fdc`.

## Test results (final tree)

| Configuration | Result | Log |
|---|---|---|
| CPU-only Release + Python (`build-cpu`) | **29/29** | [ctest-cpu.log](port-aarch64/ctest-cpu.log) |
| CUDA parity (`build-cuda`, `KAN_CUDA_FMA=OFF`, sm_121 native, Python, benchmarks) | **47/47** | [ctest-cuda-parity.log](port-aarch64/ctest-cuda-parity.log) |
| CUDA performance (`build-cuda-fma`, `KAN_CUDA_FMA=ON`) | **47/47** | [ctest-cuda-fma.log](port-aarch64/ctest-cuda-fma.log) |

The final CUDA logs were taken with the GPU otherwise idle. A first rerun after the review
fixes, made while a compute-sanitizer racecheck and the ASan test were running, failed
`cuda_legacy_calls_release_device_storage` in both builds
([parity log](port-aarch64/ctest-cuda-parity-concurrent-sanitizer.log); the FMA log of that run
was overwritten by the final run; 3 of 5 and 2 of 3 direct
reruns failed during that time, 5/5 passed afterwards). That test, like
`resident_fused_expansion`, measures device memory with `cudaMemGetInfo`, which on this
integrated GPU reports system-wide free memory, so concurrent processes that allocate
(here the sanitizer's target) disturb its 64 MiB bound. Run the CUDA suites on a quiet
machine; making these two checks robust on unified memory is a candidate for a later item.

The 9 Python suites pass in all three configurations (with CUDA they ran with `--cuda`); the
earlier Python failures were downstream of `_kan` not building. No compiler warnings were
observed in any configuration (build output was filtered for warnings; full build logs not
kept). The initializer pins (SplitMix64 + portable transform, 9 digests recorded on
MSVC/GCC x86) reproduce bitwise on aarch64 (`initializer` passes). The manual
`m3_large_basis_test` was not run.

## Sanitizers

| Run | Result | Log |
|---|---|---|
| memcheck `--leak-check full`, all 15 resident CUDA executables, parity and FMA builds | 0 errors, 0 bytes leaked, all tests pass | [sanitizers.txt](port-aarch64/sanitizers.txt) (first version of `scripts/sanitizers.sh`, without the per-run exit lines; the FMA deep runs were stopped there and rerun separately) |
| racecheck / synccheck / initcheck on `residual_resident`, `resident_training`, `resident_fused_expansion`, parity build | 0 hazards / 0 errors; under initcheck `resident_fused_expansion` reports 0/1 passed (its own memory assertion, see below) | [sanitizers.txt](port-aarch64/sanitizers.txt) |
| synccheck / initcheck / racecheck, same suites, FMA build | see [sanitizers-fma-deep.txt](port-aarch64/sanitizers-fma-deep.txt) (summary below) | |
| ASan (`build-asan`, RelWithDebInfo, CPU-only, no Python) | see below | [ctest-asan.log](port-aarch64/ctest-asan.log), [asan-long.log](port-aarch64/asan-long.log) |

initcheck reports 0 errors on `resident_fused_expansion`, but the suite's own assertion fails
under initcheck (2.63 parity and 2.70 FMA in the logs, 2.54 in a script smoke rerun; "row tensors" against the 2.5 limit): the test measures the arena by
`cudaMemGetInfo`, and the tool's allocations are counted. The test passes without the tool
(5/5 runs). The same `cudaMemGetInfo` method on an integrated GPU sees system memory, so concurrent
processes disturb it (observed for `cuda_legacy_calls_release_device_storage`, see Test
results).

FMA build deep runs ([sanitizers-fma-deep.txt](port-aarch64/sanitizers-fma-deep.txt)):
synccheck, initcheck and racecheck on the three suites all completed with 0 errors / 0 hazards
and all tests passing, except the same initcheck-only `resident_fused_expansion` memory
assertion; the `residual_resident` racecheck took 1 248 s (the Windows M3 FMA racecheck had
been aborted after about 3 h without a result, so this is the first completed FMA racecheck of
that suite).

ASan (`build-asan`, RelWithDebInfo, `KAN_ENABLE_ASAN=ON`, CPU-only): CTest 18/20 passed with no
AddressSanitizer report; the two long training demonstrations exceeded their 900 s CTest
timeouts under ASan (normally 23 s and 34 s) and were run directly without a timeout
([asan-long.log](port-aarch64/asan-long.log)): `initializer_training` 5/5 passed in 3 519 s,
no report; `residual_training` 5/5 passed in 5 030 s, no report.

## Golden comparison

Dumps were generated with `scripts/golden_dumps.sh` (harnesses linked with `-O2
-ffp-contract=off`), each run twice with identical output (deterministic on this GPU/driver/
cuBLAS).

Update 2026-10-14 (backlog M5): the golden and FP32 harnesses append Hermite-function fixtures.
The current local reference is [backlog/M5-bench/golden/](backlog/M5-bench/golden/sha256-linux.txt).
The files and hashes below remain the pre-M5 reference, which the harnesses reproduce byte for
byte with `-DKAN_GOLDEN_NO_HERMITE_FUNCTION` ([M5 evidence](backlog/M5.md#validation)).
Update 2026-10-14 (backlog C13): the C3 FP32 dumps change in their B-spline gradient lines (FP32
input VJP on the derivative-coefficient form); the current reference is
[backlog/C13-bench/golden/](backlog/C13-bench/golden/sha256-linux.txt) (the other six dumps are
unchanged from M5's).

The x86 references are recorded as SHA-256 hashes and byte counts only (M3 phase 2
`golden.txt`, C6–C8, C3); no x86 dump file is in the repository, so the comparison with x86 is
match/mismatch, not a quantified deviation.

Two differences are pure output format and were normalized
([msvc_norm.py](port-aarch64/golden/msvc_norm.py)) before comparing: MSVC's text-mode stdout
writes CRLF, and MSVC prints `%a` (used in the `x=` labels of the default dump) with a fixed
13-digit mantissa (`0x1.8000000000000p+0`) where glibc prints the shortest form (`0x1.8p+0`).
The normalizer rewrites every `%a` token (the default dump's `x=` labels and every value of the
C3 FP32 dump; the golden and rational dumps print values as raw hex bits). Its Windows-form
hashes are in [sha256-windows-form.txt](port-aarch64/golden/sha256-windows-form.txt); none
equals an x86 hash.

| Dump | Linux bytes | Normalized bytes | x86 bytes | x86 SHA-256 | Result |
|---|---:|---:|---:|---|---|
| golden default, parity | 168 110 | 176 087 | 176 087 | `6d6dc2e8…797e` | same size and structure (3 052 lines, identical tokens and exception messages), values differ |
| golden default, FMA | 168 110 | 176 087 | — | `13c4cdb4…c5e4` | values differ |
| golden `--layers` (CPU only; parity = FMA) | 33 335 | 33 628 | 33 628 | `c1f44d97…dc9` | same size, values differ |
| C3 FP32 all-family dump, parity / FMA | 17 510 785 / 17 563 189 | 26 221 643 / 26 221 642 | not recorded | `49801919…895` / `7eb68a83…b74` | differ (also after normalization) |
| C6–C8 rational dump, parity / FMA | 126 245 707 / 127 112 114 | 126 247 745 / 127 114 158 | not recorded | `018eb7ce…22a` / `57c6bd27…b74` | differ (also after CRLF) |

Interpretation (not quantified against x86):
- The CPU-only `--layers` dump differs although it matches in size, so CPU results differ. The
  fixtures themselves are generated with `std::sin`/`std::cos` (`golden.cpp:224,228,242,298,345`),
  and the library calls `exp`, `log`, `sin`, `cos` and `tanh` (`src/detail/*_formulas.hpp`,
  `host_device.hpp`). The likely cause (inferred from the hash mismatch, not isolated) is that
  MSVC's CRT and glibc 2.39 libm round some of these results differently; a 1-ulp difference
  in any fixture value propagates to every downstream value. Basic arithmetic and `sqrt` are
  IEEE-exact on both and unfused on both after fix 1. `long double` (binary128 on aarch64
  Linux, binary64 under MSVC) appears only in two tolerance-based test references
  (`residual_formula_test.cpp`, a comment in `basis_test.cpp`), not in the library or the
  dump harnesses.
- Resident sections additionally depend on cuBLAS kernel selection for sm_121 (different
  summation orders), FP32 results also on the FP32 kernels' tile choices; the FP64 full-step
  checksums printed by `residual_train_bench` agree with the RTX 3090 records to all 10 printed
  digits (all 12 FP64 cells), the FP32 ones by up to 1.0e-6 relative (1024-wide).
- Within this machine, parity vs FMA (`golden_diff.py`): 3 052 lines, 2 949 identical, 103
  numerically different, 0 structural; `resident mixed` 2.39e-14, `resident endpoints` 2.23e-16
  (x86, C1-bench/golden-diff.txt: 101 lines, 2.30e-14 and 2.23e-16) — the same sections at the
  same magnitude.

A quantified x86 comparison needs the x86 dumps: on Windows, build `golden.cpp`,
`f32_dump.cpp`, `rational_dump.cpp` against the current tree (`M3-bench/phase2/dumps.sh`), then
`python docs/evidence/backlog/golden/golden_diff.py <x86 dump> <msvc_norm.py output>` (the
default and `--layers` dumps; `golden_diff.py` does not parse the `%a`-formatted FP32 dump).

**Local reference set** ([port-aarch64/golden/](port-aarch64/golden/)): `golden-parity.txt`,
`golden-fma.txt`, `golden-layers.txt` (Linux form, LF), their and the large dumps' SHA-256 in
[sha256-linux.txt](port-aarch64/golden/sha256-linux.txt), and the Windows-form hashes in
`sha256-windows-form.txt`. The FP32 (17.5 MB) and rational (126 MB) dumps are recorded as hashes
only, like the earlier evidence; regenerate them with `scripts/golden_dumps.sh`. Later items on
this machine compare against these hashes (byte identity) and, for the small dumps, value by
value with `golden_diff.py`.

## GB10 baselines (parity build, one pass)

`scripts/bench_baseline.sh build-cuda docs/evidence/port-aarch64/bench 3`: frozen m2/m3/m4
(`--warmups 2 --repeats 7`), `residual_train_bench` (M3 phase 2; C9 matched protocol, status
interval 1), C9 `train_bench` step mode (matched/realistic, interval 1/64) and C6–C8
`rational_train_bench`, three timing windows each, one process per cell, no A/B interleaving.
Raw CSVs, `nvidia-smi dmon` samples and the GPU state before/after are in
[port-aarch64/bench/](port-aarch64/bench/); medians in
[summary.md](port-aarch64/bench/summary.md). Conditions: three other processes kept 15 GB of
GPU memory resident but were idle (0% utilization before the run; `gpu-after.txt`, taken
right after the last benchmark, shows this run's own 96%); a single-threaded CPU ASan test ran
concurrently on one of 20 cores. These are reference numbers, not accepted
performance evidence.

Full training step, ms/step, median of 3 windows, branch off (`residual_train_bench`), against
the RTX 3090 M3 phase 2 record (phase 2, branch off, ABBA median of 9):

| Case | FP64 GB10 | FP64 3090 | FP32 GB10 | FP32 3090 | TF32 GB10 | TF32 3090 |
|---|---:|---:|---:|---:|---:|---:|
| 64x64x32x16, b1024 | 1.259 | 0.879 | 0.141 | 0.153 | 0.122 | — |
| 16x24x8, b1024 | 0.233 | 0.229 | 0.061 | 0.080 | 0.057 | — |
| 256x256x256x10, b8192 | 115.2 | 80.0 | 5.29 | 2.94 | 4.39 | — |
| 1024x1024x1024, b4096 | 766.1 | 559.0 | 23.87 | 14.98 | 11.87 | 9.82 |
| rational [3/2] Smooth 256x256x256x10, b2048 | 216.6 | 135.0 | 41.67 | 15.80 | 41.58 | — |
| trainable RBF 256x256x256x10, b2048 | 50.23 | 33.78 | 2.98 | 1.61 | 2.54 | — |

Branch on (same harness): FP64 1.484 / 0.278 / 131.7 / 875.9 / 220.7 / 54.74 ms, FP32
0.180 / 0.072 / 6.42 / 27.73 / 41.61 / 3.33 ms, TF32 0.162 / 0.072 / 5.34 / 13.92 / 41.51 /
2.84 ms (case order as above).

C9 `train_bench`, step mode, interval 1 → 64 (ms/step): FP32 matched 0.143→0.133,
0.064→0.054, 5.30→5.29, 23.87→23.88; FP32 realistic 0.169→0.136, 0.081→0.059, 5.94→5.49,
25.84→24.83; TF32 matched 1024-wide 11.92→11.88, realistic 13.88→13.10; FP64 matched
1.262 / 0.229 / 114.9 / 763.3 (interval 1). C6–C8 rational step (ms): FP64 guarded / absolute
/ smooth 5.82 / 5.88 / 6.17 (64x64x32x16, b1024) and 207.0 / 207.6 / 216.6 (256-wide, b2048);
FP32 0.875 / 1.084 / 1.086 and 32.7 / 41.5 / 41.6 (b8192: 144 / 177 / 178). Frozen suites:
per-case medians in [summary.md](port-aarch64/bench/summary.md) (e.g. m2 case 3, Chebyshev
64x64x32x16 b1024: resident 1.47 ms, CPU 70.5 ms).

Observations (no retuning in this step):
- FP64 Chebyshev large cases are 1.37–1.44x the 3090's time, consistent with the measured
  DGEMM rate (0.40 vs ≈0.53 TFLOP/s achieved on the 3090): still FP64-ALU bound; trainable RBF
  1.49x, rational 1.60x; FP64 64x64x32x16 1.43x, 16x24x8 equal.
- FP32/TF32 large cases are 1.2–1.85x slower (TF32 1024-wide 1.21x, FP32 1024-wide 1.59x,
  256-wide 1.80x, trainable RBF 1.85x): the thresholds, tiles and the 4 MiB cuBLAS
  workspace were chosen on GA102 (82 SMs, 936 GB/s); GB10 has 48 SMs and about a quarter of
  the bandwidth, and FP32 steps were partly memory bound on the 3090 (C1/C3 profiles).
- The rational FP32 step is 2.6x slower (41.7 vs 15.8 ms) and TF32 does not help it: the
  rational kernels (not GEMMs) dominate, and their per-SM division/SFU throughput and the
  48-SM count matter most. This is the largest relative regression and the first candidate
  for profiling on this machine.
- Small FP32 cases (16x24x8 0.77x, 64x64x32x16 0.92x) are faster than on the 3090, where launch and graph
  overhead dominate (the 3090 ran under Windows WDDM; not profiled here).
- Unified memory: host↔device "transfers" are on the same LPDDR5x; the realistic-protocol and
  `*_transfer_full` overheads are small here (e.g. FP32 1024-wide realistic N=64 24.8 vs
  matched 23.9 ms). Pinned staging and PCIe-motivated choices (R9 `direct_upload_bytes`, C9
  double buffering) were measured against PCIe on the 3090.

## Adaptability inventory (GA102-tuned constants in `src/resident.cu`)

Line numbers at this tree. None were changed; later items can derive them from device
properties (SM count, shared memory per block/SM, L2 size, bandwidth, FP64 rate) or re-measure.

| Line | Constant | Value | Tuned for / rationale recorded in the source |
|---|---|---|---|
| 40–41 | `blocks()` grid cap | 65535 blocks, 256 work items per block | generic launch helper; no SM-count dependence |
| 100–101 | `stage_threads`, `stage_bytes` | 256 threads, 32 KiB shared stage per block | C1 coalescing tiles; bounds `stage_rows`, the C3 recompute (≤ 85 terms) and `residual_tile_inputs`; GB10 allows 99 KiB per block |
| 102–110 | `stage_rows<double>` = 0 | FP64 keeps direct (untiled) row access | "FP64 basis kernels are FP64-ALU bound on GA102; the 32 KiB stage costs occupancy (m2 Gaussian RBF 1.9x slower)" |
| 221–222 | `small_forward_contraction` | 2^23 (FP64), 2^24 (FP32) multiply-adds | warp-dot vs cuBLAS crossover measured on the 3090 (C2 dot_vs_gemm, C1 small_shapes_f32) |
| 228–230 | `small_parameter_vjp`, `small_parameter_work` | 2^15 coefficients+outputs; FP32 work 2^22 | tiled parameter VJP vs SGEMM+SGEMV crossover (C1) |
| 231–232 | `parameter_tile_rows`, `parameter_tiles` | 64 rows, 64 tiles | batch tiling of the small parameter VJP (grid fill on 82 SMs) |
| 343 | `nonlinear_tiles` | 64 | trainable-RBF center/width reduction tiles (C1 FP32 portion of C12) |
| 460–462 | `residual_tile`, `residual_tile_inputs` | 32 samples; inputs bounded by `stage_bytes` | M3 phase 2 residual dot kernel (long-scoreboard / block count at batch 1024 on the 3090) |
| 567 | `commit_max_blocks` | 160 | commit/rollback kernel grid (about 2 blocks per GA102 SM) |
| 569 | `stage_threads_max`, `stage_chunk` | 8 host threads, 2^18 values | host-side conversion of staged batches (i5-12400, 6 cores; this host has 20) |
| 576 | `mse_threads`, `mse_max_blocks` | 256, 256 | deterministic MSE reduction grid (fixed grid keeps the sum order; changing it changes results) |
| 893 | `norm_block` | 64 threads | LayerNorm row kernels ("spreading few warps over every SM", 1024 rows x 64 features → 128 blocks) |
| 900 | `norm_row_lanes`, `norm_rows_per_tile`, `norm_max_tiles` | 8, 16, 64 | LayerNorm gain/bias VJP tiling |
| 965 | `blas_workspace_bytes` | 4 MiB | carriers' cuBLAS workspace (kernel selection and results depend on it) |
| 966 | `residual_workspace_bytes` | 32 MiB | residual branch split-K workspace ("105 µs at 4 MiB, 54 µs at 32 MiB, FP32 RTX 3090") |
| 1441–1445 | `small_residual_work`, `small_residual` | reuse `small_forward_contraction` and `residual_tile_inputs` | residual forward kernel vs cuBLAS |
| 1453 | residual group cap | 65535 | grid limit |
| 1915 | `direct_upload_bytes` | 1 MiB | R9 FP64 packed vs direct upload, measured over PCIe (62 → 45 ms); GB10 uploads within unified memory |

Note for later items: changing `mse_max_blocks`, `parameter_tiles`, `nonlinear_tiles`,
`norm_*` tiles or the cuBLAS workspaces changes summation order and hence results (golden
hashes), so device-derived values must stay deterministic for a given device.

## Contract-relevant observations

- "CPU is unfused" now holds for every in-tree target on GCC/Clang regardless of the target
  ISA; consumers are not affected (directory scope, nothing exported).
- TF32 tolerance: new bias-VJP allowance (fix 3), recorded in `docs/CONTRACT.md`.
- Bitwise results are per GPU architecture, driver and cuBLAS version (as CONTRACT already
  states for the residual branch); the x86 golden hashes do not apply to this machine. Later
  items here use the local reference set above.
- `resident_fused_expansion` measures the arena with `cudaMemGetInfo`, which on an integrated GPU
  reports system memory; valid without tools, invalid under initcheck.

## Independent review

A fresh reviewer agent examined the diff and this evidence (no files changed by it). Findings
and resolution:

| Finding | Resolution |
|---|---|
| The TF32 bias allowance also applied to the last layer, whose U is the uploaded `dy` (allowance 0.25 against gradients of 0.02–0.05: the check was vacuous there) | Last layer excluded; verified passing in both builds |
| 2e-3 was 24x the measured error, justified only by consistency with M3 | Lowered to `2^-11` (TF32 unit roundoff, 6x headroom), described as an empirical scale in the test and CONTRACT |
| `scripts/build.sh` rejected the documented `--config RelWithDebInfo` | Accepts `RelWithDebInfo` and `MinSizeRel` |
| `scripts/golden_dumps.sh` could hash empty dumps after a link or run failure | Skips a tag whose harnesses fail to link, removes failed dumps, exits non-zero |
| `scripts/sanitizers.sh` discarded exit status; the FMA deep log came from an unrecorded loop | `--error-exitcode`, exit status and duration per run, non-zero exit on failure; the deep log's header names the loop (same format) |
| RED paragraph claimed 46/47 for a log showing 45/47; FMA RED value and TF32 diagnosis had no artifacts | Corrected; after-fix-1–2 CTest logs for both builds and the TF32 probe sources/outputs added under `port-aarch64/` |
| "RTX 3090 / CUDA 12" was wrong (CUDA 13.1 there); "most probably ran without tensor ops" was speculation | Corrected; the cause of the cross-machine difference is stated as unknown |
| Benchmark-condition, checksum-range, ratio-range, bandwidth and initcheck figures over- or understated | Corrected against the CSVs/logs |
| Nits: `-Xcompiler` only valid for nvcc; set-but-empty `CUDAARCHS`; README; harness include guess; bench script status; `.sh` line endings | `COMPILE_LANG_AND_ID:CUDA,NVIDIA`; empty `CUDAARCHS` treated as unset; README documents the `native` default; `link_harness.sh` reads `CMAKE_HOME_DIRECTORY`; `bench_baseline.sh` records failures; `.gitattributes` keeps `*.sh` LF |

Verified by the reviewer: no flag reaches consumers (installed `KANTargets.cmake` carries only
`cxx_std_20`); the MSVC path is untouched (`if(NOT MSVC)`, cl as nvcc host on Windows,
`build.ps1` passes architecture 86); `this->capacity` is portable; FP32/FP64 checks unchanged;
CTest counts, memcheck results, golden hashes and the 3090 comparison numbers match their
sources.

Re-review of the revisions: code, scripts and normalizer verified (the normalizer reproduces the
Windows-form hashes). Its remaining points were resolved: the initcheck paragraph now refers to
the observed `cudaMemGetInfo` flake, the missing FMA log of the concurrent run is stated, the
headroom is given from the larger of the two measured steps, `scripts/sanitizers.sh` no longer
fails on the known initcheck-only `resident_fused_expansion` assertion (only on sanitizer
errors there), and `scripts/golden_dumps.sh` hashes only the dumps written by the current run.

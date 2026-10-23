# Platform check: back on Windows x86-64 / RTX 3090 (not a backlog item)

Date: 2026-10-23. Branch `cpp-foundation`, base `e423684` (M5 plan). Purpose: build and test
the tree on the original development machine after the work on GB10
([port-aarch64](port-aarch64.md), 2026-10-12 to 10-23), fix what does not build, and check
whether the device-dependent choices added on GB10 (backlog C5, C12, M7) hold on sm_86.
Analysis only: no selection rule or kernel was changed. GB10 remains the primary
development and reference machine (owner, 2026-10-23); this machine is a validation target.

| Item | Value |
|---|---|
| OS / CPU | Windows 11 Pro 10.0.26200, Intel i5-12400 (6 cores) |
| GPU | RTX 3090, sm_86, 82 SMs, L2 6 MiB, FP64 1:64, 99 KiB opt-in shared memory per block, driver 591.86 ([gpu.txt](port-windows-3090/gpu.txt)) |
| Toolchain | MSVC 19.50 (Visual Studio 18), CUDA 13.1 (V13.1.115), CMake 4.4.3, Ninja (VS), Python 3.13.10 `.venv`, numpy 2.5.2 |
| Build | `scripts\build.ps1 -Cuda [-CudaFma] -AllowUnsupportedCudaCompiler -Python -Benchmarks` (VS 18 is outside nvcc's supported range, as in every earlier Windows record) |

## Build defects and fixes (commit `a339745`)

RED ([red-build-parity.log](port-windows-3090/red-build-parity.log)): both CUDA builds fail
in two translation units; the CPU-only build succeeds.

1. `resident.cu(3984): error: invalid combination of type specifiers`. The CUB headers added
   by C5 include CCCL's `cuda/std/__thread/threading_support_win32.h`, which includes
   `windows.h`; the Windows SDK's `rpcndr.h` defines `#define small char`, so the local
   `const bool small` of the parameter-VJP launcher (present since 2026-10-02, compiled before
   the CUB include) no longer compiles. Fix: renamed to `small_vjp`.
2. `wavelet_resident_test.cpp(146): error C2027: use of undefined type 'std::array'`: the
   M7 suite uses `std::array` without `<array>` (libstdc++ provides it transitively). Fix:
   the include.

Both are host-only. SASS of `resident.cu` before/after ([sass-identity.txt](port-windows-3090/sass-identity.txt),
built on GB10 with `nvcc -arch=sm_121` and `-arch=sm_86`, `--fmad=false` and `true`,
`C5-bench/sass_cmp.py`): 292/292 kernels identical in all four. clang-format 18.1.8: no change.

Cause of the gap: CI (`.github/workflows/cpu.yml`) builds CPU-only on Ubuntu and Windows, so
no CUDA translation unit was compiled with MSVC between 2026-10-12 and today.

## Tests (final tree)

| Configuration | Result | Log |
|---|---|---|
| CPU-only Release + Python | 36/36 | — |
| CUDA parity (`KAN_CUDA_FMA=OFF`, sm_86) + Python | 59/59 | [ctest-cuda-parity.log](port-windows-3090/ctest-cuda-parity.log) |
| CUDA FMA + Python | 59/59 | [ctest-cuda-fma.log](port-windows-3090/ctest-cuda-fma.log) |

`package_*` suites configure a nested MSVC project and must run inside the vcvars
environment (outside it they fail with `rc` not found); they also fail when the build
directory path exceeds CMake's 250-character object path limit.

The uncommitted M5 phase-1 tree from GB10 (server working copy at `e423684`, applied to a
local worktree with the fix above) also builds here: CPU-only all quantum suites pass
(`pqc`, `pqc_layer`, `pqc_allocation`, `quantum_source_check`, `pqc_stub_python`, including
the `long double` closed-form reference and the pinned PQC initializer digest), CUDA
parity 63/63 without `package_*`. PQC CPU throughput (`pqc_cpu_bench`, MSVC `/O2`, one
thread, [pqc-cpu/](port-windows-3090/pqc-cpu/pqc-cpu-3090host.csv)) is 1.4x (n = 1) to 2.4x
(n = 4) below the Cortex-X925/GCC record forward, 1.4-1.6x backward; checksums agree to
about 1e-15 (libm).

## Device-dependent choices on sm_86

Complete `train_step`s with the automatic choice against the forced alternatives of each
private test hook, interleaved in one process per run (harness sources unchanged:
`C5-bench/spline_bench.cpp`, `C12-bench/rbf_step.cpp`, `M7-bench/wavelet_step.cpp`, linked
with `-DC12_HOOK -DM7_HOOK`), parity build. Raw CSVs, configs, script and summary in
[sweeps/](port-windows-3090/sweeps/summary.txt).

- **C12 (trainable-RBF reduction):** auto within 0.99-1.00 of the faster of column/tile in
  all 7 configurations.
- **M7 (wavelet launch cost model):** auto within 1.01-1.07 of the best of 8 forced
  geometries in 10 of 11 configurations, 1.18 at 64x64x64x8 batch 512 (0.03 ms). The same
  metric on the GB10 record (`M7-bench/sweep-2.csv`) is 0.99-1.11; the minimum of many noisy
  medians is biased low, so this is not a regression.
- **C5 (sparse B-spline selection): wrong in both directions.** The rule evaluated with this
  device's attributes ([rule-eval.txt](port-windows-3090/sweeps/rule-eval.txt),
  `sweeps/attr.py`) reproduces the measured sparse layer count in all 23 configurations, so
  the mechanism is known:
  - *Regressions (auto sparse, sparse slower; dense would be the pre-C5 behaviour):* FP32
    256- and 1024-wide K = 24 degree 3: 1.18x; FP32 degree 1 K = 12: 1.45x; FP32 64x64x32x16
    K = 96: 1.14x. With 6 MiB of L2 the upstream of these layers exceeds half the L2 (GB10:
    24 MiB), selecting the out-of-L2 ratio 6, fitted on GB10's 243 GB/s; on GA102 (936 GB/s)
    the dense rows are cheaper and the FP32 out-of-L2 crossover lies between K/(p+1) = 6 and 12.
  - *Missed gains (auto dense, as pre-C5):* FP64 256x256x256x10 batch 2048 K = 24: sparse 2.05x
    faster (out of L2 here, so the unmeasured, extrapolated FP64 ratio 8 applies); FP64 K = 12
    1.10x, degree 1 K = 6 1.21x; FP64 64x64x32x16 K = 48 1.36x, where the per-SM work floor
    (82 SMs) keeps the second and third layers dense.

## Existing paths: before vs after the GB10 period

Same-session ABBA of the frozen suites, `37a07f8` (last tree before the port, built here in a
worktree) against this head, `--backend all --warmups 2 --repeats 7`
([abba/](port-windows-3090/abba/summary.txt), `compare_bench.py`): resident geomean basis
1.015, local 1.013, rational 0.968; cpu_full 1.026 / 0.935 / 0.977; 0 checksum mismatches.
Largest single-case ratios are small cases (Chebyshev 16x24x8 batch 32 cpu_full 0.24 -> 0.40
ms) and the Gaussian RBF FP64 64x64x32x16 cases (1.21-1.23), whose samples are bimodal on
this machine in every record. An earlier comparison against the C3 CSVs (other session) showed
1.2-2.6x outliers that this ABBA does not reproduce: cross-session drift.

Measurement note for this machine (WDDM): some cases depend on the process history. Rational
case 3 (FP64 64x64x32x16 batch 1024, resident) runs in 3.6 ms alone and in 17 ms after cases
0-2 of the same process, reproducibly; the C6-C8 and C9 records show the same jumps; GB10
records are stable (4.1-4.3 ms). Wall-clock comparisons here need isolated per-case runs or
repeated, robust statistics.

## Open points (recorded, not acted on)

- C5's ratios and floors are GB10 measurements expressed through device attributes; they do
  not transfer to sm_86. This belongs to device adaptation (owner's planned post-M5 task).
- CI has no CUDA compile job; a compile-only CUDA build on Windows and Linux would have caught
  both defects.
- CCCL warns about MSVC's traditional preprocessor (`/Zc:preprocessor` suggested).

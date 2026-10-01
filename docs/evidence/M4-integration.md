# M4 integration and toolchain evidence

2026-10-01, Windows x64, MSVC19.50.35724.0, CUDA13.1.115 with explicit local
unsupported-host-compiler override, RTX3090 sm86, Python3.13.10, NumPy2.5.2,
pybind11 3.0.2, GCC/UCRT64 15.2.0, gcovr8.6. Baseline M3 replay passed14/14 at
`b3777f8`. Final CPU numerical source is `4b808d2`; initial corrected resident
source is `c5a54e3`. Subsequent GPU tuning requires its own final replay.

## Actually executed integration checks

| Command | Observed result |
| --- | --- |
| `scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler -Python -Benchmarks -BuildDirectory build-m4-final -Test` | **18/18 CTest** at corrected resident `c5a54e3` |
| `scripts/build.ps1 -Python -BuildDirectory build-m4-python -Test` after CPU underflow correction | **13/13**, dedicated M4 Python3pass+1explicitGPUskip |
| `scripts/build.ps1 -Asan -Configuration Debug -BuildDirectory build-asan-m4 -Test` after CPU underflow correction | **10/10**, no reported memory error |
| GCC Debug coverage build `build-coverage-m4`, CTest after CPU underflow correction | **10/10**, warning-free |
| `.venv/Scripts/python.exe -m gcovr --root . --filter 'src/' --exclude-unreachable-branches --exclude-throw-branches --txt --json-summary build-coverage-m4/summary.json --fail-under-line 80 build-coverage-m4` | **98.3% lines572/582,90.8% branches740/815,100% functions38/38** |

GCC configuration uses Ninja with explicit
`CMAKE_MAKE_PROGRAM=C:/Program Files/Microsoft Visual Studio/18/Professional/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe`,
`CMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe`, Debug and
`KAN_ENABLE_COVERAGE=ON`. Prefix PATH with `C:/msys64/ucrt64/bin` for matching
compiler/gcov. The first attempt omitted the Ninja path and failed configuration;
the corrected command above built, tested and produced actual coverage. No empty
coverage result is counted. Build/coverage reports stay in ignored directories.

The eighteen integrated targets are ten CPU/integration targets, five real-device
CUDA targets and three Python suites. Package custom-include and default-config
tests build isolated CPU libraries, install and consume the resulting package,
including M4 rational values/VJPs. Explicit installed CUDA/Python acceptance and
post-tuning integrated results follow after the profiling step.

Public use and mathematical/lifecycle contracts are updated in README.md and
CONTRACT.md. The only implementation-independent numerical review finding so far
was representable derivative loss through intermediate underflow; retained CPU
and actual-device RED/GREEN commits correct it with rare log-space paths.

## Final source replay and installed use

After GPU tuning `0071628` and CPU allocation refinement `c80b7d3`, all gates were
replayed against both final numerical implementations:

| Final command / gate | Observed result |
| --- | --- |
| Same full CUDA/Python/benchmarks helper command for `build-m4-final` | **19/19 CTest** |
| Same CPU-only Python helper for `build-m4-python` | **14/14 CTest** |
| Same ASAN Debug helper for `build-asan-m4` | **11/11**, no reported memory error |
| GCC Debug coverage rebuild and CTest | **11/11**, warning-free |
| Same gcovr command | **98.1% lines574/585,91.0% branches731/803,100% functions39/39** |
| `compute-sanitizer --tool memcheck --error-exitcode 1 build-m4-final/<suite>.exe` for m4_resident_test, resident_test, resident_review_test, m3_resident_test | **7/7,5/5,4/4,7/7;0errors each** |
| `cmake --install build-m4-final --prefix build-m4-install` | Installs CPU/CUDA libraries, public rational headers and Python module |
| VS18 Release installed consumer `build-m4-consumer`, `ctest -C Release` | **1/1**, actual installed CPU/CUDA rational values, VJPs and SGD |
| `PYTHONPATH=build-m4-install/python` and existing/M3/M4 Python suites with `--cuda` | **5/5,4/4,4/4** |

Consumer configuration:
`cmake -S tests/consumer -B build-m4-consumer -G "Visual Studio 18 2026" -DCMAKE_PREFIX_PATH=K:/PycharmProjects/auxiliary_projects/KAN/build-m4-install -DKAN_CONSUMER_CUDA=ON`,
then `cmake --build build-m4-consumer --config Release --parallel` and
`ctest --test-dir build-m4-consumer -C Release --output-on-failure`.
Private `src/rational_internal.hpp` is not required by the installed public API.

The final nineteen targets are eleven CPU/integration, five GPU and three Python.
Retained final CTest/ASAN/CPU-Python/coverage/installed-consumer transcripts,
coverage summary and four sanitizer diagnostics live under `m4/final-*`.
Sanitizer resolves through the CUDA13.1 `compute-sanitizer.bat` launcher on PATH;
the initial guessed `.exe` path did not exist, and no result from that failed
invocation is counted. The actual launcher commands above completed successfully.

The final independent retained numerical probe covers all289degree pairs,
1445scalar identities,95mixed rational/RBF parameter finite differences,
strict arrays/snapshots, repeated GPU trajectories, zero/subnormal derivatives,
warp batch/block tails, failed-backward recovery and next-execution pole guards.
Its review/evidence gate is recorded in M4-review.md. CPU/GPU measured tuning
is separately recorded in M4-cpu-performance.md and M4-benchmark.md; this
integration replay adds no performance claim.

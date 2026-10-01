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

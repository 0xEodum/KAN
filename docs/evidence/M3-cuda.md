# M3 resident CUDA evidence

2026-10-01 executable RED: Release MSVC/CUDA architecture 86 build using
`./scripts/build.ps1 -Cuda -Benchmarks -AllowUnsupportedCudaCompiler
-BuildDirectory build-m3-cuda`. `m3_resident_test.exe` returned exit 1, **0/5**.
[Exact output](m3/resident-red.txt) records unsupported localized bases, missing
regularization validation, and absent learned-width candidate validation.
The latter two execute the old resident implementation on real RTX 3090 hardware.
Only the optional backward argument was introduced for compilable RED tests;
the old implementation deliberately ignores it at this revision.

Tests cover independent spline hat values, mixed-family CPU/device VJPs and
repeated SGD, learned snapshots, zero-batch L2, invalid lambda, log-width
underflow rejection before any network mutation, reusable failed gradients,
and explicit CPU snapshot/refine/device reconstruction.

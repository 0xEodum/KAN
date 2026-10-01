# M4 resident executable RED

Frozen fixture/test/protocol commit `73e3f47` preceded CUDA implementation.
Initial executable RED (`m4/cuda-red.txt`) records 0/4 with the pending CPU
rational constructor. After CPU GREEN, the unchanged resident implementation
was rebuilt and replayed: `m4/cuda-resident-red.txt` records 0/4 because its
basis-only construction calls `basis()` on a rational layer. This independently
isolates absent resident support from the CPU prerequisite.

Commands: `scripts/build.ps1 -Cuda -Benchmarks -AllowUnsupportedCudaCompiler
-BuildDirectory build-m4-cuda`, then `build-m4-cuda/m4_resident_test.exe`.
Release MSVC, CUDA 13.1, architecture 86, real RTX 3090, no host fallback.

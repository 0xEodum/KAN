# Independent large-launch reduction regression: RED

2026-10-01: independent reviewer found the tiled RBF finish kernel lacked a
grid-stride loop while its launch helper caps blocks at 65535. Default-sized
parity tests and matched diagnostic timings passed, but a valid large basis
could leave the tail of nonlinear gradients unwritten.

The retained test calls the real resident API with **8,388,481 trainable Gaussian
terms**, batch one, one input/output, all centers1, log widths0, coefficients0.1,
input0 and upstream1. Thus each log-width VJP is independently known to be
`0.2*exp(-1)`, including the tail. It checks four final entries; zero batch would
not reliably expose uninitialized device storage.

`scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler -Python -Benchmarks
-BuildDirectory build-m3-final`, then `build-m3-final/m3_large_basis_test.exe`:
successful build, executable exit1 with
`FAIL unwritten nonlinear gradient at 8388479: actual=0 expected=0.0735759`.
The test initially needed explicit private CUDA runtime linkage to compile its
free-memory query; that setup error is corrected in the retained CMake target.

This manual regression requires substantial GPU memory (explicitly checks 10GiB
free) and is excluded from default CTest. The RTX3090 executed the actual failing
case. This checkpoint retains the otherwise parity-tested tiled implementation
with its defect; performance acceptance remains pending a verified correction.

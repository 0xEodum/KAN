# CUDA TDD evidence (M1)

Runtime RED: `./scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler -Test`
on 2026-09-30 configured and compiled the CUDA target and test suite successfully.
`CUDA device available: yes`; CUDA suite 1/9 passed, with eight failures reporting
`CUDA forward is not implemented` or `CUDA backward is not implemented` from the
intentional runtime stubs. All four other CTest suites passed; CTest returned failure.
The one passing case exercises the no-device path only when there is no device;
it returned without device operations on this GPU host.

The tests require forward, input/parameter/bias VJP parity with CPU on five varied
batch/topology/term-count combinations; analytic endpoint derivatives and unclipped
inputs; zero-batch behavior; invalid sizes and nonfinite data; dimension overflow;
unsupported families; overflow in output and each gradient category; and repeated
and concurrent independent calls.

Toolchain: CUDA 13.1.115, MSVC 19.50.35724.0 / VS 18, architecture 86 (RTX 3090).
CUDA's host header rejects MSVC >= 1950, so this local build explicitly opts into
`--allow-unsupported-compiler`. A standalone stub probe and the complete RED build
both compiled with this override. This validates the observed toolchain only;
it is not a claim of official CUDA support for this MSVC release.

GREEN is pending implementation.

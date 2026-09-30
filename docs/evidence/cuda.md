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

GREEN: the same full build/test command returned exit 0, with all 5/5 CTest
suites passing. CUDA reported device available and 8/8 device cases passed on
the RTX 3090. Forward uses the value and derivative recurrence, including a
high-degree derivative-overflow test even when the basis values remain finite.
All input, coefficient, and bias gradients match CPU within relative-plus-absolute
tolerance 2e-11 on the tested shapes. No performance or bitwise-equivalence claim.

Implementation uses custom double CUDA kernels, one thread per output or gradient
element, sequential per-element reductions without floating-point atomics,
and per-call RAII device buffers and nonblocking streams. The API validates
configuration, checked element and byte counts, shapes, finite host data and
parameters, and finite returned outputs/gradients. Empty batches allocate no
device buffers and return zero parameter gradients. Streams synchronize before
buffers are destroyed on normal return and exception cleanup.

`compute-sanitizer --tool memcheck --error-exitcode 1 ./build-cuda/cuda_test.exe`
returned exit 0 with device available, 8/8 cases passed, and
`ERROR SUMMARY: 0 errors`.

Additional RED: with `CUDA_VISIBLE_DEVICES=-1` scoped to a separate process,
the initial default suite incorrectly returned exit 0 and 9/9 after skipping
the eight GPU cases. A check requiring default suite failure returned exit 1
with `RED: default CUDA parity suite incorrectly succeeded without hardware`.
The test entry point now requires hardware for the default GPU suite. Repeating
the hidden-GPU default invocation returned exit 1 and
`FAIL real CUDA hardware is required for the GPU parity suite` (GREEN).

In that same hidden-GPU environment,
`./build-cuda/cuda_test.exe --expect-no-device` returned exit 0, 1/1 passed,
and confirmed `std::runtime_error` for forward and backward without a device.
This explicit mode runs only the no-device assertion, independently of the
real-GPU suite, and does not claim GPU parity validation.

## Independent-review FMA regression

The independent reviewer found a concrete exception-contract mismatch:
Chebyshev size 2, one input/output, coefficients `{-DBL_MAX, DBL_MAX}`,
bias 0 and input 2. CPU raises overflow for `DBL_MAX * 2`; CUDA's default
fused multiply-add instead computes that product plus `-DBL_MAX` in one
operation and returns finite `DBL_MAX`.

The new test `cuda_preserves_unfused_intermediate_overflow_contract` first
asserts CPU overflow and then requires CUDA overflow. Running the normal full
build/test command against the default-fused backend returned failure on the
real GPU: CUDA 8/9 passed; the new case reported
`expected exception was not thrown`. All four other CTest suites passed.
This is the executable RED checkpoint for the independently reported issue.
Correction is pending a uniform CUDA-only `--fmad=false` compile option;
the demonstrated numerical exception semantics justify that option.

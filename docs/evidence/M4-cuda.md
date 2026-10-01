# M4 resident CUDA execution evidence

The rational constructor is distinct from fixed bases. Resident construction
validates the CPU snapshot, stores numerator/bias/denominator parameters in its
permanent arena, and reserves edge-major P/Q/input-derivative caches for its
maximum batch. The copied configuration remains immutable; learned parameters
come exclusively from device storage. Rational forward and complete nonlinear
input/numerator/denominator/bias VJPs run on device. Mixed basis/rational networks
retain the existing stream, status checks, state invalidation and atomic
whole-network SGD region swap. Coefficient L2 penalizes numerator parameters
only. Zero batch preserves the explicit regularizer with zero data VJPs.

Finite unsafe denominators set a distinct device status and raise domain_error;
nonfinite intermediates/results raise overflow_error. Forward invalidates prior
outputs/gradients before execution; failed backward invalidates gradient/SGD
access, retaining the valid forward output for a corrected upstream retry.
Finite SGD candidates may introduce an unsafe sample, rejected on subsequent
forward as specified. Setters reconstruct snapshots using rational parameters.

Executable RED precedes implementation: frozen test/benchmark `73e3f47`, initial
0/4 RED, then isolated 0/4 resident construction RED `1d7797f` after CPU GREEN.
Baseline resident support `0185229` passed 4/4 and integrated 15/15 on RTX 3090.
Independent review exposed representable denominator derivatives lost through
power/quotient underflow; `9938c85` retains executable 4/5 RED before the fix
`c5a54e3`. The rare logarithmic branch mirrors CPU for subnormal powers,
quotients and scaled input derivatives; ordinary values retain Horner and
analytic quotient evaluation. Device tests include negative-z degree 16,
P/Q underflow, finite Q whose square would overflow, and scale1e-320.

Final tests additionally cover Padé 1/1 scalar identities and explicit VJPs,
mixed rational/Chebyshev trajectories for degree pairs 0/0,0/3,4/1,16/16,
numerator-only L2, exact/removable poles, relative guard equality/cancellation,
overflow invalidation/recovery, atomic SGD failure and zero batch. The capped
warp launch is exercised with 600001 outputs and 1200002 parameters to verify
the parameter tail beyond 65535 blocks. Allocation counts remain two throughout.
Independent reviewer separately verifies mixed trainable-RBF/rational finite
differences and five-step parameter/state parity; see retained reviewer files.

Release reproduction: `scripts/build.ps1 -Cuda -Benchmarks
-AllowUnsupportedCudaCompiler -BuildDirectory build-m4-cuda -Test` imports the
MSVC environment required by installed consumer tests. Standalone ctest without
that imported environment cannot find rc/mt; this is not a numerical failure.
Compute Sanitizer command: `compute-sanitizer --tool memcheck --error-exitcode
99 build-m4-cuda/m4_resident_test.exe`. Actual retained logs show all resident
tests passed and zero sanitizer errors. Profiling/tuning and frozen matched
complete-call measurements are recorded in [M4-benchmark.md](M4-benchmark.md).

# M2 frozen benchmark protocol

Protocol frozen before any M2 optimization claim, 2026-10-01. The executable
`benchmarks/m2_benchmark.cpp` fixes all six M1 basis families, seven terms,
topologies 16 -> 24 -> 8 and 64 -> 64 -> 32 -> 16, batch sizes 32 and 1024,
double storage/arithmetic, deterministic sine inputs, nonzero deterministic
coefficients, cosine biases, fixed upstream VJP and learning rate 0.001.
Jacobi alpha/beta 0.25/0.5, Fourier frequency 1.25, RBF width 0.65 and seven
uniform centers in [-1,1] are fixed in source. There is no RNG or data generation
inside the measured calls. The upstream includes 1/batch scaling.

Each backend starts from identical parameters, executes two warmup training
steps then seven measured steps, and retains the same evolving parameter trajectory.
The final pre-update output and gradients, and final post-update parameters,
are independently compared elementwise with CPU (2e-10 absolute-plus-relative
tolerance). Weighted checksums and maximum absolute discrepancy accompany each row.
Every measured sample is retained; median and interpolated interquartile range
are reported in milliseconds. Timer is steady_clock on the host and all GPU work
is synchronized before stopping it. No event or kernel-only time is a speedup metric.

- `cpu_full`: CPU Network forward, Network backward (which recomputes forward),
  and Network SGD, including their validation and host allocations.
- `m1_host_full`: the same public forward/backward/SGD workload using the M1
  Chebyshev host layer CUDA calls and CPU network SGD. Includes repeated device
  allocation, uploads/downloads, stream creation and synchronization. The explicit
  wrapper recomputes activations in backward to match CPU Network behavior.
- `m2_resident_full`: uploaded input/upstream and resident parameters; public
  forward, backward, SGD and synchronization including scalar status traffic and
  validation. Caches activations per the resident API. Full tensor downloads occur
  after timing. This row must not be described as a host transfer inclusive result.
- `m2_transfer_full`: the same M2 calls plus input/upstream uploads each step and
  output/all-gradient downloads, timed. Parameter download for independent final
  verification is outside timing, as CPU/M1 already hold parameters on the host.

Setup is measured separately once per row, including network construction and,
for M2, resident allocation, parameter/input/upstream uploads and synchronization.
Workspace allocation count is recorded and must remain constant. Setup times are
single observations, not robust statistics. Legacy comparisons are Chebyshev only;
unsupported M1 families are omitted, never silently run on CPU. CPU is the serial
reference, not a claim to compare against an optimized BLAS or other KAN library.

Reproduction: build Release CUDA with `KAN_BUILD_BENCHMARKS=ON`; run
`./scripts/benchmark-m2.ps1`. Baseline can be built with `KAN_BENCH_RESIDENT=OFF`.
Profiling invocations and matched before/after evidence will be recorded below.

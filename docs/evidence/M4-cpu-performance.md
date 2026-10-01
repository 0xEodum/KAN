# M4 CPU allocation refinement: matched complete-call evidence

2026-10-01. This approved M4 refinement implements
`M4-cpu-performance-plan.md`. A single private fixed-array numerical evaluator
is shared by Layer and the public owned-vector wrapper; mathematical recurrences,
relative guard, derivative checks in forward, underflow recovery and exceptions
are preserved. Layer uses its already validated owned configuration/parameters
and finite batch input. All array slots are initialized for portable return
semantics. No public interface or dependency changes.

Executable allocation RED `2f01c81` counts complete one-layer
forward/backward/SGD across widths 1/8/32 and batches 1/64. Before the change,
allocations rise from 12 to 262152; the required bound below 256 fails.
GREEN `c80b7d3` uses **8 allocations in every case**, with no sample/edge heap
growth. `scripts/build.ps1 -BuildDirectory build-m4-cpu -Test` passes **11/11**
CPU/package targets. Red/green logs are retained under `m4/cpu-allocation-*`.

The frozen fixture `m4/cpu_allocation_probe.cpp` measures complete network
forward/backward/SGD with topology 64->64->32->16, batch1024, rational degrees6/4,
double precision, fixed data/parameters and learning rate0.001. Each independent
run executes one warmup and three measured trajectory steps. An immutable
baseline executable linked before the source change and a final executable
linked at `c80b7d3` execute in balanced **baseline/final/final/baseline** order.
Both count allocations with the same global operator-new probe. Snapshot I/O is
outside timing. The machine is an Intel Core i5-12400, 6 cores/12 threads,
using MSVC Visual Studio18 Professional Release builds.

| Complete-call metric | Baseline CPU `4b808d2` | Final CPU `c80b7d3` |
|---|---:|---:|
| Pooled median, six measured samples | 4613.84455 ms | 1521.9724 ms |
| IQR | 18.0214 ms | 23.190925 ms |
| Heap allocations per step | 40894504 | 40 |
| Cumulative allocated bytes per step | 1804788788 | 5432372 |

The accepted complete-call median improves **67.0129%** (about3.03x).
These are CPU fixture-specific measurements, separate from GPU tuning claims.
One earlier corrected-source sequence overlapped a CUDA rebuild/sweep and was
discarded; the retained CSV files contain the subsequent isolated four runs.

All four runs serialize every final output, global/per-layer input VJP,
numerator/denominator/bias VJP and learned parameter after four trajectory steps.
Their **392416 doubles are bit-identical**, with maximum absolute error0 and
the same SHA256:
`eefedb1bcfa72bc93dd761021b9c39c3ca4d65e3bb5d8814fd1d3579bd5ed66b`.
Binary snapshots/executables remain locally in `build-m4-cpu`; the durable
manifest records their hashes and sizes, field order, exact source hashes,
fixture settings and every raw timing sample. The source reproduces snapshots
from either recorded CPU commit without requiring a GPU.

Artifacts: four `m4/cpu-{baseline,final}-{a,b}.csv` files,
`m4/cpu-performance-manifest.json`, `m4/cpu-performance-verification.log`,
`m4/cpu_allocation_probe.cpp` and the executable allocation test source.
Probe compile command (developer environment):
`cl /O2 /MD /EHsc /std:c++20 /I include docs/evidence/m4/cpu_allocation_probe.cpp <build>/kan.lib /Fe<probe>.exe`.
Run each executable as `<probe>.exe 1 3 <snapshot>.bin`.

This closes the CPU allocation refinement. Root-owned integrated bindings,
sanitizers/coverage and final independent acceptance must be replayed against
the final shared evaluator before M4 is marked DONE.

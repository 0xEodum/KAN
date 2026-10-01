# M4 independent acceptance review

2026-10-01. Reviewer inspected ROADMAP.md and the complete M4 plan, then reviewed
the typed rational CPU/Layer/Network/Python interfaces and resident implementation
independently of their implementation lanes. Review source includes CPU correction
`4b808d2` and corrected resident `c5a54e3`, followed by final CPU allocation
refinement `c80b7d3` and tuned resident `0071628`.

## Independent executable numerical review

The retained probe is `m4/independent_review.py`. Executed commands:

```powershell
.venv/Scripts/python.exe docs/evidence/m4/independent_review.py build-m4-python/python
.venv/Scripts/python.exe docs/evidence/m4/independent_review.py build-m4-final/python
.venv/Scripts/python.exe docs/evidence/m4/independent_review.py build-m4-install/python
```

Observed results are retained in `m4/independent_review_cpu.txt` and
`m4/independent_review.txt`, plus `m4/independent_review_installed.txt`. All pass
against the final numerical implementations. The CPU-only module reports the GPU
trajectory as False; the CUDA module reports True and executes those checks.
The probe does not merely replay the ordinary acceptance suites:

- 1,445 scalar checks independently construct power sums, quotient input
  derivatives and both parameter partial families, covering all 289 degree pairs
  0..16, five scaled inputs per pair and independently chosen coefficients.
- A rational -> shared trainable RBF -> rational network has all six input
  derivatives and 95 numerator/denominator/bias/center/log-width parameters
  independently central-differenced, preserving nonlinear carrier boundaries.
- Five GPU SGD steps compare output, all VJP families, all parameters, shared
  nonlinear state and explicit numerator L2 with CPU; allocations stay fixed.
- Scalar bindings reject float32, non-native endian, wrong rank, strided,
  deliberately unaligned and nonfinite arrays. Config, parameter and network
  snapshots own their values.
- Representable denominator VJPs survive zero and nonzero subnormal powers,
  degree 16 and negative z. Large finite Q preserves finite nonzero derivatives.
- A forward-safe backward-only overflow invalidates gradients/SGD; replacing
  the upstream permits recovery. A finite SGD candidate introducing a pole is
  committed atomically, then fails the next execution's guard as specified.
- The tuned warp reduction is independently challenged with batches
  1/31/32/33/37/0/41, capacity43 and a62-parameter final block tail. All data/L2
  VJPs match CPU across changed batch sizes; allocation counts remain unchanged.

## Findings and resolution

The substantive review finding was representable denominator derivatives lost
when a power underflows before multiplication by a large numerator. For example,
m=0,n=2,z=1e-200,P=1e300,Q=1 returned zero for db2 instead of -1e-100. A nonzero
subnormal degree-16 power similarly lost about 1.1e-5 relative accuracy before
amplification. Retained CPU/device RED tests precede the fixes. The rare log-space
paths now restore finite numerator/denominator/input derivatives, including
underflowed P/Q and explicit subnormal scale. See M4-cpu-underflow.md and the
actual-device RED/GREEN evidence. Independently rerun probes confirm the fix.

The typed rational edge is not a BasisKind or fixed linear expansion. Q's
constant is fixed at one; nonlinear b VJPs are returned and updated explicitly.
Horner magnitude recurrence implements the stated relative cancellation guard,
including zero numerator/upstream. Shapes, moved states, atomic setters and
network candidate SGD preserve the existing contracts. Resident caches and
candidate regions are allocated at construction, host math fallback is absent,
and state invalidation occurs before numerical execution. No remaining numerical,
API, ownership or lifecycle blocker was found in the reviewed source.

The final CPU refinement shares one private fixed-array numerical evaluator
between Layer and the public validated owned-vector wrapper. Configured spans
stay within17/16 slots, and all guards/derivatives still execute. Reviewer and
coordinator caught the need to initialize unused slots before a potentially
non-elided struct return; final `RationalTerms r{}` initializes all slots.
No duplicate mathematics or new installed-header dependency is introduced.

The tuned device kernel assigns a uniform parameter index to each full warp.
Its tail condition and grid-stride loop are uniform across all32 lanes, making
the full shuffle mask valid. Partial sample batches contribute zero in unused
lanes; n=0 never executes denominator division; only lane0 commits. The capped
launch has a retained1200002-parameter regression and the independent tail probe.

## Evidence gate cross-check

Final retained build records agree with M4-integration.md: CPU-only Python14/14,
AddressSanitizer11/11, GCC11/11 and integrated CUDA/Python19/19. The actual final
coverage summary reports574/585 lines98.1%,731/803 branches91.0%,39/39 functions
100%; rational.cpp is66/69 lines95.7%. Final actual-device Compute Sanitizer
reports7/7 M4 resident,5/5 resident,4/4 resident review and7/7 M3 resident, with
zero errors for each. Installed CPU package consumers and the actual installed
CUDA C++ consumer pass. The independent probe also passes using the installed
Python module, including its actual-device checks.

Benchmark source is the frozen M4 fixture and timed execution includes complete
forward/backward/SGD calls and status synchronization. Verification compares every
element of the final pre-update output/VJPs and final parameters against CPU.
The steady executor also uses an untimed identical evolving replay and requires
exact final parameter equality with the timed run. This verifies final-state
trajectory equivalence; the benchmark does not claim per-step tensor comparison.
Separate resident tests and the independent probe check short trajectories at
every step. No kernel-only speedup is accepted. Nsight Compute's actual log has
ERR_NVGPUCTRPERM; it cannot substantiate occupancy or counter-based utilization.

The actual Nsight Systems exports identify nonlinear parameter reduction as the
measured device bottleneck and show its cost falling from470.591ms to240.786ms
over81 calls; profiled times are not used for speedup claims. Independently
recomputed raw balanced samples agree with resident29.95135->20.78005ms
(-30.6207%) and transfer28.80385->20.88545ms (-27.4908%),14 samples per source/mode.
All compared VJPs/parameters meet the tolerance and allocation count remains2.
The frozen benchmark source has no diff against `73e3f47`.

The CPU allocation regression passes8 allocations for all tested layer
widths/batches. The matched network fixture drops40894504 allocations to40;
independent raw CSV/manifest recomputation agrees with4613.84455->1521.9724ms
(-67.0129%), six samples per source. All392416 output/VJP/final-parameter doubles
in four binary snapshots are finite and bit-identical. Source, CSV, executable
and snapshot hashes were independently checked. The earlier overlapping CPU
sequence was discarded; only the subsequent exclusive four runs substantiate
this claim. CPU and GPU tuning keep their separate recorded source provenance.

Independent raw checks also confirm M2/M3/M4 numerical sweeps76/36/36 rows, all
within tolerance with fixed resident allocations. The M4 matrix's overlapping
timings remain diagnostic only. `m4/independent_provenance.py` reproduces these
checks and verifies every recorded source/evidence/local-binary/trace hash and
size in the final manifest; its output is `m4/independent_provenance.txt`.

All numerical, ownership, lifecycle, installation, sanitizer, coverage and
matched-performance exit gates are satisfied. No blocking finding remains.
The completed manifest and its historical source blobs passed the independent
hash/size check. Independent acceptance is **PASS**. The coordinator must seal
the manifest again after committing the final reviewer log, then commit/push
the seal before ROADMAP is changed to M4 DONE. M5/M6 implementation remains
outside this review.

# M4 CPU allocation refinement and matched-call acceptance

2026-10-01. Live coordinator-requested allocation probing at CPU source
`4b808d2` identifies an avoidable M4 execution cost. On the frozen case11
64->64->32->16, degrees6/4, one complete forward/backward/SGD call uses
40,894,504 heap allocations at batch1024 (40,894,464 come from per-edge evaluator
vectors),1,804,788,788 cumulatively allocated bytes and an initial diagnostic
4617.11ms. Batch32 uses1,277,992 allocations. These timings are diagnostic,
not accepted before/after performance claims.

This refinement stays inside M4: use one private fixed-array rational evaluator
shared by Layer execution and the public owned-vector evaluator. Preserve all
existing values, derivatives, overflow/guard checks, owned public results and
interfaces. Layer may rely on its already validated owned parameters and input
to avoid repeated per-edge configuration/data validation. No new public API or
dependency. CPU numerical loops must not allocate heap storage per edge/sample.

Acceptance: retained executable allocation regression RED then GREEN (ordinary
bulk output/gradient/SGD storage is allowed; complete one-layer execution must
use fewer than256 allocations independently of batch/edge count), all existing
CPU/binding/ASAN/coverage gates and independent numerical review. Retain probe
source and counts. Freeze deterministic complete-call before/after fixture and
use balanced baseline/final/final/baseline samples on the same machine, verify
output/VJPs/learned parameters before accepting a CPU speedup. Baseline GPU tuning
comparisons finish using their unchanged CPU source before this refactor begins.
GPU timing acceptance and CPU allocation acceptance are recorded separately.

# Stage decision: review backlog B complete, M5 next

Date: 2026-10-22. Branch `cpp-foundation`. Follows the
[2026-10-01 stage decision](backlog-gate.md) that made the M1–M4 review backlog stage B
and gated roadmap M5 on it.

## Change

- ROADMAP stage **B: M1–M4 review backlog** goes from `NEXT` to `DONE`.
- Roadmap **M5: experimental quantum carriers** goes from `PLANNED` to `NEXT`; its
  dependency on B is satisfied.
- No scope, default or acceptance criterion changes. M5's scope stays as ROADMAP states
  it: typed PQC/Fock carrier interfaces and simulator adapters, physical normalization,
  measurement/gradient semantics and independently verified QUBO/quadratization, with no
  string hashing or symbolic parsing in numerical kernels. M5 has no detailed acceptance
  plan yet. As with M2–M4, the plan (scope, defaults, tests before implementation) is
  M5's first task and is recorded in `docs/evidence`. The planned source location is in
  [CONTRACT.md](../CONTRACT.md#extension-boundaries) (`src/carriers/quantum/`).
- `AGENTS.md` is unchanged. Rule 3 ("M5 must not start while any backlog item remains
  open") is now satisfied rather than removed.

## Basis

[BACKLOG.md](../BACKLOG.md) lists 29 items and all of them are closed. Each closure has a
verified fix, its evidence and an entry in the status journal:

- R1–R9 (code organization);
- M1–M7 (mathematics and trainability);
- C1–C13 (CUDA). C11 was closed by the owner. C13 was added on 2026-10-14 by owner
  decision.

The last item was R6 (source formatting), closed on 2026-10-22 with
[evidence](backlog/R6.md). It was the second and final item of pass 9 (M6 → R6). The
passes followed the gate's rules (at most three items per pass, P0 items first,
dependencies honoured), with two recorded exceptions: pass 4 began with R7 out of
priority order (owner-approved, to avoid conflicting with parallel C1 work), and pass 8
also took C13 after the owner added it, for four items.

## Carried forward (not blocking)

Owner questions raised by closed items remain recorded in their evidence and in the
journal. They do not reopen an item:

- C9: the `status_interval` default;
- C6–C8: the reciprocal-multiply and CPU-bound options;
- C13: the TF32 degree-1 deviation;
- M6: the C5 sparse-selection crossover, the refit workspace, per-input dictionary
  centers, the refit solver and sm_86 launch sizes.

The recorded follow-ups are also not open items: a `resident.cu` split (R4) and further
GEMM fusion (C3).

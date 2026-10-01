# Stage decision: review backlog gates M5

Date: 2026-10-01. Branch `cpp-foundation`.

## Change

The M1–M4 review (see [BACKLOG.md](../BACKLOG.md)) found carrier-type, trainability
and CUDA-efficiency debt. The repository owner decided that the whole backlog is
closed before M5 (quantum carriers) starts:

- ROADMAP gains stage **B: M1–M4 review backlog** with status `NEXT`.
- M5 moves from `NEXT` to `PLANNED` and depends on B.
- `AGENTS.md` rule 3 states that M5 must not start while any backlog item is open.

## Rationale

M5 adds a third nonlinear carrier family. Without R1–R3 it would extend the flat
`BasisConfig`, the `rational_` branch in `Layer` and the duplicated CPU/CUDA formulas.
The review recommends the same order: R → C1/C2 → M5.

## Execution rules

- At most three items per pass, chosen by priority and absence of open dependencies.
- An item is closed only after its tests and evidence pass; the closure is recorded in
  the BACKLOG status journal with a link to its evidence.
- Contract or default changes made while closing an item (for example R1 changing the
  public basis configuration type) are recorded in `docs/evidence`.

## First pass

C11 (Nsight Compute counter access) was closed by the owner. The first pass takes
R3 (single host/device formula source) and R1 (typed per-family basis configuration).
Neither depends on another open item. R2 depends on R1 and goes to the next pass.

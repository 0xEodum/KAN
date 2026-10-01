# Project Agent Instructions

Before doing any work in this workspace:

1. Read `docs\ROADMAP.md` completely. It is the source of truth for scope, ordering,
   architectural invariants, acceptance criteria, and the current handoff.
2. Verify the workspace and recorded evidence instead of assuming backlog facts
   are current.
3. Work only on the single `NEXT` stage, or on the nearest `READY` stage
   when no stage is in progress. Respect all declared dependencies.
4. Do not mark a stage `DONE` without satisfying its tasks, exit criteria, and
   evidence requirements.
5. Record scope, default, dependency, or acceptance-criteria changes in the
   `docs\evidence` directory.
6. When working on CUDA code, carry out profiling to identify bottlenecks and resolve them. 
   The GPU should be utilised to its full potential.
7. Commit work incrementally as it is completed: each finished, verified step
   (RED test, fix, evidence, backlog update) gets its own conventional commit
   (`feat:`, `fix:`, `test:`, `docs:` …) and is pushed to the tracked branch.
   Do not leave completed work uncommitted at the end of a session. Commit only
   files that belong to the step; regenerated `artifacts/` are committed only
   when recorded as evidence per `BACKLOG.md`.
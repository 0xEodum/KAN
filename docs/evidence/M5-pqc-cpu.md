# M5 phase 1: PQC carrier on the CPU (exact, adjoint and parameter-shift)

Date: 2026-10-23. Branch `cpp-foundation`, base `e423684` (M5 plan). Machine: GB10 (sm_121,
aarch64, 10 Cortex-X925 + 10 Cortex-A725 cores, 128 GB unified memory), CUDA 13.0.88, GCC 13.3,
`.venv` numpy 2.5.2, pybind11 3.0.2, clang-format 18.1.8, gcovr 8.6. Specification: the
[M5 plan](M5-plan.md), phase 1 (defaults D1-D13, the `PqcEdges` API, edge function, adjoint
order, normalization bound, RED list 1-10, exit criteria, isolation gates). Phase 2a (resident
PQC) and later phases are not started.

**Status: not complete.** The phase-1 review fixes R3 and R5
([2026-10-23 decision](roadmap-2026-10-23.md#4-plan-review-dispositions)) and the independent
review (below) are outstanding.

**Commit history (reconstructed).** The work was done in a working tree and not committed
when it was written; the "RED commit" and "GREEN commits" named below did not exist until
2026-10-23, when they were reconstructed on top of `d995853` (which includes the MSVC build
fix `a339745`): `test:` RED commit from the retained RED working tree, its quantum suites
rebuilt (CUDA parity + Python) and failing with exactly the assertions and messages of
[M5-pqc-cpu-red.log](M5-pqc-cpu-red.log); one `feat:` GREEN commit with the final code and
tests (the intermediate GREEN and performance steps described below were not committed
separately and cannot be recovered as commits); this evidence commit. The reconstructed tree
equals the original working tree except for `a339745` and the 2026-10-23 roadmap documents
(checked with `diff -r` against a backup of the working tree).

## Result

| | Change | Effect |
|---|---|---|
| API | `include/kan/pqc.hpp`: `Rotation`, `Entangler`, `PqcReadout`, `GradientMethod`, `PqcConfig`, `PqcEvaluation`, `validate_pqc`, `evaluate_pqc`, `pqc_state`, `pqc_terms`/`pqc_angles`/`pqc_encodings`; `PqcEdges` and `PqcGradients` appended to `Carrier` / `NonlinearGradients` (existing indices kept); `Layer(inputs, outputs, PqcConfig)`; `kan::set_pqc_parameters` (`kan/families.hpp`) | Existing carriers, results, 8/8 golden dumps and 292/292 kernels unchanged |
| Simulator | Integer gate tape (`src/carriers/quantum/tape.hpp`, at most 815 entries, structurally inert angles marked), shared host/device state-vector formulas (`pqc_formulas.hpp`): fused forward (one 2x2 matrix per qubit and layer), Born readouts, gate-by-gate adjoint | Dense Kronecker-product reference: amplitudes 6.8e-16, expectations 1.2e-15; unit norm within 0.04 of `(4G + 2^n) 2^-52` |
| Gradients | Adjoint (default, the layer backward) and parameter shift (`evaluate_pqc`) for every angle, encoding weight and bias and the input | Agree within 1.4e-15; both match central FD within `1e-7 |fd| + 1e-9` (library and reference) |
| CPU loops | `src/backends/cpu/pqc_edges.cpp`: per call one tape, per edge stack arrays, 8-sample vector lanes (`lanes.hpp`) bitwise equal to one sample at a time | 13 allocations per full call at every width and batch; forward 1.4-1.9x, backward 1.3-1.8x the first version |
| Initializers | D8: coefficients, then angles `pi (2u - 1)`, moments per readout/entangler; pinned digest `c8bec86f1c294f6f` | `E[y^2]/target` 1.03 (n = 1), 1.00 (n = 2); existing digests unchanged |
| D13 stubs | Resident construction rejects PQC layers (`std::invalid_argument`, every precision); read-only `kan.PqcEdges` snapshot class, no Python construction; `shots > 0` rejected | Both CUDA builds and the Python suites green |
| D3 | CTest `quantum_source_check` (the plan's regular expression over the quantum sources; reference headers include standard headers only) | 4 sources clean; the pattern is self-tested on 13 forbidden and 6 allowed snippets |

## Implementation notes

- **Tape** (D3). `compile_tape(PqcConfig, Tape&)` writes `{op, axis, qubit, other, slot, inert}`
  entries in application order: `W_0`, then per layer `E`, `S_l` (q ascending), `W_l` (q ascending,
  axis 0 first). The tape is compiled once per call into a stack aggregate (CPU), never stored
  in the carrier. Inert marks: a Z rotation while its qubit is |0> up to a phase (only Z
  rotations and CZ before it on the qubit; a CNOT changes the target's state only when its control
  has left |0>), or a Z rotation after which its qubit sees only Z rotations, CZ and CNOTs it
  controls (Z_q commutes with those and with the diagonal readouts). Inert slots get exactly zero
  derivatives on both paths and are skipped. The reverse pass ends at the first entry with a
  derivative (`first_derivative`; the remaining gates only rebuild a discarded state).
- **Forward (fused).** Per edge the block matrices `W_(l,q)` (2x2 complex, `W <- R_k W`); per
  sample `W_(0,q)`, then per layer the entangler and `M_(l,q) = W_(l,q) R_enc(phi)` (the product
  specialized per encoding axis), one state pass each. This is the tape's unitary (gates on
  different qubits commute) with fewer roundings; it is the only forward (layer, `evaluate_pqc`,
  `pqc_state`, the backward's state). The parameter-shift path rebuilds the shifted slot's block.
- **Adjoint.** As the plan's pseudocode, gate by gate from the last gate: for a parametrized,
  non-inert gate `d = Im <lambda|sigma|psi>` over the pairs in ascending order, then `U^dagger`
  on psi and on lambda; CNOT/CZ on both. `lambda = (u o_k) psi_k` with the diagonal observable
  `o_k = sum_t sign_t(k) c_t`. Nothing is stored per gate; states are recovered by inversion.
- **Sample lanes.** `kan::detail::Lanes<8>` (fixed arrays with element-wise operators) is the
  amplitude scalar of the shared formulas on the CPU; every lane executes the scalar operation
  sequence (no reassociation, `-ffp-contract=off`), and lane results are accumulated in
  ascending sample order, so results are bitwise those of a one-sample loop with the same
  accumulation order (the 12 bench checksums are identical before and after the lane change).
- **Overflow.** An encoding angle `w x + b` that overflows makes the state nonfinite:
  `Layer::forward/backward` raise `std::overflow_error` (the layer's result checks),
  `evaluate_pqc` and `pqc_state` check their results likewise.

## Deviations from the plan and additions (AGENTS rule 5)

1. **Fused forward instead of the gate-by-gate forward** of the adjoint pseudocode's first line
   (`psi <- U_G ... U_1 |0...0>`). Same unitary, different (fewer) roundings; the reverse pass is
   gate by gate as specified. Reason: measured, the single-qubit passes are the cost (each a load
   and store of the whole state); fusing B + 1 passes into one gave 1.4x forward at n = 4 (table
   below). CONTRACT states the fused order. Phase 2a measures the FP32 inversion drift against
   this forward.
2. **Closed-form tolerance (RED test 2).** "1e-15 absolute" holds up to r = 2 (5 rotations; measured
   worst 6.7e-16 over 2000 trials). For the r = 5 case (11 rotations) the library's worst error
   against a long-double `cos` was 1.18e-15, so that case is held to `G 2^-52` (2.4e-15), and the
   test's own reference is now summed in long double (its double sum alone was off by up to
   3.5e-15). Recorded in the GREEN commit; the other closed forms keep 1e-15.
3. **Inert derivatives are exactly zero on the parameter-shift path too** (the plan requires it for
   the adjoint; the shift path skips the same structurally zero slots). Finite differences of
   those slots are checked against 0.
4. **Additional public functions:** `kan::pqc_state` (the final amplitudes, needed by the Born-rule
   and dense-reference tests through the public API, and later by the Python NumPy reference) and
   `kan::pqc_terms`, `pqc_angles`, `pqc_encodings` (layout counts). `set_pqc_parameters` is in
   `kan/families.hpp` with the other family setters. `evaluate_pqc` and `pqc_state` raise
   `std::overflow_error` for nonfinite results.
5. **Python stub** (D13): a minimal read-only `kan.PqcEdges` snapshot class (coefficients) that
   cannot be constructed; nothing can produce a PQC layer in Python before phase 7, so the test
   checks the class, the absence of `PqcConfig` and that existing carriers are unaffected.
6. **RED-test amendments in the GREEN commits** (all reasons above or test defects): the Z-encoding
   constant edge is compared within 4 ulp of 1 (`|phase|^2 = c^2 + s^2` rounds); the overflow test
   uses the constructor circuit (`m = 1`, random angles let terms cancel); n = 4 and n = 5 layer
   cases were added to the layer FD test after coverage showed the layer loops for those qubit
   counts unexercised; the digest placeholder of the RED commit was pinned.

## Validation

| Check | Result |
|---|---|
| RED ([M5-pqc-cpu-red.log](M5-pqc-cpu-red.log), CUDA parity build of the RED commit) | `pqc` 0/8, `pqc_layer` 1/13 (L2, consistent with zero stubs), `quantum_source_check` 2/3 (sources missing), `pqc_allocation` passes (stub), `pqc_resident_stub` and `pqc_stub_python` pass (the D13 stubs are final in the RED commit) |
| New suites | `pqc` (8: validation, evaluation validation and overflow, closed forms, 200-circuit dense reference, Born/normalization up to r = 32, adjoint vs shift vs FD, FD of the reference, inert zeros), `pqc_layer` (13: construction and `cos(r x)`, symmetric point, forward order, layer FD with/without residual for n = 1..5, finiteness and overflow, set_carrier/setter atomicity, SGD atomicity, L2, mixed Chebyshev -> TanhMap -> PQC -> wavelet network FD, draw order and moments, pinned digest, VarianceScaling, deterministic training), `pqc_allocation` (1), `quantum_source_check` (3), `pqc_resident_stub` (2), `pqc_stub_python` (3) |
| CTest CPU (`build-cpu`, with Python) | 42/42 |
| CTest CUDA parity / FMA | 66/66 / 66/66 |
| Reconstructed head (`1ae17a8`, GB10, 2026-10-23) | CTest CPU 41/41, CUDA parity 65/65, FMA 65/65 (the optional format CTest is not registered when clang-format is not on PATH; `scripts/check_format.sh` with the pinned 18.1.8: 112 files, exit 0); SASS of `resident.cu` against `a339745`: 292/292 identical in both builds; golden: the 8 hashes below reproduced |
| RTX 3090, MSVC 19.50 + CUDA 13.1 ([port-windows-3090](port-windows-3090.md#tests-final-tree)) | Same tree content: CPU-only all quantum suites pass, including the `long double` closed-form reference (binary64 under MSVC) and the pinned digest; CUDA parity 63/63 without the `package_*` suites; PQC CPU throughput (i5-12400) 1.4-2.4x below the Cortex-X925 forward |
| ASan (`build-asan`, CPU suites) | 25/25; the new suites clean; `residual_training` and `initializer_training` (unchanged 3000-epoch demonstrations) exceed the 900 s CTest limit under ASan and pass with `--timeout 5400` |
| Coverage (GCC, `--merge-lines`: template instantiations merged) | `pqc_edges.cpp` 127/127, `pqc_formulas.hpp` 141/141, `tape.hpp` 69/69, `lanes.hpp` 21/21, `pqc.cpp` 124/125 (line 93, the result's initializer, which gcov attributes to an exception path) ([coverage.txt](m5/pqc-cpu/coverage.txt)); library 98% (2684/2715, without the two long demonstrations) |
| Format | `scripts/check_format.sh` exit 0 |
| SASS ([sass-identity.txt](m5/pqc-cpu/sass-identity.txt), C5 `sass_cmp.py`, base `e423684`) | `resident.cu`: 292/292 kernels identical in both builds, none added or removed; `cuda.cu`: no kernels |
| Golden ([golden-sha256.txt](m5/pqc-cpu/golden-sha256.txt)) | all 8 dumps byte-identical to the [C13 reference](backlog/C13-bench/golden/sha256-linux.txt) |
| CPU-only build | `KAN_ENABLE_CUDA=OFF`; `libkan.a` has no CUDA, cuBLAS or cuStateVec symbol |
| Existing initializer digests | unchanged (`initializer` suite) |
| Frozen suites ABBA (base `e423684` vs final, same session) | resident_full geomean 1.001 (basis), 0.998 (local), 0.999 (rational); cpu_full 0.991-0.999; 0 checksum mismatches (below) |

### Measured numbers

- Dense reference (200 random circuits, n <= 4, r <= 4, all axes, entanglers 95/47/31/27, both
  readouts): largest amplitude difference 6.8e-16, expectation 1.2e-15 (bound 1e-13).
- Normalization (120 circuits up to n = 5, r = 32): largest `|sum p - 1|` is 0.037 of
  `(4G + 2^n) 2^-52`; every `p_k <= 1 + bound`.
- Adjoint vs parameter shift (40 random circuits, n <= 3, r <= 3): largest difference 1.4e-15
  (largest derivative 0.98).
- Initializer moments (`E[y^2]/(gain^2/3)`, 16 x 16 layer, 256 samples, 16 seeds): asserted n = 1
  r = 4: 1.031, n = 2 r = 4: 0.999; recorded n = 1 r = 1: 1.030, r = 2: 1.009; n = 2 r = 2: 1.016;
  n = 2 r = 4 Parity: 0.975; n = 2 r = 4 CnotChain: 1.057; n = 3 r = 4 CnotRing: 1.019; n = 3
  r = 4 CzChain Parity: 0.903.
- Training toy (PQC 1 -> 4, n = 1, r = 3, Chebyshev readout, 3000 full-batch SGD steps on 64
  points): holdout MSE 1.77e-1 -> 2.98e-5 (target variance 2.15e-1; threshold 1% of it, fixed in
  the RED commit), bitwise identical over two runs.
- Allocation probe: 13 allocations for forward + backward + SGD at widths 1, 8, 32 and batches 1,
  64 (n = 3, r = 4, CNOT ring).

### CPU throughput

[pqc_cpu_bench.cpp](m5/pqc-cpu/pqc_cpu_bench.cpp), one thread pinned to a Cortex-X925 core
(`taskset -c 19`; the A725 cores run `sincos` 2.7x slower), width x width layers, batch 256,
CnotChain for n > 1, median of 7 calls. "Circuits" are edge-sample evaluations
(`width^2 batch`); backward is `Layer::backward` (it recomputes the forward state). CSVs:
[first version](m5/pqc-cpu/cpu-first.csv) (gate-by-gate, one sample at a time),
[lanes](m5/pqc-cpu/cpu-lanes.csv), [fused](m5/pqc-cpu/cpu-fused.csv),
[final](m5/pqc-cpu/cpu-final.csv).

| n | r | width | forward Mcircuits/s first -> final | backward first -> final | backward / forward time |
|---|---|---|---|---|---|
| 1 | 2 | 16 | 26.01 -> 40.18 (1.54x) | 11.56 -> 19.48 (1.69x) | 2.06 |
| 1 | 4 | 16 | 12.40 -> 22.48 (1.81x) | 6.42 -> 11.27 (1.76x) | 1.99 |
| 2 | 2 | 16 | 9.46 -> 17.38 (1.84x) | 4.55 -> 6.32 (1.39x) | 2.75 |
| 2 | 4 | 16 | 5.31 -> 9.84 (1.85x) | 2.56 -> 3.60 (1.40x) | 2.74 |
| 4 | 2 | 16 | 2.99 -> 4.18 (1.40x) | 0.86 -> 1.12 (1.30x) | 3.73 |
| 4 | 4 | 16 | 1.72 -> 2.49 (1.45x) | 0.49 -> 0.64 (1.32x) | 3.87 |
| 1 | 2 | 64 | 25.75 -> 39.97 (1.55x) | 11.36 -> 19.88 (1.75x) | 2.01 |
| 1 | 4 | 64 | 12.34 -> 21.96 (1.78x) | 6.29 -> 11.18 (1.78x) | 1.96 |
| 2 | 2 | 64 | 9.30 -> 16.98 (1.83x) | 4.48 -> 6.26 (1.40x) | 2.71 |
| 2 | 4 | 64 | 5.23 -> 9.72 (1.86x) | 2.55 -> 3.56 (1.40x) | 2.73 |
| 4 | 2 | 64 | 2.96 -> 4.15 (1.40x) | 0.86 -> 1.12 (1.30x) | 3.72 |
| 4 | 4 | 64 | 1.70 -> 2.47 (1.45x) | 0.49 -> 0.64 (1.32x) | 3.85 |

Profiling (perf is unavailable: `perf_event_paranoid = 4`; ablation instead): `sincos` costs 4.7 ns
on an X925 core (GCC already merges `cos`/`sin` of one argument), about 9 ns of the 39 ns per
n = 1, r = 2 circuit of the first version; the rest was per-gate interpretation overhead (one-pair
states). At n = 4 a gate costs a load and store of the whole state, about at the throughput limit
of the gate kernel (isolated: 27 ns per 8-lane Y rotation). Steps, each measured on the same
bench (CSVs above) and kept only when faster:

1. 8-sample lanes (bitwise identical): n = 1 forward 1.4-1.8x, n = 2 1.6x; n = 4 1.0x. Inlining
   all three axis variants into one interpreter made the lane kernel 2x slower than in isolation;
   per-axis out-of-line host kernels (`KAN_PQC_GATE`, inlined on device) recovered part of it.
2. Fused forward with a generic `W S` product: n = 4 forward +35%, but n = 1 regressed (the generic
   product multiplies structural zeros); specializing `W R_enc` per axis: n = 1 recovered to 1.54x
   the first version, n = 2 1.84x, n = 4 1.40x.
3. A single-pass fused adjoint step (derivative and both inverse rotations in one loop) was
   bitwise identical but 20-25% slower for n >= 2 (64 live doubles per pair exceed the 32 NEON
   registers); reverted to three passes per gate.
4. Ending the reverse pass at the first gate with a derivative: 1-3%, bitwise identical.

The checksums of the final version differ from the first version by at most 2.1e-15 (the fused
forward's rounding). Backward remains 2-3.9x the forward: per gate an inner product and two
inverse rotations, plus the recomputed forward.

### Frozen suites

`basis` and `local` ABBA (base, final, final, base), `rational` one pair, `--backend all`, pinned
to cores 15-19 ([frozen/](m5/pqc-cpu/frozen/), `compare_bench.py`; GPU state in
[gpu-state.txt](m5/pqc-cpu/frozen/gpu-state.txt)):

| suite | backend | cases | geomean (final / base) | min | max |
|---|---|---|---|---|---|
| basis | cpu_full | 24 | 0.991 | 0.980 | 1.003 |
| basis | legacy_full | 4 | 1.005 | 0.996 | 1.019 |
| basis | resident_full | 24 | 1.001 | 0.969 | 1.100 |
| basis | transfer_full | 24 | 1.001 | 0.988 | 1.016 |
| local | cpu_full | 12 | 0.999 | 0.993 | 1.003 |
| local | resident_full | 12 | 0.998 | 0.982 | 1.007 |
| local | transfer_full | 12 | 0.996 | 0.981 | 1.009 |
| rational | cpu_full | 12 | 0.993 | 0.972 | 1.015 |
| rational | resident_full | 12 | 0.999 | 0.980 | 1.005 |
| rational | transfer_full | 12 | 1.001 | 0.987 | 1.011 |

0 checksum mismatches. The existing CPU and resident paths are unchanged in code (one more
variant alternative in the per-call dispatch); the ratios are run-to-run noise.

Reproduce (repository root, Linux; E = docs/evidence/m5/pqc-cpu):

```bash
cmake --build build-cpu -j20 && ctest --test-dir build-cpu          # likewise build-cuda, build-cuda-fma
ctest --test-dir build-asan -L "quantum|carriers|core|init|local|rational" -E format
g++ -std=c++20 -O3 -ffp-contract=off -Iinclude $E/pqc_cpu_bench.cpp build-cpu/libkan.a -o /tmp/pqcb && taskset -c 19 /tmp/pqcb 7
cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DKAN_ENABLE_COVERAGE=ON && cmake --build build-coverage -j16
ctest --test-dir build-coverage -L cpu -E "format|residual_training|initializer_training|package"
python -m gcovr --root . --object-directory build-coverage --filter src/ --merge-lines --txt
scripts/golden_dumps.sh <out> parity=build-cuda fma=build-cuda-fma
python3 docs/evidence/backlog/C5-bench/sass_cmp.py <e423684 resident.cu.o> build-cuda/CMakeFiles/kan_cuda.dir/src/backends/cuda/resident.cu.o
```

## For phase 2a (PQC on the resident executor)

- The tape (`Tape`, `TapeEntry`, at most 815 entries, plus the per-layer entangler list, block
  axes and `first_derivative`) and every formula in `pqc_formulas.hpp` are `KAN_HOST_DEVICE`
  templates on the qubit count and the scalars; gate kernels are out of line on the host and
  `KAN_FORCE_INLINE` on the device (`KAN_PQC_GATE`). `compile_tape` is host code (it reads
  `std::vector` block axes); cache the tape in device memory at construction (D3).
- Per-edge data the kernels need once per call: angle half-angles and the block matrices
  `W_(l,q)` (`block_matrix`), the diagonal observable `o_k` (backward). Per sample: encoding
  half-angles, `times_rotation`, `apply_matrix`, the reverse pass (`run_adjoint`).
- The D13 stub is `make_plan(const Layer&, const PqcEdges&, std::size_t)` in `resident.cu`
  (throws); `pqc_resident_stub_test` and `pqc_stub_python_test` are to be replaced by the
  phase-2a and phase-7 suites.
- FP32 inversion drift (plan phase 2a test 5) is to be measured against this fused forward.
- CPU accumulation orders the device must match within the parity tolerances: forward ascending
  i then t; parameter VJPs ascending b; `dx[b,i]` ascending o; derivative sums over pairs
  ascending.

## Independent review

Pending. Next steps: the R3 and R5 review fixes as `fix:`/`test:` commits, then a fresh
independent reviewer over the source and this evidence; findings and resolutions are recorded
here before phase 1 is marked complete in the ROADMAP.

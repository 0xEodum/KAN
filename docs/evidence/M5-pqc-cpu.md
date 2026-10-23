# M5 phase 1: PQC carrier on the CPU (exact, adjoint and parameter-shift)

Date: 2026-10-23. Branch `cpp-foundation`, work base `e423684` (M5 plan), commit base `d995853`
(see the commit history below). Machine: GB10 (sm_121,
aarch64, 10 Cortex-X925 + 10 Cortex-A725 cores, 128 GB unified memory), CUDA 13.0.88, GCC 13.3,
`.venv` numpy 2.5.2, pybind11 3.0.2, clang-format 18.1.8, gcovr 8.6. Specification: the
[M5 plan](M5-plan.md), phase 1 (defaults D1-D13, the `PqcEdges` API, edge function, adjoint
order, normalization bound, RED list 1-10, exit criteria, isolation gates). Phase 2a (resident
PQC) and later phases are not started.

**Status: complete (2026-10-23).** The phase-1 review fixes R3 and R5
([2026-10-23 decision](roadmap-2026-10-23.md#4-plan-review-dispositions)) are made
([below](#review-fixes-r3-and-r5)); the independent review (two passes, [below](#independent-review))
found nothing blocking, and its findings are resolved or recorded.

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
| Gradients | Adjoint (default, the layer backward) and parameter shift (`evaluate_pqc`) for every angle, encoding weight and bias and the input | Agree within a gate-count-scaled bound (measured 1.443e-15, 0.028 of it, n <= 3, r <= 3; 9.7e-15, 0.010 of it, up to n = 5, r = 32); both match central FD within `1e-7 |fd| + 1e-9` (library and reference, shallow circuits) |
| CPU loops | `src/backends/cpu/pqc_edges.cpp`: per call one tape, per edge stack arrays, 8-sample vector lanes (`lanes.hpp`) bitwise equal to one sample at a time | 13 allocations per full call at every width and batch; forward 1.4-1.9x, backward 1.3-1.8x the first version |
| Initializers | D8: coefficients, then angles `pi (2u - 1)`, moments per readout/entangler; pinned digest `c8bec86f1c294f6f` | `E[y^2]/target` 1.03 (n = 1), 1.00 (n = 2); existing digests unchanged |
| D13 stubs | Resident construction rejects PQC layers (`std::invalid_argument`, every precision); read-only `kan.PqcEdges` snapshot class, no Python construction; `shots > 0` rejected | Both CUDA builds and the Python suites green |
| D3 | CTest `quantum_source_check` (the plan's regular expression over the quantum sources; reference headers include standard headers only) | 4 sources clean; the pattern is self-tested on 13 forbidden and 6 allowed snippets |

## Implementation notes

- **Tape** (D3). `compile_tape(PqcConfig, Tape&)` writes `{op, axis, qubit, other, slot, inert}`
  entries in application order: `W_0`, then per layer `E`, `S_l` (q ascending), `W_l` (q ascending,
  axis 0 first). The tape is compiled once per call into a stack aggregate (CPU), never stored
  in the carrier. Inert marks: a Z rotation while its qubit is |0> up to a phase (before it on
  the qubit only Z rotations, CZ, CNOTs it controls, and CNOTs on it as the target only of a
  control in |0>: a CNOT changes the target's state only when its control has left |0>), or a Z
  rotation after which its qubit sees only Z rotations, CZ and CNOTs it
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
   test's own reference phase is summed in double-double, as plan review finding 22 specified (its
   double sum alone was off by up to 3.5e-15). Recorded in the GREEN commit; the other closed forms
   keep 1e-15. The GREEN commit summed in `long double`, which is binary64 under MSVC; review
   finding F1 replayed that with a binary64 reference (seed 11, r = 5: 2.37e-15 against 2.44e-15;
   over 20000 trials 2/104/49 failures at r = 1/2/5). Since `e59f217` the reference uses a
   test-local TwoSum double-double (no `long double`, independent of the library) and
   `cos(hi) - sin(hi) lo`; a probe over 2000 seeds x 20 trials gave worst errors 6.66e-16 /
   8.88e-16 / 1.443e-15 at r = 1 / 2 / 5, identical to a binary128 reference, 0 failures
   (binary64 reference: 6 / 191 / 104 failures).
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
   uses the constructor circuit (`m = 1`, random angles let terms cancel); an overflowing
   encoding angle `w x + b` (`steep`, w = 10, x = 1e308) raises `std::overflow_error` from
   `Layer::forward` and `Layer::backward` and leaves the layer unchanged; three
   `std::overflow_error` assertions were added to `pqc_evaluation_validates_parameters`
   (`evaluate_pqc` adjoint, parameter shift and `pqc_state` with w x = 1e300 x 1e300); n = 4 and n = 5 layer
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
| Review-fix head (`m5p1-r3r5` at `9684236`, code and tests; later commits documentation only; GB10, 2026-10-23) | CTest CPU 42/42 including the registered format test (clang-format 18.1.8 on PATH); CUDA parity build (`KAN_ENABLE_CUDA=ON`, Release): `ctest -L quantum` 6/6, full suite 66/66 serially. Under `ctest -j4`, `resident_fused_expansion` failed once and passed when rerun: it measures device free memory with `cudaMemGetInfo`, so concurrent GPU tests perturb it. No CUDA or library code changed. `scripts/check_format.sh`: 112 files, exit 0. `quantum_source_check` 3/3, now including `lanes.hpp` (5 sources clean). Printed ratios: symmetric-point shift 0 of its bound; adjoint vs shift 0.028 (shallow) and 0.010 (deep) of theirs |
| RTX 3090, MSVC 19.50 + CUDA 13.1 ([port-windows-3090](port-windows-3090.md#tests-final-tree)) | Same tree content: CPU-only all quantum suites pass, including the `long double` closed-form reference (binary64 under MSVC) and the pinned digest; CUDA parity 63/63 without the `package_*` suites; PQC CPU throughput (i5-12400) 1.4-2.4x below the Cortex-X925 forward (this run predates `e59f217`: the closed-form reference was still `long double`). At the phase-1 head with the review S1 change, MSVC CPU-only `ctest -L quantum` 5/5: symmetric point exactly 0 (as on glibc), adjoint vs shift 1.110e-15 (0.028 of the bound) and deep 7.772e-15 (0.014), dense reference 8.951e-16 / 1.554e-15, normalization 0.043 of the bound, pinned digest and moments as on GCC |
| ASan (`build-asan`: RelWithDebInfo, `-DKAN_ENABLE_ASAN=ON`, CPU suites) | 25/25; the new suites clean; under Debug + ASan `pqc_layer` takes about 224 s (review F4) and needs `--timeout`; `residual_training` and `initializer_training` (unchanged 3000-epoch demonstrations) exceed the 900 s CTest limit under ASan and pass with `--timeout 5400` |
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
- Adjoint vs parameter shift (40 random circuits, n <= 3, r <= 3): largest difference 1.443e-15,
  0.028 of the gate-count-scaled bound (largest derivative 0.98); 16 deep random circuits (four at
  n = 5, r = 32, one per entangler; G up to 815): 9.7e-15, 0.010 of the bound (see R3 below).
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
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKAN_ENABLE_ASAN=ON && cmake --build build-asan -j16
ctest --test-dir build-asan -L "quantum|carriers|core|init|local|rational" -E format   # Debug + ASan: add --timeout 600
g++ -std=c++20 -O3 -ffp-contract=off -Iinclude -Isrc $E/tape_bench.cpp build-cpu/libkan.a -o /tmp/tapeb && taskset -c 19 /tmp/tapeb 7
g++ -std=c++20 -O3 -ffp-contract=off -Iinclude $E/pqc_cpu_bench.cpp build-cpu/libkan.a -o /tmp/pqcb && taskset -c 19 /tmp/pqcb 7
cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DKAN_ENABLE_COVERAGE=ON && cmake --build build-coverage -j16
ctest --test-dir build-coverage -L cpu -E "format|residual_training|initializer_training|package"
python -m gcovr --root . --object-directory build-coverage --filter src/ --merge-lines --txt
scripts/golden_dumps.sh <out> parity=build-cuda fma=build-cuda-fma
python3 docs/evidence/backlog/C5-bench/sass_cmp.py <e423684 resident.cu.o> build-cuda/CMakeFiles/kan_cuda.dir/src/backends/cuda/resident.cu.o
```

## Review fixes R3 and R5

Branch `m5p1-r3r5` from `f4e3c6e`. Test and evidence changes only; no library code changed
(`src/carriers/quantum/tape.hpp` changed in a comment only, review F6).

### R3: parameter-shift near-zero tolerance

**Derivation** (comment at `shift_zero_bound`, `pqc_layer_test.cpp`; u = 2^-52, G the tape
length `(r+1) n B + r n + r * entangling`). The shift result is
`d = (m(fl(p + q)) - m(fl(p - q))) / 2`, `q = fl(pi/2)`, at a point where the exact derivative
is 0.

1. In one parameter, `m(p) = a + b cos p + c sin p` (the rotation enters through `cos p/2` and
   `sin p/2` on both sides), so `m(p + s) - m(p - s) = 2 sin(s) m'(p)`, which is 0 for every s.
   The inexact q costs nothing.
2. Argument rounding: `fl(p +- q) = p +- q + eta`, `|eta| <= (u/2)(|p| + q)`, and
   `|m'| <= sqrt(b^2 + c^2) <= 1` because `|m| <= 1`. This contributes at most `(u/2)(|p| + 2)`.
3. Each expectation's rounding: the state error is `|e| <= 2 G u` (the per-gate budget behind
   the normalization bound `(4G + 2^n) u`), so `|m^ - m| <= 2|e| + |e|^2 + 2^n u <= (4G + 2^n) u`
   with `|M| <= 1`. Half the difference of two such errors is no larger.
4. The final subtraction and halving add at most `u/2 |d|`.

Total: **`|d| <= (4G + 2^n + |p| + 2) 2^-52`**. Step 3's `2 G u` is an assumed first-order
per-gate budget (the one behind the normalization bound), not a proven forward-error bound for
the fused 2x2 complex products; the bound, like the adjoint-vs-shift bounds below, is a
first-order estimate calibrated by measurement (worst measured ratios 0, 0.028 and 0.010:
35-100x margin; review S2). This matches the brief's starting point, which
was checked here rather than assumed.

**Cases and outcome.** `pqc_symmetric_point_derivatives` now covers 19 circuits at the
constructor point (Y encoding, `c = 1`): the 3 shallow ones, plus `n = 3, r = 8` and the D9 limit
`n = 5, r = 32` with block ZYZ, every entangler and both readouts. Adjoint Z derivatives are still
asserted exactly 0.

- **Not RED.** The deep cases pass the old fixed `4 * 2^-52` (commit `1ec8fce`). Every non-inert
  Z-angle shift derivative is exactly 0, so the worst ratio to `4 * 2^-52` is 0, and all other
  assertions hold at depth.
- **Why depth does not matter at this point.** With the Z angle at 0, the shifted arguments are
  exactly `+-q`, and every other gate (Y rotations, Y encoding, CNOT, CZ) is real. So the `+q` and
  `-q` states are bitwise complex conjugates: negation is exact, `-ffp-contract=off` holds, and
  glibc `cos` is even and `sin` odd bitwise. The two Born readouts are therefore identical. The
  review's premise ("passes only because shallow") does not hold here.
- **The bound is still replaced** (`dc1e792`), because that bitwise symmetry rests on libm and
  compiler properties IEEE 754 does not guarantee (MSVC, device `sincos` in phase 2a). The worst
  ratio `|d| / bound` is printed and is 0.
- **Rejected alternative: Z angles at `fl(k pi)` with random Y angles.** It would make the old
  bound fail (up to 4.6x), but it is not a valid RED. `fl(k pi) - k pi` moves the point off the
  symmetry, and the adjoint shows the same magnitudes there (`adj/bound ~ shift/bound`, 2e-3 to
  2.5e-2). Those values are true derivatives, not shift rounding.

**Adjoint vs parameter shift** (`pqc_test`, `f2b6801`). The fixed 1e-12 is not derivable for
deep circuits: the input derivative `sum_j w_j d_j` scales with `sum |w_j|`, up to about 130 at
r = 32. It is replaced by first-order per-quantity bounds:

| Quantity | Bound |
|---|---|
| Angles, biases | `(12G + 2^(n+1) + |p| + 2) u`: the shift bound plus the adjoint's `(8G + 2^n) u` (forward state within `2Gu`; the reverse pass recovers psi and lambda through at most G more inverse gates, each within `4Gu`; then a sum over `2^(n-1)` pairs) |
| Weights | `|x| (bound + u)` |
| Input | `sum |w_j| (max bound + 2 E u)` |

A new suite runs 16 random deep circuits: four at `n = 5, r = 32`, one per entangler, with G up
to 815. It has no finite differences, because an expectation rounding of about 1e-15 divided by
2h is about 5e-10, at the 1e-9 floor of the FD rule. The FD checks stay shallow and away from
symmetry points.

| Check | Worst \|difference\| | Ratio to its bound | Old bound |
|---|---|---|---|
| Symmetric point, shift Z derivatives (19 circuits, up to G = 815) | 0 | 0 | 4·2^-52: ratio 0 |
| Adjoint vs shift, 40 random circuits n <= 3, r <= 3 | 1.443e-15 | 0.028 | 1e-12 |
| Adjoint vs shift, 16 deep random circuits (G up to 815) | 9.7e-15 | 0.010 | 1e-12 (passes; not derivable) |

No measured worst comes within 10x of its bound.

### R5: tape preparation measured separately

The tape is compiled once per `Layer::forward` / `Layer::backward` call. Per edge, the angle
half-angles and block matrices are formed once (`prepare_edge`); this work is parametric and
cannot be shared across edges.

[tape_bench.cpp](m5/pqc-cpu/tape_bench.cpp) times:
- `compile_tape`: loops of 20000 calls;
- a labelled replica of `prepare_edge`'s two loops (the original is in an anonymous namespace):
  sweeps over the layer's edges of at least 20000 edges;
- complete calls.

Setup: the configurations of `cpu-final.csv` plus the D9 limit (n = 5, r = 32, ZYZ, CnotRing), at
width 1, 16 and 64 and batch 1 and 256. One thread on core 19 (Cortex-X925), medians of 7, three
runs ([tape-prep.csv](m5/pqc-cpu/tape-prep.csv), column `run`). The run-to-run spread of the
forward medians is at most 6.2% (width 1, batch 1; at most 3.3% elsewhere), and the checksums are identical across runs and equal to
`cpu-final.csv`.

Preparation share = `(compile_tape + width^2 prepare_edge) / call`. A *realistic* call was fixed
before measuring as batch 256, width 16 or 64.

| n | r | compile_tape | prepare_edge | compile share, realistic call (fwd) | prepare share, realistic (fwd / bwd) | prepare share, batch 1, width 16-64 (fwd / bwd) |
|---|---|---|---|---|---|---|
| 1 | 2 | 22 ns | 62 ns | <= 0.0014% | 0.98% / 0.48% | 24.6% / 14.2% |
| 1 | 4 | 36 ns | 102 ns | <= 0.0013% | 0.90% / 0.45% | 22.9% / 13.7% |
| 2 | 2 | 46 ns | 122 ns | <= 0.0012% | 0.83% / 0.30% | 21.4% / 9.4% |
| 2 | 4 | 77 ns | 205 ns | <= 0.0011% | 0.79% / 0.29% | 20.6% / 9.1% |
| 4 | 2 | 97 ns | 249 ns | <= 0.0006% | 0.41% / 0.11% | 11.8% / 3.5% |
| 4 | 4 | 164 ns | 420 ns | <= 0.0006% | 0.42% / 0.11% | 11.8% / 3.4% |
| 5 | 32 | 1.49 us | 3.42 us | <= 0.0004% | 0.24% / 0.05% | 7.2% / 1.7% |

**Conclusion.**
- `compile_tape`, the subject of R5, is negligible. It is at most 0.0014% of a realistic call and
  at most 0.35% at width 1, batch 256. It reaches 3-7% only for a single-edge, single-sample call,
  which lasts 0.3-50 us in total.
- Per-edge preparation is at most 0.98% of a realistic forward call. **No library change.**
- At batch 1, per-edge preparation is 7-25% of a forward call. There it is amortized over one
  sample, and 7 of the 8 sample lanes are idle. This is irreducible within a call: the half-angles
  depend on the edge's parameters. Avoiding it would require caching across calls with unchanged
  parameters (inference), which is outside phase 1. It is reported here, not optimized.

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

A fresh reviewer agent (read-only, no part in the implementation) reviewed `0fa47b6..f4e3c6e`
against the plan (with the 2026-10-23 amendment), CONTRACT and AGENTS, then `f4e3c6e..b144569`
(the R3/R5 work and its own findings' fixes) in a second pass. **Nothing blocking or high in
either pass.**

Verified by probes and runs (first pass): gate algebra, rotation conventions and readouts by
derivation; inert marking and `first_derivative` exhaustively (all 4536 configurations with
n 1..4, r 1..3, every encoding, block, entangler and readout; adjoint and shift against central
differences of the dense reference, 649 134 checks: no slot wrongly inert, worst error 1.1e-9);
`Lanes<8>` bitwise equal to one-sample evaluation (14 386 forward and 5 760 backward values);
independence of the dense reference; protocol atomicity, D8 draw order, D13 messages; every
number printed by the suites equals the evidence; the RED log reproduced from `0fa47b6`.
Second pass: the R3 and adjoint-vs-shift bounds are not vacuous (a defect dropping one slot's
contribution fails in 81-97% of slots; the rest are below 1e-13), the double-double reference
is exact TwoSum, the R5 replica of `prepare_edge` matches the original token for token, and the
"not RED" statement and the rejected `fl(k pi)` alternative are accurate.

| Finding | Severity | Resolution |
|---|---|---|
| F1 closed-form reference in `long double` (binary64 under MSVC: passes only by the seed) | MEDIUM | `e59f217`: double-double TwoSum reference; 40 000 probe trials, 0 failures, equal to binary128 |
| F2 source check skipped `lanes.hpp`; textual pattern easy to bypass | LOW | `0d950bb`: `lanes.hpp` scanned, `using namespace std` and std aliases caught |
| F3 deviation 6 incomplete | LOW | `b144569` |
| F4 ASan build type not stated (`pqc_layer` times out under Debug + ASan) | LOW | `b144569`: RelWithDebInfo stated, timeout noted |
| F5 base commit | NIT | `b144569` |
| F6 inert comment omitted CNOTs the qubit controls (code correct) | NIT | `3a0dfa1` (comment, CONTRACT, evidence) |
| F7 stale MSVC comment | NIT | `43496ff` |
| F8 1.5e-15 vs 1.4e-15 | NIT | `3a0dfa1` |
| F9 allocation and L2 tests pass against the stubs (never RED) | NIT | Already stated in the RED row; no change |
| S1 pattern still missed wide/pmr strings, `from_chars`, `strto*`, `ato*`, `<charconv>`, `<format>`, `std :: string` | LOW | Pattern extended with 14 new self-test snippets and 3 more allowed ones (`fbc7ded`); sources still clean on GCC and MSVC |
| S2 "derived" bounds rest on an assumed `2Gu` per-gate budget | LOW | Worded as first-order estimates calibrated by measurement in the test (`fbc7ded`), CONTRACT and here |
| S3 no MSVC run of the double-double test | NIT | Run recorded in the RTX 3090 row (this commit) |
| S4 input bound uses the largest `|w x + b|` over all slots (slightly loose) | NIT | Not changed: at most 1.3x looser than the old 1e-12 on the shallow cases and still detecting (above) |

Not checked by the reviewer: CUDA builds at the final head (no library code changed after
`f4e3c6e` except a comment; recorded runs above), frozen-suite timings, coverage numbers.

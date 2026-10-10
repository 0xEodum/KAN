# CUTLASS fusion experiment protocol

Date: 2026-10-10. Branch: `experiment/cutlass-kan-fusion`.
Frozen control: `37a07f8` (current M3 closure). Stage B; no M5 work or backlog
reopening. The owner explicitly authorized this experiment series, dependency
downloads and use of the RTX 3090. Production defaults remain the control.

## Questions and bounded search

1. Can a library GEMM mainloop make C3 fusion worthwhile? Measure materialized
   CUTLASS GEMM as a diagnostic control, then Chebyshev virtual-operand forward
   and coefficient VJP, and input-VJP epilogue fusion where feasible.
2. Does a fused SiLU residual input-VJP epilogue help complete training steps?
3. Do benefits depend on width, depth, batch, residual branch and precision?

Start with a small correctness/profile screen, at most eight kernel configurations
per operation and two hours of measured GPU runs. Freeze selected configurations
before the confirmation matrix. Keep unsuccessful variants and reasons. CUTLASS
3.9.2, commit `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`, is downloaded
under ignored `build-cutlass-deps/`; the experiment uses its C++ headers, not
Python DSL, Hopper or Blackwell-only instructions. Record installation smoke
test, CUDA/compiler/driver versions and source hashes with results.

## Complete-call matrix

Chebyshev K=7: tiny 16->24->8/b1024; small 64->64->32->16/b1024;
irregular 63->95->17/b257; deep six-layer 64-wide/b1024;
medium 256->256->256->10/b2048; wide 256->256->256->10/b8192;
large 1024->1024->1024/b4096. Residual branch off/on. FP32 and TF32
measured separately; FP64 is an unchanged control and correctness reference.
Use actual dispatch (including small custom kernels), not forced GEMM-only
timings as the acceptance metric.

Two protocols: resident input/target MSE forward+loss+backward+SGD graph replay;
and host-batch MSE training (four cyclic batches, including validation,
conversion, upload and synchronization). Status interval 1 for both. All
variants use identical initialization, data, learning rate, steps and warmup.
Confirm with three independent seeds, interleaved ABBA execution, at least
five timing windows per process and a common step count per paired case.
Report wall ms/step, median/range, paired ratios, memory and GPU state. Do not
infer general speedups from one kernel, one shape or the best seed.

## Correctness and promotion

Before timing, compare output, input gradient, every parameter gradient and
updated parameters to the control; include irregular/tail dimensions, zero
batch, L2, graph/eager equivalence, overflow/rollback and unsupported-family
fallbacks. Existing production gates must remain green with the experiment
disabled. A reordered GEMM may differ in FP32 bits: record max absolute and
normalized error and reject outside FP32 `1e-4` / TF32 `5e-3` elementwise
`abs(candidate-reference)/(1+abs(reference))` for finite moderate inputs.
These experiment gates do not relax production bitwise/regression contracts.

Run longer actual MSE training for three seeds and compare learning curves,
final loss and parameter drift on small/deep and medium networks. Report
divergence or numerical rejection, even when a rejected variant is faster.
Use Nsight Systems for complete steps and Nsight Compute for dominant new
kernels. Run Compute Sanitizer on a bounded representative correctness case.
Adopt no default without repeated full-step benefit and contract compliance;
an opt-in experimental result is a valid negative outcome.

## Reproduction and evidence

Store harness sources, build/run scripts, raw CSV/logs, summaries and SHA-256
manifest here. Large profiler binaries and downloaded dependencies stay in
ignored build directories. Commit and push protocol, verified implementations,
and measured conclusions as separate steps.

# M3 independent implementation and acceptance review

2026-10-01. Reviewer authored the CPU basis lane, and independently reviewed the
other implementers' CPU layer/network, Python and resident CUDA lanes. This
document does not claim an independent review of the reviewer's own basis code.
Read the complete current `ROADMAP.md` and M3 plan before reviewing.

## Layer/network review

Read `src/layer.cpp`, `src/network.cpp` and their public headers. Reviewed Boehm
insertion copy/blend ranges, right-hand repeated-knot spans, degree-zero handling,
upper endpoints, stable huge-domain interpolation weights, data-driven median
and midpoint selection, atomic candidates, shape-invalidated gradients,
shared RBF VJP summation and coefficient-only L2.

Three separate adversarial C++ test cases in ignored
`build-m3-review-basis/review.cpp` compiled directly with MSVC C++20 `/W4`
and passed **3/3**, then passed **3/3** again with `/fsanitize=address /Zi`
and no sanitizer diagnostics. The source SHA256 is
`6F0983FE35DD53F0FFC15B710ACAAA973173EA452C9659103F8183E298CA2B9F`.

- 340 successive insertions across every degree 0..16, including existing
  knots raised to full allowed multiplicity and deterministic random new
  knots, compared values and input VJPs to the unchanged original model at
  exact domain endpoints, old/new knots and random interior values.
- Independent adaptation checked tied spans choose the lowest, boundary-only
  samples choose the selected span midpoint, empty/outside/nonfinite sets fail
  atomically, and an adjacent-double degree-zero span cannot be refined.
- Shared trainable zero-batch gradients and coefficient-only regularization
  shapes/values were checked. Invalid third-layer width updates preserve every
  earlier/later coefficient, bias, center and log width. Moved-layer M3 methods
  reject state; unrepresentable coefficient penalties raise overflow.

No blocking layer/network implementation findings were found.

## Python review

Read the new binding changes and strict ndarray helpers. Ran a separate Python
prober in ignored `build-m3-review-basis/review.py` against the actual CPU
extension. Returned center-gradient arrays and basis configurations are owned
snapshots: mutating a returned array/config does not mutate its source. Rank,
float32 and strided parameter vectors reject. Layer/network regularizer input
arrays have the correct zero-batch shape. Existing network gradient arrays keep
their original topology after indexed refinement, and applying them to the
refined model rejects.

Independently replayed `tests/m3_python_test.py` against `build-m3-python/python`:
three tests passed and the GPU test was explicitly skipped in CPU mode. Replayed
against the CUDA extension in `build-m3-final/python` with `--cuda`: **4/4**
passed on the actual device. These include nonlinear finite differences, held-out
localized training, strict dtype handling, refinement and resident parity/L2.
The existing two-array `evaluate_basis` return is retained for compatibility;
new nonlinear layer VJPs are exposed as owned gradient arrays.

No blocking Python implementation findings were found.

## Resident and benchmark source review

Read the new device spline/wavelet/RBF evaluation, shared nonlinear reduction,
arena offsets, parameter snapshots, zero/nonzero batch L2 and all-layer candidate
width checks before region swap. Degree-bounded local spline scratch is indexed
within 18 slots including degree 16 and full repeated knots. Immutable grid data
and nonlinear derivative storage are setup allocations, and explicit refinement
reconstruction keeps numerical execution resident.

Independently replayed the real-device `build-m3-cuda/m3_resident_test.exe`:
**7/7** passed. Inspected retained `m3/resident-green.txt` (**4/4** CUDA targets)
and `m3/resident-memcheck.txt` (**0 errors**, all seven M3 cases passed).

Reviewed `benchmarks/m3_benchmark.cpp` and its frozen protocol. Each backend
starts with identical deterministic inputs/parameters, synchronized complete
calls are timed, all outputs/VJPs and nonlinear parameter snapshots are compared,
and the resident untimed trajectory replay must reproduce the actual timed
final parameters. Source inspection alone does not establish performance.

## Tiled nonlinear reduction follow-up

Read the profiled replacement shared-parameter reduction while timing runs were
isolated. The flattened edge index, deterministic per-lane loop/tree/tile order,
64-tile persistent scratch bound, compact indexing across changing batch sizes,
zero-batch initialization and all-network candidate swap were sound. Nonfinite
lane/tile results remain observable at the reported reduction outputs.

One blocking large-dimension finding was sent to both implementers: the finish
kernel initially handled one index per thread while the shared launch helper
caps blocks at 65535. More than 8388480 basis terms could therefore leave trailing
nonlinear gradients unwritten; capacity-zero persistent storage at that threshold
is approximately 8.6 GB and can fit the validation GPU. The required correction
is a grid-stride finish loop, preserving normal-size reduction behavior.
The coordinator retained an actual failing 8,388,481-term regression at `accdf49`.
Correction `f36d22d` adds the grid-stride finish loop and capacity-bounded scratch;
the same case passes with unchanged allocation counts. The separate final
auditor independently reviewed the correction, numerical probes and complete
evidence at `b7e763b`: [final review](M3-final-review.md), **PASS**, no blocking
findings. This append records that later audit rather than attributing it to the
initial basis-lane reviewer. See [M3 final evidence](M3.md) for acceptance closure.

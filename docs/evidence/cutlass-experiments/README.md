# CUTLASS experiment reproduction

See [PLAN.md](PLAN.md) for the frozen control, matrix and acceptance gates.
Production defaults and public precision contracts are unchanged. Set
`KAN_CUTLASS_EXPERIMENT=OFF` (default) for the normal build; the CPU build
does not include CUTLASS headers or acquire any CUDA dependency.

## Setup

```powershell
git clone --depth 1 --branch v3.9.2 https://github.com/NVIDIA/cutlass.git build-cutlass-deps/cutlass
git -C build-cutlass-deps/cutlass rev-parse HEAD
# Must be ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e
& docs/evidence/cutlass-experiments/build.ps1
python docs/evidence/cutlass-experiments/run.py screen
python docs/evidence/cutlass-experiments/run.py confirm
python docs/evidence/cutlass-experiments/run.py learn
python docs/evidence/cutlass-experiments/run.py summarize
```

Validated installation: RTX 3090 SM86; CUDA toolkit 13.1.115; driver 591.86;
MSVC 19.50.35724 (explicit nvcc unsupported-host-compiler override). CUTLASS
requires MSVC `/Zc:__cplusplus`, supplied only to the experimental translation
unit, as in upstream CUTLASS's CMake setup. Upstream headers are unmodified.
The default experiment build uses `KAN_CUDA_FMA=ON`; `build.ps1 -NoFma`
builds the parity configuration separately.

## Modes

`KAN_EXPERIMENT_MODE` and `KAN_EXPERIMENT_TILE` are read once at resident
executor construction, only in an experimental build. Harness CLI sets them
before construction. Mode 0 is the unchanged cuBLAS/custom-kernel control.

| Mode | Operation changed | Purpose |
|---:|---|---|
| 1 | Large Chebyshev K=7 forward: materialized Phi + CUTLASS GEMM | Library mainloop diagnostic without basis fusion |
| 2 | Chebyshev K=7 input VJP: CUTLASS GEMM + in-CTA derivative reduction | Avoid the global U*C tensor and separate input finish |
| 3 | Large Chebyshev K=7 forward: virtual Phi iterator | Isolate on-the-fly basis cost; Phi is still materialized for dC |
| 4 | Virtual forward + virtual dC + mode 2 + mode 5 | Elide Phi materialization on layers where both contractions take the large path |
| 5 | Large residual U*W GEMM + in-CTA derivative multiplication and accumulation | Avoid the residual intermediate tensor |

Three SIMT configurations: tile 0 = CTA64x128, warp32x64; tile 1 =
CTA128x128, warp64x64; tile 2 = CTA64x64, warp32x32. K tile 8, two-stage
CUTLASS pipeline, SM80 implementation compiled for SM86. These are actual
CUTLASS mainloops/epilogues, not the previous hand-written C3 SGEMM.

The changed contractions use full FP32 SIMT arithmetic in both FP32 and TF32
executors; unchanged contractions keep the executor's cuBLAS precision.
TF32 comparisons therefore include a mixed path, not a claim of native TF32
CUTLASS Tensor Core fusion. Tensor Core mainloops/custom virtual-operand
pipelines and other GPU generations are outside this bounded SIMT series.

Input fusion writes accumulator tiles to CTA shared memory using CUTLASS's
generic CUDA-store epilogue, then reduces whole seven-term inputs. Adjacent
column tiles overlap only in unused tail columns. Each output has one owner;
there are no numerical atomic reductions. Residual fusion uses the same
mechanism for elementwise accumulation. Both keep existing parameter-gradient
checks; output/input checks are retained in the new kernels.

Small forward/parameter contractions and unsupported basis families retain
their existing paths. Phi/scratch arena reservations are retained for changing
batch sizes and fallback paths: reduced materialization traffic does not imply
reduced reserved memory. No device allocation is introduced during execution.

## Tail regression

[RED](raw/virtual-tail-red.txt): the first virtual iterator incorrectly assumed
full-K-first traversal. CUTLASS's predicated iterator traverses the residue
first, then full K tiles. The virtual iterator now follows that traversal.
The `tail` correctness case exercises both forward and dC residue tiles;
screen logs record the GREEN output/gradient/SGD/graph/rollback checks.

`check` verifies candidate against mode 0 in the same precision, with L2,
eager/graph bitwise equivalence, zero batch and atomic overflow rollback.
It reports every changed bit and maximum error rather than treating tolerance
as bitwise parity. `bench` measures complete MSE updates with synchronization,
and `train` records a 1000-update learning curve including first graph capture.

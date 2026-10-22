# KAN numerical contract (M1 through M4, backlog R1-R3, M1, M2, M3, M4, M5, M6 and M7)

An edge is a learned univariate function. A basis layer computes
`y[b,o] = bias[o] + sum_i sum_k coefficients[o,i,k] * basis_k(x[b,i])`.
Arrays are contiguous, batch-major; coefficients use `(o * inputs + i) * basis_size(basis) + k`.
All arithmetic and storage in M1 use `double`. Bias and coefficients initialize to zero.
Topology and parameters are owned values, never global state. Initialization for training
is explicit: zero initialization is useful for a single linear-in-coefficients layer, but
multilayer training needs nonzero parameters to propagate gradients; constructors stay
zero-initialized and the opt-in initializers are described under "Initializers (backlog M4)".

`BasisConfig` is a `std::variant` of one configuration type per family
(`ChebyshevConfig`, `LegendreConfig`, `JacobiConfig`, `HermiteConfig`, `FourierConfig`,
`GaussianRbfConfig`, `TrainableRbfConfig`, `BSplineConfig`, `MexicanHatConfig`); each
holds only its family's parameters, and a value-initialized `BasisConfig` is
`ChebyshevConfig{4}`. `basis_size(config)` is the number of terms, never a polynomial
degree: explicit `size` for polynomial and Fourier families, derived for localized
families (one term per center; `knots.size()-degree-1` for splines). Chebyshev
uses T_n, Legendre uses P_n, Jacobi uses P_n^(alpha,beta), Hermite uses physicists'
H_n by default or the normalized Hermite functions psi_n (`HermiteConfig::normalization`,
backlog M5, below). Polynomials are evaluated by recurrence, including endpoints and outside [-1,1].
There is no implicit clipping, normalization, tanh, or extrapolation policy; inputs are
brought into a basis domain only by an explicit input map layer (see "Input maps").
Fourier is `[1, cos(w*x), sin(w*x), cos(2*w*x), sin(2*w*x), ...]` and size is odd.
Gaussian RBF is `exp(-((x-center)/width)^2)` with explicit finite centers and width > 0.
Jacobi alpha and beta are finite and greater than -1; angular frequency is finite and positive.
Irrelevant parameters cannot be expressed: each configuration type has only its own.
Localized configurations need at least one term and equal-length parameter vectors.
Each evaluator returns values and derivatives with respect to x.

**Hermite normalization (backlog M5).** `HermiteConfig{size, normalization}` with
`kan::HermiteNormalization::Physicists` (default; every existing configuration, result and
golden dump is unchanged) or `Function`: `psi_n(x) = H_n(x) exp(-x²/2) / sqrt(2^n n! sqrt(pi))`,
n = 0..size-1, orthonormal in L²(R) (Lebesgue measure) and bounded by `pi^(-1/4)` for every n
and x (physicists' terms grow like `sqrt(2^n n!)`). The normalization is fixed configuration:
part of `operator==`, so `ResidentNetwork::upload_parameters` rejects a change; an unknown
enumerator value is rejected by validation (`std::invalid_argument`). It selects a separate
family tag (`BasisKind::HermiteFunction`): the resident executor instantiates its own basis and
FP32 Φ'-recompute kernels, no kernel branches on the normalization, and the existing
instantiations are SASS-identical. Evaluation (`hermite_function_terms`,
`src/carriers/polynomial/polynomial_formulas.hpp`) uses the stable recurrence
`psi_(n+1) = (sqrt(2) x psi_n - sqrt(n) psi_(n-1)) * (1/sqrt(n+1))` and
`psi_n' = sqrt(2n) psi_(n-1) - x psi_n` on scaled values: the recurrence starts at `2^-B`
(B = 512 in FP64, 64 in FP32), rescales by `2^-B` whenever a value exceeds 1, and multiplies
every term by `pi^(-1/4) exp(-x²/2) 2^(B+E)`, computed without intermediate underflow (Cody–Waite
reduction of x²/2 with `fma` for the high part and the exact low part of x·x, `ldexp` for the power of two) and recomputed
only when a rescale changes E. Inputs where the plain weight `exp(-x²/2)` underflows (FP64
|x| > 38.6, FP32 |x| > 14.4) therefore still give the O(0.1) values of high-degree terms near
their turning points sqrt(2n+1); no intermediate overflows and no `inf·0` arises. For
`x² >= max(8·size, T)` (T = 3072 in FP64, 448 in FP32) every |psi_n| and |psi_n'| is provably
below half the smallest subnormal (`|psi_n| <= exp(-x²/4)` there) and the row is exactly zero.
Accuracy against an independent 700-digit reference
([M5 evidence](evidence/backlog/M5.md#accuracy)): FP64 with size 1024 on |x| <= 60, max
absolute error 7.6e-14 (values) and 1.9e-14·sqrt(2n+1) (derivatives), relative error <= 3e-13
in the monotone tail; FP32 resident with size 128 on |x| <= 26, 5.2e-6 and 1.9e-6·sqrt(2n+1),
1.0e-5 in the tail. Those sizes and ranges are the tested limits; `size` itself is not capped.
psi_n is concentrated on |x| <= sqrt(2n+1): on inputs that cover that range the design matrix
is perfectly conditioned (cond(ΦᵀΦ) = 1.000 for size 8–48 on an equispaced grid), while inputs
concentrated near 0 (for example N(0, 1/2)) leave high-degree terms nearly dependent; scale
inputs with an input map when using large sizes.
Python: `kan.HermiteNormalization.PHYSICISTS/FUNCTION`, `kan.HermiteConfig(size, normalization)`.

Layer backward is a vector-Jacobian product, summing parameter gradients over the batch.
It does not average or mutate parameters. Caller supplies any loss scaling.
Network backward recomputes intermediate activations, returning gradients in layer order.
SGD applies `parameter -= learning_rate * gradient`; learning rate is finite and positive.
Updates validate all parameter gradients and the complete resulting parameter vectors
before mutating, including across a network. Input gradients are not consumed by SGD.
Inputs, upstream gradients, and parameters must be finite. Nonfinite computed results
raise `std::overflow_error`; invalid configuration, sizes and data raise `std::invalid_argument`.
Dimension multiplication is checked before allocation (`std::overflow_error`).
Layer dimensions are positive; batch zero accepts only empty inputs/upstream and returns
empty outputs/input gradients and zero parameter gradients. Networks are nonempty,
adjacent dimensions match, and can mix basis families and input maps.
Moved-from layers and networks remain assignable; their numerical/parameter-update
operations raise `std::invalid_argument`. A network rejects moved-from layer values.
Accessors are safe but their moved-from values are unspecified.

CUDA is an optional separate `kan::cuda` target; CPU has no CUDA dependency.
No device returns `available() == false`; actual operations fail explicitly with
`std::runtime_error`. `available()` is the supported device query for every CUDA API.
It is declared in `kan/cuda_runtime.hpp` (backlog R8), which both `kan/resident.hpp` and
the legacy `kan/cuda.hpp` include, so resident code needs no legacy header; the header has
no CUDA dependency and the definition (`src/backends/cuda/cuda_runtime.cpp`) is in `kan::cuda`.

**Legacy synchronous layer API (M1, deprecated by backlog R7).** `kan::cuda::forward`
and `kan::cuda::backward` keep their signatures and are `[[deprecated]]` in favour of
`kan::cuda::ResidentNetwork`. Each call validates shapes and finiteness on the host
(`std::invalid_argument`; size overflow raises `std::overflow_error`) before any device
allocation, then builds a one-layer `ResidentNetwork` with capacity `batch`, uploads,
runs forward (and, for `backward`, forward plus backward with no L2 term), downloads and
releases it. Device memory is therefore owned per invocation and released on return and
on exceptions; the call pays the executor construction (stream, arena, cuBLAS handle)
each time and makes no performance claim. It accepts every carrier the resident
executor supports, not only Chebyshev (contract change of R7; M1 rejected the other
families with `std::invalid_argument`): `LayerGradients::nonlinear` holds the trainable
RBF or rational VJPs, and unsafe guarded rational denominators raise `std::domain_error`.
Numerics, summation order and the nonfinite-result contract are those of the resident
executor (below); results match the CPU within `|a-e| <= 1e-12|e| + 1e-13 max|e|` per
entry, not bitwise (see [R7 evidence](evidence/backlog/R7.md)).

## Persistent CUDA execution (M2)

`kan::cuda::ResidentNetwork` is a move-only executor built from an owned snapshot
of a CPU `Network`, with a fixed maximum batch capacity. It supports every M1
basis family and mixed compatible networks. Storage is double precision unless the
opt-in FP32 precision policy below (backlog C1) is selected.
Since R7 the deprecated M1 layer functions are a thin adapter over this executor.

Each resident executor owns its CUDA stream and preallocated device storage for
parameters, inputs, upstream gradients, activations, basis values/derivatives,
input/parameter gradients, and candidate SGD parameters. Construction uploads the
model; `upload_input` and `upload_output_gradient` explicitly transfer finite host
data. No numerical call allocates device storage. Batches above capacity fail.
`workspace_allocations()` reports the executor's construction-time device allocation
count, which remains unchanged by subsequent operations.

Uploads, computations, and downloads complete before returning. Computations check
a small device error status on the host without copying full tensors. The only
asynchronous calls are the training steps of backlog C9 (below), whose status check is
deferred by an explicit, documented interval. Different instances can run independently;
callers must serialize operations on the same instance. Host input buffers need
only remain alive for their upload call. Downloads return independently owned values.

After input upload, `forward()` saves the activations for `backward()`. Backward
requires a successful current forward and a correctly sized uploaded upstream.
`download_output()` and `download_gradients()` require their corresponding current
successful computation. Input upload invalidates output/gradients and the uploaded
upstream. Upstream upload invalidates gradients. SGD requires current gradients
and a finite positive learning rate. It validates every candidate parameter on
the GPU before committing any layer, preserving network-wide atomicity. Successful
SGD invalidates output/gradients while retaining input/upstream for another iteration.
A successful `upload_parameters` (below) does the same.
Batch zero produces empty outputs/input gradients and zero parameter gradients.
Invalid lifecycle and moved-from operations raise `std::logic_error`; invalid
host shapes/data raise `std::invalid_argument`, dimension/numerical overflow raises
`std::overflow_error`, and absent hardware/runtime failures raise `std::runtime_error`.
All M1 mathematical domain and finite-result
requirements apply; CPU/GPU equivalence is tolerance-based.

**Parameter upload (backlog R9).** `upload_parameters(network)` replaces the executor's
trainable state with the parameters of a host `Network`, so one executor can be reused for
weights trained on the CPU or restored from elsewhere: KAN-layer coefficients (rational
numerators included) and biases, trainable RBF centers and log widths, rational
denominators, wavelet translations and log scales (backlog M7), SiLU residual-branch weights
(backlog M3), and LayerNorm gain and bias. Structure is what the executor was built for and
is never uploaded; it must match exactly, otherwise `std::invalid_argument` names the first
differing layer: the number of layers; per position the kind (KAN layer or input map) and
dimensions; the carrier (`BasisEdges`, `TrainableRbfEdges`, `RationalEdges`, `WaveletEdges`) or map kind; whether
a KAN layer has the residual branch ("residual branch presence");
and the fixed configuration, compared with `operator==`: the whole `BasisConfig` of a
`BasisEdges` layer (family, size, Jacobi alpha/beta, Fourier frequency, Gaussian centers and
width, B-spline degree and knots, Mexican-hat centers and scales), the term count of a
trainable RBF, the `WaveletConfig` (wavelets per edge), the whole `RationalConfig` (degrees, center, scale, epsilon, denominator
policy), `AffineMap` and `TanhMap` values, the LayerNorm epsilon and whether it has
gain/bias. Spline knots are configuration, not state: `insert_knot`/`adapt_grid` change the
term count, and knots moved at an equal count (`set_carrier`) change the fixed function
space the coefficients refer to; both are rejected, and such a model needs a new executor.
The executor's own `refit_grids` (backlog M6) changes its knots in place; its snapshot
takes the refit knots, so `download_parameters()` returns them and `upload_parameters`
then compares against them (the network before the refit is rejected).
A `download_parameters()` result always matches its executor, and the round trip is exact
in every precision (downloads are exact widenings). Values are validated with the
construction rules of the executor's precision: finite, and in FP32 magnitudes at most
`FLT_MAX` and trainable RBF widths positive after rounding (values below the FP32 range round
as at construction), with the construction messages. All checks run on the host before
anything changes: a rejected call leaves parameters, outputs, gradients, input, upstream and
batch unchanged. A CUDA runtime failure during the copy raises `std::runtime_error` and
leaves the parameters unspecified. Success invalidates the output and gradients (`backward()`
and `download_output()` need a new `forward()`) and keeps the uploaded input and upstream,
like SGD. The call is valid at any point after construction, also before any input. It
allocates no device memory (`workspace_allocations()` is unchanged) and writes the active
parameter region: FP32 executors and FP64 regions up to 1 MiB convert on the host and make
one host-to-device copy; larger FP64 regions are copied tensor by tensor without host
staging (at most four copies per KAN layer, five with the residual branch, two per LayerNorm
map) after validating all of
them. Measured on the RTX 3090: 0.16 ms (FP64) / 0.12 ms (FP32) for a 0.4 MB
64x64x32x16 Chebyshev network, about 7% of construction; 45 ms / 40 ms for the 117 MB
1024x1024x1024 network, about 40% of construction, of which the PCIe copy is 36 / 18 ms
([R8/R9 evidence](evidence/backlog/R8-R9.md)). Python: `ResidentNetwork.upload_parameters(network)`
(`ValueError` for rejections, `RuntimeError` for lifecycle errors), releasing the GIL.

**Training steps (backlog C9).** `train_step(learning_rate, coefficient_l2 = 0, loss =
Loss::OutputGradient)` runs forward, the loss gradient, `backward(coefficient_l2)` and
`sgd(learning_rate)` as one captured CUDA graph, replayed without host synchronization.
Arguments are validated like `backward`/`sgd` (`std::invalid_argument`, before anything is
enqueued). `Loss::OutputGradient` needs an uploaded input and upstream (`std::logic_error`
otherwise) and uses the upstream like `backward()`, which stays resident for the next step;
the parameters after the step are **bitwise** those of `forward(); backward(l2); sgd(rate)` in
every precision. `Loss::MeanSquaredError` needs an input, a target (`upload_target(target)`,
shape batch x outputs, validated like the upstream; an input upload invalidates it) and a
nonempty batch (`std::invalid_argument`); it computes on the device the upstream
`(y - t)*(2/N)` in the executor precision, `N = batch*outputs`, bitwise what the eager
sequence computes from that upstream, and the loss `sum (y - t)^2 / N` (block tree sums
added in a fixed order: deterministic), returned by `download_loss()` for the most recent
training step if it used this loss (`std::logic_error` otherwise); the value belongs to that
step's forward pass, before its update. The step consumes the uploaded upstream (its own
gradient replaces it: an `OutputGradient` step afterwards needs a new upload). After any
step outputs and gradients are stale, as after `sgd()`. A step does not compute the gradient
with respect to the network input, which it never exposes (as PyTorch for an input without
`requires_grad`): a nonfinite network-input gradient therefore does not fail a training
step, whereas `backward()` reports it.

`train_step(input, target, batch, learning_rate, coefficient_l2 = 0)` trains one MSE step on
a new host batch. Shapes, capacity, a nonempty batch and the values (finite; FP32 magnitudes at most
`FLT_MAX`) are checked with the upload messages before the call returns, and a rejected batch
changes nothing. Accepted data is converted into one of two page-locked host slots (split over
up to 8 host threads for batches above 2^18 values) before the call returns, so the caller may
reuse its buffers at once. It is copied on a separate stream into the input/target regions the
running steps do not read: the input copy starts while the target converts, and the step waits
for its target inside the graph, before the loss, so the target copy overlaps the forward
pass. The batch then becomes the executor's input and target, for later eager calls and
resident MSE steps. A slot is reused only after the step two staged batches earlier has
finished. The host slots, copy stream and events are created on first use (host memory, not
counted by `workspace_allocations()`); the second input and target regions, the loss and its
block sums are part of the construction-time arena. No training call allocates device memory.

*Deferred status.* The device status of training steps is checked once every
`status_interval()` steps (`set_status_interval(steps)`, default 1, `steps = 0` raises
`std::invalid_argument`) and by `check_status()`. Every other non-const call also checks it
first: uploads, `forward`, `backward`, `sgd`, `upload_parameters`, downloads, `download_loss`,
`synchronize`, `set_status_interval`. The const queries `status_interval()` and `trained_steps()`
do not. A step writes its forward and loss, backward and SGD-candidate results into three
status words that stay set until the check. A commit kernel at the end of the step counts the
step as committed when all words are clear. Otherwise it records the phase bits of the
first failing step (forward before backward before SGD, the order in which the eager calls
would have raised) and copies the current parameters over the candidates. The region that
becomes active then holds the parameters of the last good step. The check reads this block
once and raises for the **first** failing step of the interval, with the exception the eager
call would have raised (`std::overflow_error` for nonfinite results, `std::domain_error` for
guarded rational poles). The message names the step's index since construction and the number
of later steps of the interval. The failing step and all later steps of the interval commit no update: they run
but roll back. Afterwards the parameters are those after the last good step, outputs,
gradients and the loss are stale, the status is clear, the last batch stays the input/target,
and training can continue. `trained_steps()` counts committed steps as of the last check, and
equals the failing step's index right after such an exception. With the default interval each
step's own call reports it; a larger interval reports up to `interval - 1` steps later and
costs that much rolled-back work on failure, never a silently corrupted model. The destructor
cannot report a pending failure: call `check_status()` (or any checking call) before
discarding an executor whose last steps matter. Eager calls report and clear the status as
before.

*Graphs.* A captured step bakes in the batch, the active parameter region (the regions
alternate every update), the input/target region, the loss, the L2 weight and whether it waits
for a staged target. It is re-captured when any of these changes; at most 8 graphs are cached,
least recently used first out. The learning rate is a parameter of the SGD candidate kernel,
updated in the instantiated graph (`cudaGraphExecKernelNodeSetParams`, affecting later
launches only), so a learning-rate schedule never re-captures. Graphs read parameters, inputs,
targets and upstreams from the arena at replay, so `upload_parameters`, `upload_input`,
`upload_target` and `upload_output_gradient` take effect at the next step without invalidation.
Capture is thread-local (`cudaStreamCaptureModeThreadLocal`), so executors on other threads
are unaffected. Python: `kan.Loss.OUTPUT_GRADIENT / MEAN_SQUARED_ERROR` (declared in every
build), `train_step(learning_rate, coefficient_l2=0.0, loss=...)`,
`train_step(input, target, learning_rate, coefficient_l2=0.0)` (strict float64 arrays),
`upload_target`, `download_loss()`, the `status_interval` property, `check_status()` and
`trained_steps`. Measurements against PyTorch are in the [C9 evidence](evidence/backlog/C9.md).

**Contraction engine (backlog C2).** For `BasisEdges` and `TrainableRbfEdges` layers the
resident executor computes the dense contraction and its VJPs with cuBLAS:
`Y = Phi*C^T + b`, `dC = U^T*Phi + lambda*C`, `db = U^T*1` and `W = U*C`, from which
`dx = sum_k Phi' (.) W` and the trainable-RBF center/log-width VJPs are reduced (FP32/TF32
B-spline layers contract derivative coefficients instead, see "Spline input VJP"). A forward
contraction of at most `2^23` multiply-adds (`batch*outputs*inputs*terms`) instead runs one
warp per output, with the bias and the finiteness check in the same launch, because cuBLAS
executes such tiny products as a single latency-bound block. For the same reason a layer
with at most `2^15` coefficients plus outputs reduces its coefficient and bias VJPs in at
most 64 batch tiles and sums the tiles in a fixed order; larger layers use cuBLAS. `W` lives in
one scratch region shared by all expansion layers (the largest `capacity*inputs*terms`).
The derivative rows `Phi'` are kept from forward to backward only by the FP64 executor and
by `TrainableRbfEdges`; FP32/TF32 `BasisEdges` layers (backlog C3) store only `Phi` and the
backward pass recomputes `Phi'` from the layer input with the forward's formula, so the
results are unchanged (bitwise in both builds) and each such layer reserves one
`capacity*inputs*terms` region less (rows too long for the staged tile, above 85 terms,
keep the stored rows). B-spline layers (backlog C13) evaluate the degree-`p-1` basis
instead of `Phi'` in that finish and store no `Phi'` rows at any length (beyond the stage the
basis kernel writes only the values, from the sparse form of the same evaluation: identical
values, derivatives still evaluated and checked). The cuBLAS handle is created at construction and runs on the executor's stream with a
workspace inside the construction-time arena, so numerical calls make no `cudaMalloc` and
no arena growth and `workspace_allocations()` is unchanged (the handle's own
library-internal state, created once with it, is not counted). cuBLAS sums in its own (fused multiply-add, tiled) order: results are no
longer bitwise identical to the CPU, which stays the FP64 reference, and agree within
floating-point tolerance (see [C2 evidence](evidence/backlog/C2.md) for the measured
deviation). Results are deterministic for one GPU, driver and cuBLAS version. The
nonfinite check covers every result tensor (outputs, coefficient/bias/nonlinear VJPs and
input VJPs); an intermediate product that overflows inside a fused contraction and is
cancelled by the accumulator is not reported if the computed result is finite.
`kan::cuda` therefore links `CUDA::cublas` and requires CUDA 12 or newer; the installed
package finds it through `CUDAToolkit`.

**Trainable-RBF center/log-width reduction (backlog C12).** The sums over the rows
`r = (sample, input)` of `W[r,k]*(-dPhi/dx)` and `W[r,k]*dPhi/dlog-width` have a fixed order:
the rows are split into at most 64 tiles of interleaved 256-row chunks, each of a tile's 256
lanes sums its rows in order, a fixed tree sums the lanes and the tiles are summed in order.
Two kernels compute exactly this order, one block per (term, tile) reading columns or one
block per (tile, group of terms) reading whole row chunks into shared-memory accumulators,
so their results are bitwise equal in both builds; the executor takes the second when its
grid has at least one block per SM, or 64 blocks on larger devices (a per-launch choice that
never changes results). FP32/TF32 layers with at most 85 terms keep the C1 staged kernel and
its own order (unchanged). No arena memory is added; a term group needs
`2*256*group*sizeof(T)` bytes of shared memory (groups of whole cache lines when all terms
exceed the per-block limit).

**Sparse B-spline path (backlog C5).** At any input only `degree+1` (P) of a clamped
spline's terms can be nonzero. A resident `BasisEdges` layer with a `BSplineConfig` can
therefore keep, per input and sample, the index `g` of the first such term (32-bit,
`[inputs][capacity]`; 0 outside the domain) and those P values and input derivatives
(`[inputs][P][capacity]`, samples fastest; zero outside the domain) instead of the dense
`Phi` (and `Phi'`) rows, and contract only them: `y = bias + sum_i sum_j phi_j*C[o,i,g+j]`,
`dC[o,i,k] = sum` over the samples with `g <= k < g+P` of `U[b,o]*phi`, plus `lambda*C`,
`db = sum_b U`, `dx = sum_j phi'_j * sum_o U[b,o]*C[o,i,g+j]` (FP64; FP32/TF32 layers keep
the `P-1` degree-`p-1` values instead of the derivatives and contract the derivative
coefficients, `dx = sum_j B_{g+1+j,p-1} * sum_o U[b,o]*C*[o,i,g+j]`, see "Spline input
VJP"). Kernels: a tiled basis
kernel writes the sparse rows (and, with the residual branch, `S` and `silu'`); the forward
and input-VJP kernels map lanes to samples and read C tiles staged in shared memory; the
coefficient and bias VJPs walk each input's samples in the order of a segmented radix sort
by `g` (CUB `DeviceSegmentedRadixSort`, CCCL 3.0.1 in CUDA 13.0, which shares the stable
`DeviceRadixSort` implementation; determinism does not depend on stability), one thread per (output, input) with a register window of the P coefficients, and
coefficient ranges split across blocks for layers with few outputs (the split does not change
any sum). Every sum has a fixed order and no floating-point atomics are used: results are
deterministic for one GPU, driver and CUDA version and independent of the split; a graph
step stays bitwise the eager sequence. Precision: FP64 and FP32 kernels; under
`TensorFloat32` the sparse kernels compute in plain FP32 (no TF32 rounding), so a sparse
layer is at least as accurate there as the FP32 executor. Parity with the CPU is the
existing tolerance (FP64 `1e-11*|e| + 1e-12*max|e|` in the C5 suite, FP32/TF32 the C1
tolerances); results differ from the dense path by summation order only.

*Selection* is per layer at construction, so the arena holds either layout and graphs,
`upload_parameters` and rollback see a fixed one (knot insertion already requires a new
executor). A B-spline layer takes the sparse path when all of: degree 0..3 (the instantiated
kernel widths; higher degrees keep the dense rows); the kernels' shared memory fits the
device's per-block opt-in limit (the input VJP stages 8, 4, 2 or 1 inputs per block as the
rows grow; on a 99 KiB device this allows up to 1856 terms, the forward up to 1267 (FP32) /
704 (FP64)) and `inputs*capacity` fits 32-bit sort indices; the dense
forward contraction would run on cuBLAS at capacity (above the small-kernel threshold) and
has at least `2^22` (FP32), `2^23` (TF32) or `2^21` (FP64) multiply-adds per SM at capacity;
`terms/(degree+1)` reaches 4 (FP32), 5 (TF32) or 2 (FP64 on a device whose
`cudaDevAttrSingleToDoublePrecisionPerfRatio` is at least 8; 2 otherwise);
and, when the layer's upstream (`capacity*outputs` values) exceeds half the L2 cache,
6 (FP32), 20 (TF32) or 8 (FP64, extrapolated, not measured; 6 at a full FP64 rate), because the coefficient VJP
rereads the upstream once per input (FP32 4, TF32 5 and 20 since backlog C13, whose cheaper
dense input VJP moved the crossovers; at these values two measured boundary configurations run
at 0.97-0.98 of the dense step: [C13 evidence](evidence/backlog/C13.md)). The other constants come from a GB10 sweep of complete
training steps ([C5 evidence](evidence/backlog/C5.md)) and have been validated on GB10 only; SM count, L2 size, FP64 rate and shared-memory
limit are read from the device. Memory: a sparse layer reserves `2P` (FP64) or `2P-1`
(FP32/TF32) scalars and four 32-bit
indices per input and sample, `2*inputs` segment offsets and the sort's temporary storage instead of
`capacity*inputs*terms` values (`Phi`, plus `Phi'` in FP64), and needs no share of the
`W = U*C` scratch or of the small parameter-VJP tiles; allocation count unchanged.
*Status:* the sparse rows, outputs, coefficient, bias and input VJPs are checked like the
dense ones. One difference: the dense FP64 path forms `W = U*C` for every term, so an overflowing
`U*C` at a term whose basis derivative is zero becomes `inf*0 = NaN` in `dx` and is reported;
the sparse path never forms that product, so such a step succeeds with the (finite) input
VJP (FP32/TF32: both paths, see "Spline input VJP"). A test-only hook (`src/backends/cuda/resident_testing.hpp`, not installed) reports each
executor's layout and can force either path for every supported layer.

**Spline input VJP (backlog C13).** The derivative of a clamped spline is a spline of
one degree less on the same knots: `d/dx sum_k c_k B_{k,p}(x) = sum_{j=1}^{K-1} C*_j
B_{j,p-1}(x)` with the derivative coefficients `C*_j = (c_j - c_{j-1}) s_j`, `s_j = p /
(t_{j+p} - t_j)` (`spline_derivative_coefficient` and `spline_derivative_scale` in
`src/carriers/local/local_formulas.hpp`; `s_j = 0` for a zero knot gap, whose `B_{j,p-1}`
vanishes; the halved division for a gap beyond the range).
Its weights `B_{j,p-1}(x)` are nonnegative and sum to one in the domain. The signed
derivatives `B'_{k,p}`, of order `p/(knot spacing)` and summing to zero, instead multiply
the rounding of `W = U*C` by `sum_k |B'_k|`, which grows with the term count: under TF32,
for coefficients smooth over the grid, the dense input VJP exceeded the TF32 tolerance from
about 48 terms (2.6x at 48, 7.0x at 192; [C13 evidence](evidence/backlog/C13.md)). FP32/TF32
B-spline layers therefore compute, in every backward pass that needs the input VJP, `C*`
from the current `C` (one kernel; graphs, uploads and rollback need nothing) and
- dense: `C*` in rows of `terms-1`, `W* = U*C*` (cuBLAS, TF32 tensor ops under
  `TensorFloat32`, in the W scratch), then per (sample, input) `dx = sum_j B_{g+1+j,p-1}(x) W*[g+j]` over the
  `p` nonzero degree-`p-1` values of `x` (`spline_lower_terms`: the Cox-de Boor values at the
  start of the last step of the forward's evaluation, so span, side at repeated knots and
  right endpoint are the forward's);
- sparse: `C*` in the layout of `C` (rows of `terms`, the last zero; rows of `terms-1` and
  differencing while staging were both slower), the same sum over `C*` tiles staged as the C
  tiles were (`P-1` terms per sample, stored degree-`p-1` rows).
Memory: no `Phi'` rows (dense layers above 85 terms previously stored them), the knot
scales (`terms-1` values) per such layer, after its knots, and one region
shared by these layers for `C*` (the largest `outputs*inputs*(terms-1)`, or `*terms` for a
sparse layer), reserved after every other region and only when such a layer of degree >= 1
exists; allocation count unchanged; FP64 and other families' arenas are unchanged.
Degree zero gives `dx = 0` without a contraction. Error of the input VJP against the exact
`sum_j B_{j,p-1} sum_o U C*_j` at the executor's operands, per entry: at most
`u * sum_j B_{g+1+j,p-1}(x) sum_o |U[b,o]| |C*[o,i,j]|` with `u = 2^-9` (TF32: operand
rounding of `U` and `C*`, safe for truncating tensor-op kernels, plus FP32 accumulation) or
`(outputs+8)*2^-24` (FP32: any summation order, `C*` and basis rounding); it does not grow with
the term count. The CTest suite `local_spline_gradient_resident` checks exactly this bound
(both paths, degrees 1..3 and 5, up to 259 terms). FP64 layers keep `W = U*C` and `Phi'`
(results unchanged); the CPU, the FP64 reference, is unchanged. *Status:* the input VJP is
checked as before; an FP32/TF32 B-spline layer never forms `U*C`, so an overflowing `U*C`
at a term of zero derivative is not reported (the C5 sparse difference now holds for both
paths), while a `C*` beyond the FP32 range (a derivative coefficient beyond the range; a
difference `c_j - c_{j-1}` that overflows is halved first, so representable `C*` are formed) is reported by the input VJP of every sample whose
degree-`p-1` terms include it (eager `backward` raises `std::overflow_error`; a training step
raises and rolls back). The forward and its status are unchanged (the sparse basis kernel
still evaluates and checks the derivatives it no longer stores).
*Other families (TF32).* `dx = sum_k Phi'_k W_k` with `W = U*C` from tensor-op GEMMs, error
at most about `2^-10 * sum_k |Phi'_k| sum_o |U||C_k|` (the operand rounding of `C`; the
rounding of `U` only perturbs the upstream). Measured against the C1 TF32 tolerance over
sizes 8..256, five coefficient profiles (including least-squares fits of smooth functions)
and two input widths and distributions, by a rule fixed before the matrix ran (a family fails
when more than one cell with a well-conditioned forward exceeds the tolerance): Jacobi
exceeded it (up to 1.85x) for fitted coefficients at every size from 16 and its layers
therefore run their `W = U*C` GEMM with FP32 math (`CUBLAS_DEFAULT_MATH`, restored after
the call, also on an exception; host-side, captured into graphs) under `TensorFloat32`
(measured: 1.04x of the 256-wide Jacobi step, 1.15x at 1024 wide). Single cells exceed it without triggering the rule: Legendre size 16 with fitted
coefficients (1.23x; the other 31 cells pass) and normalized Hermite functions at sizes
128/256 with a non-decaying synthetic coefficient profile (1.11x/1.77x; fitted coefficients
0.17x). Hermite physicists' polynomials at fitted coefficients and Mexican hats fitted to
functions with a nonzero mean exceed it as well, but so do their TF32 forward outputs: the
representation is ill-conditioned there, which no input-VJP precision repairs; the FP32
input VJP of the 256-center Mexican-hat fit also exceeds the C1 FP32 tolerance (6.7x, FP32
forward 0.08x; unchanged by C13, not addressed here).
*Reference.* The derivative of a degree-1 spline (or at an interior knot of multiplicity
`p`) jumps at its knots, so any forward rounding that moves an activation across such a knot
changes that layer's input VJP by the full jump, and with it every upstream layer's input
and parameter VJPs. Under TF32 (forward rounding about `2^-11` of the terms) the network input
VJP of the C5 two-layer case (degrees 3 and 1) differs from the CPU evaluated at its own
activations by 0.46 (expected 0.398, -0.060 on the dense path) before and after C13; against
the CPU evaluated at the executor's own upstream it is within the C1 tolerance (0.17 of it).
Comparisons across such layers use the executor's upstream (the C5 suite therefore compares
the TF32 dense and sparse paths with the CPU only).

**Precision policy (backlog C1).** `ResidentNetwork(network, capacity, precision)` takes a
`kan::cuda::Precision`: `Float64` (default; unchanged FP64 storage and kernels, the parity
reference), `Float32` (FP32 storage of parameters, gradients, activations and workspaces;
every basis, rational and input-map formula evaluated in FP32 from the same shared
`KAN_HOST_DEVICE` source, now templated on the scalar type; cuBLAS SGEMM/SGEMV) or
`TensorFloat32` (`Float32` whose cuBLAS contractions use TF32 tensor-op math: operands
rounded to 10 mantissa bits, FP32 accumulation). `precision()` reports it. The host
interface stays `double` for every precision: uploads round to the executor precision,
downloads (outputs, gradients, `download_parameters`) are exact widenings of the device
values, so a downloaded FP32 network holds FP32-representable parameters. FP32 additionally
requires, with `std::invalid_argument` at construction or upload: every uploaded value and
every configuration scalar the kernels read has magnitude at most `FLT_MAX` (values below
the FP32 range round to subnormals or zero); quantities the CPU requires to be positive
(widths, scales, frequency, LayerNorm epsilon, tanh scale, exponentiated RBF log widths,
the learning rate) stay positive after rounding; affine scales stay nonzero; Jacobi
`alpha, beta > -1`; distinct B-spline knots stay distinct. A learning rate or L2 weight
beyond the FP32 range is rejected the same way. Nonfinite FP32 results (including any
result beyond `FLT_MAX`, finite in FP64) raise `std::overflow_error` as before; the
log-space paths use the FP32 normal range. The guarded rational pole test uses the relative
threshold `max(epsilon, n*2^-23)` (n the denominator degree): FP32 Horner evaluation cannot
resolve `|Q|` below about `n*2^-24` of the guard bound, so FP32 reports poles that the FP64
executor at `epsilon = 1e-8` accepts. FP32 results agree with the FP64 CPU reference evaluated
at the executor's parameters per entry within `2e-4*|e| + 5e-5*max|e| + 1e-37`, TF32 within
`1e-2*|e| + 1e-2*max|e|` (the suite's tolerances; measured deviations in the
[C1 evidence](evidence/backlog/C1.md)), where the TF32 bias VJP of every layer but the last
adds `2^-11*sum_b |U|` (its upstream U is itself a TF32 contraction output; the rounding does
not cancel with the batch sum; an empirical scale, needed on GB10 with cuBLAS 13.1,
[aarch64 port](evidence/port-aarch64.md)); the input VJPs of B-spline layers satisfy the
tighter per-entry bound of "Spline input VJP" (backlog C13), and gradients upstream of a
spline layer with a discontinuous derivative are compared at the executor's own upstream
(same paragraph); trajectories over
several SGD steps diverge further because SGD amplifies rounding. The FP32 small-kernel thresholds are `2^24` forward
multiply-adds and, for the parameter VJP, at most `2^15` coefficients plus outputs and at
most `2^22` `batch*(coefficients+outputs)`. Python: `kan.Precision.FLOAT64/FLOAT32/TF32`
(declared in every build) and `kan.ResidentNetwork(network, capacity, precision=...)`; arrays
stay strict float64.

**FMA build option (backlog C10).** CMake `KAN_CUDA_FMA` (default `OFF`): `OFF` is the parity
build, compiling the CUDA kernels with `--fmad=false` as before; `ON` is the performance
build, where nvcc contracts multiply-adds into FMA (`scriptsuild.ps1 -CudaFma`). In the
performance build an overflowing intermediate product can be cancelled inside an FMA without
being reported, as already stated for the cuBLAS contractions, and FP64 resident results move
within the existing tolerance. (The `cuda_preserves_unfused_intermediate_overflow_contract`
case still passes in the performance build only because the warp kernel accumulates its two
products in different lanes; that is not a guarantee.) cuBLAS always uses FMA.

Python bindings expose the same mathematics through an optional `kan` module.
Numerical tensor arguments must be NumPy C-contiguous float64 arrays of the declared
shape. No implicit float32 promotion or layout conversion is performed. Returned
arrays own their storage; expensive computation releases the GIL. Serialize access
to shared model instances and do not mutate borrowed input arrays during a call.
Building CPU
static libraries needs neither Python nor pybind11. Python package/wheel distribution
and model serialization remain outside M2.

## Localized and adaptive bases (M3)

`BSplineConfig` uses explicit clamped nondecreasing finite knots and degree 0..16;
it has `knots.size()-degree-1` terms, at least degree+1. Endpoints have exactly
degree+1 repetitions and a positive domain `[knots[degree],knots[size]]`.
Interior multiplicity is at most degree+1; full multiplicity permits a jump.
Values and input derivatives are zero outside the domain. Interior knots use
right-hand values/derivatives; the upper endpoint uses the inward left-hand
convention. Degree-zero derivatives are zero, including at jumps (a convention,
not a claim of differentiability there). Cox-de Boor terms with zero denominators
are zero. No implicit extrapolation or clipping occurs.

`MexicanHatConfig` has explicit translations `centers` and positive `scales` of
equal length, one term each. For `q=(x-center)/scale`, each term is
`A*(1-q*q)*exp(-q*q/2)`, where `A=2/(sqrt(3)*pi^(1/4)*sqrt(scale))`.
Its derivative is `A/scale*q*(q*q-3)*exp(-q*q/2)`. These are L2-normalized
continuous wavelets; configuration is fixed, with learned edge coefficients.
Extreme representable tails use log-space evaluation to preserve derivatives
even when the basis value underflows. Nonfinite mathematical results fail.

`GaussianRbfConfig` is fixed, using a scalar width. `TrainableRbfConfig` makes the
centers and `log_widths` (equal length, each exponent finite and positive) nonlinear
trainable parameters. Centers/log widths are shared
across a layer's edges, not per edge. At each term, the center derivative is the
negative input derivative, and log-width derivative is `2*q*q*exp(-q*q)`.
`BasisValues` returns these vectors only for trainable RBFs. `LayerGradients::nonlinear`
holds their VJPs as `TrainableRbfGradients`, summing over batches and edges; fixed
families hold `std::monostate`. `kan::set_rbf_parameters(layer, centers, log_widths)`
requires a `TrainableRbfEdges` layer and vectors of its current term count, validates
and atomically replaces both.
SGD validates finite candidate vectors and finite positive exponentiated widths
before committing any parameter; candidate width overflow/underflow raises
`overflow_error`, while invalid user configuration/gradients raises
`invalid_argument`. Network SGD retains whole-network atomicity.

`kan::insert_knot(layer, x)` accepts a strictly interior spline knot whose new
multiplicity is allowed. Boehm insertion adds one term and transforms every edge's
coefficients, preserving the represented function and its derivatives wherever
the declared derivative convention applies, within floating-point tolerance.
`kan::adapt_grid(layer, samples)` validates all samples, counts in-domain samples per nonzero
span (upper endpoint belongs to the final span), chooses the most populated span
(ties choose the lowest), and inserts the median (mean of middle pair for even
count) if strictly inside that span, otherwise its midpoint. Empty/outside-only
samples or a span without a representable interior value fail explicitly.
Updates are atomic; stale gradient shapes are rejected after refinement. The
`Network::insert_knot/adapt_grid` conveniences forward to them and accept an explicit layer index and samples in that layer's input domain.
Samples do not implicitly propagate through preceding layers; the network-input forms
of backlog M6 (`Network::adapt_grid(index, input, batch)`, `Network::refit_grids`) do,
explicitly (see "Per-input grids and quantile refit").

`regularization(lambda)` returns the coefficient L2 penalty
`lambda/2*sum(coefficients^2)` and its parameter VJP `lambda*coefficients`; a layer with
the SiLU residual branch adds `lambda/2*sum(w^2)` and `lambda*w` (backlog M3, below).
Lambda is finite and nonnegative. Input gradients are empty; bias and trainable
RBF gradients are correctly shaped zero vectors. Network penalties sum layers.
Add this VJP to a loss VJP explicitly before CPU SGD. Resident `backward(lambda=0)`
adds coefficient L2 gradients on GPU. Resident execution supports all eight
families, nonlinear VJPs and candidate width validation with persistent storage.
Grid refinement changes storage shape: explicitly download a model, refine it,
and reconstruct the resident executor (`upload_parameters` rejects a refined network). Numerical execution never reallocates or
silently falls back to the host. Python exposes these operations and snapshots;
`regularization` returns `(value, gradients)`. Its compatible `evaluate_basis`
continues returning `(values,input_derivatives)`; nonlinear gradients are
accessible through layer/network backward.

## Rational edges (M4)

A typed `RationalConfig` selects a nonlinear rational Layer, separately from
`BasisConfig`. Each edge is `r(x)=P(z)/Q(z)`, `z=(x-center)/scale`,
`P=sum(a[k]*z^k,k=0..m)`, `Q=1+sum(b[k-1]*z^k,k=1..n)`.
Degrees m,n are independently 0..16, default 3,2. Center is finite (default zero),
scale finite and positive (default one). Fixing Q's constant to one removes common
scale ambiguity. All a/b values are trainable per edge. Zero initialization gives
P=0,Q=1. This Padé-compatible parameterization does not automatically construct a
Taylor-series approximant. Supplied [1/1] exponential coefficients, for example,
are `a={1,0.5}, b={-0.5}`. Horner evaluation and explicit scaling condition the
polynomial evaluation; inputs are not clipped.

`RationalEvaluation` exposes value, input derivative, and numerator/denominator
partial derivatives. With primes denoting derivatives with respect to z:
`dr/dx=(P'/Q-r*Q'/Q)/scale`, `dr/da[k]=z^k/Q`,
`dr/db[k-1]=-r*z^k/Q`. Layer backward contracts these partials with upstream
gradients and sums over the batch. Center/scale/epsilon are fixed configuration.
Rare log-space paths preserve representable parameter/input derivatives when
powers or intermediate quotients become zero or subnormal; the ordinary Horner
path handles normal values. These paths do not hide overflowing intermediates.

For every executed sample/edge, require
`abs(Q)>epsilon*(1+sum(abs(b[k-1]*z^k)))`, epsilon finite and strictly between zero
and one (default 1e-8). A finite denominator failing this relative cancellation
guard raises `std::domain_error`, including exact or removable poles, zero
numerators and zero upstreams. Nonfinite intermediates/results raise
`std::overflow_error`. No clipping or pole removal changes the represented
function. The guard checks executed samples; it does not prove pole freedom
between them. Finite setters/SGD candidates can therefore fail later execution.

### Denominator policy (backlog M2)

`RationalConfig::denominator_policy` (`kan::DenominatorPolicy`, Python
`kan.DenominatorPolicy.GUARDED/ABSOLUTE/SMOOTH`) selects the denominator. With
`S(z)=sum(b[k-1]*z^k,k=1..n)` and the gain `g=dQ/dS`:

| Policy | Q | g | Poles |
|---|---|---|---|
| `Guarded` (default) | `1+S` | `1` | relative guard above, `domain_error` |
| `Absolute` (safe PAU, Molina et al. 2019) | `1+abs(S)` | `sign(S)`, `sign(0)=0` | none, `Q>=1` |
| `Smooth` | `1+S^2` | `2S` | none, `Q>=1` |

The derivatives are, for every policy, `Q'=g*S'`, `dr/dx=(P'/Q-r*Q'/Q)/scale`,
`dr/da[k]=z^k/Q` and `dr/db[k-1]=-r*g*z^k/Q`. For `Absolute`, `sign(0)=0` is the
subgradient at `S=0`: the midpoint of the one-sided derivatives (the same
convention as `abs` in PyTorch autograd). Consequently `b=0` is stationary under
both safe policies (`S` vanishes identically and every `dr/db` is zero; for `Smooth`
this is a true critical point): denominators of a safe layer must be initialized
nonzero, or they never train (`kan::initialize` does so for every policy, see backlog M4). The safe policies never report a pole and ignore
`epsilon`, which is still validated for every policy; all finiteness checks remain
(`S`, `S^2`, `Q`, `Q'` and every derivative intermediate). The log-space paths cover
the safe policies too; `Smooth` restores `r*Q'/Q` from `log|g|+log|S'|` when
`Q'=2*S*S'` underflows. A safe policy changes the represented function: supplied
Padé coefficients (such as the [1/1] exponential above) describe a `Guarded` edge.
An unknown enumerator raises `std::invalid_argument` ("invalid rational
configuration"). CPU loops dispatch the policy once per call; resident CUDA has one
forward and one parameter-VJP kernel instantiation per policy and caches `g` per
sample/edge for the safe policies only (the `Guarded` layout has no `g` region; see the
rational resident caches below).

Numerator layout is `(outputs,inputs,m+1)` and denominator layout is
`(outputs,inputs,n)`; bias is per output. A rational layer holds `RationalEdges`
(`config`, numerator `coefficients`, `denominators`); `Layer::coefficients()` exposes
numerator a and `set_parameters` replaces a and bias. `kan::set_rational_parameters`
atomically replaces a,b,bias. RBF setters and spline operations reject rational
layers. The constrained rational constructor preserves existing
`Layer(inputs,outputs,{})` basis usage. Rational gradients hold `RationalGradients`
in `LayerGradients::nonlinear`, correctly shaped including zero batch. SGD validates shapes/data and all
finite candidate vectors before network-wide commit. Numerator coefficient L2
keeps its existing definition (plus residual-branch weights, backlog M3); denominators
and bias are unpenalized.

Resident CUDA executes mixed rational/basis networks with persistent a/b storage,
analytic VJPs and atomic GPU SGD. Unsafe denominators are reported to the host as
domain_error; a failed execution invalidates its output/gradient state. Numerical
calls do not allocate GPU storage or fall back to CPU evaluation. CPU/GPU parity
remains tolerance-based. Since R7 the deprecated synchronous layer API also runs
rational layers, through this executor.

*Resident rational execution (backlog C6-C8).* Per rational layer the arena holds the
arguments `z` per input and sample (`capacity*inputs`) and edge-major caches of `P`, `Q` and
`dr/dx` (plus `g` for the safe policies) per edge and sample (`capacity*inputs*outputs`
each); every rational kernel maps consecutive threads to consecutive samples, so these
accesses are coalesced. As on the CPU, the forward pass reports a nonfinite
parameter-VJP intermediate (`z^k`, `z^k/Q`, `dr/da_k`, `dr/db_k` for `k <= max(m,n)`) of
every executed sample, also when the output and the upstream are finite or zero, as
`std::overflow_error` of that forward pass (in a training step: its forward phase); the
backward pass recomputes these values unchecked. The forward pass evaluates them in full
only where a cheap bound (largest rounded power, `|z^k/Q|`, `|r z^k/Q|` and for `Smooth`
`|g z^k/Q|` against `max/4` of the precision) does not prove them finite; the reported
status is exactly that of the full evaluation. A Horner chain is checked at its end, which
reports exactly what a check after every step would. The parameter VJP runs one warp per
edge, which accumulates all numerator, denominator (and for input 0 the bias) sums over
the samples in a fixed order: results are deterministic and bitwise those of the
pre-C6 executor in both builds ([C6-C8 evidence](evidence/backlog/C6-C8.md)).

Python exposes one class per basis configuration type (`kan.ChebyshevConfig(size=...)`,
`kan.BSplineConfig(degree=..., knots=...)`, ...; each has a `size` property and value
equality; being mutable, they are unhashable), `kan.basis_size`, and `Layer.carrier`
returns a read-only snapshot of the carrier (`kan.BasisEdges` / `kan.TrainableRbfEdges`
with `basis` and `coefficients`, `kan.RationalEdges` with `config`, `coefficients` and
`denominators`). Family operations are module functions: `kan.insert_knot(layer, x)`,
`kan.adapt_grid(layer, samples)`, `kan.set_rbf_parameters(layer, centers, log_widths)`,
`kan.set_rational_parameters(layer, coefficients, denominators, bias)`.
`LayerGradients.centers/log_widths/denominators` stay available as arrays, empty
(shape `(0,)`) for other carriers. Vector attributes are copies as well: assign a whole
list (`cfg.knots = [...]`) rather than mutating the returned list in place.
Python exposes `RationalConfig` (including `denominator_policy`, a
`kan.DenominatorPolicy` enum member), the rational Layer constructor, owned config,
parameter and gradient snapshots, and `kan.set_rational_parameters`. Rational
denominator arrays have shape `(outputs,inputs,n)`, even for n=0; basis-layer
denominator arrays have shape `(0,)`. `evaluate_rational(config,x,a,b)` accepts
strict one-dimensional float64 arrays and returns `(value,input_derivative,da,db)`.
`domain_error` maps to Python ValueError. All existing strict array/layout and
owned-snapshot rules apply.

## Edge carriers (R2)

A `Layer` holds dimensions, a per-output bias and exactly one `kan::Carrier`
(`include/kan/carrier.hpp`), a `std::variant` of:

- `BasisEdges{basis, coefficients}`: linear in its parameters. A layer is the
  expansion `Phi: R^I -> R^(I*K)` followed by the dense contraction
  `Y = Phi * C^T + bias`, `C` being `outputs x (I*K)` in the coefficient layout
  above. Holds every fixed family; a `TrainableRbfConfig` is rejected.
- `TrainableRbfEdges{basis, coefficients}`: the same expansion and contraction,
  plus nonlinear shared centers/log widths with their own VJPs.
- `RationalEdges{config, coefficients, denominators}`: nonlinear per-edge P/Q.
- `WaveletEdges{config, coefficients, translations, log_scales}`: per-edge trainable
  Mexican-hat wavelets, nonlinear in the per-edge-term translations and log scales (backlog M7,
  below); no expansion is shared between edges, so it has no contraction form.

`Layer(inputs, outputs, BasisConfig)` selects `TrainableRbfEdges` for a
`TrainableRbfConfig` and `BasisEdges` otherwise; `Layer(inputs, outputs, RationalConfig)`
selects `RationalEdges`, `Layer(inputs, outputs, WaveletConfig)` selects `WaveletEdges`.
`carrier()` returns the carrier; `terms()` is the number of
coefficients per edge; `coefficients()`/`bias()` and `set_parameters` act on every
carrier's per-edge coefficient tensor (the tensor the coefficient L2 penalizes, together
with the layer's residual-branch weights).
`set_carrier(carrier[, bias])` validates the configuration, the shapes for the layer's
dimensions and finite parameters, then replaces the carrier (and bias) atomically.
Carriers and configurations are values with `operator==`.

`LayerGradients{input, coefficients, bias, nonlinear, residual}`: `nonlinear` is a
`NonlinearGradients` variant whose alternative corresponds to the carrier
(`std::monostate`, `TrainableRbfGradients{centers, log_widths}`,
`RationalGradients{denominators}`, `WaveletGradients{translations, log_scales}`). SGD rejects a gradient whose alternative does not
match the carrier with `std::invalid_argument`. `residual` is the residual-branch weight
gradient (backlog M3), empty for a layer without the branch.

Family-specific operations are free functions in `include/kan/families.hpp`
(`insert_knot`, `adapt_grid`, `set_rbf_parameters`, `set_rational_parameters`,
`set_wavelet_parameters`); each
requires its carrier and basis type, raises `std::invalid_argument` otherwise, and
commits through `set_carrier`.

Every Layer operation dispatches on the carrier once per call. The CPU loops of each
carrier live in `src/backends/cpu/` (`linear_engine.hpp` holds the expansion and
contraction shared by `BasisEdges` and `TrainableRbfEdges`; since backlog C5 a B-spline
`BasisEdges` layer uses its sparse form, which skips the exactly-zero terms and is bitwise
the dense loops, signed zeros and nonfinite results included: a forward output whose bias is
`-0` and an edge whose `|u|*max|c|` overflows take the dense loop); the resident executor
holds one execution plan per carrier. A new carrier adds a `Carrier` and a
`NonlinearGradients` alternative, its overloads in `src/backends/cpu/edge_ops.hpp` and a
resident plan; Layer and Network do not change.

## Input maps (backlog M1)

Polynomial bases grow like `(2|x|)^n` outside `[-1,1]`. B-splines have compact support
and are zero, with zero gradient, outside it. Gaussian RBF and Mexican-hat bases have
decaying tails rather than compact support; their values and derivatives can underflow
to zero far from their centers in floating-point arithmetic. Nothing rescales
inputs implicitly; the explicit tool is an input map, a network layer kind of shape
`features -> features` (`include/kan/input_map.hpp`). `kan::InputMap(features, map)` holds
one `InputMapKind = std::variant<AffineMap, TanhMap, LayerNormMap>`:

- `AffineMap{scale, shift}`: `y[b,i] = scale[i]*x[b,i] + shift[i]`, one finite value per
  feature each, scales nonzero (a zero scale would make a feature silently constant).
  Fixed (not trained): its purpose is to place inputs in a basis domain, and training
  could move them out again. `dx = scale*u`.
- `TanhMap{scale}`: `y = tanh(scale*x)` in `(-1,1)`, fixed finite positive scale (default 1).
  `dx = u*scale*(1-y)*(1+y)`, evaluated from the output; saturated outputs give exactly 0.
- `LayerNormMap{epsilon, gain, bias}`: per sample over the features,
  `mean = sum(x)/n`, `var = sum((x-mean)^2)/n` (population, two passes),
  `xhat = (x-mean)/sqrt(var+epsilon)`, `y = gain*xhat + bias`, or `y = xhat` when gain and
  bias are both empty. Epsilon is fixed, finite and positive (default 1e-5); gain and bias
  are both empty or both of length `features`, and then trainable. With `w = u*gain`
  (or `u`): `dx = (w - mean(w) - xhat*mean(w*xhat)) / sqrt(var+epsilon)`,
  `dgain = sum_b u*xhat`, `dbias = sum_b u`. One feature gives `xhat = 0`.

Means are computed as `sum * (1/n)` on both backends. `InputMap::forward/backward/sgd`
follow the Layer rules: finite inputs, upstreams and parameters (`invalid_argument`),
nonfinite results including overflowing row moments (`overflow_error`), batch zero, and
SGD that validates gradient shapes (`InputMapGradients{input, gain, bias}`, gain/bias
empty for fixed maps) and finite candidates before committing. `set_map` validates and
replaces the map atomically. A moved-from map has zero features and its operations raise
`invalid_argument`. `affine_from_range(samples, batch, features, lower=-1, upper=1)`
builds the fixed map sending each feature's sample `[min,max]` onto `[lower,upper]` (up to
rounding; a constant feature gets scale 1 and the midpoint; an overflowing span or a
scale that underflows to zero raises `overflow_error`), and
`affine_from_moments(samples, batch, features)` the standardizing map `(x-mean)/std`
(constant feature: scale 1, centered). Both validate the sample shape and finiteness.

A `Network` is a nonempty sequence of `NetworkLayer = std::variant<Layer, InputMap>`
with matching adjacent dimensions; `Network(std::vector<Layer>)` remains valid.
`NetworkGradients::layers` holds `NetworkLayerGradients =
std::variant<LayerGradients, InputMapGradients>`, the alternative matching each
position, and network SGD rejects a mismatched alternative with `invalid_argument`.
Layer indices of every Network API are positions in `layers()`, maps included;
`insert_knot`/`adapt_grid` at an input map raise `invalid_argument`. The coefficient L2
penalizes KAN layer coefficients and residual-branch weights only: a map contributes zero value and zero gain/bias
gradients (no input gradient), like RBF and rational nonlinear parameters. Every
operation dispatches on a layer's kind once per call. `Network::inputs()/outputs()`
give the network's dimensions.

The resident executor supports input maps anywhere in a network: elementwise affine and
tanh kernels, LayerNorm row kernels (fixed-order group shuffles) and a tiled fixed-order
gain/bias reduction, without floating-point atomics. Trainable gain/bias live in the
parameter regions and take part in the network-wide atomic SGD; fixed map parameters are
uploaded at construction. CPU/GPU parity is tolerance-based. Python exposes
`kan.AffineMap`, `kan.TanhMap`, `kan.LayerNormMap` (value classes with equality; their
vector attributes are copies, so assign whole lists), `kan.InputMap`
(`features`, `map` snapshot, `set_map`, `forward`, `backward`, `sgd`),
`kan.InputMapGradients` (`input`, `gain`, `bias`), `kan.affine_from_range(samples, lower,
upper)` and `kan.affine_from_moments(samples)`; `kan.Network` accepts and returns mixed
lists of `Layer` and `InputMap`, and `NetworkGradients.layers` mixes `LayerGradients` and
`InputMapGradients`.

## Initializers (backlog M4)

`include/kan/initializers.hpp`. Constructors are unchanged (zero coefficients, bias and
denominators); an initializer is applied only by `kan::initialize(Layer&, Initializer)` or
`kan::initialize(Network&, Initializer)`, with `Initializer = std::variant<VarianceScaling,
NoiseInit>`. Both carry `distribution` (`kan::Distribution::Uniform` / `Normal`), a 64-bit
`seed` and `denominators` (`DenominatorInit{bound = 0.5, radius = 1}`). Validation, all with
`std::invalid_argument` before anything changes: gain / noise scale finite and positive, a
known distribution, `bound` in (0, 1), `radius` finite and positive, a valid (not moved-from)
layer, and for a `Guarded` rational layer `epsilon*(1+bound) < 1-bound`. Results that leave
the double range raise `std::overflow_error`. Success replaces coefficients, bias (zero) and
rational denominators through `set_carrier`; trainable RBF centers/log widths are kept (they
define the reference measure). The network form validates every layer on copies before
assigning, initializes the KAN layer at position p with seed `layer_seed(seed, p)` and resets
LayerNorm maps that have gain/bias to gain 1, bias 0; affine and tanh maps are unchanged.

**Determinism.** The generator is SplitMix64: draw j of a layer is
`mix(seed + (j+1)*0x9E3779B97F4A7C15)`; `layer_seed(seed, p)` is `mix(seed + (p+1)*gamma)`.
A uniform raw draw is `2u-1`, `u = (draw >> 11)*2^-53`, in [-1, 1); a normal raw draw is the
Marsaglia polar method (pairs, rejecting `s >= 1` and `s = 0`) with a logarithm implemented
from IEEE basic operations. Every derived quantity (moments, scales, the exponential of
trainable log widths) uses basic operations, `sqrt`, `frexp`, `ldexp` and `nearbyint` only,
so the same configuration and seed give bitwise-identical parameters with MSVC and GCC
(pinned digests in `tests/init/initializer_test.cpp`). Draw order: coefficients in layout order,
then denominators in layout order, then (layers with the residual branch only) residual
weights in layout order. A parameter is a per-term factor times one raw draw. Normal draws
come in pairs from one generator: a pending second value of the coefficient draws survives
the (uniform) denominator draws and is the first residual draw.

**VarianceScaling{gain}.** `kan::reference_moments(BasisConfig)` returns the variance of the
family's reference measure and `m_k = E[phi_k(x)^2]` under it:

| Family | Reference measure | m_k | variance |
|---|---|---|---|
| Chebyshev | arcsine on [-1,1] (orthogonality weight) | 1, then 1/2 | 1/2 |
| Legendre | uniform on [-1,1] | 1/(2k+1) | 1/3 |
| Jacobi | normalized weight (1-x)^α(1+x)^β | h_k/h-mass, by its ratio recurrence | 4(α+1)(β+1)/((α+β+2)²(α+β+3)) |
| Hermite (physicists') | N(0, 1/2) (weight e^{-x²}) | 2^k k! | 1/2 |
| Hermite functions (backlog M5) | N(0, 1/2), the same input measure | Γ(k+1/2)/(√2 π k!): 1/√(2π), then ·(2k-1)/(2k) | 1/2 |
| Fourier | uniform on one period [-π/ω, π/ω] | 1, then 1/2 | (π/ω)²/3 |
| B-spline | uniform on the domain [t_p, t_K] | (p+1)-point Gauss-Legendre per span (exact) | L²/12 |
| Gaussian RBF (fixed, trainable) | uniform on [min c, max c] | 8-point Gauss-Legendre, 36 panels on c ± 9w | L²/12 |
| Mexican hat | uniform on [min c, max c] | 8-point Gauss-Legendre, 36 panels on c ± 12s | L²/12 |
| Per-edge wavelets (backlog M7) | uniform on [min(τ − s), max(τ + s)] over the layer's wavelets | per wavelet, closed form `(G(q_hi) − G(q_lo))/L`, `G(q) = erf(q)/2 + (q − 2q³)e^{−q²}/(3√π)` | L²/12 |

The Hermite functions are orthonormal under the Lebesgue measure, which is not a probability
measure; they keep the physicists' N(0, 1/2), so switching the normalization does not change the
input distribution the initializer assumes (orthogonality under the measure is not needed, see
below). Coincident centers use [c-h, c+h], h the largest width or scale. With independent zero-mean
coefficients `Var_c E_x[y²] = sum_i sum_k sigma_k² m_k` without cross terms, so
`sigma_k² = gain²·variance / (inputs·terms·m_k)` (`m_k = 0` gives 0) gives every term the same
share and `E[y²] = gain²·variance` for inputs drawn from the measure (variance-preserving).
Uniform draws are `sqrt(3)·sigma_k·(2u-1)`, normal draws `sigma_k·z`. Per-edge wavelets
(`WaveletEdges`, backlog M7) use one moment per wavelet, `sigma_j² = gain²·variance /
(inputs·size·m_j)`, with `s = exp(l)` by the portable exponential and `m_j` in closed form (erf by
its positive series `(2/√π) q e^{−q²} Σ (2q²)^n/(2n+1)!!` and the portable exponential, `G = ±1/2`
beyond `|q| = 7`: bitwise portable, about 0.12 µs per wavelet, within 1.4e-15 of a long-double
reference); the measure spans every wavelet's central lobe `[τ - s, τ + s]` (`[-1, 1]` for the
constructor's placement). Their translations and log scales are kept, like trainable RBF centers;
`NoiseInit` uses `G = size`. With one wavelet per edge all edges of an input share ψ, whose mean on
the measure is far from zero, so a single draw's `E[y²]` spreads over about 0.8–1.4 of the target
(the guarantee is in expectation over initialization; the test averages 16 seeds). Rational edges use z
uniform on [-radius, radius] (x on center ± radius·scale, variance (radius·scale)²/3) and the
edge's own drawn denominator: `mu_k = E[u^{2k}/Q(radius·u)²]` by 16 panels of 8-point
Gauss-Legendre, `sigma_k = sqrt(gain²·variance/(inputs·(m+1)·mu_k)) / radius^k`. A layer whose
inputs leave the measure keeps the guarantee only approximately: put an input map between
layers of bounded-domain families (Chebyshev, Legendre, Jacobi, B-spline); measured second
moments over 8 layers stay within 0.3–1.5 times the target for every family.

**NoiseInit{scale = 0.3}** (pykan `KANLayer`, master `ecde4ec`): amplitude
`a = scale / (G·sqrt(inputs))`, G = terms - degree for B-splines (grid intervals), the term
count for other families and m+1 for rational numerators; uniform draws `(a/2)·(2u-1)` are
pykan's `U(-a/2, a/2)`, normal draws `(a/sqrt(12))·z` have the same variance. Differences from
pykan: the noise is put on the coefficients rather than on the G+1 grid values followed by a
least-squares fit; `scale_sp = 1/sqrt(inputs)` is folded into the amplitude; the SiLU base
branch is opt-in per layer (backlog M3, below) — its weights are initialized as pykan's
`scale_base` only for layers that have it, and deep NoiseInit networks without it propagate
almost no signal (see the M4 evidence); pykan's generator is torch's, ours is SplitMix64.

**Rational denominators** (both initializers). `beta_k = ±(bound/n)·(1+|ξ|)/2` with sign and ξ
from one uniform draw, so `|beta_k|` is in [bound/(2n), bound/n], never zero, and
`b_k = beta_k / radius^k`. Then for every `|z| <= radius`, `|S(z)| <= sum|b_k||z|^k <= bound`:
`Q = 1+S >= 1-bound > epsilon(1+bound) >= epsilon(1+sum|b_k z^k|)` for `Guarded` (the relative
guard cannot fire on the domain), `Q` in [1, 1+bound] for `Absolute` and [1, 1+bound²] for
`Smooth`; no safe-policy denominator starts at the stationary point b = 0. A `b_k` that is
zero or nonfinite in double (radius^k out of range) raises `std::overflow_error`. Outside the
domain no bound is claimed.

Python: `kan.Distribution.UNIFORM/NORMAL`, `kan.DenominatorInit(bound, radius)`,
`kan.VarianceScaling(gain, distribution, seed, denominators)`, `kan.NoiseInit(scale, distribution,
seed, denominators, residual_mean, residual_spread)`
(value classes with equality), `kan.initialize(layer_or_network, initializer)` (in place,
GIL released, `ValueError`/`OverflowError`), `kan.reference_moments(config)` returning
`(variance, moments)` and `kan.layer_seed(seed, position)`. Initialized networks run on the
resident executor by construction or `upload_parameters`; no executor change.

## Residual branch (backlog M3)

`include/kan/layer.hpp`. A `Layer` optionally holds `SiluResidual{weights}`, layout
`(outputs, inputs)` (the carrier's edge order), and computes

```
y[b,o] = bias[o] + carrier_o(x_b) + sum_i w[o,i] * silu(x[b,i]),   silu(x) = x * sigmoid(x)
```

for every carrier (`BasisEdges`, `TrainableRbfEdges`, `RationalEdges`): the edge function is
`phi_{o,i}(x) = w[o,i]·silu(x) + carrier_{o,i}(x)`, the base branch of the original KAN
(pykan's `scale_base·silu(x)`). It is off by default (constructors are unchanged).
`residual()` returns `const std::optional<SiluResidual>&`; `set_residual(optional)` validates
the shape (`outputs*inputs`) and finite weights with `std::invalid_argument` and enables,
replaces or (`std::nullopt`) disables the branch atomically. The branch belongs to the layer,
not to the carrier: `set_carrier`, `set_parameters`, `insert_knot`, `adapt_grid`,
`set_rbf_parameters` and `set_rational_parameters` keep it unchanged. Copies are values;
`SiluResidual` has `operator==`.

Purpose: B-splines are exactly zero, with zero input gradient, outside their compact
support. Gaussian RBF and Mexican-hat bases have decaying tails; sufficiently far from
their centers, both values and derivatives can underflow to zero. If every carrier
derivative vanishes at a layer's inputs, no input gradient crosses that layer without
the branch. `tests/core/residual_test.cpp` checks zero gradients without the branch and
nonzero gradients with it on the tested B-spline and remote Gaussian-RBF inputs,
including the preceding layer's coefficient gradients. The branch itself does not
guarantee a nonzero gradient for every input or upstream.

**Formulas** (single host/device source `src/core/residual_formulas.hpp`, templated on the
scalar): with `e = exp(-|x|)`, `sigma = 1/(1+e)` and `1-sigma = e/(1+e)` for `x >= 0`,
`sigma = e/(1+e)` and `1-sigma = 1/(1+e)` for `x < 0`; `silu = x·sigma`,
`silu' = sigma·(1 + x·(1-sigma))`. No intermediate is nonfinite for any finite double or float
`x` (no exp overflow, no cancellation for `1-sigma`); `silu(±DBL_MAX)` is `DBL_MAX` / `-0`.

**Order of evaluation (CPU).** Forward: the carrier's output (bias included, unchanged code
path), then per sample and output `r = sum_i w[o,i]·silu(x[b,i])` accumulated from 0 in
ascending i, then `y += r`. Backward, after the carrier's VJPs:
`dw[o,i] = sum_b u[b,o]·silu(x[b,i])` (ascending b) and
`dx[b,i] += silu'(x[b,i])·t`, `t = sum_o u[b,o]·w[o,i]` (ascending o). silu and silu' are
evaluated once per sample and input. Nonfinite results raise `std::overflow_error`, as for the
carriers. The coefficient, bias and nonlinear VJPs are unchanged.

**SGD** updates `w -= rate·dw` with the other parameters, validated before any commit;
`LayerGradients::residual` must hold `outputs*inputs` finite values for a layer with the branch
and be empty otherwise (`std::invalid_argument`).

**L2.** `regularization(lambda)` penalizes every linear parameter of the edge functions:
value `lambda/2·(sum c² + sum w²)` (coefficients first, unchanged, then the weights), gradient
`lambda·w` in `residual`. Leaving w unpenalized would let a fit escape into the unpenalized
branch. Denominators, RBF centers/widths and bias stay unpenalized.

**Initializers.** `kan::initialize` keeps the branch's presence: a layer without it is
initialized exactly as before (the nine M4 digests are unchanged). For a layer with it,
`NoiseInit` draws pykan's `scale_base = (residual_mean + residual_spread·(2u-1))/sqrt(inputs)`
(defaults `residual_mean = 0`, `residual_spread = 1`, pykan's MultKAN `scale_base_mu/sigma`;
Normal draws `(residual_mean + residual_spread/sqrt(3)·z)/sqrt(inputs)`, equal mean and
variance), after the carrier's draws; `residual_mean` must be finite and `residual_spread`
finite and `>= 0` (`std::invalid_argument`). `VarianceScaling` sets `w = 0`: the carrier keeps
the `E[y²] = gain²·variance` guarantee under the reference measure, in expectation over
initialization. Zero weights do not force `dw = sum_b u·silu(x)` to vanish: the branch
can learn on the first step, depending on its inputs and upstream (the sum can also
be zero). Its contribution to `dx` is zero while `w = 0`, so it starts passing input
gradients only after a weight update makes that contribution nonzero.
pykan's trainable `scale_sp` is not a separate parameter here: it
multiplies the carrier and is a reparametrization folded into the coefficients.

**Python.** `Layer.residual` returns an owned `(outputs, inputs)` float64 array or `None`;
`Layer.set_residual(weights)` takes a strict C-contiguous float64 array of that shape or `None`
(`TypeError` for other types, `ValueError` for shape or nonfinite data, GIL released);
`LayerGradients.residual` is `(outputs, inputs)`, or shape `(0,)` without the branch (as
`denominators`). Gradients record the branch's presence: `sgd` with a gradient of the other
kind raises `ValueError`. `NoiseInit` takes `residual_mean` and `residual_spread`.

**Resident executor (phase 2).** `kan::cuda::ResidentNetwork` executes the branch for every
carrier in every precision (`Float64`, `Float32`, `TensorFloat32`), in the eager calls and in the
captured training steps; the deprecated `kan::cuda::forward/backward(Layer)` adapters carry it.

- *Layout.* A KAN layer's block in the parameter, gradient and SGD-candidate regions is
  `[coefficients | bias | nonlinear | residual]`: `W` (outputs, inputs) is a trailing part, empty
  without the branch, so layers without it keep their blocks and the executor's arena is
  unchanged for networks without the branch (results bitwise as before in both builds). SGD
  candidates, their validation and the C9 commit/rollback cover `W` like every other parameter: a
  failing training step leaves `W` at the last good step. Each layer with the branch reserves two
  `capacity*inputs` workspace regions, `S = silu(X)` and `silu'(X)`, written by its forward and read
  by its backward; when a branch takes a cuBLAS path at capacity the arena also holds one 32 MiB
  cuBLAS workspace used by the branch's GEMMs only (the carriers keep their 4 MiB). All of it is
  inside the construction-time arena (`workspace_allocations()` unchanged).
- *Order.* Forward computes `Y += S*W^T` after the carrier's output, backward `dW = U^T*S +
  lambda*W` and `dX += silu'(X) (.) (U*W)` after the carrier's VJPs. On the device the sums are
  not separate as on the CPU: for `BasisEdges`/`TrainableRbfEdges` the small forward contraction
  accumulates the branch into the carrier's per-lane partial sums before one reduction and then
  adds the bias, and the cuBLAS path adds `S*W^T` to `Phi*C^T` before the bias; rational layers
  add the branch to their finished output. Small shapes use warp/tile kernels and tiled
  reductions under the carriers' thresholds (forward `batch*outputs*inputs` at most
  `2^23`/`2^24` multiply-adds for FP64/FP32 and, rational only, a 32-sample S tile in shared
  memory; dW the expansion carriers' parameter-VJP rule), larger ones cuBLAS (`TF32` math in
  `TensorFloat32`). Parity with the CPU is tolerance-based, as for the carriers: FP64 within
  `1e-11*|e| + 1e-12*max|e|` in the suite; FP32/TF32 within the C1 tolerances, where TF32 adds
  `2e-3*sum_b |U||S|` for the weight VJP (TF32 operand rounding under cancellation).
  Results are deterministic for one GPU, driver and cuBLAS version.
- *L2.* `backward(l2)` and training steps add `lambda*w` to the weight VJPs on the device (the
  CPU's `regularization` gradient); biases, RBF centers/widths and denominators stay unpenalized.
- *Training steps.* A step skips the network input gradient: for a first layer with the branch
  only its `dX` is skipped; `dW` is computed. `Loss::OutputGradient` steps stay bitwise the eager
  `forward(); backward(l2); sgd(rate)` sequence.
- *Status.* `S`, `silu'`, the branch's outputs and both VJPs are checked like the carrier results
  (`std::overflow_error`, attributed to the forward or backward phase of a training step). The
  formulas never produce a nonfinite intermediate for finite input (x = +-1e3 FP64, +-100 FP32 are
  tested); an overflowing `S*W^T` or `U^T*S` is reported like a carrier overflow. In the
  performance build (`KAN_CUDA_FMA=ON`) nvcc may contract `1 + x*(1-sigma)` and the branch's
  multiply-adds into FMA; results stay within the tolerance and no stated contract depends on
  the unfused rounding (`rounded_product` is not used).
- *Upload and download.* Branch presence is structure: `upload_parameters` with a network whose
  layer has the branch where the executor's has not, or the reverse, raises
  `std::invalid_argument` naming the layer's "residual branch presence", executor unchanged.
  The weights are values, validated with the construction rules (finite; FP32 magnitudes at most
  `FLT_MAX`). Construction accepts the branch on any layer. `download_parameters()` returns the
  layers with the branch and their current weights (exact widening), `download_gradients()`
  fills `LayerGradients::residual` (empty without the branch).

## Per-edge trainable wavelets (backlog M7)

`include/kan/wavelet.hpp`, carrier `kan::WaveletEdges` (Wav-KAN, Bozorgasl and Chen 2024). Every
edge (o,i) holds `config.size >= 1` wavelets k, each with its own coefficient `w`, translation `τ`
and log scale `l` (layout `(outputs, inputs, size)`, index `j = (o·inputs + i)·size + k`):

```
phi_{o,i}(x) = sum_k w_j psi_j(x),   psi_j(x) = A s^(-1/2) (1 - q^2) exp(-q^2/2),
q = (x - τ_j)/s,  s = exp(l_j),  A = 2/(sqrt(3) pi^(1/4))
```

the L2-normalized Mexican hat of `MexicanHatConfig` with τ and s trainable per edge and term.
`size = 1` is Wav-KAN's edge `w ψ((x - τ)/s)` up to the factor `s^(-1/2)`, a reparametrization of
`w` that keeps the M3 normalization (and its moments) and changes only `d/dl`; `size > 1` is a
per-edge trainable wavelet dictionary (the M3 dictionary made per-edge and trainable). Scope
decision (M7 evidence): per-edge parameters as the backlog states, not per-input shared ones (that
would be a Mexican-hat `TrainableRbfEdges`, with a GEMM form); the cost is measured below.
`coefficients` is the carrier's per-edge coefficient tensor (`coefficients()`, `set_parameters`,
the L2 penalty); τ and l are nonlinear parameters (unpenalized, like RBF centers/widths).

- *Configuration and validation.* `WaveletConfig{size = 1}` (value type, `operator==`);
  `validate_wavelet` rejects `size = 0`. A log scale is valid when `l` is finite and `exp(l)` and
  `exp(-l)` are finite and nonzero (`kan::valid_log_scale`, |l| below about 709.78);
  translations must be finite. Invalid user data raise `std::invalid_argument` (`set_carrier`,
  `set_wavelet_parameters`, `evaluate_wavelet`).
- *Construction.* `Layer(inputs, outputs, WaveletConfig)`: zero coefficients and bias; term k of
  every edge at `τ = -1 + (2k+1)/size`, `s = 1/size` (`l = -log(size)`): equispaced hats on [-1, 1]
  whose zero crossings meet; `size = 1` gives Wav-KAN's `τ = 0, s = 1`. Distinct starting τ matter:
  terms of an edge that start equal stay equal under SGD.
- *Derivatives* (one wavelet, unit coefficient, `kan::evaluate_wavelet(x, τ, l)` returns all four):
  `dψ/dx = A s^(-3/2) q (q² - 3) E`, `dψ/dτ = -dψ/dx`, `dψ/dl = -A s^(-1/2) E (q² - r1)(q² - r2)`
  (`= A s^(-1/2) E (-q⁴ + 3.5 q² - 0.5)`, roots `r1,2 = (7 ± sqrt 41)/4` correctly rounded; the
  factored form keeps relative accuracy near the roots), `E = exp(-q²/2)`.
- *Evaluation* (single host/device source `wavelet_scales`, `wavelet_terms` in
  `src/carriers/local/local_formulas.hpp`). Per wavelet once per call: `r = exp(-l)`,
  `a = A sqrt(r)`; per sample `q = (x - τ) r`, then one exponential. Where `q²` is not finite the
  distance is retried as `x r - τ r` (an overflowed `x - τ`), otherwise the result is the exact zero
  tail; where `E` is below the normal range (`q² > 1416` FP64, `> 175` FP32) all three are formed in
  log space (out-of-line `wavelet_tail_terms`), which keeps the derivatives of narrow wavelets whose
  `E` underflows (tested at `s = exp(-700)`, 40 scales out). No intermediate is NaN for finite data.
  Accuracy against long-double closed forms: relative `1e-15 (8 + q²)` (the rounding of `q²/2`) with
  an absolute floor `1e-15 a` near the polynomial zeros.
- *CPU order.* Forward `y[b,o] = bias[o]` then `+= w_j ψ_j(x[b,i])` in ascending i, then k;
  backward per sample, output, input and term (`t = w dψ/dx`): `dx += u t`, `dw += u ψ`,
  `dτ -= u t`, `dl += u (w dψ/dl)`, the parameter sums in ascending b. Nonfinite intermediates or
  results raise `std::overflow_error`.
- *SGD, L2, residual branch.* SGD updates `w`, `τ`, `l` with the other parameters after validating
  everything; a candidate log scale outside the valid range raises `std::overflow_error` with the
  layer unchanged (as an RBF width). `regularization` penalizes `w` (and the residual weights) only;
  `WaveletGradients` of the penalty are zero vectors. The residual branch works as for every carrier.
  `set_parameters` keeps τ and l; `set_wavelet_parameters(layer, coefficients, translations,
  log_scales, bias)` replaces all of them atomically (`std::invalid_argument` for another carrier).
- *Initializers.* See the initializer section (lobe-span measure, per-wavelet moments, τ and l kept,
  `NoiseInit` `G = size`); pinned digest `b36eaec5aa0cc030` (GCC, aarch64; the quadrature of the
  first M7 version gave `ebfe9f8b648306d3`).
- *Python.* `kan.WaveletConfig(size=1)`, `kan.Layer(inputs, outputs, wavelet=...)`,
  `kan.WaveletEdges` snapshot (`config`, `coefficients`, `translations`, `log_scales`, each
  `(outputs, inputs, size)`), `kan.set_wavelet_parameters`, `kan.evaluate_wavelet(x, translation,
  log_scale)` returning `(value, d/dx, d/dtranslation, d/dlog_scale)`, `kan.valid_log_scale`,
  `LayerGradients.translations` / `.log_scales` (shape `(0,)` for other carriers).

**Resident executor.** `WaveletEdges` run in every precision, eager and in captured training steps.
Parameter block `[coefficients | bias | translations | log scales | residual]`. Structure for
`upload_parameters`: the `WaveletConfig` (a different `size` is rejected); translations and log
scales are state. FP32 rejects translations beyond `FLT_MAX` and log scales whose `exp(±l)` leave
the float range (`std::invalid_argument`, construction and upload). SGD candidates of the log scales
are checked on the device (`validate_log_scale_kernel`), so a failing step rolls back (C9) and eager
SGD stays atomic. A finite candidate outside the range raises `std::overflow_error` "resident
wavelet log-scale candidate leaves the valid range" (status bit 4, the CPU's cause). A nonfinite
candidate, including one that a finite gradient pushes to ±inf, reports "nonfinite resident
numerical result" as before; the CPU names the range in the latter case, with the same exception
type. Trainable RBF widths report "nonfinite" in both cases (their kernels are unchanged).

- *Kernels* (`src/backends/cuda/resident.cu`). Per forward: `wavelet_scales_kernel` (r, a per
  wavelet; reused by the backward), the layer input transposed to `xt [i·capacity + b]` and
  `wavelet_forward_kernel` (block = output, sample chunk and input split, thread = S samples, lanes =
  consecutive samples; parameters at block-uniform addresses; unsplit, the sum from the bias in
  ascending i, k: the CPU order; with z > 1 input splits each split sums its inputs from zero and
  `wavelet_forward_finish_kernel` adds bias + split 0 + split 1 + ... in order). Per backward: the upstream transposed to `ut [o·capacity + b]`, `wavelet_bias_kernel`
  (warp per output, fixed lane order and shuffle tree), `wavelet_backward_kernel` (block = input,
  partial slot and output split; one evaluation per sample and wavelet gives dx, accumulated over
  (o, k) in the CPU order, and the three sample sums, reduced per wavelet by a fixed lane butterfly,
  then over the 8 warps in order, into the slot's partial plane), `wavelet_input_finish_kernel` (dx
  over the output splits, in order) and `wavelet_finish_kernel` (parameter VJPs over the slots, in
  order, `+ λw`). No floating-point atomics: results are deterministic for one device, batch and
  build. The direct path is branch-free; a group of samples whose warp has a rare sample (log-space
  or retried distance) is recomputed with `wavelet_terms` before use, so every lane gets the
  formula's result (bitwise in the parity build; with `KAN_CUDA_FMA=ON` the two inlined copies may
  contract differently). Full sample chunks compile without batch-end tests; in the last, partial
  chunk lanes beyond the batch evaluate at `x = τ` with `u = 0` (direct path, exact zero
  contributions) and warps entirely beyond it skip the sample slot. Intermediates are not checked one by one: every CPU-guarded intermediate is
  summed into a reported result, so the reported status is the CPU's.
- *FP32 envelope.* The FP32 direct path computes `E = 2^(c q²)`, `c = -log2(e)/2` rounded to float,
  with one `ex2.approx.ftz` (relative error below 2^-22; the rounding of `c q²` adds at most
  `|c q²| 2^-24 < 8e-6` to the exponent) and flushes exactly where the log-space branch takes over.
  FP64 uses `exp` (the CPU formula). TF32 has no contraction in this carrier: it computes it as FP32
  (only a residual branch's GEMMs use TF32).
- *Launch geometry* (selected per call from the batch by a cost model, M7 evidence "Launch model"):
  for every instantiated S (FP32 1, 2, 4, 8; FP64 forward 1, backward 1, 2, 4) and split count z the
  estimated time `(max(1, b/n) + 0.5)·(c/slots·it·max(n·(S + 3), 20) + 50·n)` is minimized (plus 100
  for the extra reduction kernel of z > 1; ties keep the larger S, then the smaller z), with c the
  chunks of 256·S samples, `it` the wavelets a block evaluates per chunk, b the blocks per
  multiprocessor and n = min(⌈b⌉, resident blocks of that instantiation). Multiprocessors and
  resident blocks come from the CUDA runtime (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` per
  instantiation), so the choice adapts to the device; the relative costs (3, 20, 0.5, 50, 100, in
  thread evaluations) were fitted on GB10 to per-kernel times of 3080 forced geometries. Forward
  input splits are considered while `outputs·chunks·z` stays within one wave of the instantiation;
  backward output splits up to `min(outputs, 16, (inputs + outputs)/inputs)` (the dx partials stay
  within the layer's `xt + ut` size). Backward partial slots: at construction the chunks of a full
  batch at S = 4 (FP64 2), at least `wave/inputs` (`wave` = multiprocessors x resident blocks of the
  largest backward instantiation), at most what fits the layer's `xt + ut` size. The backward kernel
  is compiled for 3 resident blocks per multiprocessor in FP32 (`__launch_bounds__`; at most 80
  registers allocated, no spills on sm_86 or sm_121). A private test hook (`testing::WaveletLaunch`:
  samples, slots, output splits, input splits) forces any instantiated geometry for
  device-independent tests (others raise `std::invalid_argument`).
- *Workspace* (per wavelet layer, in the construction-time arena): `xt` (capacity x inputs), `ut`
  (capacity x outputs), r and a (one value each per wavelet), the partial planes (slots x 3 x
  wavelets), with output splits splits x capacity x inputs dx partials and, for layers whose
  forward may split its inputs, the forward partial sums (at most the largest `z·batch·outputs` the
  model may choose for a batch up to the capacity, within one wave of blocks of 256·S samples).
- *Parity.* FP64 within `1e-11|e| + 1e-12 max|e|` of the CPU in the suite (order of the parameter
  sums, of split forward sums and the libm/CUDA exponential differ); FP32/TF32 within the C1
  tolerances; training steps bitwise the eager sequence (the geometry is a function of the batch and the
  device; a sample's forward output may differ in rounding between batches or devices).
- *Cost* (complete `train_step`, GB10, M7 evidence): an evaluation per sample, output, input and
  term with no GEMM; against the fixed dictionary with the same parameters per edge (3K terms)
  1.3-5.5x the FP32 step (2.9-3.1x at 256 wide), 4.1x in TF32, 3.3-6.6x in FP64. Narrow wavelets
  whose warps mostly hold log-space samples cost more per evaluation (FP32 64-wide, placement
  s = 1/16 and 1/32 on [-1, 1]: 3.1x and 3.8x the evaluation cost at s = 1/4). A batch just above
  a multiple of 256·S costs more than its valid samples suggest (FP32 256-wide: 2049 samples 1.54x
  of 2048).

## Per-input grids and quantile refit (backlog M6)

**Per-input grids.** `BSplineConfig{degree, knots, grids}`: `knots` holds `grids` clamped
knot vectors of equal length `n` one after another, each validated as an M3 spline (degree
0..16, clamped endpoints of multiplicity exactly degree+1, interior multiplicity at most
degree+1, positive domain); `basis_size = n - degree - 1` for every grid. `grids = 1`
(the default, so `BSplineConfig{p, knots}` is unchanged) is one grid shared by every input;
a layer with `grids = inputs` gives input `i` the knots `knots[i*n, (i+1)*n)` (pykan's
per-input grid). Any other count raises `std::invalid_argument` (constructor,
`set_carrier`). Degree and term count are common to the grids, so the coefficient layout
`(outputs, inputs, terms)`, L2, initializers' draw order, `set_parameters` and the resident
contraction, sparse path and C13 derivative coefficients are unchanged; edge `(o, i)` uses
grid `i`. `grids` is part of `operator==` and of the structure `upload_parameters` checks.
`evaluate_basis` of a configuration with several grids raises `std::invalid_argument`;
`spline_grid(config, g)` returns grid `g` as a single-grid configuration,
`per_input_grids(config, inputs)` repeats a single grid (identical grids give bitwise the
shared layer's results on the CPU and the resident executor). `reference_moments` takes a
single grid; `VarianceScaling` draws input `i`'s coefficients with the moments and
variance of its own grid (uniform on its domain), so `E[y^2]` is `gain^2` times the mean
of the inputs' reference variances; `NoiseInit` uses `G = terms - degree`, common to all
grids. Only B-splines have per-input grids: Gaussian and Mexican-hat dictionaries are
translation-scale families whose per-input affine placement is an explicit `AffineMap`
(backlog M1), trainable RBFs learn their centers and widths, and per-edge wavelets
(backlog M7) learn their own translations and scales.

**Knot insertion.** `insert_knot(layer, x)` and `adapt_grid(layer, samples)` keep their M3
behavior and require one shared grid. `insert_knots(layer, knots)` inserts `knots[g]` into
grid `g` (one per grid, each strictly inside its domain, allowed multiplicity) by Boehm
insertion, with the M3 arithmetic. `adapt_grid(layer, input, batch)` takes layer inputs
`(batch, inputs)`, applies the M3 rule (median of the most populated span, else its
midpoint) per grid — to all values of the batch for a shared grid (the 1-D form's result),
to column `i` for grid `i` — inserts the knots and returns them.

**Quantile refit** (`refit_grid(layer, input, batch, GridRefit)`, pykan's
`update_grid_from_samples`; shared host/device formulas in
`src/carriers/local/grid_refit.hpp`). `GridRefit{intervals = 0, grid_eps = 0.02,
ridge = 1e-8, per_input = true, margin = 0.01}`: `grid_eps` in [0, 1], `ridge` and `margin`
finite and nonnegative (`std::invalid_argument` otherwise, also for a non-spline layer,
an input of the wrong shape, nonfinite input and batch 0). New grids: one per input
(`per_input`, or one shared grid from all values of the batch), each with `G = intervals`
(0 keeps `terms - degree`) knot intervals and the layer's degree. From the `n` sorted
samples `x_0 <= ... <= x_(n-1)` of a grid, `r = x_(n-1) - x_0`, domain
`[a, c] = [x_0 - margin*r, x_(n-1) + margin*r]`, `h = (c - a)/G`, interior knot
`j = 1..G-1`: `grid_eps*(a + h*j) + (1 - grid_eps)*x_(floor(n/G*j))` (pykan's quantile index
and mixing, computed in double as `double(n)/G*j`), endpoints `a` and `c` repeated
degree+1 times. With `margin = 0` the extreme samples are the endpoints exactly, which
leaves them on the boundary of the clamped domain (any later perturbation moves them
outside, where the spline is zero); pykan instead extends its grid `k` intervals beyond
the samples. The interior knots must be strictly increasing and finite (in the executor's
precision on the device): samples that span no interval, or quantile knots that coincide
(repeated samples with `grid_eps = 0`), raise `std::invalid_argument`. Coefficients: for
every input, with its own samples `x_b` (the batch column) and the current edge functions
`f_(o,i)`, `c = argmin sum_b (s(x_b) - f(x_b))^2 + ridge*sum c^2`, i.e. the banded normal
equations `(B^T B + ridge*I) c = B^T f` with `B^T f = M c_old` (`M` the cross Gram matrix of
the new and the current basis; samples outside the current domain have `f = 0`). The
sorted samples are grouped into runs with one first term on each grid; each run's products
are summed in sample order, runs in order; the band (half bandwidth `degree`) is factored by
Cholesky, a pivot not above `64*2^-52` of its diagonal raises `std::invalid_argument`
("singular": spans without samples need `ridge > 0`), and all outputs are solved with the
factor. The refit reproduces any edge that is a polynomial of the spline degree on the
samples' range to rounding (Marsden), and any spline when the new knots contain the old
ones and samples cover the domain. Bias and residual branch are unchanged; the layer is
replaced atomically through `set_carrier` (stale gradient shapes are rejected afterwards).

**Network refit.** `Network::refit_grids(input, batch, options)` propagates the network
input forward and refits every B-spline `BasisEdges` layer in order on its own input
activations, computed with the layers before it already refit (pykan's `update_grid`); it
returns the number of refit layers, raises `std::invalid_argument` for a network without
B-spline layers and is atomic (built on a copy). `Network::adapt_grid(index, input,
batch)` propagates the network input to layer `index` and applies `adapt_grid(layer,
activations, batch)`.

**Resident per-input layers.** Per-input knots (and FP32/TF32 knot scales) are reserved,
validated (FP32: knots distinct within each grid stay distinct) and uploaded per grid;
dense and sparse paths and all precisions run them. Knot-reading kernels have per-input
variants (`basis_grid_kernel`, `spline_grid_basis_kernel`, `GridSplineDifferences` in the
backward finish, `spline_grid_difference_kernel`); the sparse-path selection rule and the
arena are those of a shared-grid layer plus `(grids - 1)` knot vectors and knot-scale rows.
Results equal the CPU within the C5/C1 tolerances; identical grids are bitwise the
shared-grid layer. `basis_grid_kernel` stages each block's knot rows in shared memory
(rows with an odd stride; the grid is bounded by multiprocessors times the instantiation's
resident blocks from the occupancy API, and its loop step is a multiple of the inputs, or
it runs one iteration) when stage and knots fit 48 KiB, and evaluates degrees 0..3 at
compile time; dense shared-grid B-spline layers run the same kernel as one grid of every
input (same arithmetic: bitwise the previous results).

**Device refit** (`ResidentNetwork::refit_grids(GridRefit)`). Refits every B-spline layer
from the uploaded batch propagated through the executor's forward pass, as
`Network::refit_grids`, at the current term count (`intervals` 0 or the layer's) and grid
form (`per_input` must match every layer with more than one input; `std::invalid_argument`
otherwise, before anything changes); `std::logic_error` without an uploaded input,
`std::invalid_argument` for batch 0 or a network without B-spline layers. Per layer: the
input transposed to `(inputs, batch)`, a CUB segmented radix sort per input (plus a sort of
all values for a shared grid), the knots per grid (one thread each), the run keys
`(input, first new term, first current term)` run-length encoded by CUB and their offsets
scanned, one warp per run summing its products (lanes over samples in order, a fixed
shuffle tree; one thread per run, in order, for degrees above 3), one thread per input for
the band and its Cholesky factor, one thread per (input, output) for the right-hand side and
the solve, coefficients rounded to the executor precision. Spline values are evaluated in
the executor precision from its samples and knots, products and sums in double. Every sum
has a fixed order: deterministic for one device and build; FP64 results agree with the CPU
refit of the same samples within `1e-11` relative in the parity build and, in the FMA build,
for well-sampled fits (knots within an FMA contraction of the CPU's; for systems regularized
only by the ridge, e.g. fewer samples than terms, that knot rounding is amplified by the
system's conditioning: up to about 1e-8 measured at ridge 1e-8), and FP32/TF32 within their
precision (the samples are the executor's activations). Knots that coincide or leave the
executor's range (an FP32 endpoint) raise "knots coincide or leave the precision's range". Atomic: the
new coefficients go to a copy of the parameters that becomes the active region on success,
knots and knot scales are backed up and restored on failure; each layer's status is
checked after its sort and knots and again after its fit, before the next layer (messages
"the samples of an input span no interval", "knots coincide", "the least-squares system is
singular", naming the layer: `std::invalid_argument`; a nonfinite activation reaching a refit
layer, e.g. from an overflowing input map, raises `std::overflow_error` before any run is
formed). The right-hand sides are solved for as many outputs at a time as fill the device
once (multiprocessors times resident threads per multiprocessor, divided by the inputs), at
most as many as keep the right-hand sides within one sample buffer (`inputs*capacity` values). Success invalidates
output and gradients (input and upstream stay), keeps every captured graph valid (they read
knots, scales and parameters from the arena) and allocates nothing: the workspace (largest
B-spline layer at capacity: transposed and sorted samples, run keys/counts/offsets and
products, bands, right-hand sides, staged knots, backups of every layer's knots, CUB
temporary storage) is reserved after every other region when the network has a B-spline
layer and `inputs*capacity` fits the 32-bit CUB item count (and inputs and terms the
21-bit fields of the run keys); otherwise `refit_grids` raises `std::invalid_argument` naming
the reason. Measured growth of the arena: 7-9% for the networks in the
[M6 evidence](evidence/backlog/M6.md#memory). A change of the term count
still needs a host refit and a new executor. Measured cost and memory:
[M6 evidence](evidence/backlog/M6.md). Python: `kan.BSplineConfig(degree, knots, grids=1)`,
`kan.spline_grid`, `kan.per_input_grids`, `kan.GridRefit(...)`, `kan.insert_knots`,
`kan.adapt_grid(layer, samples)` (vector: a float; `(batch, inputs)`: the knots),
`kan.refit_grid(layer, input, options)`, `Network.refit_grids(input, options)`,
`Network.adapt_grid(index, samples)` (a vector in the layer's domain),
`Network.adapt_grid_from_input(index, input)` (a network input batch, propagated),
`ResidentNetwork.refit_grids(options)`.

## Extension boundaries

Basis, rational and residual-branch formulas have a single source shared by the CPU
backend and the resident CUDA kernels: `KAN_HOST_DEVICE` templates parameterized by a
finiteness guard (CPU throws, device records status). Sources are grouped by execution
properties (backlog R4). The basis family tag and flat views are in
`src/carriers/basis_types.hpp`, the compile-time and runtime family dispatch in
`src/carriers/basis_formulas.hpp`, and the evaluators with their family group:
`src/carriers/polynomial/polynomial_formulas.hpp` (Chebyshev, Legendre, Jacobi, Hermite
and the normalized Hermite functions: global three-term recurrences), `src/carriers/trigonometric/fourier_formulas.hpp`
(Fourier) and `src/carriers/local/local_formulas.hpp` (B-spline, Gaussian RBF, Mexican
hat: local support or decaying tails, with the per-edge trainable wavelets of backlog M7; `spline_span_terms` is the sparse form of the spline
recurrence, and the dense `spline_terms` scatters its result, backlog C5; `spline_lower_terms`,
`spline_derivative_scale` and `spline_derivative_coefficient` give the derivative's spline of one
degree less, backlog C13). Rational formulas are in
`src/carriers/rational/rational_formulas.hpp`; residual-branch formulas in
`src/core/residual_formulas.hpp`. Public declarations and validation live in
`include/kan/basis.hpp`/`src/carriers/basis.cpp` and
`include/kan/rational.hpp`/`src/carriers/rational/rational.cpp`. Wavelet configuration validation and `evaluate_wavelet` are in
`src/carriers/local/wavelet.cpp`, their CPU loops in `src/backends/cpu/wavelet_edges.cpp`. Family operations
(`include/kan/families.hpp`) are in `src/carriers/local/local_operations.cpp` (knots, RBF and
wavelet parameters) and `src/carriers/rational/rational_operations.cpp`. The carrier-independent
Layer protocol, including the SiLU residual branch, lives in `src/core/layer.cpp`;
topology in `src/core/network.cpp`; input maps in `src/core/input_map.cpp` with their
shared host/device formulas in `src/core/input_map_formulas.hpp`; shape and finiteness
checks in `src/core/checks.hpp`. Per-carrier CPU loops and the shared expansion +
contraction engine are in `src/backends/cpu/`. The optional `kan::cuda` target is
`src/backends/cuda/`: persistent kernels in `resident.cu`, with one basis kernel
instantiation per family and the dense contractions delegated to cuBLAS; the device
query in `cuda_runtime.cpp`; the deprecated M1 layer API in `cuda.cu` is a kernel-free
adapter over the resident executor. Initializers are in `src/init/initializers.cpp` with
portable draws and moments in `src/init/`. Private headers are included relative to
`src/` and are not installed; `include/kan/` is the public include tree. A new basis
family adds its public configuration (`include/kan/basis.hpp`), a `BasisKind` entry
(`basis_types.hpp`), its evaluator in the group that matches its execution properties (or
a new group), a dispatch case (`basis_formulas.hpp`), the configuration-to-view mapping
(`basis_view.hpp`) and validation (`basis.cpp`). Dependencies point one way: `core/` and
`carriers/` hold no CPU loops, but the CPU Layer dispatches to `backends/cpu/edge_ops.hpp`
and `evaluate_basis` uses the CPU's throwing guard (`backends/cpu/basis_host.hpp`);
`backends/cuda/` uses `carriers/` and `core/` formulas only. No symbolic parser, Eigen, Torch, Python runtime or
imported KAN implementation is required. Quantum carriers need separate
physical/measurement contracts at M5; their planned location is `src/carriers/quantum/`
(no placeholder directory is kept), with any simulator adapter as a backend under
`src/backends/`.

## Mathematical sources

- [Original KAN paper](https://arxiv.org/abs/2404.19756): learned univariate edge functions.
- [NIST DLMF 18.9](https://dlmf.nist.gov/18.9): polynomial recurrence and derivative conventions.
- [NIST DLMF 3.11](https://dlmf.nist.gov/3.11): rational and Padé approximation definitions.
- [Molina, Schramowski, Kersting: Padé Activation Units (2019)](https://arxiv.org/abs/1907.06732): safe denominator `1+|Σ b_k x^k|`; no code reused.
- [SciPy BSpline mathematical notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html): spline recurrence and partition of unity.
- [Boehm insertion references](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.insert_knot.html): exact spline refinement.
- [Ricker definition](https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.ricker.html): continuous Mexican-hat normalization.
- [Awesome KAN](https://github.com/mintisan/awesome-kan): variant discovery only.
- [Quantum-KAN](https://github.com/wtroy2/Quantum-KAN): referenced by the brief; no code reused.

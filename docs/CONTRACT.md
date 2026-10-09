# KAN numerical contract (M1 through M4, backlog R1-R3, M1, M2, M3 and M4)

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
H_n. Polynomials are evaluated by recurrence, including endpoints and outside [-1,1].
There is no implicit clipping, normalization, tanh, or extrapolation policy; inputs are
brought into a basis domain only by an explicit input map layer (see "Input maps").
Fourier is `[1, cos(w*x), sin(w*x), cos(2*w*x), sin(2*w*x), ...]` and size is odd.
Gaussian RBF is `exp(-((x-center)/width)^2)` with explicit finite centers and width > 0.
Jacobi alpha and beta are finite and greater than -1; angular frequency is finite and positive.
Irrelevant parameters cannot be expressed: each configuration type has only its own.
Localized configurations need at least one term and equal-length parameter vectors.
Each evaluator returns values and derivatives with respect to x.

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
no CUDA dependency and the definition (`src/cuda_runtime.cpp`) is in `kan::cuda`.

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
denominators, SiLU residual-branch weights (backlog M3), and LayerNorm gain and bias. Structure is what the executor was built for and
is never uploaded; it must match exactly, otherwise `std::invalid_argument` names the first
differing layer: the number of layers; per position the kind (KAN layer or input map) and
dimensions; the carrier (`BasisEdges`, `TrainableRbfEdges`, `RationalEdges`) or map kind; whether
a KAN layer has the residual branch ("residual branch presence");
and the fixed configuration, compared with `operator==`: the whole `BasisConfig` of a
`BasisEdges` layer (family, size, Jacobi alpha/beta, Fourier frequency, Gaussian centers and
width, B-spline degree and knots, Mexican-hat centers and scales), the term count of a
trainable RBF, the whole `RationalConfig` (degrees, center, scale, epsilon, denominator
policy), `AffineMap` and `TanhMap` values, the LayerNorm epsilon and whether it has
gain/bias. Spline knots are configuration, not state: `insert_knot`/`adapt_grid` change the
term count, and knots moved at an equal count (`set_carrier`) change the fixed function
space the coefficients refer to; both are rejected, and such a model needs a new executor.
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
`dx = sum_k Phi' (.) W` and the trainable-RBF center/log-width VJPs are reduced. A forward
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
keep the stored rows). The cuBLAS handle is created at construction and runs on the executor's stream with a
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
[C1 evidence](evidence/backlog/C1.md)); trajectories over several SGD steps diverge further
because SGD amplifies rounding. The FP32 small-kernel thresholds are `2^24` forward
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
Samples do not implicitly propagate through preceding layers.

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

`Layer(inputs, outputs, BasisConfig)` selects `TrainableRbfEdges` for a
`TrainableRbfConfig` and `BasisEdges` otherwise; `Layer(inputs, outputs, RationalConfig)`
selects `RationalEdges`. `carrier()` returns the carrier; `terms()` is the number of
coefficients per edge; `coefficients()`/`bias()` and `set_parameters` act on every
carrier's per-edge coefficient tensor (the tensor the coefficient L2 penalizes, together
with the layer's residual-branch weights).
`set_carrier(carrier[, bias])` validates the configuration, the shapes for the layer's
dimensions and finite parameters, then replaces the carrier (and bias) atomically.
Carriers and configurations are values with `operator==`.

`LayerGradients{input, coefficients, bias, nonlinear, residual}`: `nonlinear` is a
`NonlinearGradients` variant whose alternative corresponds to the carrier
(`std::monostate`, `TrainableRbfGradients{centers, log_widths}`,
`RationalGradients{denominators}`). SGD rejects a gradient whose alternative does not
match the carrier with `std::invalid_argument`. `residual` is the residual-branch weight
gradient (backlog M3), empty for a layer without the branch.

Family-specific operations are free functions in `include/kan/families.hpp`
(`insert_knot`, `adapt_grid`, `set_rbf_parameters`, `set_rational_parameters`); each
requires its carrier and basis type, raises `std::invalid_argument` otherwise, and
commits through `set_carrier`.

Every Layer operation dispatches on the carrier once per call. The CPU loops of each
carrier live in `src/carriers/` (`linear_engine.hpp` holds the expansion and
contraction shared by `BasisEdges` and `TrainableRbfEdges`); the resident executor
holds one execution plan per carrier. A new carrier adds a `Carrier` and a
`NonlinearGradients` alternative, its overloads in `src/carriers/edge_ops.hpp` and a
resident plan; Layer and Network do not change.

## Input maps (backlog M1)

Polynomial bases grow like `(2|x|)^n` outside `[-1,1]` and localized bases (B-spline,
RBF, Mexican hat) are zero, with zero gradient, outside their support. Nothing rescales
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
(pinned digests in `tests/initializer_test.cpp`). Draw order: coefficients in layout order,
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
| Fourier | uniform on one period [-π/ω, π/ω] | 1, then 1/2 | (π/ω)²/3 |
| B-spline | uniform on the domain [t_p, t_K] | (p+1)-point Gauss-Legendre per span (exact) | L²/12 |
| Gaussian RBF (fixed, trainable) | uniform on [min c, max c] | 8-point Gauss-Legendre, 36 panels on c ± 9w | L²/12 |
| Mexican hat | uniform on [min c, max c] | 8-point Gauss-Legendre, 36 panels on c ± 12s | L²/12 |

Coincident centers use [c-h, c+h], h the largest width or scale. With independent zero-mean
coefficients `Var_c E_x[y²] = sum_i sum_k sigma_k² m_k` without cross terms, so
`sigma_k² = gain²·variance / (inputs·terms·m_k)` (`m_k = 0` gives 0) gives every term the same
share and `E[y²] = gain²·variance` for inputs drawn from the measure (variance-preserving).
Uniform draws are `sqrt(3)·sigma_k·(2u-1)`, normal draws `sigma_k·z`. Rational edges use z
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

Purpose: B-spline, Gaussian RBF and Mexican-hat carriers are exactly zero, with zero input
gradient, outside their support, so without the branch no gradient crosses a layer whose
inputs left the support (`tests/residual_test.cpp` asserts both zero without and nonzero with
the branch, including the first layer's coefficient gradients behind such a layer).

**Formulas** (single host/device source `src/detail/residual_formulas.hpp`, templated on the
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
the exact `E[y²] = gain²·variance` guarantee, and `w = 0` is not stationary, since
`dw = sum_b u·silu(x)`. pykan's trainable `scale_sp` is not a separate parameter here: it
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

## Extension boundaries

Basis, rational and residual-branch formulas have a single source shared by the CPU
backend and the resident CUDA kernels: `KAN_HOST_DEVICE` templates in
`src/detail/basis_formulas.hpp`, `src/detail/rational_formulas.hpp` and
`src/detail/residual_formulas.hpp`, parameterized by a finiteness guard (CPU throws,
device records status). Public declarations and validation live in
`include/kan/basis.hpp`/`src/basis.cpp` and `include/kan/rational.hpp`/`src/rational.cpp`;
the carrier-independent Layer protocol, including the SiLU residual branch, lives in `src/layer.cpp`, per-carrier CPU
loops in `src/carriers/`, family operations in `src/families.cpp`; initializers in `src/initializers.cpp` with portable draws and moments in `src/init/`; input maps in `src/input_map.cpp`
with their shared host/device formulas in `src/detail/input_map_formulas.hpp`; topology in `src/network.cpp`; persistent
kernels in `src/resident.cu`, with one basis kernel instantiation per family and the
dense contractions delegated to cuBLAS; the device query in `src/cuda_runtime.cpp`; the
deprecated M1 layer API in `src/cuda.cu` is a kernel-free adapter over the resident executor. No symbolic parser, Eigen, Torch,
Python runtime or imported KAN implementation is required. Quantum carriers need
separate physical/measurement contracts at M5.

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
